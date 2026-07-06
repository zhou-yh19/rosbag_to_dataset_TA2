# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

from typing import Tuple, Any, Dict, Union, Callable, Iterable
import gc
import math
import os
import tensorflow_datasets as tfds

from tensorflow_datasets.core import download
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import utils
from tensorflow_datasets.core import writer as writer_lib
from tensorflow_datasets.core import example_serializer
from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import file_adapters

Key = Union[str, int]
# The nested example dict passed to `features.encode_example`
Example = Dict[str, Any]
KeyExample = Tuple[Key, Example]


def _str2bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(min_value, value)


class MultiThreadedDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """Streaming rosbag -> TFDS builder.

    Note: despite the class name (kept from the upstream RLDS builder template),
    example generation runs sequentially in a single thread. N_WORKERS and
    MAX_PATHS_IN_MEMORY only control how bag paths are chunked to bound memory
    usage; they do not parallelize generation.
    """
    N_WORKERS = 4                  # memory-safe default; can be overridden by env
    MAX_PATHS_IN_MEMORY = 8        # memory-safe default; can be overridden by env
    PARSE_FCN = None               # needs to be filled with path-to-record-episode parse function

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        split_paths = self._split_paths()
        return {split: type(self).PARSE_FCN(paths=split_paths[split]) for split in split_paths}

    def _generate_examples(self):
        pass  # this is implemented in global method to enable multiprocessing

    def _download_and_prepare(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
            self,
            dl_manager: download.DownloadManager,
            download_config: download.DownloadConfig,
    ) -> None:
        """Generate all splits and returns the computed split infos."""
        assert self.PARSE_FCN is not None       # need to overwrite parse function
        n_workers = _env_int("ROSBAG_TFDS_N_WORKERS", self.N_WORKERS)
        max_paths_in_memory = _env_int(
            "ROSBAG_TFDS_MAX_PATHS_IN_MEMORY",
            self.MAX_PATHS_IN_MEMORY,
        )
        force_disable_shuffling = _str2bool(
            os.environ.get("ROSBAG_TFDS_DISABLE_SHUFFLING", "1"),
            default=True,
        )

        # TFDS>=4.9 expects an ExampleWriter object at SplitBuilder construction.
        example_writer = writer_lib.ExampleWriter(file_format=self.info.file_format)
        split_builder = ParallelSplitBuilder(
            split_dict=self.info.splits,
            features=self.info.features,
            dataset_size=self.info.dataset_size,
            max_examples_per_split=download_config.max_examples_per_split,
            beam_options=download_config.beam_options,
            beam_runner=download_config.beam_runner,
            example_writer=example_writer,
            shard_config=download_config.get_shard_config(),
            split_paths=self._split_paths(),
            parse_function=type(self).PARSE_FCN,
            n_workers=n_workers,
            max_paths_in_memory=max_paths_in_memory,
        )
        split_generators = self._split_generators(dl_manager)
        split_generators = split_builder.normalize_legacy_split_generators(
            split_generators=split_generators,
            generator_fn=self._generate_examples,
            is_beam=False,
        )
        dataset_builder._check_split_names(split_generators.keys())

        # Start generating data for all splits
        path_suffix = file_adapters.ADAPTER_FOR_FORMAT[
            self.info.file_format
        ].FILE_SUFFIX

        split_info_futures = []
        for split_name, generator in utils.tqdm(
                split_generators.items(),
                desc="Generating splits...",
                unit=" splits",
                leave=False,
        ):
            filename_template = naming.ShardedFileTemplate(
                split=split_name,
                dataset_name=self.name,
                data_dir=self.data_path,
                filetype_suffix=path_suffix,
            )
            future = split_builder.submit_split_generation(
                split_name=split_name,
                generator=generator,
                filename_template=filename_template,
                disable_shuffling=(self.info.disable_shuffling or force_disable_shuffling),
                nondeterministic_order=download_config.nondeterministic_order,
            )
            split_info_futures.append(future)

        # Finalize the splits (after apache beam completed, if it was used)
        split_infos = [future.result() for future in split_info_futures]

        # Update the info object with the splits.
        split_dict = splits_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)


class _SplitInfoFuture:
    """Future containing the `tfds.core.SplitInfo` result."""

    def __init__(self, callback: Callable[[], splits_lib.SplitInfo]):
        self._callback = callback

    def result(self) -> splits_lib.SplitInfo:
        return self._callback()


def iter_encoded_examples_from_generator(paths, fcn, split_name, total_num_examples, features):
    """Yield encoded examples one-by-one to keep memory usage bounded."""
    generator = fcn(paths)
    for idx, sample in enumerate(
        utils.tqdm(
            generator,
            desc=f'Generating {split_name} examples...',
            unit=' examples',
            total=total_num_examples,
            leave=False,
            mininterval=1.0,
        ),
        start=1,
    ):
        if sample is None:
            continue
        key, example = sample
        try:
            example = features.encode_example(example)
        except Exception as e:  # pylint: disable=broad-except
            utils.reraise(e, prefix=f'Failed to encode example:\n{example}\n')
        yield key, example
        if idx % 8 == 0:
            gc.collect()


class ParallelSplitBuilder(split_builder_lib.SplitBuilder):
    def __init__(self, *args, split_paths, parse_function, n_workers, max_paths_in_memory, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_paths = split_paths
        self._parse_function = parse_function
        self._n_workers = n_workers
        self._max_paths_in_memory = max_paths_in_memory

    def _build_from_generator(
            self,
            split_name: str,
            generator: Iterable[KeyExample],
            filename_template: naming.ShardedFileTemplate,
            disable_shuffling: bool,
    ) -> _SplitInfoFuture:
        """Split generator for example generators.

        Args:
          split_name: str,
          generator: Iterable[KeyExample],
          filename_template: Template to format the filename for a shard.
          disable_shuffling: Specifies whether to shuffle the examples,

        Returns:
          future: The future containing the `tfds.core.SplitInfo`.
        """
        total_num_examples = None
        serialized_info = self._features.get_serialized_info()
        serializer = example_serializer.ExampleSerializer(serialized_info)
        writer = writer_lib.Writer(
            serializer=serializer,
            filename_template=filename_template,
            hash_salt=split_name,
            disable_shuffling=disable_shuffling,
            example_writer=self._example_writer,
            shard_config=self._shard_config,
        )

        del generator  # regenerated below from paths, chunked to bound memory
        paths = self._split_paths[split_name]
        total_chunks = max(1, math.ceil(len(paths) / max(1, self._max_paths_in_memory)))
        example_idx = 0
        max_examples = getattr(self, "_max_examples_per_split", None)
        reached_limit = False
        print("Generating with streaming writer (no in-memory output accumulation).")
        for i, path_groups in enumerate(
            chunk_max(paths, self._n_workers, self._max_paths_in_memory),
            start=1,
        ):
            print(f"Processing chunk {i} of {total_chunks}.")
            for worker_paths in path_groups:
                if not worker_paths:
                    continue
                for key, encoded_example in iter_encoded_examples_from_generator(
                    paths=worker_paths,
                    fcn=self._parse_function,
                    split_name=split_name,
                    total_num_examples=total_num_examples,
                    features=self._features,
                ):
                    if disable_shuffling and not isinstance(key, int):
                        key = example_idx
                    writer.write(key, encoded_example)
                    example_idx += 1
                    if max_examples is not None and example_idx >= max_examples:
                        reached_limit = True
                        break
                gc.collect()
                if reached_limit:
                    break
            if reached_limit:
                print(f"Reached max_examples_per_split={max_examples}; stopping generation early.")
                break

        print("Finishing split conversion...")
        shard_lengths, total_size = writer.finalize()

        split_info = splits_lib.SplitInfo(
            name=split_name,
            shard_lengths=shard_lengths,
            num_bytes=total_size,
            filename_template=filename_template,
        )
        return _SplitInfoFuture(lambda: split_info)


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL, t)) for t in zip(*DL.values())]

def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si:si + (d + 1 if i < r else d)]

def chunk_max(l, n, max_chunk_sum):
    max_chunk_sum = max(1, int(max_chunk_sum))
    for start in range(0, len(l), max_chunk_sum):
        yield list(chunks(l[start:start + max_chunk_sum], n))
