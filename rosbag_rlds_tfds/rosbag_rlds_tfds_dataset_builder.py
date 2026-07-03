# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

from __future__ import annotations

from typing import Iterator, Tuple, Any, Dict, List
import gc
import itertools
import os
import pickle
import shutil
import tempfile
from pathlib import Path
import importlib.util

import numpy as np
import tensorflow_datasets as tfds
from rosbag_rlds_tfds.conversion_utils import MultiThreadedDatasetBuilder


def _load_convert_module():
    """Load scripts/convert_rosbag_to_rlds.py for shared conversion logic."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "convert_rosbag_to_rlds.py"
    if not module_path.exists():
        raise FileNotFoundError(f"convert script not found: {module_path}")

    spec = importlib.util.spec_from_file_location("convert_rosbag_to_rlds_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CONVERT_MOD = _load_convert_module()
BaseRosbagConverter = _CONVERT_MOD.MultiVideoRosBag2RLDSConverter
STATE_ACTION_DIM = int(_CONVERT_MOD.STATE_ACTION_DIM)
CHASSIS_DIM = int(_CONVERT_MOD.CHASSIS_DIM)
CAMERA_KEYS = ("left_color", "right_color", "head_camera")


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


def _camera_feature() -> tfds.features.Image:
    return tfds.features.Image(
        shape=(224, 224, 3),
        dtype=np.uint8,
        encoding_format="png",
        doc="Camera RGB image resized to 224x224 and encoded as PNG.",
    )


def _camera_calibration_feature() -> tfds.features.FeaturesDict:
    return tfds.features.FeaturesDict(
        {
            "found": tfds.features.Scalar(dtype=np.bool_),
            "K": tfds.features.Tensor(shape=(9,), dtype=np.float64),
            "D": tfds.features.Sequence(tfds.features.Scalar(dtype=np.float64)),
            "P": tfds.features.Tensor(shape=(12,), dtype=np.float64),
            "width": tfds.features.Scalar(dtype=np.int32),
            "height": tfds.features.Scalar(dtype=np.int32),
            "distortion_model": tfds.features.Text(),
        }
    )


def _build_features() -> tfds.features.FeaturesDict:
    image_features = {k: _camera_feature() for k in CAMERA_KEYS}
    camera_meta_features = {k: _camera_calibration_feature() for k in CAMERA_KEYS}
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            "state": tfds.features.Tensor(
                                shape=(STATE_ACTION_DIM,),
                                dtype=np.float32,
                                doc="Robot state vector (62D).",
                            ),
                            "chassis_state": tfds.features.Tensor(
                                shape=(CHASSIS_DIM,),
                                dtype=np.float32,
                                doc="Chassis state vector (9D).",
                            ),
                            **image_features,
                        }
                    ),
                    "action": tfds.features.Tensor(
                        shape=(STATE_ACTION_DIM,),
                        dtype=np.float32,
                        doc="Robot action vector (62D).",
                    ),
                    "chassis_action": tfds.features.Tensor(
                        shape=(CHASSIS_DIM,),
                        dtype=np.float32,
                        doc="Chassis action vector (9D).",
                    ),
                    "timestamp": tfds.features.Scalar(dtype=np.float64),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }
            ),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "episode_idx": tfds.features.Scalar(dtype=np.int64),
                    "n_steps": tfds.features.Scalar(dtype=np.int32),
                    "fps": tfds.features.Scalar(dtype=np.int32),
                    "task": tfds.features.Text(),
                    "source_bag": tfds.features.Text(),
                    "recording_time": tfds.features.Text(),
                    "image_height": tfds.features.Scalar(dtype=np.int32),
                    "image_width": tfds.features.Scalar(dtype=np.int32),
                    "feature_names_state_action": tfds.features.Sequence(tfds.features.Text()),
                    "feature_names_chassis": tfds.features.Sequence(tfds.features.Text()),
                    "camera_keys": tfds.features.Sequence(tfds.features.Text()),
                    "camera_topics": tfds.features.Sequence(tfds.features.Text()),
                    "cameras": tfds.features.FeaturesDict(camera_meta_features),
                }
            ),
        }
    )


class _EpisodeCollector(BaseRosbagConverter):
    """Reuse rosbag->episode logic, but spool episodes to disk to cap RAM usage."""

    def __init__(self, input_directory: str, fps: int):
        super().__init__(input_directory=input_directory, output_directory="/tmp/unused_tfds", fps=fps)
        self._spool_dir = Path(tempfile.mkdtemp(prefix="rosbag_tfds_episode_spool_"))
        self._spooled_episode_files: List[Path] = []

    def save_episode_rlds(
        self,
        episode_index: int,
        episode_frames: list,
        task_description: str,
        source_bag: str = "",
        recording_time: str = "",
    ):
        episode_example, n_steps = self._build_rlds_episode_example(
            episode_index=episode_index,
            episode_frames=episode_frames,
            task_description=task_description,
            source_bag=source_bag,
            recording_time=recording_time,
        )
        self._stats_rlds_total_frames += n_steps
        spool_path = self._spool_dir / f"episode_{episode_index:09d}_{len(self._spooled_episode_files):06d}.pkl"
        with spool_path.open("wb") as f:
            pickle.dump((episode_index, episode_example), f, protocol=pickle.HIGHEST_PROTOCOL)
        self._spooled_episode_files.append(spool_path)
        del episode_example
        gc.collect()
        return True

    def iter_spooled_episodes(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for spool_path in self._spooled_episode_files:
            with spool_path.open("rb") as f:
                episode_index, episode_example = pickle.load(f)
            try:
                yield episode_index, episode_example
            finally:
                del episode_example
                spool_path.unlink(missing_ok=True)
                gc.collect()
        self._spooled_episode_files.clear()

    def cleanup(self):
        shutil.rmtree(self._spool_dir, ignore_errors=True)


def _parse_example(
    rosbag_file: str,
    fps: int,
    task_description: str,
    enforce_video_topics: bool,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Parse one rosbag file into one or multiple RLDS episodes."""
    bag_path = Path(rosbag_file).resolve()
    rosbag = {
        "name": bag_path.parent.name,
        "path": str(bag_path.parent),
        "bag_file": str(bag_path),
    }
    collector = _EpisodeCollector(input_directory=str(bag_path.parent), fps=fps)
    try:
        collector.convert_single_bag(
            rosbag=rosbag,
            task_description=task_description,
            ENFORCE_FOUR_VIDEO_TOPICS_FLAG=enforce_video_topics,
        )

        for episode_index, episode_example in collector.iter_spooled_episodes():
            key = f"{bag_path.stem}_episode_{episode_index:06d}"
            yield key, episode_example
            del episode_example
            gc.collect()
    finally:
        collector.cleanup()


# Module-level episode counter: _generate_examples is invoked once per path
# chunk, so numbering must live at module scope to stay monotonic across
# chunks and bags within one build.
_EPISODE_IDX_COUNTER = itertools.count()


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of rosbag files."""
    task_description = os.environ.get("ROSBAG_TASK", "task description")
    fps = int(os.environ.get("ROSBAG_FPS", "30"))
    enforce_video_topics = _str2bool(
        os.environ.get("ROSBAG_ENFORCE_VIDEO_TOPICS", "0"),
        default=False,
    )

    for rosbag_file in paths:
        for key, example in _parse_example(
            rosbag_file=rosbag_file,
            fps=fps,
            task_description=task_description,
            enforce_video_topics=enforce_video_topics,
        ):
            episode_metadata = example.get("episode_metadata")
            if isinstance(episode_metadata, dict) and "episode_idx" in episode_metadata:
                episode_metadata["episode_idx"] = next(_EPISODE_IDX_COUNTER)
            yield key, example


class RosbagRldsTfds(MultiThreadedDatasetBuilder):
    """TFDS dataset builder: convert ROS bag directly to RLDS during tfds build."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial direct rosbag-to-RLDS TFDS conversion.",
    }

    N_WORKERS = _env_int("ROSBAG_TFDS_N_WORKERS", 4)   # memory-safe default, can override by env
    MAX_PATHS_IN_MEMORY = _env_int(
        "ROSBAG_TFDS_MAX_PATHS_IN_MEMORY",
        8,
    )  # conservative default to avoid RAM spikes on large rosbags
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes


    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(features=_build_features())

    # def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    #     del dl_manager
    #     split_paths = self._split_paths()
    #     return {
    #         split_name: _generate_examples(paths=paths)
    #         for split_name, paths in split_paths.items()
    #     }

    def _generate_examples(self, paths):
        # Required by GeneratorBasedBuilder abstract interface.
        yield from _generate_examples(paths=paths)

    def _split_paths(self):
        """Discover rosbags like `discover_rosbags` and put all files into train split."""
        input_directory = Path(
            os.environ.get("ROSBAG_ROOT", "./rosbag")
        ).expanduser()
        multibag_flag = _str2bool(
            os.environ.get("ROSBAG_MULTIBAG", "1"),
            default=True,
        )

        rosbags: List[Dict[str, str]] = []

        if multibag_flag is True:
            # multi rosbag
            rosbag_folders: List[str] = []
            root_path = input_directory
            root = Path(root_path).resolve()

            if not root.exists():
                raise ValueError(f"Path does not exist: {root_path}")
            if not root.is_dir():
                raise ValueError(f"Path is not a directory: {root_path}")

            for dirpath, dirnames, filenames in os.walk(root):
                current_dir = Path(dirpath)
                file_extensions = {Path(f).suffix.lower() for f in filenames}
                has_bag = ".db3" in file_extensions or ".mcap" in file_extensions
                has_yaml = ".yaml" in file_extensions or ".yml" in file_extensions

                if has_bag and has_yaml:
                    rosbag_folders.append(str(current_dir))
                    dirnames.clear()

            if rosbag_folders:
                # One entry per bag directory: _parse_example opens the bag
                # directory, which replays every storage file listed in
                # metadata.yaml, so split bags must not be enumerated per file.
                for rosbag_dir in sorted(rosbag_folders):
                    rosbag_dir = Path(rosbag_dir)
                    db3_files = sorted(rosbag_dir.glob("*.db3"))
                    mcap_files = sorted(rosbag_dir.glob("*.mcap"))
                    if db3_files and mcap_files:
                        print(f"Warning: {rosbag_dir} contains both .db3 and .mcap files; "
                              f"using .db3 and ignoring .mcap")
                    bag_files = db3_files or mcap_files
                    if bag_files:
                        rosbags.append(
                            {
                                "name": rosbag_dir.name,
                                "path": str(rosbag_dir),
                                "bag_file": str(bag_files[0]),
                            }
                        )
        else:
            # one rosbag
            root = Path(input_directory).resolve()
            if root.is_file() and root.suffix.lower() in {".db3", ".mcap"}:
                rosbags.append(
                    {
                        "name": f"episode_{0:03d}",
                        "path": str(root.parent),
                        "bag_file": str(root),
                    }
                )
            elif root.is_dir():
                # The directory holds a single bag (one metadata.yaml), possibly
                # split into several storage files.
                db3_files = sorted(root.glob("*.db3"))
                mcap_files = sorted(root.glob("*.mcap"))
                if db3_files and mcap_files:
                    print(f"Warning: {root} contains both .db3 and .mcap files; "
                          f"using .db3 and ignoring .mcap")
                bag_files = db3_files or mcap_files
                if bag_files:
                    rosbags.append(
                        {
                            "name": "episode_000",
                            "path": str(bag_files[0].parent),
                            "bag_file": str(bag_files[0]),
                        }
                    )
            else:
                raise ValueError(f"Path is neither rosbag file nor directory: {root}")

        # Deduplicate by bag directory (one bag = one directory, whatever the
        # number of storage files) while keeping deterministic order.
        seen = set()
        train_files: List[str] = []
        for rosbag in rosbags:
            bag_dir = str(Path(rosbag["bag_file"]).resolve().parent)
            if bag_dir not in seen:
                seen.add(bag_dir)
                train_files.append(rosbag["bag_file"])

        print(f"Discovered {len(train_files)} bag files for train split:")
        for bag_file in train_files:
            print(f"  - {bag_file}")

        if not train_files:
            raise FileNotFoundError(
                f"No rosbag files found under: {input_directory}. "
                "Expected bag directories containing metadata(.yaml/.yml) and .db3/.mcap files."
            )

        # Per request: put all discovered bags into train only.
        return {"train": train_files}
