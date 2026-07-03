#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Split a LeRobot dataset sequentially into train and test subsets.

The first ``--num-train-episodes`` episodes (sorted by ``episode_index``) become
the train split; the remainder become the test split. Each split is written as a
self-contained dataset alongside the source: ``<dataset>_train`` and
``<dataset>_test``. Episodes are renumbered from 0 in each split, parquet
``index``/``episode_index`` columns are rewritten, and meta files
(``info.json``, ``episodes.jsonl``, ``episodes_stats.jsonl``, ``tasks.jsonl``)
are regenerated. ``episodes_stats.jsonl`` reuses the source per-episode stats
(values are unchanged when episodes are partitioned) with updated indices.

Usage example:
    python scripts/split_train_test.py --dataset /path/to/dataset --num-train-episodes 80
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_CHUNK_SIZE = 1000


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def detect_video_keys(info: dict) -> tuple[str | None, list[str]]:
    template = info.get("video_path")
    keys: list[str] = []
    for name, feature in info.get("features", {}).items():
        if feature.get("dtype") == "video":
            keys.append(name)
    return template, keys


def find_source_parquet(dataset: Path, episode_index: int, chunk_size: int) -> Path | None:
    candidate = (
        dataset / "data" / f"chunk-{episode_index // chunk_size:03d}" / f"episode_{episode_index:06d}.parquet"
    )
    if candidate.exists():
        return candidate
    for path in dataset.rglob(f"episode_{episode_index:06d}.parquet"):
        return path
    return None


def find_source_video(
    dataset: Path,
    video_template: str,
    video_key: str,
    episode_index: int,
    chunk_size: int,
) -> Path | None:
    episode_chunk = episode_index // chunk_size
    candidates = [
        dataset / video_template.format(
            episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index
        ),
        dataset / video_template.format(
            episode_chunk=0, video_key=video_key, episode_index=episode_index
        ),
        dataset / "videos" / f"chunk-{episode_chunk:03d}" / video_key / f"episode_{episode_index}.mp4",
        dataset / "videos" / f"chunk-{episode_chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    videos_root = dataset / "videos"
    if videos_root.exists():
        for path in videos_root.rglob(f"episode_{episode_index:06d}.mp4"):
            if video_key in path.parts:
                return path
    return None


def build_dest_parquet(
    dataset: Path, data_template: str | None, episode_index: int, chunk_size: int
) -> Path:
    if data_template:
        path = dataset / data_template.format(
            episode_chunk=episode_index // chunk_size, episode_index=episode_index
        )
    else:
        path = (
            dataset / "data" / f"chunk-{episode_index // chunk_size:03d}" / f"episode_{episode_index:06d}.parquet"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_dest_video(
    dataset: Path, video_template: str, video_key: str, episode_index: int, chunk_size: int
) -> Path:
    path = dataset / video_template.format(
        episode_chunk=episode_index // chunk_size,
        video_key=video_key,
        episode_index=episode_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def copy_episode(
    src_dataset: Path,
    dst_dataset: Path,
    old_index: int,
    new_index: int,
    new_start_frame: int,
    chunk_size: int,
    data_template: str | None,
    video_template: str | None,
    video_keys: list[str],
) -> None:
    src_parquet = find_source_parquet(src_dataset, old_index, chunk_size)
    if src_parquet is None:
        raise FileNotFoundError(f"Parquet not found for episode {old_index}")

    df = pd.read_parquet(src_parquet)
    if "episode_index" in df.columns:
        df["episode_index"] = new_index
    if "index" in df.columns:
        df["index"] = [new_start_frame + i for i in range(len(df))]

    dst_parquet = build_dest_parquet(dst_dataset, data_template, new_index, chunk_size)
    df.to_parquet(dst_parquet, index=False)

    if not (video_template and video_keys):
        return

    for video_key in video_keys:
        src_video = find_source_video(src_dataset, video_template, video_key, old_index, chunk_size)
        if src_video is None:
            print(f"[Warning] Video not found: {video_key}, episode {old_index}, skipped.")
            continue
        dst_video = build_dest_video(dst_dataset, video_template, video_key, new_index, chunk_size)
        shutil.copy2(str(src_video), str(dst_video))


def write_split(
    src_dataset: Path,
    dst_dataset: Path,
    split_episodes: list[dict],
    src_stats_map: dict[int, dict],
    info: dict,
    chunk_size: int,
    data_template: str | None,
    video_template: str | None,
    video_keys: list[str],
    split_name: str,
) -> tuple[int, int]:
    new_episodes: list[dict] = []
    new_stats: list[dict] = []
    cum_frames = 0

    for new_index, episode in enumerate(split_episodes):
        old_index = episode["episode_index"]
        ep_copy = episode.copy()
        ep_copy["episode_index"] = new_index
        new_episodes.append(ep_copy)

        copy_episode(
            src_dataset,
            dst_dataset,
            old_index,
            new_index,
            cum_frames,
            chunk_size,
            data_template,
            video_template,
            video_keys,
        )

        if old_index in src_stats_map:
            stat = src_stats_map[old_index].copy()
            stat["episode_index"] = new_index
            new_stats.append(stat)

        cum_frames += episode.get("length", 0)

    n_episodes = len(new_episodes)
    total_frames = cum_frames

    meta_dir = dst_dataset / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(new_episodes, meta_dir / "episodes.jsonl")
    save_jsonl(new_stats, meta_dir / "episodes_stats.jsonl")

    src_tasks = src_dataset / "meta" / "tasks.jsonl"
    if src_tasks.exists():
        shutil.copy2(str(src_tasks), str(meta_dir / "tasks.jsonl"))

    new_info = json.loads(json.dumps(info))
    new_info["total_episodes"] = n_episodes
    new_info["total_frames"] = total_frames
    new_info["total_chunks"] = math.ceil(n_episodes / chunk_size) if chunk_size else 0
    new_info["splits"] = {split_name: f"0:{n_episodes}"}
    if video_keys:
        new_info["total_videos"] = n_episodes * len(video_keys)
    elif "total_videos" in new_info:
        new_info["total_videos"] = 0

    with (meta_dir / "info.json").open("w", encoding="utf-8") as fh:
        json.dump(new_info, fh, indent=4, ensure_ascii=False)

    return n_episodes, total_frames


def split_dataset(dataset_path: Path, num_train_episodes: int) -> None:
    dataset = dataset_path.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    meta_dir = dataset / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    data_template = info.get("data_path")
    video_template, video_keys = detect_video_keys(info)

    episodes = load_jsonl(meta_dir / "episodes.jsonl")
    if not episodes:
        raise RuntimeError("episodes.jsonl is empty or does not exist.")
    episodes = sorted(episodes, key=lambda x: x["episode_index"])

    stats_records = load_jsonl(meta_dir / "episodes_stats.jsonl")
    src_stats_map = {r["episode_index"]: r for r in stats_records if "episode_index" in r}

    total = len(episodes)
    train_dst = dataset.parent / f"{dataset.name}_train"
    test_dst = dataset.parent / f"{dataset.name}_test"

    if num_train_episodes >= total:
        print(
            f"[Info] Total episodes ({total}) <= --num-train-episodes ({num_train_episodes})."
            f" All episodes go to train; no test split will be created."
        )
        train_eps = episodes
        test_eps: list[dict] = []
    else:
        train_eps = episodes[:num_train_episodes]
        test_eps = episodes[num_train_episodes:]

    if train_dst.exists():
        raise FileExistsError(f"Train output directory already exists: {train_dst}")
    if test_eps and test_dst.exists():
        raise FileExistsError(f"Test output directory already exists: {test_dst}")

    print(f"[Info] Splitting: {len(train_eps)} train / {len(test_eps)} test (total {total}).")

    train_dst.mkdir(parents=True)
    n_tr, f_tr = write_split(
        dataset, train_dst, train_eps, src_stats_map, info,
        chunk_size, data_template, video_template, video_keys, "train",
    )
    print(f"[Complete] Train: {n_tr} episodes, {f_tr} frames -> {train_dst}")

    if test_eps:
        test_dst.mkdir(parents=True)
        n_te, f_te = write_split(
            dataset, test_dst, test_eps, src_stats_map, info,
            chunk_size, data_template, video_template, video_keys, "test",
        )
        print(f"[Complete] Test:  {n_te} episodes, {f_te} frames -> {test_dst}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a LeRobot dataset sequentially into train/test subsets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Source dataset root (containing meta/, data/, optionally videos/).",
    )
    parser.add_argument(
        "--num-train-episodes",
        required=True,
        type=int,
        help="Number of episodes assigned to the train split. Remaining episodes go to test."
        " If >= total, all episodes go to train and no test split is created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_train_episodes <= 0:
        raise ValueError("--num-train-episodes must be positive")
    split_dataset(args.dataset, args.num_train_episodes)


if __name__ == "__main__":
    main()
