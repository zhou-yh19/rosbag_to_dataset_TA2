#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Delete specified episodes from a LeRobot dataset and reindex remaining episodes.

Usage example:
    python scripts/delete_wrong_episodes.py --dataset /path/to/dataset --episode-indices 2 5 10
"""
import argparse
import contextlib
import json
import math
import os
import shutil
from pathlib import Path

import pandas as pd

# Constants
DEFAULT_CHUNK_SIZE = 1000

def load_jsonl(path: Path) -> list[dict]:
    """Read JSONL file and return a list of dictionaries."""
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
    """Write a list of dictionaries back to a JSONL file."""
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def find_parquet_path(dataset: Path, episode_index: int, chunk_size: int) -> Path | None:
    """Infer the parquet file location based on episode index, search recursively if not found."""
    candidates = [
        dataset / "data" / f"chunk-{episode_index // chunk_size:03d}" / f"episode_{episode_index:06d}.parquet",
        dataset / "parquet" / f"episode_{episode_index:06d}.parquet",
        dataset / "data" / f"episode_{episode_index:06d}.parquet",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: recursive search
    with contextlib.suppress(FileNotFoundError):
        for path in dataset.rglob(f"episode_{episode_index:06d}.parquet"):
            return path
    return None


def build_parquet_path(dataset: Path, episode_index: int, chunk_size: int) -> Path:
    """Construct the target parquet path corresponding to the given episode index."""
    chunk_dir = dataset / "data" / f"chunk-{episode_index // chunk_size:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return chunk_dir / f"episode_{episode_index:06d}.parquet"


def detect_video_keys(info: dict) -> tuple[str | None, list[str]]:
    """Extract video path template and corresponding feature key list from info.json."""
    template = info.get("video_path")
    video_keys: list[str] = []

    features = info.get("features", {})
    for name, feature in features.items():
        if feature.get("dtype") == "video":
            video_keys.append(name)

    return template, video_keys


def find_video_path(
    dataset: Path,
    video_template: str,
    video_key: str,
    episode_index: int,
    chunk_size: int,
) -> Path | None:
    """Attempt to locate video file based on template and multiple fallback patterns."""
    episode_chunk = episode_index // chunk_size
    candidates = [
        dataset
        / video_template.format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        ),
        dataset
        / video_template.format(
            episode_chunk=0,
            video_key=video_key,
            episode_index=episode_index,
        ),
        dataset / "videos" / f"chunk-{episode_chunk:03d}" / video_key / f"episode_{episode_index}.mp4",
        dataset / "videos" / f"chunk-{episode_chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    videos_root = dataset / "videos"
    if videos_root.exists():
        for root, _, files in os.walk(videos_root):
            for file in files:
                if file.endswith(".mp4") and video_key in Path(root).parts and f"{episode_index:06d}" in file:
                    return Path(root) / file
    return None


def build_video_path(
    dataset: Path,
    video_template: str,
    video_key: str,
    episode_index: int,
    chunk_size: int,
) -> Path:
    """Construct the new path for the video file."""
    episode_chunk = episode_index // chunk_size
    target_path = dataset / video_template.format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def rewrite_parquet(
    dataset: Path,
    old_index: int,
    new_index: int,
    start_frame: int,
    chunk_size: int,
) -> None:
    """Read old parquet, update indices and write to new location."""
    source_path = find_parquet_path(dataset, old_index, chunk_size)
    if source_path is None:
        print(f"[Warning] Could not find parquet file for episode {old_index}, skipped.")
        return

    df = pd.read_parquet(source_path)
    if "episode_index" in df.columns:
        df["episode_index"] = new_index
    if "index" in df.columns:
        df["index"] = [start_frame + i for i in range(len(df))]

    destination_path = build_parquet_path(dataset, new_index, chunk_size)
    df.to_parquet(destination_path, index=False)

    if destination_path != source_path and source_path.exists():
        source_path.unlink()
        # Clean up empty directories
        with contextlib.suppress(OSError):
            source_path.parent.rmdir()


def move_videos(
    dataset: Path,
    video_template: str | None,
    video_keys: list[str],
    old_index: int,
    new_index: int,
    chunk_size: int,
) -> None:
    """Move/rewrite video files based on index mapping."""
    if not video_template or not video_keys:
        return

    for video_key in video_keys:
        source_path = find_video_path(dataset, video_template, video_key, old_index, chunk_size)
        if source_path is None:
            print(f"[Info] Video file not found: {video_key}, episode {old_index}, skipped.")
            continue

        destination_path = build_video_path(dataset, video_template, video_key, new_index, chunk_size)
        if destination_path == source_path:
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))

        # Delete old empty directories
        with contextlib.suppress(OSError):
            source_path.parent.rmdir()


def remove_target_files(
    dataset: Path,
    target_index: int,
    chunk_size: int,
    video_template: str | None,
    video_keys: list[str],
) -> None:
    """Delete parquet and video files corresponding to the specified episode."""
    path = find_parquet_path(dataset, target_index, chunk_size)
    if path and path.exists():
        path.unlink()
        with contextlib.suppress(OSError):
            path.parent.rmdir()

    if video_template and video_keys:
        for video_key in video_keys:
            video_path = find_video_path(dataset, video_template, video_key, target_index, chunk_size)
            if video_path and video_path.exists():
                video_path.unlink()
                with contextlib.suppress(OSError):
                    video_path.parent.rmdir()


def compute_frame_starts(episodes: list[dict]) -> tuple[dict[int, int], int]:
    """Calculate the starting frame index and total frame count for each episode based on the episodes list."""
    starts: dict[int, int] = {}
    running = 0

    for episode in sorted(episodes, key=lambda x: x["episode_index"]):
        starts[episode["episode_index"]] = running
        running += episode.get("length", 0)
    return starts, running


def update_info(
    info_path: Path,
    info: dict,
    episodes: list[dict],
    total_frames: int,
    chunk_size: int,
    video_keys: list[str],
) -> None:
    """Update core statistics in info.json."""
    info["total_episodes"] = len(episodes)
    info["total_frames"] = total_frames
    info["total_chunks"] = math.ceil(len(episodes) / chunk_size) if chunk_size else 0
    info["splits"] = {"train": f"0:{len(episodes)}"}

    if video_keys:
        info["total_videos"] = len(episodes) * len(video_keys)

    with info_path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=4, ensure_ascii=False)


def remove_episodes(dataset: Path, episode_indices: list[int]) -> None:
    """Delete multiple specified episodes, then reorder indices."""
    meta_dir = dataset / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"

    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    video_template, video_keys = detect_video_keys(info)

    episodes = load_jsonl(episodes_path)
    if not episodes:
        raise RuntimeError("episodes.jsonl is empty or does not exist.")

    # Convert to set for fast lookup
    indices_to_remove = set(episode_indices)

    # Verify that all episodes to be deleted exist
    existing_indices = {episode.get("episode_index") for episode in episodes}
    missing_indices = indices_to_remove - existing_indices
    if missing_indices:
        raise ValueError(f"The following episode_index not found in episodes.jsonl: {sorted(missing_indices)}")

    print(f"[Info] Deleting {len(indices_to_remove)} episodes: {sorted(indices_to_remove)}")

    # Build new episodes list and index mapping
    # Retained episodes are renumbered in original order as 0, 1, 2, ...
    new_episodes: list[dict] = []
    index_mapping: dict[int, int] = {}
    new_index = 0

    for episode in sorted(episodes, key=lambda x: x["episode_index"]):
        old_idx = episode["episode_index"]
        if old_idx in indices_to_remove:
            continue  # Skip episodes to be deleted

        # Renumber: consecutive numbering starting from 0
        episode = episode.copy()
        episode["episode_index"] = new_index
        new_episodes.append(episode)
        index_mapping[old_idx] = new_index
        new_index += 1

    print(f"[Info] After deletion, {len(new_episodes)} episodes will be retained, renumbered from 0 to {len(new_episodes) - 1}")

    # Save updated episodes.jsonl
    save_jsonl(new_episodes, episodes_path)

    # Update episodes_stats.jsonl
    stats_records = load_jsonl(stats_path)
    if stats_records:
        new_stats: list[dict] = []
        for record in stats_records:
            idx = record.get("episode_index")
            if idx in indices_to_remove:
                continue  # Skip statistics for episodes to be deleted
            if idx in index_mapping:
                record = record.copy()
                record["episode_index"] = index_mapping[idx]
            new_stats.append(record)
        new_stats.sort(key=lambda x: x["episode_index"])
        save_jsonl(new_stats, stats_path)

    # Delete all files corresponding to target episodes
    for target_index in indices_to_remove:
        remove_target_files(dataset, target_index, chunk_size, video_template, video_keys)

    # Recalculate frame start indices
    frame_starts, total_frames = compute_frame_starts(new_episodes)

    # Rewrite parquet / video files (those that need to be moved)
    for old_index, new_index in sorted(index_mapping.items()):
        start = frame_starts[new_index]
        # If index has changed, need to rewrite parquet and move video files
        if old_index != new_index:
            rewrite_parquet(dataset, old_index, new_index, start, chunk_size)
            move_videos(dataset, video_template, video_keys, old_index, new_index, chunk_size)
        # If index hasn't changed (old_index == new_index), this episode was before the deleted episodes
        # Frame start position unchanged, no need to rewrite parquet and move files

    # Update info.json
    update_info(info_path, info, new_episodes, total_frames, chunk_size, video_keys)

    print(
        f"[Complete] Deleted {len(indices_to_remove)} episodes: {sorted(indices_to_remove)}."
        f" Current episode count: {len(new_episodes)}, frames: {total_frames}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete specified episodes in the dataset, then reorder indices.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Dataset root directory path (containing meta, data, videos subdirectories).",
    )
    parser.add_argument(
        "--episode-indices",
        required=True,
        nargs="+",
        type=int,
        help="List of episode_index to delete (0-based), multiple can be specified, e.g.: --episode-indices 2 5 10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    # Deduplicate and sort
    episode_indices = sorted(set(args.episode_indices))
    if not episode_indices:
        raise ValueError("Must specify at least one episode_index to delete")

    remove_episodes(dataset, episode_indices)


if __name__ == "__main__":
    main()
