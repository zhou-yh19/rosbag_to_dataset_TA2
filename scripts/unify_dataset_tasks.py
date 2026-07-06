#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Unify all tasks of a LeRobot dataset into a single task.

Rewrites:
  meta/tasks.jsonl              -> one row, task_index = 0
  meta/info.json                -> total_tasks = 1
  meta/episodes.jsonl           -> every episode's tasks = [unified_task]
  meta/episodes_stats.jsonl     -> stats.task_index zeroed
  data/chunk-*/episode_*.parquet -> task_index column set to 0

Other files (norm_stats.json, videos/) are copied as-is.

Usage:
    python scripts/unify_dataset_tasks.py \
        --input  /model/dataset_test/rolling_coffee_can \
        --output /model/dataset_test/rolling_coffee_can_single_task \
        --task   "catch rolling coffee can"
"""
import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def zero_task_index_stats(stat: dict, frame_count: int) -> dict:
    """Reset a task_index stats dict so min=max=mean=0, std=0, count=N."""
    new_stat = dict(stat)
    new_stat["min"] = [0.0]
    new_stat["max"] = [0.0]
    new_stat["mean"] = [0.0]
    new_stat["std"] = [0.0]
    if "count" in new_stat:
        new_stat["count"] = [frame_count] if isinstance(new_stat["count"], list) else frame_count
    return new_stat


def rewrite_parquet(src: Path, dst: Path) -> None:
    table = pq.read_table(src)
    if "task_index" in table.column_names:
        idx = table.column_names.index("task_index")
        col = table.column("task_index")
        zeros = pa.array([0] * len(col), type=col.type)
        table = table.set_column(idx, "task_index", zeros)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)


def copy_tree_excluding(src_root: Path, dst_root: Path, exclude: set[Path]) -> None:
    """Copy everything under src_root to dst_root except paths in `exclude`."""
    for src in src_root.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(src_root)
        if rel in exclude:
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Source LeRobot dataset directory")
    parser.add_argument("--output", required=True, type=Path, help="Destination directory (must not exist or be empty)")
    parser.add_argument("--task", required=True, help="Unified task prompt to assign to every episode")
    args = parser.parse_args()

    src: Path = args.input
    dst: Path = args.output
    unified_task: str = args.task

    if not src.is_dir():
        raise SystemExit(f"Input directory not found: {src}")
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"Output directory already exists and is not empty: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    meta_files = {
        Path("meta/tasks.jsonl"),
        Path("meta/info.json"),
        Path("meta/episodes.jsonl"),
        Path("meta/episodes_stats.jsonl"),
    }
    parquet_files = {p.relative_to(src) for p in (src / "data").rglob("*.parquet")} if (src / "data").exists() else set()

    print(f"[copy ] static files (videos, norm_stats.json, ...) -> {dst}")
    copy_tree_excluding(src, dst, exclude=meta_files | parquet_files)

    # 1. tasks.jsonl
    tasks_path = dst / "meta/tasks.jsonl"
    write_jsonl([{"task_index": 0, "task": unified_task}], tasks_path)
    print(f"[write] {tasks_path} -> 1 task")

    # 2. info.json
    info = json.loads((src / "meta/info.json").read_text())
    info["total_tasks"] = 1
    info_path = dst / "meta/info.json"
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False))
    print(f"[write] {info_path} -> total_tasks=1")

    # 3. episodes.jsonl
    episodes = read_jsonl(src / "meta/episodes.jsonl")
    for ep in episodes:
        ep["tasks"] = [unified_task]
    write_jsonl(episodes, dst / "meta/episodes.jsonl")
    print(f"[write] meta/episodes.jsonl -> {len(episodes)} episodes unified")

    # 4. episodes_stats.jsonl
    stats_src = src / "meta/episodes_stats.jsonl"
    if stats_src.exists():
        ep_lengths = {ep["episode_index"]: ep["length"] for ep in episodes}
        rows = read_jsonl(stats_src)
        for row in rows:
            stats = row.get("stats", {})
            if "task_index" in stats:
                stats["task_index"] = zero_task_index_stats(
                    stats["task_index"], ep_lengths.get(row["episode_index"], 0)
                )
        write_jsonl(rows, dst / "meta/episodes_stats.jsonl")
        print(f"[write] meta/episodes_stats.jsonl -> task_index zeroed for {len(rows)} episodes")

    # 5. parquet files
    if parquet_files:
        print(f"[write] rewriting {len(parquet_files)} parquet file(s) ...")
        for rel in sorted(parquet_files):
            rewrite_parquet(src / rel, dst / rel)
        print("[write] parquet done")
    else:
        print("[skip ] no parquet files under data/")

    print(f"\nDone. Unified dataset written to: {dst}")


if __name__ == "__main__":
    main()
