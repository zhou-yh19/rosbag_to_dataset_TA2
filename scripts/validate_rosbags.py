#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Validate ROS2 bag files and report data statistics without any conversion.

This script scans rosbag files, detects X/Y episode markers, and outputs a
summary of bag size, duration, episode counts, and timing statistics.

Usage:
    conda activate rosbag2lerobot
    unset PYTHONPATH

    # Validate a single rosbag directory
    python scripts/validate_rosbags.py \
        --input_directory ./data/rosbags \
        --task "叠衣服" \
        --fps 30

    # Validate multiple rosbag directories
    python scripts/validate_rosbags.py \
        --multibag \
        --input_directory ./data/rosbags \
        --task "叠衣服" \
        --fps 30
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import logging
from datetime import datetime

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter, Info
    from sensor_msgs.msg import Joy
except ImportError as e:
    print(f"ROS2 dependencies not found: {e}")
    sys.exit(1)

MIN_EPISODE_FRAMES = 30


def discover_rosbags(input_directory: Path, multibag: bool):
    """Discover all ROS2 bags in the input directory."""
    rosbags = []

    if multibag:
        root = input_directory.resolve()
        if not root.exists():
            raise ValueError(f"Path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        rosbag_folders = []
        for dirpath, dirnames, filenames in os.walk(root):
            current_dir = Path(dirpath)
            file_extensions = {Path(f).suffix.lower() for f in filenames}
            has_bag = '.db3' in file_extensions or '.mcap' in file_extensions
            has_yaml = '.yaml' in file_extensions or '.yml' in file_extensions
            if has_bag and has_yaml:
                rosbag_folders.append(current_dir)
                dirnames.clear()

        # One entry per bag directory: the reader opens the directory and replays
        # every storage file listed in metadata.yaml, so split bags (multiple
        # .db3/.mcap segments) must not be enumerated per file
        for rosbag_dir in sorted(rosbag_folders):
            db3_files = sorted(rosbag_dir.glob("*.db3"))
            mcap_files = sorted(rosbag_dir.glob("*.mcap"))
            if db3_files and mcap_files:
                print(f"Warning: {rosbag_dir} contains both .db3 and .mcap files; "
                      f"using .db3 and ignoring .mcap")
            db3_files = db3_files or mcap_files
            if db3_files:
                rosbags.append({
                    'name': rosbag_dir.name,
                    'path': str(rosbag_dir),
                    'bag_file': str(db3_files[0]),
                })
    else:
        # the directory holds a single bag (one metadata.yaml), possibly split
        # into several storage files
        db3_files = sorted(input_directory.glob("*.db3"))
        mcap_files = sorted(input_directory.glob("*.mcap"))
        if db3_files and mcap_files:
            print(f"Warning: {input_directory} contains both .db3 and .mcap files; "
                  f"using .db3 and ignoring .mcap")
        db3_files = db3_files or mcap_files
        if db3_files:
            rosbags.append({
                'name': "episode_000",
                'path': str(db3_files[0].parent),
                'bag_file': str(db3_files[0]),
            })

    return rosbags


def get_bag_size_bytes(bag_path: str) -> int:
    return sum(f.stat().st_size for f in Path(bag_path).rglob('*') if f.is_file())


def scan_single_bag(rosbag: dict, fps: int, logger: logging.Logger):
    """Scan a single bag for Joy markers, return per-bag statistics.

    Returns dict with keys:
        bag_name, bag_size_bytes, bag_duration_s,
        total_segments, valid_segments, valid_durations (list of float seconds),
        topics_found (set of topic names)
    """
    bag_file = rosbag['bag_file']
    storage_id = 'mcap' if bag_file.endswith('.mcap') else 'sqlite3'
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=rosbag['path'], storage_id=storage_id),
        ConverterOptions('', ''),
    )

    topic_types = reader.get_all_topics_and_types()
    topics_found = {tm.name for tm in topic_types}

    joy_topic = '/xr/left_hand_inputs'
    reader.set_filter(StorageFilter(topics=[joy_topic]))

    x_button, y_button = 2, 3
    frame_duration = 1.0 / fps

    is_recording = False
    previous_buttons = None
    start_time = None
    bag_first_ts = None
    bag_last_ts = None

    total_segments = 0
    valid_segments = 0
    valid_durations = []
    frame_count = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        timestamp_s = timestamp / 1e9
        if bag_first_ts is None:
            bag_first_ts = timestamp_s
        bag_last_ts = timestamp_s

        if topic != joy_topic:
            continue

        buttons = deserialize_message(data, Joy).buttons
        if len(buttons) <= max(x_button, y_button):
            continue

        if previous_buttons is not None:
            x_rising = previous_buttons[x_button] == 0 and buttons[x_button] == 1
            y_rising = previous_buttons[y_button] == 0 and buttons[y_button] == 1

            if x_rising:
                is_recording = True
                start_time = timestamp_s
                frame_count = 0

            if y_rising and is_recording:
                is_recording = False
                end_time = timestamp_s
                duration = end_time - start_time
                total_segments += 1

                estimated_frames = int(duration * fps)
                if estimated_frames >= MIN_EPISODE_FRAMES:
                    valid_segments += 1
                    valid_durations.append(duration)
                    logger.info(
                        f"    [有效] 片段 #{total_segments}: "
                        f"{duration:.2f}s (~{estimated_frames} frames)"
                    )
                else:
                    logger.info(
                        f"    [过短] 片段 #{total_segments}: "
                        f"{duration:.2f}s (~{estimated_frames} frames, "
                        f"需 >={MIN_EPISODE_FRAMES} frames)"
                    )

        previous_buttons = list(buttons)

    del reader

    # The reader filter above passes only the Joy topic, so the first/last
    # timestamps seen in the loop underestimate the bag span; prefer the bag
    # metadata duration and fall back to the Joy span.
    try:
        metadata = Info().read_metadata(rosbag['path'], storage_id)
        bag_duration_s = metadata.duration.total_seconds()
    except Exception:
        bag_duration_s = (bag_last_ts - bag_first_ts) if (bag_first_ts and bag_last_ts) else 0.0

    return {
        'bag_name': rosbag['name'],
        'bag_size_bytes': get_bag_size_bytes(rosbag['path']),
        'bag_duration_s': bag_duration_s,
        'total_segments': total_segments,
        'valid_segments': valid_segments,
        'valid_durations': valid_durations,
        'topics_found': topics_found,
    }


def fmt_duration(secs: float) -> str:
    secs = int(secs)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


def fmt_size(size_bytes: int) -> str:
    gb = size_bytes / 1e9
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{size_bytes / 1e6:.2f} MB"


def setup_logging(log_file_path: Path | None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate ROS2 bag data and report statistics")
    parser.add_argument("--multibag", action="store_true",
                        help="Input directory contains multiple rosbag sub-directories")
    parser.add_argument("--input_directory", default="./data/rosbags",
                        help="Directory containing ROS2 bag(s)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (used to estimate frame counts)")
    parser.add_argument("--task", default="task description",
                        help="Task description")
    parser.add_argument("--log_file", default=None,
                        help="Log file path (default: auto-generated in scripts/logs/)")
    args = parser.parse_args()

    if args.log_file is None:
        logs_dir = Path(__file__).parent / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = logs_dir / f"rosbag_validation_{timestamp}.log"
    else:
        log_file_path = Path(args.log_file)

    logger = setup_logging(log_file_path)
    logger.info(f"日志文件: {log_file_path}")

    input_directory = Path(args.input_directory)
    if not input_directory.exists():
        logger.error(f"输入目录不存在: {input_directory}")
        sys.exit(1)

    scan_start = time.time()

    rosbags = discover_rosbags(input_directory, args.multibag)
    if not rosbags:
        logger.error("未发现任何 rosbag！")
        sys.exit(1)

    logger.info(f"发现 {len(rosbags)} 个 rosbag:")
    for rb in rosbags:
        logger.info(f"  - {rb['name']}: {rb['bag_file']}")

    total_size_bytes = 0
    total_bag_duration_s = 0.0
    all_total_segments = 0
    all_valid_segments = 0
    all_valid_durations = []
    all_topics = set()

    for i, rosbag in enumerate(rosbags):
        logger.info(f"\n--- [{i+1}/{len(rosbags)}] 扫描 {rosbag['name']} ---")
        try:
            stats = scan_single_bag(rosbag, args.fps, logger)
        except Exception as e:
            logger.error(f"扫描失败: {rosbag['name']}: {e}")
            continue

        total_size_bytes += stats['bag_size_bytes']
        total_bag_duration_s += stats['bag_duration_s']
        all_total_segments += stats['total_segments']
        all_valid_segments += stats['valid_segments']
        all_valid_durations.extend(stats['valid_durations'])
        all_topics |= stats['topics_found']

        logger.info(
            f"    大小: {fmt_size(stats['bag_size_bytes'])}  "
            f"时长: {fmt_duration(stats['bag_duration_s'])}  "
            f"片段: {stats['total_segments']} (有效 {stats['valid_segments']})"
        )

    scan_elapsed = time.time() - scan_start

    if all_valid_durations:
        dur_mean = sum(all_valid_durations) / len(all_valid_durations)
        dur_var = sum((d - dur_mean) ** 2 for d in all_valid_durations) / len(all_valid_durations)
        dur_std = math.sqrt(dur_var)
        dur_min = min(all_valid_durations)
        dur_max = max(all_valid_durations)
        total_valid_duration = sum(all_valid_durations)
    else:
        dur_mean = dur_var = dur_std = dur_min = dur_max = total_valid_duration = 0.0

    sep = "=" * 56
    logger.info(f"\n{sep}")
    logger.info(f"  数据验证摘要 / Data Validation Summary")
    logger.info(sep)
    logger.info(f"  任务                 : {args.task}")
    logger.info(f"  目标 FPS             : {args.fps}")
    logger.info(f"  Rosbag 数量          : {len(rosbags)}")
    logger.info(f"  原始 rosbag 总大小   : {fmt_size(total_size_bytes)}")
    logger.info(f"  原始 rosbag 总时长   : {fmt_duration(total_bag_duration_s)}  ({total_bag_duration_s:.1f}s)")
    logger.info(f"  总数据片段           : {all_total_segments} 段")
    logger.info(f"  有效数据片段         : {all_valid_segments} 段  (>= {MIN_EPISODE_FRAMES} frames)")
    logger.info(f"  有效数据总时长       : {fmt_duration(total_valid_duration)}  ({total_valid_duration:.1f}s)")
    if all_valid_durations:
        logger.info(f"  有效数据时长均值     : {dur_mean:.2f}s")
        logger.info(f"  有效数据时长标准差   : {dur_std:.2f}s")
        logger.info(f"  有效数据时长方差     : {dur_var:.2f}s²")
        logger.info(f"  有效数据时长范围     : {dur_min:.2f}s ~ {dur_max:.2f}s")
        logger.info(f"  各片段时长           : {', '.join(f'{d:.2f}s' for d in all_valid_durations)}")
    logger.info(f"  扫描耗时             : {fmt_duration(scan_elapsed)}  ({scan_elapsed:.1f}s)")
    logger.info(sep)

    logger.info(f"\n  发现的 ROS topics ({len(all_topics)}):")
    for t in sorted(all_topics):
        logger.info(f"    {t}")
    logger.info(sep)


if __name__ == "__main__":
    main()
