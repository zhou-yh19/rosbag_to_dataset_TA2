#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Convert ROS2 bag files with X/Y episode markers to HDF5 format.

This script processes rosbag files that contain multiple episodes marked by operator
button presses (X button to start, Y button to end recording).

The robot's camera streams are H.265 with inter-coded GOPs. Raw packets are
buffered per episode (starting at the last IDR keyframe before the X press so
the whole GOP chain is decodable), then decoded in one ffmpeg pass per camera
— NVDEC GPU-accelerated when available, software decode otherwise. Frames are
sampled on the episode's fps grid (frame for row k = latest camera frame at
row k's sampling instant), JPEG-encoded, and stored in HDF5 (RobotWin style).

The HDF5 output structure for each episode file is:
    episode_XXXXXX.hdf5
    ├── /meta (attrs: task, fps, episode_idx, n_frames, source_bag, recording_time)
    │   └── /cameras
    │       ├── left_color  (attrs: K, D, P, width, height, distortion_model)
    │       └── right_color (attrs: K, D, P, width, height, distortion_model)
    └── /data
        ├── timestamp                        (N,)    float64
        ├── observation/
        │   ├── state                        (N, 62) float32  (attrs: names)
        │   ├── chassis_state                (N, 9)  float32  (attrs: names)
        │   └── images/
        │       ├── left_color               (N,) vlen uint8  ← JPEG bytes per frame
        │       ├── right_color              (N,) vlen uint8
        │       └── head_camera              (N,) vlen uint8
        ├── action                           (N, 62) float32  (attrs: names)
        └── chassis_action                   (N, 9)  float32  (attrs: names)

Usage:
    First, ensure the correct Python environment is activated:
    ```bash
    conda activate rosbag2lerobot
    unset PYTHONPATH
    ```

    Process multiple rosbag files:
    ```bash
    python scripts/convert_rosbag_to_hdf5.py \
        --multibag \
        --input_directory ./data/rosbags \
        --output_directory ./data/hdf5_output \
        --fps 45 \
        --task "task description" \
        --start_episode_idx 0 \
        --jpeg_quality 85
    ```

    Process a single rosbag file:
    ```bash
    python scripts/convert_rosbag_to_hdf5.py \
        --input_directory ./data/rosbags \
        --output_directory ./data/hdf5_output \
        --fps 45 \
        --task "task description" \
        --start_episode_idx 88

    Notes:
    - images are stored as per-frame JPEG bytes (decode with cv2.imdecode).
    - image datasets are append-friendly:
      maxshape=(None,), chunks=(1024,), dtype=vlen uint8
    - repeated frames reuse previously encoded bytes to reduce CPU overhead.
    ```
"""

import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path
import logging
import traceback
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import h5py
import cv2

# Shared packet-buffering / GPU frame-extraction machinery (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from rosbag_video_extraction import (
        detect_gpu_count,
        extract_grid_frames,
        packets_from_last_idr,
    )
except ImportError as e:
    print(f"rosbag_video_extraction module not found next to this script: {e}")
    sys.exit(1)

# ROS2 imports
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from sensor_msgs.msg import JointState, CameraInfo
    from geometry_msgs.msg import Pose
    from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
    from sensor_msgs.msg import Joy
except ImportError as e:
    print(f"ROS2 dependencies not found: {e}")
    sys.exit(1)

# Constants
STATE_ACTION_DIM = 62
CHASSIS_DIM = 9   # 3 motors × (position + velocity + effort)
MIN_EPISODE_LENGTH = 30
ACTION_OFFSET_RATIO = 1.0 / 3.0
# Warn when consecutive sampled frames are this many frame periods apart in real
# bag time (the synthetic output timestamps would silently hide the gap).
TIME_GAP_WARN_RATIO = 3

class MultiVideoRosBag2HDF5Converter:
    """Enhanced converter for multiple ROS2 bags to HDF5 dataset with multiple episodes."""

    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        fps: int = 45,
        start_episode_idx: int = 0,
        jpeg_quality: int = 85,
        jpeg_workers: int = 0,
        overwrite: bool = False,
    ):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.fps = fps
        self.frame_duration = 1.0 / self.fps
        self.overwrite = bool(overwrite)

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Video topics mapping
        self.video_topics = {
            'left_color':  '/left/color/image_raw/ffmpeg',
            'right_color': '/right/color/image_raw/ffmpeg',
            'head_camera': '/xr_video_topic/ffmpeg',
        }
        # Video topics set
        self.video_topics_set = set(self.video_topics.values())
        # Camera info topics mapping (camera_key -> camera_info topic)
        # head_camera (/xr_video_topic/ffmpeg) has no dedicated camera_info in recorded bags.
        self.camera_info_topics = {
            'left_color':  '/left/color/camera_info',
            'right_color': '/right/color/camera_info',
        }
        # Cache for camera intrinsics, populated once per bag before main loop
        self._camera_info_cache: dict = {}
        # State topics set
        self.state_topics_set = {
            '/left_arm/joint_states', '/left_arm/current_ee_pose', '/left_gripper/joint_states',
            '/right_arm/joint_states','/right_arm/current_ee_pose','/right_gripper/joint_states',
        }
        # Chassis topics
        self.chassis_state_topic = '/chassis/joint_states'
        self.chassis_action_topic = '/chassis/joint_cmd'
        self.chassis_topics_set = {self.chassis_state_topic, self.chassis_action_topic}
        # Action topics set
        self.action_topics_set = {
            '/left_arm/joint_cmd', '/left_arm/target_ee_pose', '/left_gripper/joint_cmd',
            '/right_arm/joint_cmd','/right_arm/target_ee_pose','/right_gripper/joint_cmd',
        }
        # All topics set
        self.all_topics_set = (self.video_topics_set | self.state_topics_set
                               | self.action_topics_set | self.chassis_topics_set)

        # JPEG encoding settings (RobotWin-like per-frame compressed bytes).
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.jpeg_workers = max(0, int(jpeg_workers))
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        self._jpeg_executor = (
            ThreadPoolExecutor(max_workers=self.jpeg_workers, thread_name_prefix="jpeg")
            if self.jpeg_workers > 0 else None
        )
        # HDF5 dtype for variable-length compressed frame bytes.
        self._vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))

        # Episode tracking
        self.num_xy_pairs = 0
        # state tracking
        self._cur_state_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        # chassis tracking
        self._cur_chassis_state_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        self._cur_chassis_action_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        # action tracking
        self._cur_action_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)

        # Reverse topic -> camera_key map (avoids a linear scan per message)
        self.topic_to_camera = {topic: cam for cam, topic in self.video_topics.items()}
        # Actual frame dimensions per camera, taken from the FFMPEGPacket msgs
        self._video_dims = {}
        # Lazy state/action deserialization: raw CDR bytes are cached per
        # topic and only the latest message per topic is deserialized when a
        # frame is actually captured (identical result, ~10x fewer deserializes)
        self._dirty_state = {}
        self._dirty_action = {}

        # Raw HEVC packet buffering. GOP != 1: every episode must include the
        # packets from the last IDR before its X press, and every packet of
        # the episode, so the decoder can rebuild the reference chain.
        self._window_size = 60
        self._recent_packets = {ck: deque(maxlen=self._window_size) for ck in self.video_topics.keys()}
        self._ep_packets = None      # {camera_key: [ {'data','ts'}, ... ]} while recording
        self._ep_row_ticks = []      # grid tick index of each written row

        # NVDEC-accelerated decode when a GPU is present (CPU fallback inside
        # the extractor). Decoding has no NVENC-style session cap.
        self.n_gpus = detect_gpu_count()
        self._episode_seq = 0
        self.logger.info(f"Detected {self.n_gpus} NVIDIA GPU(s) for video decode")

        # Global episode index across all bags (used for output naming).
        self.start_episode_idx = max(0, int(start_episode_idx))
        self.total_episodes_saved = self.start_episode_idx

        # Statistics tracking
        self._stats_rosbag_size_bytes = 0       # total raw rosbag size in bytes
        self._stats_rosbag_duration_s  = 0.0    # total raw rosbag duration in seconds
        self._stats_hdf5_total_frames  = 0      # total frames saved across all episodes
        self._stats_num_xy_pairs_total = 0      # total X/Y marker pairs (attempted episodes)

        # Streaming episode writer state (keeps memory O(1) over episode length)
        self._ep_n_time_gaps = 0
        self._ep_max_time_gap_s = 0.0
        self._ep_prev_frame_target_t = None
        self._ep_h5 = None
        self._ep_episode_index = None
        self._ep_out_path = None
        self._ep_tmp_path = None
        self._ep_n_frames = 0
        self._ep_ds_timestamp = None
        self._ep_ds_state = None
        self._ep_ds_chassis_state = None
        self._ep_ds_action = None
        self._ep_ds_chassis_action = None
        self._ep_images_grp = None
        self._ep_image_ds = {}
        self._ep_blank_image_bytes = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Discard any unfinished episode file and release background workers."""
        if self._ep_h5 is not None:
            self._finalize_episode_hdf5(keep=False)
        if self._jpeg_executor is not None:
            self._jpeg_executor.shutdown(wait=True)
            self._jpeg_executor = None

    def _camera_hw_from_meta(self, camera_key: str) -> tuple[int, int]:
        # Actual bitstream dimensions (from FFMPEGPacket) are authoritative
        dims = self._video_dims.get(camera_key)
        if dims is not None:
            return int(dims[1]), int(dims[0])
        info = self._camera_info_cache.get(camera_key)
        if info is not None and info.get("height", 0) > 0 and info.get("width", 0) > 0:
            return int(info["height"]), int(info["width"])
        h, w, _ = self.get_camera_resolution(camera_key)
        return h, w

    def _encode_jpeg(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return None
        if frame_bgr.dtype != np.uint8 or not frame_bgr.flags.c_contiguous:
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", frame_bgr, self._jpeg_params)
        if not ok:
            return None
        if enc.dtype != np.uint8:
            enc = enc.astype(np.uint8, copy=False)
        if enc.ndim != 1:
            enc = enc.reshape(-1)
        return enc

    def _get_or_make_blank_bytes(self, camera_key: str):
        blank = self._ep_blank_image_bytes.get(camera_key)
        if blank is not None:
            return blank
        h, w = self._camera_hw_from_meta(camera_key)
        blank_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        blank = self._encode_jpeg(blank_bgr)
        self._ep_blank_image_bytes[camera_key] = blank
        return blank

    def _begin_episode_hdf5(self, episode_index: int, task_description: str, source_bag: str, recording_time: str):
        """Open a temporary episode HDF5 and prepare appendable datasets."""
        if self._ep_h5 is not None:
            self._finalize_episode_hdf5(keep=False)

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._ep_out_path = self.output_directory / f"episode_{episode_index:06d}.hdf5"
        # ".hdf5.tmp" (not ".tmp.hdf5") so unfinished files never match *.hdf5 globs
        self._ep_tmp_path = self.output_directory / f"episode_{episode_index:06d}.hdf5.tmp"
        self._ep_episode_index = episode_index
        self._ep_n_time_gaps = 0
        self._ep_max_time_gap_s = 0.0
        self._ep_prev_frame_target_t = None
        if self._ep_tmp_path.exists():
            self._ep_tmp_path.unlink()

        feature_names = self.setup_features()
        chassis_feature_names = self.setup_chassis_features()

        self._ep_h5 = h5py.File(self._ep_tmp_path, 'w', rdcc_nbytes=64 * 1024 * 1024)
        meta = self._ep_h5.create_group("meta")
        meta.attrs["task"] = task_description
        meta.attrs["fps"] = self.fps
        meta.attrs["image_encoding"] = "jpeg"
        meta.attrs["jpeg_quality"] = self.jpeg_quality
        meta.attrs["episode_idx"] = episode_index
        meta.attrs["n_frames"] = 0
        meta.attrs["source_bag"] = source_bag
        meta.attrs["recording_time"] = recording_time

        cameras_grp = meta.create_group("cameras")
        for camera_key, info in self._camera_info_cache.items():
            cam_ds = cameras_grp.create_group(camera_key)
            cam_ds.attrs["K"] = info["K"]
            cam_ds.attrs["D"] = info["D"]
            cam_ds.attrs["P"] = info["P"]
            cam_ds.attrs["width"] = info["width"]
            cam_ds.attrs["height"] = info["height"]
            cam_ds.attrs["distortion_model"] = info["distortion_model"]

        data_grp = self._ep_h5.create_group("data")
        self._ep_ds_timestamp = data_grp.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(1024,),
            compression="lzf",
        )

        obs_grp = data_grp.create_group("observation")
        self._ep_ds_state = obs_grp.create_dataset(
            "state",
            shape=(0, STATE_ACTION_DIM),
            maxshape=(None, STATE_ACTION_DIM),
            dtype=np.float32,
            chunks=(1024, STATE_ACTION_DIM),
            compression="lzf",
        )
        self._ep_ds_state.attrs["names"] = feature_names

        self._ep_ds_chassis_state = obs_grp.create_dataset(
            "chassis_state",
            shape=(0, CHASSIS_DIM),
            maxshape=(None, CHASSIS_DIM),
            dtype=np.float32,
            chunks=(1024, CHASSIS_DIM),
            compression="lzf",
        )
        self._ep_ds_chassis_state.attrs["names"] = chassis_feature_names
        self._ep_images_grp = obs_grp.create_group("images")

        self._ep_ds_action = data_grp.create_dataset(
            "action",
            shape=(0, STATE_ACTION_DIM),
            maxshape=(None, STATE_ACTION_DIM),
            dtype=np.float32,
            chunks=(1024, STATE_ACTION_DIM),
            compression="lzf",
        )
        self._ep_ds_action.attrs["names"] = feature_names

        self._ep_ds_chassis_action = data_grp.create_dataset(
            "chassis_action",
            shape=(0, CHASSIS_DIM),
            maxshape=(None, CHASSIS_DIM),
            dtype=np.float32,
            chunks=(1024, CHASSIS_DIM),
            compression="lzf",
        )
        self._ep_ds_chassis_action.attrs["names"] = chassis_feature_names

        self._ep_image_ds = {}
        self._ep_blank_image_bytes = {}
        self._ep_n_frames = 0
        self._ep_row_ticks = []

    def _ensure_episode_image_dataset(self, camera_key: str):
        ds = self._ep_image_ds.get(camera_key)
        if ds is not None:
            return ds

        h, w = self._camera_hw_from_meta(camera_key)

        ds = self._ep_images_grp.create_dataset(
            camera_key,
            shape=(0,),
            maxshape=(None,),
            dtype=self._vlen_uint8,
            chunks=(1024,),
        )
        ds.attrs["encoding"] = "jpeg"
        ds.attrs["jpeg_quality"] = self.jpeg_quality
        ds.attrs["height"] = h
        ds.attrs["width"] = w
        ds.attrs["channels"] = 3
        self._ep_image_ds[camera_key] = ds
        return ds

    def _append_episode_frame(self, frame_data: dict):
        """Append one scalar frame sample to current episode HDF5.

        Images are NOT written here: raw packets are buffered during the
        episode and decoded/JPEG-encoded in one pass at episode end
        (see _write_episode_images).
        """
        if self._ep_h5 is None:
            return

        i = self._ep_n_frames
        n = i + 1

        self._ep_ds_timestamp.resize((n,))
        self._ep_ds_state.resize((n, STATE_ACTION_DIM))
        self._ep_ds_chassis_state.resize((n, CHASSIS_DIM))
        self._ep_ds_action.resize((n, STATE_ACTION_DIM))
        self._ep_ds_chassis_action.resize((n, CHASSIS_DIM))

        self._ep_ds_timestamp[i] = i / self.fps
        self._ep_ds_state[i] = frame_data["observation.state"]
        self._ep_ds_chassis_state[i] = frame_data["observation.chassis_state"]
        self._ep_ds_action[i] = frame_data["action"]
        self._ep_ds_chassis_action[i] = frame_data["chassis_action"]

        self._ep_row_ticks.append(int(frame_data["tick_index"]))

        self._ep_n_frames = n
        if n % 256 == 0:
            self._ep_h5.flush()

    def _extract_camera_jpegs(self, camera_key: str, first_tick_time: float,
                              n_grid: int, row_ticks: list, gpu_index: int) -> list:
        """Decode one camera's episode packets and JPEG-encode the row frames.

        Returns a list of JPEG byte arrays aligned with the episode rows
        (row i shows the latest camera frame at row i's grid tick).
        """
        packets = self._ep_packets.get(camera_key) or []
        if not packets:
            blank = self._get_or_make_blank_bytes(camera_key)
            return [blank] * len(row_ticks)

        width, height = self._video_dims[camera_key]
        needed = set(row_ticks)
        by_tick = {}
        pending = {}
        frames = extract_grid_frames(
            packets, first_tick_time, n_grid, self.fps,
            out_width=width, out_height=height,
            gpu_index=gpu_index, use_gpu=self.n_gpus > 0, logger=self.logger,
        )
        for tick, frame in enumerate(frames):
            if tick not in needed:
                continue
            if self._jpeg_executor is None:
                by_tick[tick] = self._encode_jpeg(frame)
            else:
                # copy: the generator may reuse/overwrite its buffer
                pending[tick] = self._jpeg_executor.submit(self._encode_jpeg, frame.copy())
                # Bound in-flight raw-frame copies (4K frames are ~22MB each):
                # resolve the oldest submissions once the queue grows
                while len(pending) > 24:
                    oldest = next(iter(pending))
                    by_tick[oldest] = pending.pop(oldest).result()
        for tick, future in pending.items():
            by_tick[tick] = future.result()

        blank = None
        result = []
        for tick in row_ticks:
            enc = by_tick.get(tick)
            if enc is None:
                if blank is None:
                    blank = self._get_or_make_blank_bytes(camera_key)
                enc = blank
            result.append(enc)
        return result

    def _start_episode_packets(self, start_time: float):
        """Begin buffering episode packets, primed from each camera's last IDR.

        The sliding window holds >= one GOP of recent packets; starting the
        episode buffer at the last IDR before the X press gives the decoder a
        complete reference chain (no info loss, no undecodable leading frames).
        """
        self._ep_packets = {}
        for camera_key in self.video_topics.keys():
            primed = packets_from_last_idr(self._recent_packets[camera_key])
            self._ep_packets[camera_key] = list(primed)
            if primed:
                self.logger.info(
                    f"  📹 {camera_key}: primed {len(primed)} pkts from IDR at "
                    f"{primed[0]['ts']:.3f}s (X at {start_time:.3f}s)"
                )
            else:
                self.logger.warning(
                    f"  ⚠️ {camera_key}: no IDR in recent window — episode may "
                    f"start with blank frames until the next IDR"
                )

    def _write_episode_images(self, start_time: float):
        """Decode all cameras (in parallel) and fill the episode image datasets."""
        row_ticks = self._ep_row_ticks
        if not row_ticks:
            return
        first_tick_time = start_time + self.frame_duration
        n_grid = row_ticks[-1] + 1
        gpu_index = self._episode_seq % max(1, self.n_gpus)
        self._episode_seq += 1

        with ThreadPoolExecutor(max_workers=len(self.video_topics)) as cam_pool:
            futures = {
                camera_key: cam_pool.submit(
                    self._extract_camera_jpegs, camera_key,
                    first_tick_time, n_grid, row_ticks, gpu_index,
                )
                for camera_key in self.video_topics.keys()
            }
            for camera_key, future in futures.items():
                jpegs = future.result()
                ds = self._ensure_episode_image_dataset(camera_key)
                ds.resize((len(jpegs),))
                for i, enc in enumerate(jpegs):
                    ds[i] = enc if enc is not None else np.empty((0,), dtype=np.uint8)

    def _reset_message_caches(self):
        """Zero the state/action caches so a new episode cannot start with
        values left over from a previous episode or bag."""
        self._cur_state_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        self._cur_action_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        self._cur_chassis_state_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        self._cur_chassis_action_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        self._dirty_state.clear()
        self._dirty_action.clear()

    def _note_frame_time_gap(self, frame_target_t: float):
        """Track real bag-time gaps between consecutive sampled frames.

        The output /data/timestamp is synthetic (i/fps), so message stalls in the
        source bag would otherwise be invisible in the saved file.
        """
        if self._ep_prev_frame_target_t is not None:
            gap_s = frame_target_t - self._ep_prev_frame_target_t
            if gap_s >= TIME_GAP_WARN_RATIO * self.frame_duration:
                self._ep_n_time_gaps += 1
                self._ep_max_time_gap_s = max(self._ep_max_time_gap_s, gap_s)
                self.logger.warning(
                    f"⚠️  TIME GAP in episode {self._ep_episode_index}: {gap_s:.2f}s "
                    f"(~{gap_s / self.frame_duration:.0f} frame periods) of source messages "
                    f"missing before sampled frame {self._ep_n_frames} — the synthetic "
                    f"timestamps in the output hide this gap"
                )
        self._ep_prev_frame_target_t = frame_target_t

    def _finalize_episode_hdf5(self, keep: bool) -> int:
        """Close current episode file. Keep+rename if valid, otherwise delete tmp."""
        if self._ep_h5 is None:
            return 0

        n = self._ep_n_frames
        self._ep_h5["meta"].attrs["n_frames"] = n
        self._ep_h5["meta"].attrs["n_time_gaps"] = self._ep_n_time_gaps
        self._ep_h5["meta"].attrs["max_time_gap_s"] = float(self._ep_max_time_gap_s)
        self._ep_h5.flush()
        self._ep_h5.close()

        if keep:
            if self._ep_out_path.exists():
                if not self.overwrite:
                    # h5 already closed above; keep the tmp file on disk for
                    # inspection and clear the handle so close() won't touch it
                    self._ep_h5 = None
                    raise FileExistsError(
                        f"Refusing to overwrite existing {self._ep_out_path}; "
                        f"rerun with --overwrite or a higher --start_episode_idx"
                    )
                self._ep_out_path.unlink()
            self._ep_tmp_path.rename(self._ep_out_path)
            self._stats_hdf5_total_frames += n
            logging.info(f"✅ Saved episode {self._ep_episode_index} with {n} frames ({n / self.fps:.1f}s)")
            if self._ep_n_time_gaps > 0:
                self.logger.warning(
                    f"⚠️  Episode {self._ep_episode_index} saved WITH {self._ep_n_time_gaps} time gap(s), "
                    f"max {self._ep_max_time_gap_s:.2f}s — see /meta attrs n_time_gaps / max_time_gap_s"
                )
        else:
            try:
                if self._ep_tmp_path.exists():
                    self._ep_tmp_path.unlink()
            except Exception:
                pass

        self._ep_h5 = None
        self._ep_episode_index = None
        self._ep_out_path = None
        self._ep_tmp_path = None
        self._ep_n_frames = 0
        self._ep_n_time_gaps = 0
        self._ep_max_time_gap_s = 0.0
        self._ep_prev_frame_target_t = None
        self._ep_ds_timestamp = None
        self._ep_ds_state = None
        self._ep_ds_chassis_state = None
        self._ep_ds_action = None
        self._ep_ds_chassis_action = None
        self._ep_images_grp = None
        self._ep_image_ds = {}
        self._ep_blank_image_bytes = {}
        self._ep_row_ticks = []
        return n


    def discover_rosbags(self, MULTIBAG_FLAG, verbose: bool = False):
        """Discover all ROS2 bags in the input directory."""
        rosbags = []

        if MULTIBAG_FLAG is True:
            # multi rosbag
            rosbag_folders = []
            root_path = self.input_directory
            root = Path(root_path).resolve()
            
            if not root.exists():
                raise ValueError(f"Path does not exist: {root_path}")
            
            if not root.is_dir():
                raise ValueError(f"Path is not a directory: {root_path}")
            
            folder_count = 0
            
            for dirpath, dirnames, filenames in os.walk(root):
                current_dir = Path(dirpath)
                folder_count += 1
                
                file_extensions = {Path(f).suffix.lower() for f in filenames}
                has_bag = '.db3' in file_extensions or '.mcap' in file_extensions
                has_yaml = '.yaml' in file_extensions or '.yml' in file_extensions

                if has_bag and has_yaml:
                    rosbag_folders.append(str(current_dir))
                    dirnames.clear()

            if rosbag_folders:
                # One entry per bag directory: the reader opens the directory and
                # replays every storage file listed in metadata.yaml, so split bags
                # (multiple .db3/.mcap segments) must not be enumerated per file
                for rosbag_dir in sorted(rosbag_folders):
                    rosbag_dir = Path(rosbag_dir)
                    db3_files = sorted(rosbag_dir.glob("*.db3"))
                    mcap_files = sorted(rosbag_dir.glob("*.mcap"))
                    if db3_files and mcap_files:
                        self.logger.warning(
                            f"{rosbag_dir} contains both .db3 and .mcap files; "
                            f"using .db3 and ignoring .mcap")
                    db3_files = db3_files or mcap_files
                    if db3_files:
                        rosbags.append({
                            'name': rosbag_dir.name,
                            'path': str(rosbag_dir),
                            'bag_file': str(db3_files[0]),
                        })

        else:
            # one rosbag: the directory holds a single bag (one metadata.yaml),
            # possibly split into several storage files
            db3_files = sorted(self.input_directory.glob("*.db3"))
            mcap_files = sorted(self.input_directory.glob("*.mcap"))
            if db3_files and mcap_files:
                self.logger.warning(
                    f"{self.input_directory} contains both .db3 and .mcap files; "
                    f"using .db3 and ignoring .mcap")
            db3_files = db3_files or mcap_files
            if db3_files:
                rosbags.append({
                    'name': "episode_000",
                    'path': str(db3_files[0].parent),
                    'bag_file': str(db3_files[0])
                })

        self.logger.info(f"Discovered {len(rosbags)} bags:")
        for rosbag in rosbags:
            self.logger.info(f"  - {rosbag['name']}: {rosbag['bag_file']}")

        return rosbags

    def get_camera_resolution(self, camera_key):
        """Get camera resolution - different cameras may have different resolutions."""
        if camera_key == 'head_camera':
            # Head camera is higher resolution
            return (2160, 4320, 3) 
        else:
            return (480, 848, 3)  # Stereo cameras

    def setup_features(self):
        """Setup feature names for state/action dimensions."""
        feature_names = [
            # Joint positions (16)
            "left_joint1_position", "left_joint2_position", "left_joint3_position", "left_joint4_position",
            "left_joint5_position", "left_joint6_position", "left_joint7_position", "left_gripper_position",
            "right_joint1_position", "right_joint2_position", "right_joint3_position", "right_joint4_position",
            "right_joint5_position", "right_joint6_position", "right_joint7_position", "right_gripper_position",
            # Joint velocities (16)
            "left_joint1_velocity", "left_joint2_velocity", "left_joint3_velocity", "left_joint4_velocity",
            "left_joint5_velocity", "left_joint6_velocity", "left_joint7_velocity", "left_gripper_velocity",
            "right_joint1_velocity", "right_joint2_velocity", "right_joint3_velocity", "right_joint4_velocity",
            "right_joint5_velocity", "right_joint6_velocity", "right_joint7_velocity", "right_gripper_velocity",
            # Joint efforts (16)
            "left_joint1_effort", "left_joint2_effort", "left_joint3_effort", "left_joint4_effort",
            "left_joint5_effort", "left_joint6_effort", "left_joint7_effort", "left_gripper_effort",
            "right_joint1_effort", "right_joint2_effort", "right_joint3_effort", "right_joint4_effort",
            "right_joint5_effort", "right_joint6_effort", "right_joint7_effort", "right_gripper_effort",
            # ee poses (14)
            "left_ee_position_x", "left_ee_position_y", "left_ee_position_z",
            "left_ee_orientation_x", "left_ee_orientation_y", "left_ee_orientation_z","left_ee_orientation_w",
            "right_ee_position_x", "right_ee_position_y", "right_ee_position_z",
            "right_ee_orientation_x", "right_ee_orientation_y", "right_ee_orientation_z","right_ee_orientation_w",
        ]
        return feature_names

    def _collect_camera_info(self, rosbag: dict) -> dict:
        """Scan the bag once and extract the first CameraInfo message for each camera.

        Returns a dict keyed by camera_key with values:
            {'K': (9,) float64, 'D': list[float], 'P': (12,) float64,
             'width': int, 'height': int, 'distortion_model': str}
        """
        result = {}
        topic_to_key = {topic: key for key, topic in self.camera_info_topics.items()}
        needed = set(topic_to_key.keys())

        try:
            bag_file = rosbag['bag_file']
            storage_id = 'mcap' if bag_file.endswith('.mcap') else 'sqlite3'
            storage_options = StorageOptions(uri=rosbag['path'], storage_id=storage_id)
            reader = SequentialReader()
            reader.open(storage_options, ConverterOptions('', ''))

            # Only read camera_info topics
            reader.set_filter(StorageFilter(topics=list(needed)))

            while reader.has_next() and len(result) < len(self.camera_info_topics):
                topic, data, _ = reader.read_next()
                if topic in needed:
                    camera_key = topic_to_key[topic]
                    if camera_key in result:
                        continue
                    msg: CameraInfo = deserialize_message(data, CameraInfo)
                    info = {
                        'K':                 np.array(msg.k,  dtype=np.float64),
                        'D':                 np.array(msg.d,  dtype=np.float64),
                        'P':                 np.array(msg.p,  dtype=np.float64),
                        'width':             msg.width,
                        'height':            msg.height,
                        'distortion_model':  msg.distortion_model,
                    }
                    result[camera_key] = info
            del reader
        except Exception as e:
            self.logger.warning(f"Could not collect camera_info from {rosbag['name']}: {e}")

        # Log which cameras were found / missing
        for camera_key in self.camera_info_topics:
            if camera_key in result:
                info = result[camera_key]
                self.logger.info(
                    f"  📷 {camera_key}: {info['width']}×{info['height']}  "
                    f"K=[{info['K'][0]:.2f}, {info['K'][4]:.2f}, {info['K'][2]:.2f}, {info['K'][5]:.2f}]"
                )
            else:
                self.logger.warning(f"  ⚠️  camera_info not found for {camera_key}")

        return result

    def setup_chassis_features(self):
        """Setup feature names for chassis state/action dimensions (3 motors × pos/vel/effort)."""
        return [
            "chassis_motor1_position", "chassis_motor2_position", "chassis_motor3_position",
            "chassis_motor1_velocity", "chassis_motor2_velocity", "chassis_motor3_velocity",
            "chassis_motor1_effort",   "chassis_motor2_effort",   "chassis_motor3_effort",
        ]

    def _get_camera_key_from_topic(self, topic):
        """Map ROS topic to camera key."""
        for camera_key, camera_topic in self.video_topics.items():
            if topic == camera_topic:
                return camera_key
        return None

    def extract_joint_data(self, msg:JointState):
        """Extract joint state data."""
        positions = list(msg.position) if msg.position else []
        velocities= list(msg.velocity) if msg.velocity else []
        efforts   = list(msg.effort)   if msg.effort   else []

        # Ensure we have the right number of joints (7-DOF arms)
        max_joints = 7
        for data_list in [positions, velocities, efforts]:
            if len(data_list) > max_joints:
                data_list[:] = data_list[:max_joints]
            while len(data_list) < max_joints:
                data_list.append(0.0)

        return np.array(positions + velocities + efforts, dtype=np.float32)

    def extract_gripper_data(self, msg:JointState):
        """Extract gripper data (single joint)."""
        pos = msg.position[0] if msg.position else 0.0
        vel = msg.velocity[0] if msg.velocity else 0.0
        effort = msg.effort[0] if msg.effort else 0.0
        return np.array([pos, vel, effort], dtype=np.float32)

    def extract_chassis_data(self, msg:JointState):
        """Extract chassis data (3 motors × position/velocity/effort)."""
        n_motors = 3
        positions  = list(msg.position)[:n_motors] if msg.position else []
        velocities = list(msg.velocity)[:n_motors] if msg.velocity else []
        efforts    = list(msg.effort)[:n_motors]   if msg.effort   else []
        for data_list in [positions, velocities, efforts]:
            while len(data_list) < n_motors:
                data_list.append(0.0)
        return np.array(positions + velocities + efforts, dtype=np.float32)

    def extract_ee_pose_data(self, msg:Pose):
        """Extract end-effector pose data (position + orientation)."""
        return np.array([
            msg.position.x, msg.position.y, msg.position.z, 
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ], dtype=np.float32)

    def _update_state_msg(self, topicName:str, msg:JointState):
        if   topicName == '/left_arm/joint_states':
            joint_data = self.extract_joint_data(msg)
            self._cur_state_msg[0:7] = joint_data[0:7]
            self._cur_state_msg[16:23] = joint_data[7:14]
            self._cur_state_msg[32:39] = joint_data[14:21]

        elif topicName == '/left_gripper/joint_states':
            gripper_data = self.extract_gripper_data(msg)
            self._cur_state_msg[7] = gripper_data[0]
            self._cur_state_msg[23] = gripper_data[1]
            self._cur_state_msg[39] = gripper_data[2]

        elif topicName == '/right_arm/joint_states':
            joint_data = self.extract_joint_data(msg)
            self._cur_state_msg[8:15] = joint_data[0:7]
            self._cur_state_msg[24:31] = joint_data[7:14]
            self._cur_state_msg[40:47] = joint_data[14:21]

        elif topicName == '/right_gripper/joint_states':
            gripper_data = self.extract_gripper_data(msg)
            self._cur_state_msg[15] = gripper_data[0]
            self._cur_state_msg[31] = gripper_data[1]
            self._cur_state_msg[47] = gripper_data[2]

        elif topicName == '/left_arm/current_ee_pose':
            ee_pose_data = self.extract_ee_pose_data(msg)
            self._cur_state_msg[48:55] = ee_pose_data

        elif topicName == '/right_arm/current_ee_pose':
            ee_pose_data = self.extract_ee_pose_data(msg)
            self._cur_state_msg[55:62] = ee_pose_data

    def _update_action_msg(self, topicName:str, msg:JointState):
        if   topicName == '/left_arm/joint_cmd':
            joint_data = self.extract_joint_data(msg)
            self._cur_action_msg[0:7] = joint_data[0:7]
            self._cur_action_msg[16:23] = joint_data[7:14]
            self._cur_action_msg[32:39] = joint_data[14:21]

        elif topicName == '/left_gripper/joint_cmd':
            gripper_data = self.extract_gripper_data(msg)
            self._cur_action_msg[7] = gripper_data[0]
            self._cur_action_msg[23] = gripper_data[1]
            self._cur_action_msg[39] = gripper_data[2]

        elif topicName == '/right_arm/joint_cmd':
            joint_data = self.extract_joint_data(msg)
            self._cur_action_msg[8:15] = joint_data[0:7]
            self._cur_action_msg[24:31] = joint_data[7:14]
            self._cur_action_msg[40:47] = joint_data[14:21]

        elif topicName == '/right_gripper/joint_cmd':
            gripper_data = self.extract_gripper_data(msg)
            self._cur_action_msg[15] = gripper_data[0]
            self._cur_action_msg[31] = gripper_data[1]
            self._cur_action_msg[47] = gripper_data[2]

        elif topicName == '/left_arm/target_ee_pose':
            ee_pose_data = self.extract_ee_pose_data(msg)
            self._cur_action_msg[48:55] = ee_pose_data

        elif topicName == '/right_arm/target_ee_pose':
            ee_pose_data = self.extract_ee_pose_data(msg)
            self._cur_action_msg[55:62] = ee_pose_data

    def _update_chassis_msg(self, topicName:str, msg:JointState):
        chassis_data = self.extract_chassis_data(msg)
        if topicName == self.chassis_state_topic:
            self._cur_chassis_state_msg[:] = chassis_data
        elif topicName == self.chassis_action_topic:
            self._cur_chassis_action_msg[:] = chassis_data

    def update_messages(self, topicName, data):
        """Use a single interface to update state, video packet, and action.

        Video packets are buffered raw (decode happens once per episode at
        save time); state/action messages just cache their raw bytes — only
        the latest message per topic matters, so deserialization is deferred
        to the capture ticks (see _materialize_state/_materialize_action).
        """
        if topicName in self.video_topics_set:
            msg_class = self.topic_msg_class.get(topicName)
            if msg_class is None:
                return
            msg = deserialize_message(data, msg_class)
            camera_key = self.topic_to_camera[topicName]
            if camera_key not in self._video_dims:
                self._video_dims[camera_key] = (int(msg.width), int(msg.height))
            self._cur_video_packet = (camera_key, bytes(msg.data))

        elif topicName in self.state_topics_set or topicName == self.chassis_state_topic:
            if topicName in self.topic_msg_class:
                self._dirty_state[topicName] = data

        elif topicName in self.action_topics_set or topicName == self.chassis_action_topic:
            if topicName in self.topic_msg_class:
                self._dirty_action[topicName] = data

    def _materialize_state(self):
        """Apply the latest cached raw message of each state/chassis-state topic."""
        if self._dirty_state:
            for topic, raw in self._dirty_state.items():
                msg = deserialize_message(raw, self.topic_msg_class[topic])
                if topic == self.chassis_state_topic:
                    self._update_chassis_msg(topic, msg)
                else:
                    self._update_state_msg(topic, msg)
            self._dirty_state.clear()

    def _materialize_action(self):
        """Apply the latest cached raw message of each action/chassis-action topic."""
        if self._dirty_action:
            for topic, raw in self._dirty_action.items():
                msg = deserialize_message(raw, self.topic_msg_class[topic])
                if topic == self.chassis_action_topic:
                    self._update_chassis_msg(topic, msg)
                else:
                    self._update_action_msg(topic, msg)
            self._dirty_action.clear()

    def add_state_and_video_packet(self, frame_data:dict):
        """Record current state into frame_data (images are extracted at save time)."""
        self._materialize_state()
        frame_data["observation.state"] = self._cur_state_msg.copy()
        frame_data["observation.chassis_state"] = self._cur_chassis_state_msg.copy()

    def add_action(self, frame_data:dict):
        self._materialize_action()
        frame_data['action'] = self._cur_action_msg.copy()
        frame_data['chassis_action'] = self._cur_chassis_action_msg.copy()

    def convert_single_bag(self, rosbag, task_description: str, ENFORCE_ALL_VIDEO_TOPICS_FLAG: bool):
        """Convert a single bag to one or multiple episodes in the dataset."""
        self.logger.info(f"\n=== Processing {rosbag['name']} ===")

        # Accumulate rosbag file size
        bag_path = Path(rosbag['path'])
        bag_size_bytes = sum(f.stat().st_size for f in bag_path.rglob('*') if f.is_file())
        self._stats_rosbag_size_bytes += bag_size_bytes
        self.logger.info(f"   Bag size: {bag_size_bytes / 1e9:.2f} GB")

        # Initialize ROS2 bag reader
        try:
            bag_file = rosbag['bag_file']
            storage_id = 'mcap' if bag_file.endswith('.mcap') else 'sqlite3'
            storage_options = StorageOptions(uri=rosbag['path'], storage_id=storage_id)
            converter_options = ConverterOptions('','')
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
        except Exception as e:
            error_msg = f"Failed to open {rosbag['bag_file']} in {rosbag['name']}: {e}\n"
            error_msg += f"Traceback (most recent call last):\n"
            error_msg += traceback.format_exc()
            self.logger.error(error_msg)
            return

        # Get topic types
        self.topic_types_dict = {}
        topic_types = reader.get_all_topics_and_types()
        for topic_metadata in topic_types:
            if topic_metadata.name in self.all_topics_set:
                self.topic_types_dict[topic_metadata.name] = topic_metadata.type

        if ENFORCE_ALL_VIDEO_TOPICS_FLAG is True:
            for video_topic_name in self.video_topics_set:
                if video_topic_name not in self.topic_types_dict:
                    self.logger.warning(
                        f"skip {rosbag['name']}：is missing video topic {video_topic_name}"
                    )
                    del reader
                    return

        # Collect camera intrinsics once before the main loop
        self._camera_info_cache = self._collect_camera_info(rosbag)

        # Cache deserialization classes once per bag (get_message is not free
        # at ~1M calls per bag)
        self.topic_msg_class = {
            t: get_message(ty) for t, ty in self.topic_types_dict.items()
        }
        self._dirty_state.clear()
        self._dirty_action.clear()

        # Read all consumed topics continuously: video packets must be
        # buffered even between episodes so each X press can rewind to the
        # last IDR keyframe (GOP != 1: starting mid-GOP is undecodable).
        needed_topics = list(self.all_topics_set | {'/xr/left_hand_inputs'})
        reader.set_filter(StorageFilter(topics=needed_topics))

        # Sliding window of recent raw packets per camera (covers >= one GOP)
        self._recent_packets = {ck: deque(maxlen=self._window_size) for ck in self.video_topics.keys()}
        self._ep_packets = None
        self._cur_video_packet = None

        x_button, y_button = 2, 3
        is_recording = False
        previous_buttons = None
        frame_data = dict()
        episode_state_target_t = None
        episode_action_target_t = None
        is_in_adding_phase = False
        start_time = None

        # Timestamp tracking for bag duration
        _bag_first_ts = None
        _bag_last_ts  = None
        # Track xy pairs and saved episodes within this bag
        _bag_xy_pairs_before   = self.num_xy_pairs
        _bag_episodes_before   = self.total_episodes_saved

        while reader.has_next():
            try:
                (topic, data, timestamp) = reader.read_next()
            except RuntimeError as e:
                error_msg = f"Error reading message from {rosbag['bag_file']}: {e}\n"
                error_msg += f"Traceback (most recent call last):\n"
                error_msg += traceback.format_exc()
                self.logger.error(error_msg)
                break
            timestamp = timestamp / 1e9
            if _bag_first_ts is None:
                _bag_first_ts = timestamp
            _bag_last_ts = timestamp

            self.update_messages(topic, data)

            # --- Maintain sliding window / episode buffer of raw video packets ---
            camera_key = self.topic_to_camera.get(topic)
            if camera_key is not None and self._cur_video_packet is not None:
                pkt = {'data': self._cur_video_packet[1], 'ts': timestamp}
                self._recent_packets[camera_key].append(pkt)
                if is_recording and self._ep_packets is not None:
                    self._ep_packets[camera_key].append(pkt)
                self._cur_video_packet = None

            if topic == '/xr/left_hand_inputs':
                try:
                    buttons = deserialize_message(data, Joy).buttons

                    if len(buttons) > max(x_button, y_button):
                        if previous_buttons is not None:
                            if   (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and not is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Start #Episode: {self.num_xy_pairs}")
                                self._reset_message_caches()
                                self._start_episode_packets(start_time)
                                self._begin_episode_hdf5(
                                    episode_index=self.total_episodes_saved,
                                    task_description=task_description,
                                    source_bag=rosbag['name'],
                                    recording_time=datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                                )

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                frame_data.clear()

                            elif (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Re-record #Episode: {self.num_xy_pairs}")
                                self._reset_message_caches()
                                self._start_episode_packets(start_time)
                                # Drop previous unfinished recording and restart.
                                self._finalize_episode_hdf5(keep=False)
                                self._begin_episode_hdf5(
                                    episode_index=self.total_episodes_saved,
                                    task_description=task_description,
                                    source_bag=rosbag['name'],
                                    recording_time=datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                                )

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                frame_data.clear()

                            if (previous_buttons[y_button] == 0 and buttons[y_button] == 1 and is_recording):
                                is_recording = False
                                end_time = timestamp
                                duration = end_time - start_time
                                self.logger.info(f"⏹️ Stop TimeRange: {start_time:.3f} to {end_time:.3f} seconds. Duration: {duration:.3f} seconds")

                                self.num_xy_pairs += 1

                                n_written = self._ep_n_frames
                                if n_written < MIN_EPISODE_LENGTH:
                                    self._finalize_episode_hdf5(keep=False)
                                else:
                                    self._write_episode_images(start_time)
                                    self._finalize_episode_hdf5(keep=True)
                                    self.total_episodes_saved += 1
                                self._ep_packets = None

                                episode_state_target_t = None
                                episode_action_target_t = None

                        previous_buttons = list(buttons)
                
                except Exception as e:
                    error_msg = f"Error processing Joy message: {e}\n"
                    error_msg += f"Traceback (most recent call last):\n"
                    error_msg += traceback.format_exc()
                    self.logger.error(error_msg)

            if is_recording:
                if   timestamp < episode_state_target_t:
                    continue

                elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                    self.add_state_and_video_packet(frame_data)
                    # Grid tick index of this row (tick 0 = start_time + 1/fps);
                    # used to pick the matching video frame at save time
                    frame_data["tick_index"] = int(round(
                        (episode_state_target_t - start_time) * self.fps
                    )) - 1
                    is_in_adding_phase = True

                elif timestamp > episode_action_target_t and is_in_adding_phase:
                    self._note_frame_time_gap(episode_state_target_t)
                    self.add_action(frame_data)
                    self._append_episode_frame(frame_data)
                    frame_data.clear()

                    time_gap = timestamp - episode_action_target_t
                    skipped_frames = time_gap // self.frame_duration

                    episode_state_target_t += (skipped_frames + 1) * self.frame_duration
                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration
                    is_in_adding_phase = False

                elif timestamp > episode_action_target_t and not is_in_adding_phase:
                    time_gap = timestamp - episode_action_target_t
                    skipped_frames = time_gap // self.frame_duration + 1

                    episode_state_target_t += skipped_frames * self.frame_duration
                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                    frame_data.clear()

        # Accumulate bag duration
        if _bag_first_ts is not None and _bag_last_ts is not None:
            bag_duration = _bag_last_ts - _bag_first_ts
            self._stats_rosbag_duration_s += bag_duration
        else:
            bag_duration = 0.0

        # Per-bag summary
        bag_xy_pairs   = self.num_xy_pairs        - _bag_xy_pairs_before
        bag_ep_saved   = self.total_episodes_saved - _bag_episodes_before
        self._stats_num_xy_pairs_total += bag_xy_pairs
        self.logger.info(
            f"   Bag duration: {bag_duration:.1f}s | "
            f"Marker pairs: {bag_xy_pairs} | "
            f"Episodes saved: {bag_ep_saved}"
        )

        # If bag ends while recording is still active, discard unfinished temp file.
        if self._ep_h5 is not None:
            self._finalize_episode_hdf5(keep=False)

        del reader


    def convert_all(self, task_description: str, MULTIBAG_FLAG: bool, ENFORCE_ALL_VIDEO_TOPICS_FLAG: bool):
        """Convert all discovered rosbags to a multi-episode HDF5 dataset."""
        self.logger.info(f"Starting multi-bag conversion: {self.input_directory}")
        _convert_start = time.time()

        # Clean up temp files left over from a previous crashed run
        # (also matches the pre-rename "*.tmp.hdf5" naming).
        for pattern in ("episode_*.hdf5.tmp", "episode_*.tmp.hdf5"):
            for stale in self.output_directory.glob(pattern):
                self.logger.warning(f"Removing stale temp file from a previous run: {stale}")
                stale.unlink()

        # Refuse to clobber existing episodes unless --overwrite is given.
        if not self.overwrite:
            existing_indices = []
            for f in self.output_directory.glob("episode_*.hdf5"):
                try:
                    existing_indices.append(int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if existing_indices and self.start_episode_idx <= max(existing_indices):
                self.logger.error(
                    f"Output directory {self.output_directory} already contains episodes up to "
                    f"index {max(existing_indices)}, and new episodes would start at "
                    f"{self.start_episode_idx}. Use --start_episode_idx {max(existing_indices) + 1} "
                    f"to append, or --overwrite to replace existing files."
                )
                return False

        # Discover all rosbags
        rosbags = self.discover_rosbags(MULTIBAG_FLAG)
        total_rosbags = len(rosbags)
        if not rosbags:
            self.logger.error("No rosbag found!")
            return False

        # Convert each rosbag
        processed_rosbags = 0
        for rosbag in rosbags:
            self.convert_single_bag(rosbag, task_description, ENFORCE_ALL_VIDEO_TOPICS_FLAG)
            processed_rosbags += 1
            self.logger.info(f"[{processed_rosbags}/{total_rosbags}] Finished processing rosbag: {rosbag.get('name')}")

        _convert_elapsed = time.time() - _convert_start

        # Compute HDF5 output size
        hdf5_size_bytes = sum(
            f.stat().st_size
            for f in self.output_directory.rglob("*.hdf5")
            if f.is_file()
        )

        # Derived statistics
        episodes_saved_this_run = self.total_episodes_saved - self.start_episode_idx
        hdf5_duration_s    = self._stats_hdf5_total_frames / self.fps if self.fps > 0 else 0.0
        valid_ratio        = (episodes_saved_this_run / self._stats_num_xy_pairs_total * 100
                              if self._stats_num_xy_pairs_total > 0 else 0.0)
        rosbag_size_gb     = self._stats_rosbag_size_bytes / 1e9
        hdf5_size_gb       = hdf5_size_bytes / 1e9

        # Format seconds as "Xm Ys" for readability
        def _fmt_duration(secs: float) -> str:
            secs = int(secs)
            m, s = divmod(secs, 60)
            return f"{m}m {s}s" if m else f"{s}s"

        sep = "=" * 52
        self.logger.info(f"\n{sep}")
        self.logger.info(f"  转换完成摘要 / Conversion Summary")
        self.logger.info(sep)
        self.logger.info(f"  任务               : {task_description}")
        self.logger.info(f"  原始 rosbag 大小   : {rosbag_size_gb:.2f} GB")
        self.logger.info(f"  原始 rosbag 时长   : {_fmt_duration(self._stats_rosbag_duration_s)}  ({self._stats_rosbag_duration_s:.1f}s)")
        self.logger.info(f"  HDF5 数据片段数    : {episodes_saved_this_run} 段")
        self.logger.info(f"  HDF5 数据总时长    : {_fmt_duration(hdf5_duration_s)}  ({hdf5_duration_s:.1f}s)")
        self.logger.info(f"  HDF5 数据大小      : {hdf5_size_gb:.2f} GB")
        self.logger.info(f"  图像编码           : JPEG (quality={self.jpeg_quality}, workers={self.jpeg_workers})")
        self.logger.info(f"  有效 episode 占比  : {episodes_saved_this_run}/{self._stats_num_xy_pairs_total}  ({valid_ratio:.1f}%)")
        self.logger.info(f"  转换耗时           : {_fmt_duration(_convert_elapsed)}  ({_convert_elapsed:.1f}s)")
        self.logger.info(f"  输出目录           : {self.output_directory}")
        self.logger.info(sep)
        return True


def setup_logging_to_file(log_file_path):
    """Setup logging and print output to file, terminal only shows process ID"""
    # Get process ID and display in terminal (before redirection)
    pid = os.getpid()
    print(f"Process ID: {pid}", file=sys.__stderr__)
    sys.__stderr__.flush()

    # Open log file (append mode)
    log_file = open(log_file_path, 'a', encoding='utf-8')

    # Write startup information to log file
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Process started - Process ID: {pid} - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()

    # Create a class to write output to file
    class FileOutput:
        def __init__(self, file):
            self.file = file

        def write(self, text):
            # Only write to file, not to terminal
            self.file.write(text)
            self.file.flush()

        def flush(self):
            self.file.flush()

    # Redirect stdout and stderr to file
    sys.stdout = FileOutput(log_file)
    sys.stderr = FileOutput(log_file)

    # Configure logging to also output to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # This will write to file because stdout has been redirected
        ],
        force=True  # Force reconfiguration
    )

    return log_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bags to HDF5 with per-timestep JPEG bytes (RobotWin style)"
    )
    parser.add_argument("--multibag",
                        action="store_true", # If not in command, defaults to false
                        help="Whether input_directory contains multiple rosbags, pass True if yes, False if no")
    parser.add_argument("--enforce_all_video_topics",
                        action="store_true",
                        help="Enforce that rosbag must have all video topics, if you don't want to enforce this, you can comment out unwanted topics in self.video_topics")
    parser.add_argument("--input_directory",
                        default="./data/rosbags",
                        help="Directory containing ROS2 bag segments")
    parser.add_argument("--output_directory", "-o",
                        default="./data/hdf5_output",
                        help="Output directory for HDF5 episode files")
    parser.add_argument("--fps", type=int,
                        default=45,
                        help="Target FPS for dataset")
    parser.add_argument("--task",
                        default="task description",
                        help="Task description")
    parser.add_argument("--start_episode_idx", type=int,
                        default=0,
                        help="Starting episode index for output naming (e.g., 88 starts from episode_000088.hdf5)")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Allow overwriting existing episode_*.hdf5 files in the output directory")
    parser.add_argument("--jpeg_quality", type=int,
                        default=85,
                        help="JPEG quality for per-frame compressed image bytes (1-100, higher=better quality/larger size)")
    parser.add_argument("--jpeg_workers", type=int,
                        default=0,
                        help="Parallel JPEG encoder worker threads (0=sync, 2/4 usually faster for multi-camera)")
    parser.add_argument("--log_file",
                        default=None,
                        help="Log file path (default: convert_rosbag_to_hdf5_YYYYMMDD_HHMMSS.log in script directory)")
    args = parser.parse_args()

    # Set log file path
    if args.log_file is None:
        script_dir = Path(__file__).parent
        logs_dir = script_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = logs_dir / f"convert_rosbag_to_hdf5_{timestamp}.log"
    else:
        log_file_path = Path(args.log_file)

    # Set up log redirection
    log_file = setup_logging_to_file(log_file_path)

    try:
        if not os.path.exists(args.input_directory):
            print(f"Error: Input directory {args.input_directory} not found!")
            sys.exit(1)

        converter = MultiVideoRosBag2HDF5Converter(
            args.input_directory,
            args.output_directory,
            args.fps,
            args.start_episode_idx,
            args.jpeg_quality,
            args.jpeg_workers,
            overwrite=args.overwrite,
        )
        success = converter.convert_all(args.task, args.multibag, args.enforce_all_video_topics)
        if not success:
            sys.exit(1)
    finally:
        # Restore original stdout and stderr
        try:
            converter.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
