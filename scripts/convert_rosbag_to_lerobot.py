#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Convert ROS2 bag files with X/Y episode markers to LeRobot dataset format v2.1.

This script processes rosbag files that contain multiple episodes marked by
operator button presses (X button to start, Y button to end recording).

The robot's camera streams are H.265 with inter-coded GOPs, so every episode
is re-encoded such that EVERY output frame is an IDR keyframe (GOP=1 — any
frame can be decoded independently, which training dataloaders rely on for
random access). The pipeline runs fully on the GPU when an NVIDIA card is
available (NVDEC decode → NVENC encode, frames never leave GPU memory) and
falls back to libx265 on CPU otherwise:

  1. Raw camera packets are muxed into a temporary MPEG-TS carrying their
     real bag timestamps, preserving true camera timing across dropped frames.
  2. One ffmpeg pass per camera re-encodes to constant fps aligned with the
     dataset ticks: video frame k is the latest camera frame at data row k's
     sampling instant, and each video ends up with exactly as many frames as
     the episode has rows.
  3. Episodes encode on background workers while the main thread keeps
     reading the bag; all cameras of an episode encode concurrently and
     finished episodes are saved in order.

Usage:
    First, ensure the correct Python environment is activated:
    ```bash
    conda activate rosbag2lerobot
    unset PYTHONPATH
    ```

    Process multiple rosbag files:
    ```bash
    rm -rf ~/.cache/huggingface/lerobot/username/dataset_name
    python scripts/convert_rosbag_to_lerobot.py \
        --multibag \
        --input_directory ./data/rosbags \
        --output username/dataset_name \
        --fps 45 \
        --task "task description"
    ```

    Process a single rosbag file:
    ```bash
    rm -rf ~/.cache/huggingface/lerobot/username/dataset_name
    python scripts/convert_rosbag_to_lerobot.py \
        --input_directory ./data/rosbags \
        --output username/dataset_name \
        --fps 45 \
        --task "task description"
    ```

    Enforce requirement for all video topics:
    ```bash
    rm -rf ~/.cache/huggingface/lerobot/username/dataset_name
    python scripts/convert_rosbag_to_lerobot.py \
        --multibag \
        --enforce_all_video_topics \
        --input_directory ./data/rosbags \
        --output username/dataset_name \
        --task "task description"
    ```

    Treat an entire bag as one continuous episode (no X/Y markers needed):
    ```bash
    python scripts/convert_rosbag_to_lerobot.py \
        --whole_bag \
        --input_directory ./data/rosbags \
        --output username/dataset_name \
        --task "task description"
    ```

    Useful options:
      --crf N             encoder quality (constant QP on GPU / CRF on CPU,
                          lower = better quality, default 23)
      --nvenc_sessions N  max concurrent GPU encode sessions (default 7;
                          lower to 5 or 3 on NVIDIA drivers older than 550/522)
      --max_episodes N    stop after N episodes (for quick testing)
      --verbose           print progress to terminal instead of a log file
"""

import os
import sys
import time
import threading
import numpy as np
import argparse
from pathlib import Path
import logging
import shutil
import tempfile
import subprocess
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction

# PyAV: used to mux raw HEVC packets into MPEG-TS with real bag timestamps
try:
    import av
    av.logging.set_level(av.logging.ERROR)
except ImportError as e:
    print(f"PyAV dependency not found: {e}")
    sys.exit(1)

# ROS2 imports
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Pose
    from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
    from sensor_msgs.msg import Joy
except ImportError as e:
    print(f"ROS2 dependencies not found: {e}")
    sys.exit(1)

# LeRobot imports
try:
    from lerobot.record import sample_frames_from_video
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"LeRobot dependencies not found: {e}")
    sys.exit(1)

# Constants
STATE_ACTION_DIM = 72  # 62 base + 10 chassis/kinco (same indices for state & action)
MIN_EPISODE_LENGTH = 30
ACTION_OFFSET_RATIO = 1.0 / 3.0


# =============================================================================
# HEVC NAL unit utilities (for IDR detection)
# =============================================================================

_NAL_START3 = b'\x00\x00\x01'


def _iter_nal_types(data: bytes):
    """Yield NAL unit types in annex-B HEVC data (handles 3- and 4-byte start codes)."""
    pos = data.find(_NAL_START3)
    n = len(data)
    while pos != -1 and pos + 3 < n:
        yield (data[pos + 3] >> 1) & 0x3F
        pos = data.find(_NAL_START3, pos + 3)


def _has_idr(data: bytes) -> bool:
    """Check if HEVC data contains an IDR NAL unit (type 19 or 20).

    Stops at the first VCL NAL (type < 32): in a single access unit the first
    VCL NAL determines the picture type, so P-frame packets bail immediately.
    """
    for nal_type in _iter_nal_types(data):
        if nal_type < 32:
            return nal_type in (19, 20)
    return False


def _detect_gpu_count() -> int:
    """Count NVIDIA GPUs visible to CUDA (respects CUDA_VISIBLE_DEVICES).

    ffmpeg's -hwaccel_device/-gpu indices live in the same masked CUDA
    enumeration, so torch.cuda.device_count() (torch is already loaded via
    lerobot) is the authoritative source; nvidia-smi is only a fallback.
    """
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 0
        return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


# Consumer NVIDIA drivers cap concurrent NVENC sessions system-wide (8 on
# driver >=550, 5 on 522-549, 3 before that). Gate ffmpeg GPU-encode
# processes so an over-subscribed episode pipeline queues instead of failing
# over to the slow CPU path. Tune with --nvenc_sessions on older drivers.
_NVENC_SESSIONS = threading.BoundedSemaphore(7)


# =============================================================================
# All-IDR Video Buffer (replaces VideoPacketBuffer)
# =============================================================================

def normalize_codec_name(codec: str) -> str:
    """Normalize encoder/codec names to PyAV-compatible codec names."""
    codec_lower = codec.lower()
    if any(name in codec_lower for name in ["hevc", "h265", "x265", "hevc_vaapi"]):
        return "hevc"
    if any(name in codec_lower for name in ["h264", "avc", "x264"]):
        return "h264"
    return codec


class AllIdrVideoBuffer:
    """
    Buffers pre-encoded video packets (plus their bag timestamps) and re-encodes
    each episode to MP4 where EVERY frame is an IDR keyframe (GOP=1).

    Per camera the encode pipeline is:
      1. Mux the raw HEVC packets into a temp MPEG-TS with real PTS taken from
         the bag timestamps (PyAV, milliseconds of work). This preserves the
         true 45fps-with-drops timing instead of trusting SPS/VUI nominal rate.
      2. One ffmpeg pass entirely on the GPU: NVDEC (hevc_cuvid) decode with
         `-hwaccel_output_format cuda` so frames never leave GPU memory, then
         NVENC (hevc_nvenc) all-IDR encode. The output is CFR-resampled to the
         dataset fps, aligned to the episode's first state tick via `-ss`, and
         capped to exactly `episode_length` frames so video frame k matches
         data row k.
    All cameras of an episode run concurrently (one ffmpeg process each); the
    stereo cameras finish well inside the head camera's encode time.

    Example:
        ```python
        buffer = AllIdrVideoBuffer(root_dir="dataset", fps=45)
        buffer.add_packet("left_color", pkt, width=2560, height=800, codec="hevc", ts=bag_t)
        buffer.encode_episode(episode_index=0, dataset_meta=meta,
                              first_tick_time=start + 1/45, episode_length=900)
        ```
    """

    def __init__(self, root_dir: Path, fps: int, crf: int = 23, preset: str = "ultrafast"):
        """
        Initialize all-IDR video buffer.

        Args:
            root_dir: Root directory for the dataset
            fps: Frames per second for video
            crf: Quantizer for hevc_nvenc (-qp) / CRF for the libx265 CPU fallback
            preset: libx265 preset for the CPU fallback (GPU path always uses p1)
        """
        self.root_dir = Path(root_dir)
        self.fps = fps
        self.crf = crf
        self.preset = preset

        # Buffer: {camera_name: [packet_dict, ...]}
        self.packets = defaultdict(list)

        # Video info: {camera_name: dict}
        self.video_info = {}

    def add_packet(
        self, camera_name: str, packet_data: bytes,
        width: int, height: int, codec: str,
        ts: float = 0.0,
    ) -> None:
        """
        Add a packet to the buffer.

        Args:
            camera_name: Name of the camera
            packet_data: Raw packet bytes
            width: Video width
            height: Video height
            codec: Codec name (e.g., "hevc", "h264")
            ts: Bag timestamp of the packet in seconds (drives output PTS)
        """
        # Store video info from first packet
        if camera_name not in self.video_info:
            normalized_codec = normalize_codec_name(codec)
            self.video_info[camera_name] = {
                'width': width,
                'height': height,
                'codec': normalized_codec,
            }

        # Buffer packet data
        self.packets[camera_name].append({
            "data": packet_data,
            "ts": ts,
        })

    def _mux_to_ts(self, packet_list, info) -> str:
        """Mux raw annex-B packets into a temp MPEG-TS with PTS from bag timestamps."""
        # Prefer RAM-backed /dev/shm: the .ts is a pure intermediate that
        # ffmpeg reads back immediately, no reason to round-trip the disk
        tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
        fd, ts_path = tempfile.mkstemp(suffix=".ts", prefix="lerobot_ep_", dir=tmp_dir)
        os.close(fd)
        container = av.open(ts_path, "w", format="mpegts")
        stream = container.add_stream(info['codec'])
        stream.width = info['width']
        stream.height = info['height']
        if info['codec'] == 'hevc':
            # We never encode through this context (packets are muxed as-is),
            # but PyAV still opens it — keep libx265's banner out of the logs
            stream.codec_context.options = {'x265-params': 'log-level=none'}
        time_base = Fraction(1, 90000)
        base_ts = packet_list[0]['ts']
        last_pts = -1
        for p in packet_list:
            pkt = av.Packet(p['data'])
            pts = int(round((p['ts'] - base_ts) * 90000))
            if pts <= last_pts:  # guard against duplicate/rounded-equal timestamps
                pts = last_pts + 1
            last_pts = pts
            pkt.pts = pts
            pkt.dts = pts
            pkt.time_base = time_base
            pkt.stream = stream
            container.mux(pkt)
        container.close()
        return ts_path

    def encode_episode(
        self, episode_index: int, dataset_meta,
        first_tick_time: float, episode_length: int, gpu_index: int = 0,
        camera_resolutions: dict = None,
    ) -> dict:
        """
        Encode all buffered cameras to all-IDR MP4s, one concurrent ffmpeg each,
        then sample stats frames from the encoded videos (still on the worker).

        Args:
            episode_index: Index of the episode (determines output paths)
            dataset_meta: Dataset metadata object with get_video_file_path()
            first_tick_time: Bag time of the first state tick (start_time + 1/fps);
                video frame 0 is aligned to this instant
            episode_length: Number of data frames; output videos are capped to
                exactly this many frames
            gpu_index: Which GPU to run decode+encode on
            camera_resolutions: {camera_name: (height, width, channels)} for
                the stats sampler

        Returns:
            {video_key: sampled_frames ndarray} for successfully sampled cameras
        """
        _t_ep = time.perf_counter()
        jobs = []
        for camera_name, packet_list in self.packets.items():
            if len(packet_list) == 0:
                continue
            video_key = f"observation.images.{camera_name}"
            video_path = self.root_dir / dataset_meta.get_video_file_path(
                episode_index, video_key
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            # Encode to <name>.mp4.tmp and rename at save time: LeRobot's
            # save_episode asserts the number of *.mp4 files on disk matches
            # the number of saved episodes, and later episodes may finish
            # encoding before earlier ones are saved.
            tmp_path = video_path.parent / (video_path.name + ".tmp")
            jobs.append((camera_name, packet_list, tmp_path))

        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as cam_pool:
            futures = [
                cam_pool.submit(
                    self._encode_camera, camera_name, packet_list, tmp_path,
                    first_tick_time, episode_length, gpu_index,
                )
                for camera_name, packet_list, tmp_path in jobs
            ]
            for future in futures:
                future.result()  # propagate camera encode failures (fatal)

        _t_sample = time.perf_counter()
        sampled = {}
        for camera_name, _packet_list, tmp_path in jobs:
            video_key = f"observation.images.{camera_name}"
            if not tmp_path.exists():
                continue
            try:
                height, width, _ = camera_resolutions[camera_name]
                sampled[video_key] = sample_frames_from_video(
                    video_path=tmp_path,
                    episode_length=episode_length,
                    fps=self.fps,
                    width=width,
                    height=height,
                )
            except Exception as e:
                logging.error(
                    f"Failed to sample frames from {camera_name}: {e}\n"
                    f"Traceback (most recent call last):\n{traceback.format_exc()}"
                )

        logging.info(
            f"  [PERF encode ep{episode_index} gpu{gpu_index}] "
            f"{len(jobs)} cams, {episode_length} frames each: "
            f"{time.perf_counter() - _t_ep:.2f}s "
            f"(sample: {time.perf_counter() - _t_sample:.2f}s)"
        )
        self.clear()
        return sampled

    def _encode_camera(
        self, camera_name, packet_list, tmp_path,
        first_tick_time, episode_length, gpu_index,
    ) -> None:
        """Mux one camera's packets and run its GPU (or fallback CPU) encode."""
        info = self.video_info[camera_name]
        ts_path = self._mux_to_ts(packet_list, info)

        # Grid origin so that output frame k = the LATEST source frame at or
        # before (first state tick + k/fps). round=up gives this floor
        # semantics — identical to the old GOP=1 pipeline, which grabbed the
        # most recent packet on hand at each capture tick (and causally
        # correct: at time T the policy can only have seen frames <= T).
        # Packets are buffered from the last IDR before the X press, so the
        # stream starts earlier than the tick; the fps filter drops the
        # decoded-but-pre-tick frames. (fps/setpts are timestamp-only
        # filters: they pass CUDA hw frames through untouched, so the
        # pipeline stays fully on the GPU.)
        grid_start = max(first_tick_time - packet_list[0]['ts'], 0.0)

        # Raw packets are fully persisted in the temp .ts now — release them
        # so in-flight episodes don't hold whole episodes of HEVC in RAM
        packet_list.clear()
        resample_vf = (
            f"fps={self.fps}:start_time={grid_start:.6f}:round=up,"
            f"setpts=PTS-STARTPTS"
        )

        gpu_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-hwaccel", "cuda",
            "-hwaccel_device", str(gpu_index),
            "-hwaccel_output_format", "cuda",   # keep decoded frames on the GPU
            "-c:v", "hevc_cuvid",
            "-i", ts_path,
            "-vf", resample_vf,
            "-frames:v", str(episode_length),
            "-c:v", "hevc_nvenc",
            "-gpu", str(gpu_index),
            "-g", "1",
            "-forced-idr", "1",
            "-strict_gop", "1",
            "-preset", "p1",
            # -qp (constant quantizer) instead of -rc vbr -cq: every frame
            # gets the identical quantizer, so no brightness/quality
            # variation between consecutive IDR frames.
            "-qp", str(self.crf),
            "-color_range", "pc",       # full range (original is yuvj420p)
            "-f", "mp4",
            str(tmp_path),
        ]
        try:
            with _NVENC_SESSIONS:
                result = subprocess.run(gpu_cmd, capture_output=True, text=True)
            if result.returncode != 0 or not tmp_path.exists():
                gpu_err = result.stderr.strip()[:300] if result.stderr else "unknown"
                logging.warning(
                    f"  GPU pipeline failed for {camera_name} ({gpu_err}), "
                    f"falling back to CPU..."
                )
                self._encode_cpu(ts_path, tmp_path, resample_vf, episode_length)
        finally:
            if os.path.exists(ts_path):
                os.unlink(ts_path)

        self._pad_tail_frames(tmp_path, camera_name, episode_length)

    def _pad_tail_frames(self, tmp_path, camera_name, episode_length) -> None:
        """Clone the final encoded packet until the video has episode_length frames.

        A camera's last packet can precede the episode's final state tick (the
        Y press lands between camera frames), leaving the CFR output 1 frame
        short. Because the output is all-IDR, duplicating the last *encoded*
        packet reproduces the exact same picture — a pure remux, no re-encode.
        """
        with av.open(str(tmp_path)) as probe:
            n_frames = probe.streams.video[0].frames
        missing = episode_length - n_frames
        if missing <= 0:
            return
        if missing > 5:
            logging.warning(
                f"  {camera_name}: video is {missing} frames short of "
                f"{episode_length} — camera dropout? Padding anyway."
            )

        padded_path = tmp_path.parent / (tmp_path.name + ".pad")
        src = av.open(str(tmp_path))
        dst = av.open(str(padded_path), "w", format="mp4")
        in_stream = src.streams.video[0]
        out_stream = dst.add_stream_from_template(in_stream)
        last = None
        for pkt in src.demux(in_stream):
            if pkt.dts is None:
                continue
            last = (bytes(pkt), pkt.pts, pkt.duration, pkt.time_base)
            pkt.stream = out_stream
            dst.mux(pkt)
        data, last_pts, duration, time_base = last
        step = duration or int(round(1 / (self.fps * time_base)))
        for i in range(1, missing + 1):
            pad = av.Packet(data)
            pad.pts = pad.dts = last_pts + i * step
            pad.duration = duration
            pad.time_base = time_base
            pad.stream = out_stream
            pad.is_keyframe = True
            dst.mux(pad)
        dst.close()
        src.close()
        os.replace(padded_path, tmp_path)
        logging.info(f"  {camera_name}: padded {missing} tail frame(s) by cloning the last IDR packet")

    def _encode_cpu(self, ts_path, tmp_path, resample_vf, episode_length) -> None:
        """CPU fallback: software decode + libx265 all-IDR encode (same alignment)."""
        cpu_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", ts_path,
            "-vf", resample_vf,
            "-frames:v", str(episode_length),
            "-c:v", "libx265",
            "-x265-params", "keyint=1:min-keyint=1:open-gop=0:log-level=error",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-color_range", "pc",       # full range (original is yuvj420p)
            "-f", "mp4",
            str(tmp_path),
        ]
        result = subprocess.run(cpu_cmd, capture_output=True, text=True)
        if result.returncode != 0 or not tmp_path.exists():
            raise IOError(f"CPU encode failed: {result.stderr.strip()[:500]}")
        logging.info(f"  CPU complete: {tmp_path}")

    def clear(self) -> None:
        """Clear all buffered packets."""
        self.packets.clear()
        self.video_info.clear()


# =============================================================================
# Main converter class
# =============================================================================

class MultiVideoRosBag2LeRobotConverter:
    """Enhanced converter for multiple ROS2 bags to single LeRobot dataset with multiple episodes."""

    def __init__(self, input_directory: str, output_repo_id: str, fps: int = 45, crf: int = 23):
        self.input_directory = Path(input_directory)
        self.output_repo_id = output_repo_id
        self.fps = fps
        self.crf = crf
        self.max_episodes = None
        self.frame_duration = 1.0 / self.fps

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Video topics mapping - all 7 cameras from your robot
        self.video_topics = {
            'left_color':  '/left/color/image_raw/ffmpeg',
            'right_color': '/right/color/image_raw/ffmpeg',
            'head_camera': '/xr_video_topic/ffmpeg',
            # 'chest_camera':'/head/color/image_raw/ffmpeg',
        }
        # Video topics set
        self.video_topics_set = set(self.video_topics.values())
        # State topics set
        self.state_topics_set = {
            '/left_arm/joint_states', '/left_arm/current_ee_pose', '/left_gripper/joint_states',
            '/right_arm/joint_states','/right_arm/current_ee_pose','/right_gripper/joint_states',
            '/chassis/joint_states', '/kinco/actual_velocity',
        }
        # Action topics set
        self.action_topics_set = {
            '/left_arm/joint_cmd', '/left_arm/target_ee_pose', '/left_gripper/joint_cmd',
            '/right_arm/joint_cmd','/right_arm/target_ee_pose','/right_gripper/joint_cmd',
            '/chassis/joint_cmd', '/kinco/cmd_velocity',
        }
        # All topics set
        self.all_topics_set = self.video_topics_set | self.state_topics_set | self.action_topics_set

        # Initialize LeRobot dataset (will be created once)
        self.dataset = None

        # Episode tracking
        self.num_xy_pairs = 0
        # state tracking
        self._cur_state_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        # video_packet tracking
        self._cur_video_packet = {camera_key: None for camera_key in self.video_topics.keys()}
        # action tracking
        self._cur_action_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)

        # Reverse topic -> camera_key map (avoids a linear scan per message)
        self.topic_to_camera = {topic: cam for cam, topic in self.video_topics.items()}
        # Reusable all-zero placeholder frames, one per camera resolution
        self._black_frame_cache = {}

        # Lazy state/action deserialization: raw CDR bytes are cached per
        # topic and only the latest message per topic is deserialized when a
        # frame is actually captured (identical result, ~10x fewer deserializes)
        self._dirty_state = {}
        self._dirty_action = {}

        # --- Encode pipeline: episodes encode on GPU in the background while
        # --- the main thread keeps reading the bag. Two workers per GPU
        # --- (one episode's 3-camera encode doesn't saturate a 4090).
        self.n_gpus = _detect_gpu_count()
        self.logger.info(f"Detected {self.n_gpus} NVIDIA GPU(s)")
        self._encode_workers = 2 * max(1, self.n_gpus)
        self._encode_pool = ThreadPoolExecutor(
            max_workers=self._encode_workers, thread_name_prefix="ep-encode"
        )
        self._pending = deque()   # episodes submitted for encode, FIFO
        self._episode_seq = 0     # round-robin GPU assignment
        self.max_pending = 2 * self._encode_workers


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
                # Find .db3 or .mcap files in rosbag directory
                for rosbag_dir in rosbag_folders:
                    rosbag_dir = Path(rosbag_dir)
                    db3_files = list(rosbag_dir.glob("*.db3")) or list(rosbag_dir.glob("*.mcap"))
                    if db3_files:
                        for db3_file in db3_files:
                            rosbags.append({
                                'name': rosbag_dir.name,
                                'path': str(rosbag_dir),
                                'bag_file': str(db3_file),
                            })

        else:
            # one rosbag
            db3_files = list(self.input_directory.glob("*.db3")) or list(self.input_directory.glob("*.mcap"))
            for i, db3_file in enumerate(sorted(db3_files)):
                rosbags.append({
                    'name': f"episode_{i:03d}",
                    'path': str(db3_file.parent),
                    'bag_file': str(db3_file)
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
        """Setup LeRobot dataset features with 48-dim data + video features."""
        features = {}

        # Complete 48-dimensional feature names
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
            # Chassis + Kinco (10 dims, neutral names — same layout in state & action)
            "chassis_motor1_position", "chassis_motor2_position", "chassis_motor3_position",
            "chassis_motor1_velocity", "chassis_motor2_velocity", "chassis_motor3_velocity",
            "chassis_motor1_effort", "chassis_motor2_effort", "chassis_motor3_effort",
            "kinco_velocity",
        ]

        # ACTION features
        features["action"] = {
            "dtype": "float32",
            "shape": (STATE_ACTION_DIM,),
            "names": feature_names,
        }

        # OBSERVATION features
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (STATE_ACTION_DIM,),
            "names": feature_names,
        }

        # VIDEO features - each camera as separate observation
        for camera_key in self.video_topics.keys():
            height, width, channels = self.get_camera_resolution(camera_key)

            features[f"observation.images.{camera_key}"] = {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": ["height", "width", "channels"],
            }

        return features

    def create_dataset_if_needed(self):
        """Create the LeRobot dataset if not already created."""
        if self.dataset is None:
            features = self.setup_features()

            self.dataset = LeRobotDataset.create(
                repo_id=self.output_repo_id,
                fps=self.fps,
                features=features,
                robot_type="teleavatar",
                use_videos=True
            )
            self.logger.info(f"Created LeRobot dataset: {self.output_repo_id}")

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

        # --- Chassis + Kinco (indices 62-71, +10 dims) ---
        elif topicName == '/chassis/joint_states':
            jd = self.extract_joint_data(msg)  # 3 motors × 3 = 9 values
            self._cur_state_msg[62:71] = jd[0:9]   # pos[0:3], vel[3:6], eff[6:9]
        elif topicName == '/kinco/actual_velocity':
            self._cur_state_msg[71] = msg.data     # Float64

    def _update_action_msg(self, topicName:str, msg):
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

        # --- Chassis + Kinco (indices 62-71, +10 dims) ---
        elif topicName == '/chassis/joint_cmd':
            jd = self.extract_joint_data(msg)  # 3 motors × 3 = 9 values
            self._cur_action_msg[62:71] = jd[0:9]  # pos[0:3], vel[3:6], eff[6:9]
        elif topicName == '/kinco/cmd_velocity':
            self._cur_action_msg[71] = msg.data    # Float64

    def _update_video_frame(self, topicName:str, msg:FFMPEGPacket):
        camera_key = self.topic_to_camera.get(topicName)
        if camera_key:
            # Use actual resolution from the FFMPEGPacket message instead of
            # hardcoded values. This ensures the video encoding parameters
            # match the actual bitstream dimensions.
            self._cur_video_packet[camera_key] = {
                'data': bytes(msg.data),
                'pts': msg.pts,
                'width': msg.width,
                'height': msg.height,
                'encoding': msg.encoding,
            }

    def update_messages(self, topicName, data):
        """Use a single interface to update state, video_frame, and action.

        Video packets are deserialized eagerly (they must be buffered
        continuously); state/action messages just cache their raw bytes —
        only the latest message per topic matters, so deserialization is
        deferred to the 30Hz capture ticks (see _materialize_state/_action).
        """
        if topicName in self.video_topics_set:
            msg_class = self.topic_msg_class.get(topicName)
            if msg_class is not None:
                self._update_video_frame(topicName, deserialize_message(data, msg_class))

        elif topicName in self.state_topics_set:
            if topicName in self.topic_msg_class:
                self._dirty_state[topicName] = data

        elif topicName in self.action_topics_set:
            if topicName in self.topic_msg_class:
                self._dirty_action[topicName] = data

    def _materialize_state(self):
        """Apply the latest cached raw message of each state topic."""
        if self._dirty_state:
            for topic, raw in self._dirty_state.items():
                self._update_state_msg(topic, deserialize_message(raw, self.topic_msg_class[topic]))
            self._dirty_state.clear()

    def _materialize_action(self):
        """Apply the latest cached raw message of each action topic."""
        if self._dirty_action:
            for topic, raw in self._dirty_action.items():
                self._update_action_msg(topic, deserialize_message(raw, self.topic_msg_class[topic]))
            self._dirty_action.clear()

    def _get_black_frame(self, camera_key):
        """Reusable zero placeholder frame (add_frame only records the shape)."""
        black = self._black_frame_cache.get(camera_key)
        if black is None:
            height, width, channels = self.get_camera_resolution(camera_key)
            black = np.zeros((height, width, channels), dtype=np.uint8)
            self._black_frame_cache[camera_key] = black
        return black

    def add_action(self, frame_data:dict):
        self._materialize_action()
        frame_data['action'] = self._cur_action_msg.copy()

    def realign_timestamps(self):
        """Re-align all timestamps in the episode_buffer to start at 0 and increase at 1/fps intervals."""
        if self.dataset.episode_buffer is None:
            return

        episode_length = self.dataset.episode_buffer["size"]
        if episode_length == 0:
            return

        aligned_timestamps = [i / self.fps for i in range(episode_length)]
        self.dataset.episode_buffer["timestamp"] = aligned_timestamps
        self.logger.info(f"⏹️ Realigned timestamps for {episode_length} frames, timestamp range: {aligned_timestamps[0]:.6f} to {aligned_timestamps[-1]:.6f} seconds")


    def _start_episode_capture(self, recent_packets, start_time):
        """Begin capturing an episode at an X press.

        Creates a fresh episode buffer + packet buffer and pre-fills the
        packet buffer from the sliding window, starting at each camera's last
        IDR before the X press so ffmpeg can decode the complete GOP.
        """
        self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        packet_buffer = AllIdrVideoBuffer(
            root_dir=self.dataset.root, fps=self.dataset.fps, crf=self.crf
        )
        _t = time.perf_counter()
        for ck in self.video_topics.keys():
            window = recent_packets[ck]
            # Find the last IDR in the window
            idr_idx = -1
            for i in range(len(window) - 1, -1, -1):
                if _has_idr(window[i]['data']):
                    idr_idx = i
                    break
            if idr_idx >= 0:
                for p in list(window)[idr_idx:]:
                    packet_buffer.add_packet(
                        camera_name=ck,
                        packet_data=p['data'],
                        width=p['width'],
                        height=p['height'],
                        codec=p['encoding'],
                        ts=p['ts'],
                    )
                self.logger.info(
                    f"  📹 {ck}: buffered {len(window) - idr_idx} pkts "
                    f"from IDR at {window[idr_idx]['ts']:.3f}s (X at {start_time:.3f}s)"
                )
        self._perf['idr_window_flush'] += time.perf_counter() - _t
        return packet_buffer

    def _submit_episode(self, packet_buffer, start_time, end_time, task_description):
        """Hand the finished episode to a background GPU-encode worker.

        The episode buffer is detached from the dataset and queued together
        with the encode future; `_drain_pending` later runs the (cheap) frame
        sampling + parquet save on the main thread, strictly in FIFO order so
        episode indices stay consistent.
        """
        duration = end_time - start_time
        self.logger.info(f"⏹️ Stop TimeRange: {start_time:.3f} to {end_time:.3f} seconds. Duration: {duration:.3f} seconds")

        self.num_xy_pairs += 1

        data_ep_len = self.dataset.episode_buffer["size"] if self.dataset.episode_buffer else 0

        if data_ep_len < MIN_EPISODE_LENGTH:
            self.dataset.clear_episode_buffer()
            packet_buffer.clear()
            self.logger.warning(f"  ⚠️ Episode too short, discarded")
            return

        self.realign_timestamps()

        # Detach the buffer: the main loop will create a fresh one at the next
        # X press, and _stage2_save reattaches this one when its encode is done.
        episode_buffer = self.dataset.episode_buffer
        self.dataset.episode_buffer = None

        # Video index must be pre-assigned: earlier episodes may still be
        # encoding, so dataset.num_episodes hasn't caught up yet.
        video_episode_index = self.dataset.num_episodes + len(self._pending)
        gpu_index = self._episode_seq % max(1, self.n_gpus)
        self._episode_seq += 1

        # Output video frame 0 is aligned to the first state tick.
        first_tick_time = start_time + self.frame_duration

        camera_resolutions = {
            cam: self.get_camera_resolution(cam) for cam in self.video_topics.keys()
        }
        future = self._encode_pool.submit(
            packet_buffer.encode_episode,
            episode_index=video_episode_index,
            dataset_meta=self.dataset.meta,
            first_tick_time=first_tick_time,
            episode_length=data_ep_len,
            gpu_index=gpu_index,
            camera_resolutions=camera_resolutions,
        )
        self._pending.append({
            'future': future,
            'buffer': episode_buffer,
            'episode_index': video_episode_index,
            'episode_length': data_ep_len,
            'video_duration': duration,
        })

    def _drain_pending(self, block: bool = False):
        """Save episodes whose encode has finished (FIFO).

        With block=False only completed heads are saved, unless the queue
        exceeds max_pending — then we wait on the oldest for backpressure.
        With block=True everything is drained (end of bag / max_episodes).
        """
        while self._pending:
            head = self._pending[0]
            must_wait = block or len(self._pending) > self.max_pending
            if not must_wait and not head['future'].done():
                break
            _t = time.perf_counter()
            # re-raises worker errors: encode failure is fatal
            head['sampled'] = head['future'].result()
            self._perf['encode_stall'] += time.perf_counter() - _t
            self._pending.popleft()
            self._stage2_save(head)

    def _stage2_save(self, item):
        """Sample stats frames from the encoded videos and save the episode."""
        episode_buffer = item['buffer']
        episode_index = item['episode_index']
        episode_length = item['episode_length']

        assert self.dataset.num_episodes == episode_index, (
            f"Episode index mismatch: dataset has {self.dataset.num_episodes}, "
            f"pending item expects {episode_index}"
        )
        # The buffer's index was assigned at creation time, before earlier
        # pending episodes were saved — fix it up to the real index.
        episode_buffer['episode_index'] = episode_index

        _t = time.perf_counter()
        sampled = item.get('sampled') or {}
        for camera_key in self.video_topics.keys():
            video_key = f"observation.images.{camera_key}"
            video_path = self.dataset.root / self.dataset.meta.get_video_file_path(
                episode_index, video_key
            )
            # Videos are encoded to <name>.mp4.tmp; move into place only now
            # so LeRobot's on-disk mp4-count assertion stays satisfied.
            tmp_path = video_path.parent / (video_path.name + ".tmp")
            if tmp_path.exists():
                os.replace(tmp_path, video_path)
            if not video_path.exists():
                logging.warning(f"Video not found: {video_path}, skipping stats for {camera_key}")
                continue
            if video_key in sampled:
                # Stats frames were already sampled on the encode worker
                episode_buffer[video_key] = sampled[video_key]
            else:
                logging.warning(f"No sampled stats frames for {camera_key}")
        # NOTE: actual stats sampling runs on the encode worker (see the
        # per-episode "[PERF encode ...]" log line); this only times the
        # rename + assignment.
        self._perf['stage2_finalize'] += time.perf_counter() - _t

        _t = time.perf_counter()
        in_flight_buffer = self.dataset.episode_buffer
        self.dataset.episode_buffer = episode_buffer
        try:
            self.dataset.save_episode()
        finally:
            self.dataset.episode_buffer = in_flight_buffer
        self._perf['dataset_save'] += time.perf_counter() - _t

        logging.info(f"✅ Saved episode {episode_index} with {episode_length} frames")
        self.logger.info(
            f"  [PERF cumulative] episode_video_dur={item['video_duration']:.1f}s | "
            + " | ".join(f"{k}={v:.2f}s" for k, v in sorted(self._perf.items()))
        )

    def convert_single_bag(self, rosbag, task_description: str, ENFORCE_ALL_VIDEO_TOPICS_FLAG: bool, WHOLE_BAG_FLAG: bool = False):
        """Convert a single bag to one or multiple episodes in the dataset."""
        self.logger.info(f"\n=== Processing {rosbag['name']} ===")
        self._perf = defaultdict(float)
        _t_bag = time.perf_counter()

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

        # Cache deserialization classes once per bag (get_message is not free
        # at ~1M calls per bag)
        self.topic_msg_class = {
            t: get_message(ty) for t, ty in self.topic_types_dict.items()
        }
        # Drop any raw messages cached from a previous bag
        self._dirty_state.clear()
        self._dirty_action.clear()

        if ENFORCE_ALL_VIDEO_TOPICS_FLAG is True:
            for video_topic_name in self.video_topics_set:
                if video_topic_name not in self.topic_types_dict:
                    del reader
                    shutil.rmtree(self.dataset.root/'images', ignore_errors=True)
                    return

        # Only pull topics we actually consume from the storage layer — this
        # skips /tf and other high-rate topics on the C++ side instead of
        # discarding them one by one in Python.
        needed_topics = list(self.all_topics_set | {'/xr/left_hand_inputs'})
        reader.set_filter(StorageFilter(topics=needed_topics))

        # =====================================================================
        # WHOLE_BAG MODE: Treat entire recording as one continuous episode.
        #
        # CRITICAL: Each camera starts buffering video packets from its OWN
        # first IDR (not from the global sync point). Otherwise, early-synced
        # cameras lose TRAIL frames during the waiting period, breaking the
        # GOP and causing snow at the start of their videos.
        # =====================================================================
        if WHOLE_BAG_FLAG:
            self.logger.info("📦 Whole-bag mode: treating entire recording as one episode")
            self.logger.info("⏳ Waiting for first IDR frame from all cameras...")

            frame_data = dict()
            packet_buffer = AllIdrVideoBuffer(root_dir=self.dataset.root, fps=self.dataset.fps, crf=self.crf)
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()

            episode_state_target_t = None
            episode_action_target_t = None
            is_in_adding_phase = False
            start_time = None
            idr_synced = False
            cameras_with_idr = set()      # cameras that have received first IDR
            expected_cameras = set(self.video_topics.keys())

            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                timestamp = timestamp / 1e9

                self.update_messages(topic, data)

                camera_key = self.topic_to_camera.get(topic)
                is_video = camera_key is not None

                # --- Per-camera IDR detection + immediate buffering ---
                # Each camera starts buffering from its OWN first IDR, even
                # while other cameras are still waiting. This preserves the
                # complete per-camera GOP structure.
                if is_video:
                    pkt = self._cur_video_packet.get(camera_key)
                    if pkt is not None:
                        # Detect first IDR for this camera
                        if camera_key not in cameras_with_idr and _has_idr(pkt['data']):
                            cameras_with_idr.add(camera_key)
                            self.logger.info(
                                f"  ✅ {camera_key} first IDR at {timestamp:.3f}s "
                                f"— buffering started"
                            )

                        # Buffer video packets (starting with its own first IDR)
                        # for any camera that has seen its first IDR
                        if camera_key in cameras_with_idr:
                            packet_buffer.add_packet(
                                camera_name=camera_key,
                                packet_data=pkt['data'],
                                width=pkt['width'],
                                height=pkt['height'],
                                codec=pkt['encoding'],
                                ts=timestamp,
                            )

                # --- Global IDR sync: all cameras have received first IDR ---
                if not idr_synced and cameras_with_idr >= expected_cameras:
                    idr_synced = True
                    start_time = timestamp
                    episode_state_target_t = start_time + self.frame_duration
                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration
                    self.logger.info(
                        f"🔴 All cameras IDR-synced at {start_time:.3f}s. "
                        f"Starting state/action capture."
                    )
                    continue  # Let next iteration handle frame sampling

                # --- Before global sync: only buffer video, skip state/action ---
                if not idr_synced:
                    continue

                # --- State/action frame sampling (fps grid, same as old script) ---
                if timestamp < episode_state_target_t:
                    continue

                elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                    # Only capture state here (video is already buffered above)
                    self._materialize_state()
                    frame_data["observation.state"] = self._cur_state_msg.copy()
                    for camera_key in self.video_topics.keys():
                        frame_data[f"observation.images.{camera_key}"] = self._get_black_frame(camera_key)
                    is_in_adding_phase = True

                elif timestamp > episode_action_target_t and is_in_adding_phase:
                    self.add_action(frame_data)
                    self.dataset.add_frame(frame_data, task_description,
                                           episode_state_target_t - start_time - self.frame_duration)
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

            # Finalize the single whole-bag episode
            end_time = timestamp
            self._submit_episode(packet_buffer, start_time, end_time, task_description)
            self._drain_pending(block=True)

            del reader
            shutil.rmtree(self.dataset.root/'images', ignore_errors=True)
            return

        # =====================================================================
        # X/Y MARKER MODE: Joy-based episode detection.
        # Buffers ALL video packets during recording (same as whole_bag mode)
        # to preserve the complete GOP for proper ffmpeg decoding.
        #
        # A sliding window of recent video packets lets each episode start
        # from the last IDR before the X press. Joy messages are used for
        # X/Y detection inline.
        # =====================================================================
        WINDOW_SIZE = 60
        recent_packets = {ck: deque(maxlen=WINDOW_SIZE) for ck in self.video_topics.keys()}

        x_button, y_button = 2, 3
        is_recording = False
        previous_buttons = None
        frame_data = dict()
        packet_buffer = None
        episode_state_target_t = None
        episode_action_target_t = None
        is_in_adding_phase = False

        while reader.has_next():
            _t = time.perf_counter()
            (topic, data, timestamp) = reader.read_next()
            self._perf['read_next'] += time.perf_counter() - _t
            timestamp = timestamp / 1e9

            _t = time.perf_counter()
            self.update_messages(topic, data)
            self._perf['update_messages'] += time.perf_counter() - _t

            # --- Maintain sliding window of recent video packets ---
            camera_key = self.topic_to_camera.get(topic)
            if camera_key:
                pkt = self._cur_video_packet.get(camera_key)
                if pkt is not None:
                    recent_packets[camera_key].append({
                        'ts': timestamp,
                        'data': pkt['data'],
                        'width': pkt['width'],
                        'height': pkt['height'],
                        'encoding': pkt['encoding'],
                    })

                    # --- Buffer ALL video packets during recording (full GOP) ---
                    if is_recording and packet_buffer is not None:
                        packet_buffer.add_packet(
                            camera_name=camera_key,
                            packet_data=pkt['data'],
                            width=pkt['width'],
                            height=pkt['height'],
                            codec=pkt['encoding'],
                            ts=timestamp,
                        )

            if topic == '/xr/left_hand_inputs':
                try:
                    buttons = deserialize_message(data, Joy).buttons

                    if len(buttons) > max(x_button, y_button):
                        if previous_buttons is not None:
                            if   (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and not is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Start #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                frame_data.clear()
                                packet_buffer = self._start_episode_capture(recent_packets, start_time)

                            elif (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Re-record #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                self.dataset.clear_episode_buffer()

                                is_in_adding_phase = False
                                frame_data.clear()
                                packet_buffer = self._start_episode_capture(recent_packets, start_time)

                            if (previous_buttons[y_button] == 0 and buttons[y_button] == 1 and is_recording):
                                is_recording = False
                                end_time = timestamp
                                self._submit_episode(packet_buffer, start_time, end_time, task_description)

                                episode_state_target_t = None
                                episode_action_target_t = None

                        previous_buttons = list(buttons)

                except Exception as e:
                    error_msg = f"Error processing Joy message: {e}\n"
                    error_msg += f"Traceback (most recent call last):\n"
                    error_msg += traceback.format_exc()
                    self.logger.error(error_msg)

            # Save episodes whose background encode has finished. Kept OUTSIDE
            # the Joy try/except so a save failure is fatal instead of being
            # swallowed per-message (which would desync episode indices).
            if self._pending:
                self._drain_pending(block=False)

                submitted = self.dataset.num_episodes + len(self._pending)
                if self.max_episodes and submitted >= self.max_episodes:
                    self.logger.info(f"Reached --max_episodes={self.max_episodes}, stopping")
                    self._drain_pending(block=True)
                    del reader
                    shutil.rmtree(self.dataset.root/'images', ignore_errors=True)
                    return

            if is_recording:
                if   timestamp < episode_state_target_t:
                    continue

                elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                    # Only capture state (video is already buffered above)
                    self._materialize_state()
                    frame_data["observation.state"] = self._cur_state_msg.copy()
                    for camera_key in self.video_topics.keys():
                        frame_data[f"observation.images.{camera_key}"] = self._get_black_frame(camera_key)
                    is_in_adding_phase = True

                elif timestamp > episode_action_target_t and is_in_adding_phase:
                    self.add_action(frame_data)
                    self.dataset.add_frame(frame_data, task_description,
                                           episode_state_target_t - start_time - self.frame_duration)
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

        # Wait for all in-flight episode encodes and save them
        self._drain_pending(block=True)

        del reader
        shutil.rmtree(self.dataset.root/'images', ignore_errors=True)
        self.logger.info(
            f"[PERF bag total] wall={time.perf_counter() - _t_bag:.2f}s | "
            + " | ".join(f"{k}={v:.2f}s" for k, v in sorted(self._perf.items()))
        )


    def convert_all(self, task_description: str, MULTIBAG_FLAG: bool, ENFORCE_ALL_VIDEO_TOPICS_FLAG: bool, WHOLE_BAG_FLAG: bool = False):
        """Convert all discovered rosbags to a multi-episode dataset."""
        self._t_start = time.time()
        self.logger.info(f"Starting multi-bag conversion: {self.input_directory}")

        # Discover all rosbags
        rosbags = self.discover_rosbags(MULTIBAG_FLAG)
        total_rosbags = len(rosbags)
        if not rosbags:
            self.logger.error("No rosbag found!")
            return False

        # Create the dataset
        self.create_dataset_if_needed()

        # Convert each rosbag
        processed_rosbags = 0
        for rosbag in rosbags:
            self.convert_single_bag(rosbag, task_description, ENFORCE_ALL_VIDEO_TOPICS_FLAG, WHOLE_BAG_FLAG)
            processed_rosbags += 1
            self.logger.info(f"[{processed_rosbags}/{total_rosbags}] Finished processing rosbag: {rosbag.get('name')}")

        # All per-bag work is drained; shut the encode workers down cleanly
        self._encode_pool.shutdown(wait=True)

        # Final summary
        elapsed = time.time() - self._t_start
        self.logger.info(f"\n🎉 Multi-bag conversion complete!")
        self.logger.info(f"   Dataset: {self.output_repo_id}")
        self.logger.info(f"   Episodes: {self.dataset.num_episodes} successful")
        self.logger.info(f"   Total time: {elapsed:.1f}s")


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
    parser = argparse.ArgumentParser(description="Convert multiple ROS2 bags to LeRobot dataset with all-IDR video re-encoding")
    parser.add_argument("--multibag",
                        action="store_true", # If not in command, defaults to false
                        help="Whether input_directory contains multiple rosbags, pass True if yes, False if no")
    parser.add_argument("--enforce_all_video_topics",
                        action="store_true", # If not in command, defaults to false
                        help="Enforce that rosbag must have four video topics, if you don't want to enforce this, you can comment out unwanted topics in self.video_topics")
    parser.add_argument("--input_directory",
                        default="./data/rosbags",
                        help="Directory containing ROS2 bag segments")
    parser.add_argument("--output", "-o",
                        default="output/dataset",
                        help="Output dataset repo ID")
    parser.add_argument("--fps", type=int,
                        default=45,
                        help="Target FPS for dataset")
    parser.add_argument("--task",
                        default="task description",
                        help="Task description")
    parser.add_argument("--whole_bag",
                        action="store_true",
                        help="Treat the entire rosbag as one continuous episode (no X/Y markers needed)")
    parser.add_argument("--crf", type=int,
                        default=23,
                        help="CRF quality for libx265 re-encoding (lower=better quality, default: 23)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Stop after N episodes (for quick testing)")
    parser.add_argument("--nvenc_sessions", type=int, default=7,
                        help="Max concurrent NVENC encode sessions; lower to 5/3 on "
                             "drivers older than 550/522 if GPU encodes fail (default: 7)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress to terminal instead of log file")
    parser.add_argument("--log_file",
                        default=None,
                        help="Log file path (default: convert_rosbag_to_lerobot_YYYYMMDD_HHMMSS.log in script directory)")
    args = parser.parse_args()

    global _NVENC_SESSIONS
    _NVENC_SESSIONS = threading.BoundedSemaphore(args.nvenc_sessions)

    if args.verbose:
        # Terminal mode: normal logging to stdout
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        if not os.path.exists(args.input_directory):
            print(f"Error: Input directory {args.input_directory} not found!")
            sys.exit(1)
        converter = MultiVideoRosBag2LeRobotConverter(args.input_directory, args.output, args.fps, crf=args.crf)
        converter.max_episodes = args.max_episodes
        converter.convert_all(args.task, args.multibag, args.enforce_all_video_topics, args.whole_bag)
    else:
        # Log file mode: redirect to file (original behavior)
        if args.log_file is None:
            script_dir = Path(__file__).parent
            logs_dir = script_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = logs_dir / f"convert_rosbag_to_lerobot_{timestamp}.log"
        else:
            log_file_path = Path(args.log_file)

        log_file = setup_logging_to_file(log_file_path)
        try:
            if not os.path.exists(args.input_directory):
                print(f"Error: Input directory {args.input_directory} not found!")
                sys.exit(1)
            converter = MultiVideoRosBag2LeRobotConverter(args.input_directory, args.output, args.fps, crf=args.crf)
            converter.convert_all(args.task, args.multibag, args.enforce_all_video_topics, args.whole_bag)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            log_file.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
