#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Convert ROS2 bag files with X/Y episode markers to RLDS-like TFRecord format.

This script follows the same recording and frame alignment logic as
`convert_rosbag_to_hdf5.py`.

Compared with low-level tf.train.Feature writing, this script defines data format via
`tfds.features` first, then encodes examples using `features.encode_example`.

The robot's camera streams are H.265 with inter-coded GOPs. Raw packets are
buffered per episode (starting at the last IDR keyframe before the X press so
the whole GOP chain is decodable), then decoded in one ffmpeg pass per camera
— NVDEC GPU-accelerated when available, software decode otherwise. Sampled
frames are cropped (head camera keeps its left half), resized to 224x224 and
stored as PNG bytes.

Output layout:
  output_directory/
    episode_000000.tfrecord
    episode_000001.tfrecord
    ...

Each TFRecord stores one serialized RLDS episode example, including
`episode_metadata` in the same record (no sidecar JSON metadata files).

Usage:
    First, ensure the correct Python environment is activated:
    ```bash
    conda activate rosbag2lerobot
    ```

    Process multiple rosbag files:
    ```bash
    python scripts/convert_rosbag_to_rlds.py \
        --multibag \
        --input_directory ./data/rosbags \
        --output_directory ./data/rlds_output \
        --fps 45 \
        --task "task description"
    ```

    Process a single rosbag file:
    ```bash
    python scripts/convert_rosbag_to_rlds.py \
        --input_directory ./data/rosbags \
        --output_directory ./data/rlds_output \
        --fps 45 \
        --task "task description"
    ```
"""

import os
import sys
import time
import gc
import numpy as np
import argparse
from pathlib import Path
import logging
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Suppress noisy TensorFlow startup logs. Keep this before importing tensorflow.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import example_serializer

# Keep TensorFlow off the GPU (it only encodes PNGs here) WITHOUT masking
# CUDA_VISIBLE_DEVICES: the ffmpeg subprocesses need the GPUs for NVDEC.
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

# Shared packet-buffering / GPU frame-extraction machinery (same directory).
# sys.path insert is required because the TFDS builder loads this file via
# importlib.spec_from_file_location, which does not add its directory to the
# module search path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from rosbag_video_extraction import (
        detect_gpu_count,
        extract_grid_frames,
        packets_from_last_idr,
    )
except ImportError as e:
    raise ImportError(
        f"rosbag_video_extraction module not found next to this script: {e}"
    ) from e


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
    raise ImportError(f"ROS2 dependencies not found: {e}") from e

# Constants
STATE_ACTION_DIM = 62
CHASSIS_DIM = 9   # 3 motors × (position + velocity + effort)
MIN_EPISODE_LENGTH = 30
# State->action capture offset in seconds (also the state-capture window
# width). Must stay < 1/fps.
ACTION_OFFSET_S = 1.0 / 90.0
# Warn when consecutive sampled frames are this many frame periods apart in real
# bag time (the synthetic step timestamps would silently hide the gap).
TIME_GAP_WARN_RATIO = 3
TARGET_IMAGE_SIZE = (224, 224)  # (height, width)


class MultiVideoRosBag2RLDSConverter:
    """Enhanced converter for multiple ROS2 bags to RLDS-like TFRecord episodes."""

    def __init__(self, input_directory: str, output_directory: str, fps: int = 45):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.fps = fps
        self.frame_duration = 1.0 / self.fps

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Ensure ROS plugin discovery env is available before opening rosbag2 readers.
        self._ensure_ros_runtime_environment()

        # Video topics mapping - all 3 cameras used by this pipeline
        self.video_topics = {
            'left_color': '/left/color/image_raw/ffmpeg',
            'right_color': '/right/color/image_raw/ffmpeg',
            'head_camera': '/xr_video_topic/ffmpeg',
        }
        # Video topics set
        self.video_topics_set = set(self.video_topics.values())
        # Camera info topics mapping (camera_key -> camera_info topic)
        # head_camera (/xr_video_topic/ffmpeg) has no dedicated camera_info in
        # recorded bags; its cameras metadata is stored with found=False.
        self.camera_info_topics = {
            'left_color': '/left/color/camera_info',
            'right_color': '/right/color/camera_info',
        }
        self.camera_info_topics_set = set(self.camera_info_topics.values())
        # Cache for camera intrinsics, populated once per bag before main loop
        self._camera_info_cache: dict = {}

        # State topics set
        self.state_topics_set = {
            '/left_arm/joint_states', '/left_arm/current_ee_pose', '/left_gripper/joint_states',
            '/right_arm/joint_states', '/right_arm/current_ee_pose', '/right_gripper/joint_states',
        }
        # Chassis topics
        self.chassis_state_topic = '/chassis/joint_states'
        self.chassis_action_topic = '/chassis/joint_cmd'
        self.chassis_topics_set = {self.chassis_state_topic, self.chassis_action_topic}
        # Action topics set
        self.action_topics_set = {
            '/left_arm/joint_cmd', '/left_arm/target_ee_pose', '/left_gripper/joint_cmd',
            '/right_arm/joint_cmd', '/right_arm/target_ee_pose', '/right_gripper/joint_cmd',
        }
        # All topics set
        self.all_topics_set = (self.video_topics_set | self.state_topics_set
                               | self.action_topics_set | self.chassis_topics_set)

        # Episode tracking
        self.num_xy_pairs = 0
        self._cur_state_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        self._cur_chassis_state_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        self._cur_chassis_action_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        self._target_image_h, self._target_image_w = TARGET_IMAGE_SIZE
        self._empty_rgb_frame = np.zeros((self._target_image_h, self._target_image_w, 3), dtype=np.uint8)
        self._empty_png_bytes = tf.io.encode_png(self._empty_rgb_frame).numpy()
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
        self._cur_video_packet = None

        # Raw HEVC packet buffering. GOP != 1: every episode must include the
        # packets from the last IDR before its X press, and every packet of
        # the episode, so the decoder can rebuild the reference chain.
        self._window_size = 60
        self._recent_packets = {ck: deque(maxlen=self._window_size) for ck in self.video_topics.keys()}
        self._ep_packets = None

        # NVDEC-accelerated decode when a GPU is present (CPU fallback inside
        # the extractor). TensorFlow itself is kept off the GPU via
        # tf.config.set_visible_devices at import time.
        self.n_gpus = detect_gpu_count()
        self._episode_seq = 0
        self.logger.info(f"Detected {self.n_gpus} NVIDIA GPU(s) for video decode")

        # Global episode index across all bags
        self.total_episodes_saved = 0

        # Statistics tracking
        self._stats_rosbag_size_bytes = 0
        self._stats_rosbag_duration_s = 0.0
        self._stats_rlds_total_frames = 0
        self._stats_num_xy_pairs_total = 0

        self.camera_shapes = {
            camera_key: self.get_camera_resolution(camera_key)
            for camera_key in self.video_topics.keys()
        }

        # Explicit RLDS schema using tfds.features
        self.rlds_episode_features = self._build_rlds_episode_features()
        self._serializer = example_serializer.ExampleSerializer(
            self.rlds_episode_features.get_serialized_info()
        )

    def _ensure_ros_runtime_environment(self):
        """Ensure AMENT plugin-discovery env vars are available for rosbag2/pluginlib."""
        if os.environ.get('AMENT_PREFIX_PATH', '').strip():
            return

        conda_prefix = os.environ.get('CONDA_PREFIX', '').strip()
        if conda_prefix:
            prefix_path = Path(conda_prefix)
            if (prefix_path / 'share' / 'ament_index').exists():
                os.environ['AMENT_PREFIX_PATH'] = str(prefix_path)
                if not os.environ.get('CMAKE_PREFIX_PATH', '').strip():
                    os.environ['CMAKE_PREFIX_PATH'] = str(prefix_path)
                self.logger.warning(
                    'AMENT_PREFIX_PATH was empty; fallback to %s for rosbag2 plugin discovery.',
                    prefix_path,
                )
                return

        raise RuntimeError(
            "Environment variable 'AMENT_PREFIX_PATH' is empty. "
            "Please run `conda activate rosbag2lerobot` (or source a ROS setup) before conversion."
        )

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
            return (2160, 4320, 3)
        else:
            return (480, 848, 3)

    def _reset_video_runtime_state(self):
        """Reset packet-buffer runtime state to avoid cross-bag accumulation."""
        self._recent_packets = {ck: deque(maxlen=self._window_size) for ck in self.video_topics.keys()}
        self._ep_packets = None
        self._cur_video_packet = None
        gc.collect()

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

    def _extract_camera_pngs(self, camera_key: str, first_tick_time: float,
                             n_grid: int, row_ticks: list, gpu_index: int) -> list:
        """Decode one camera's episode packets and PNG-encode the row frames.

        Frames are sampled on the fps grid (floor semantics), the head
        camera's side-by-side stereo image is cropped to its left half, and
        every frame is resized to 224x224 before PNG encoding. Returns a list
        of PNG bytes aligned with the episode rows.
        """
        packets = (self._ep_packets or {}).get(camera_key) or []
        if not packets:
            return [self._empty_png_bytes] * len(row_ticks)

        vf_post = "scale=%d:%d" % (self._target_image_w, self._target_image_h)
        if camera_key == 'head_camera':
            # Side-by-side stereo: keep the left half before resizing
            vf_post = "crop=iw/2:ih:0:0," + vf_post

        needed = set(row_ticks)
        by_tick = {}
        frames = extract_grid_frames(
            packets, first_tick_time, n_grid, self.fps,
            out_width=self._target_image_w, out_height=self._target_image_h,
            vf_post=vf_post,
            gpu_index=gpu_index, use_gpu=self.n_gpus > 0, logger=self.logger,
        )
        for tick, frame_bgr in enumerate(frames):
            if tick not in needed:
                continue
            frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            by_tick[tick] = tf.io.encode_png(frame_rgb).numpy()

        return [by_tick.get(tick, self._empty_png_bytes) for tick in row_ticks]

    def _inject_episode_images(self, episode_frames: list, start_time: float):
        """Decode all cameras (in parallel) and fill PNGs into episode_frames."""
        if not episode_frames:
            return
        row_ticks = [frame['tick_index'] for frame in episode_frames]
        first_tick_time = start_time + self.frame_duration
        n_grid = row_ticks[-1] + 1
        gpu_index = self._episode_seq % max(1, self.n_gpus)
        self._episode_seq += 1

        with ThreadPoolExecutor(max_workers=len(self.video_topics)) as cam_pool:
            futures = {
                camera_key: cam_pool.submit(
                    self._extract_camera_pngs, camera_key,
                    first_tick_time, n_grid, row_ticks, gpu_index,
                )
                for camera_key in self.video_topics.keys()
            }
            for camera_key, future in futures.items():
                pngs = future.result()
                for frame, png in zip(episode_frames, pngs):
                    frame[f'observation.{camera_key}'] = png
        for frame in episode_frames:
            frame.pop('tick_index', None)

    def _clear_episode_frames(self, episode_frames: list):
        """Explicitly release per-episode memory after save/skip."""
        for frame in episode_frames:
            frame.clear()
        episode_frames.clear()
        gc.collect()

    def _reset_message_caches(self):
        """Zero the state/action caches so a new episode cannot start with
        values left over from a previous episode or bag."""
        self._cur_state_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        self._cur_action_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        self._cur_chassis_state_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        self._cur_chassis_action_msg = np.zeros(CHASSIS_DIM, dtype=np.float32)
        self._dirty_state.clear()
        self._dirty_action.clear()

    def _make_camera_observation_feature(self, camera_key: str):
        """Store decoded camera image as PNG (224x224x3) for each step."""
        return tfds.features.Image(
            shape=(self._target_image_h, self._target_image_w, 3),
            dtype=np.uint8,
            encoding_format='png',
            doc=f'{camera_key} RGB image resized to 224x224 and encoded as PNG.',
        )

    def _make_camera_calibration_feature(self):
        return tfds.features.FeaturesDict({
            'found': tfds.features.Scalar(dtype=np.bool_),
            'K': tfds.features.Tensor(shape=(9,), dtype=np.float64),
            'D': tfds.features.Sequence(tfds.features.Scalar(dtype=np.float64)),
            'P': tfds.features.Tensor(shape=(12,), dtype=np.float64),
            'width': tfds.features.Scalar(dtype=np.int32),
            'height': tfds.features.Scalar(dtype=np.int32),
            'distortion_model': tfds.features.Text(),
        })

    def _build_rlds_episode_features(self):
        """Define explicit RLDS episode schema using tfds.features."""
        images_feature = {
            camera_key: self._make_camera_observation_feature(camera_key)
            for camera_key in self.video_topics.keys()
        }
        camera_meta_feature = {
            camera_key: self._make_camera_calibration_feature()
            for camera_key in self.video_topics.keys()
        }

        steps_feature = tfds.features.Dataset({
            'observation': tfds.features.FeaturesDict({
                'state': tfds.features.Tensor(
                    shape=(STATE_ACTION_DIM,),
                    dtype=np.float32,
                    doc='Robot state vector.'),
                'chassis_state': tfds.features.Tensor(
                    shape=(CHASSIS_DIM,),
                    dtype=np.float32,
                    doc='Chassis state vector.'),
                # Flatten camera PNG image observations directly under `observation`.
                **images_feature,
            }),
            'action': tfds.features.Tensor(
                shape=(STATE_ACTION_DIM,),
                dtype=np.float32,
                doc='Robot action vector.'),
            'chassis_action': tfds.features.Tensor(
                shape=(CHASSIS_DIM,),
                dtype=np.float32,
                doc='Chassis action vector.'),
            'timestamp': tfds.features.Scalar(dtype=np.float64),
            'discount': tfds.features.Scalar(dtype=np.float32),
            'reward': tfds.features.Scalar(dtype=np.float32),
            'is_first': tfds.features.Scalar(dtype=np.bool_),
            'is_last': tfds.features.Scalar(dtype=np.bool_),
            'is_terminal': tfds.features.Scalar(dtype=np.bool_),
            'language_instruction': tfds.features.Text(),
        })

        return tfds.features.FeaturesDict({
            'steps': steps_feature,
            'episode_metadata': tfds.features.FeaturesDict({
                'episode_idx': tfds.features.Scalar(dtype=np.int64),
                'n_steps': tfds.features.Scalar(dtype=np.int32),
                'fps': tfds.features.Scalar(dtype=np.int32),
                'task': tfds.features.Text(),
                'source_bag': tfds.features.Text(),
                'recording_time': tfds.features.Text(),
                'image_height': tfds.features.Scalar(dtype=np.int32),
                'image_width': tfds.features.Scalar(dtype=np.int32),
                'feature_names_state_action': tfds.features.Sequence(tfds.features.Text()),
                'feature_names_chassis': tfds.features.Sequence(tfds.features.Text()),
                'camera_keys': tfds.features.Sequence(tfds.features.Text()),
                'camera_topics': tfds.features.Sequence(tfds.features.Text()),
                'cameras': tfds.features.FeaturesDict(camera_meta_feature),
            }),
        })

    def setup_features(self):
        """Setup feature names for state/action dimensions."""
        return [
            "left_joint1_position", "left_joint2_position", "left_joint3_position", "left_joint4_position",
            "left_joint5_position", "left_joint6_position", "left_joint7_position", "left_gripper_position",
            "right_joint1_position", "right_joint2_position", "right_joint3_position", "right_joint4_position",
            "right_joint5_position", "right_joint6_position", "right_joint7_position", "right_gripper_position",
            "left_joint1_velocity", "left_joint2_velocity", "left_joint3_velocity", "left_joint4_velocity",
            "left_joint5_velocity", "left_joint6_velocity", "left_joint7_velocity", "left_gripper_velocity",
            "right_joint1_velocity", "right_joint2_velocity", "right_joint3_velocity", "right_joint4_velocity",
            "right_joint5_velocity", "right_joint6_velocity", "right_joint7_velocity", "right_gripper_velocity",
            "left_joint1_effort", "left_joint2_effort", "left_joint3_effort", "left_joint4_effort",
            "left_joint5_effort", "left_joint6_effort", "left_joint7_effort", "left_gripper_effort",
            "right_joint1_effort", "right_joint2_effort", "right_joint3_effort", "right_joint4_effort",
            "right_joint5_effort", "right_joint6_effort", "right_joint7_effort", "right_gripper_effort",
            "left_ee_position_x", "left_ee_position_y", "left_ee_position_z",
            "left_ee_orientation_x", "left_ee_orientation_y", "left_ee_orientation_z", "left_ee_orientation_w",
            "right_ee_position_x", "right_ee_position_y", "right_ee_position_z",
            "right_ee_orientation_x", "right_ee_orientation_y", "right_ee_orientation_z", "right_ee_orientation_w",
        ]

    def setup_chassis_features(self):
        return [
            "chassis_motor1_position", "chassis_motor2_position", "chassis_motor3_position",
            "chassis_motor1_velocity", "chassis_motor2_velocity", "chassis_motor3_velocity",
            "chassis_motor1_effort", "chassis_motor2_effort", "chassis_motor3_effort",
        ]

    def _collect_camera_info(self, rosbag: dict) -> dict:
        result = {}
        needed = set(self.camera_info_topics.values())

        try:
            bag_file = rosbag['bag_file']
            storage_id = 'mcap' if bag_file.endswith('.mcap') else 'sqlite3'
            storage_options = StorageOptions(uri=rosbag['path'], storage_id=storage_id)
            reader = SequentialReader()
            reader.open(storage_options, ConverterOptions('', ''))
            reader.set_filter(StorageFilter(topics=list(needed)))

            topic_to_key = {v: k for k, v in self.camera_info_topics.items()}

            while reader.has_next() and len(result) < len(needed):
                topic, data, _ = reader.read_next()
                if topic in needed and topic_to_key[topic] not in result:
                    msg: CameraInfo = deserialize_message(data, CameraInfo)
                    result[topic_to_key[topic]] = {
                        'K': np.array(msg.k, dtype=np.float64),
                        'D': np.array(msg.d, dtype=np.float64),
                        'P': np.array(msg.p, dtype=np.float64),
                        'width': msg.width,
                        'height': msg.height,
                        'distortion_model': msg.distortion_model,
                    }
            del reader
        except Exception as e:
            self.logger.warning(f"Could not collect camera_info from {rosbag['name']}: {e}")

        for camera_key in self.camera_info_topics:
            if camera_key in result:
                info = result[camera_key]
                self.logger.info(
                    f"  Camera {camera_key}: {info['width']}x{info['height']}  "
                    f"K=[{info['K'][0]:.2f}, {info['K'][4]:.2f}, {info['K'][2]:.2f}, {info['K'][5]:.2f}]"
                )
            else:
                self.logger.warning(f"  camera_info not found for {camera_key}")

        return result

    def _get_camera_key_from_topic(self, topic):
        for camera_key, camera_topic in self.video_topics.items():
            if topic == camera_topic:
                return camera_key
        return None

    def extract_joint_data(self, msg: JointState):
        positions = list(msg.position) if msg.position else []
        velocities = list(msg.velocity) if msg.velocity else []
        efforts = list(msg.effort) if msg.effort else []

        max_joints = 7
        for data_list in [positions, velocities, efforts]:
            if len(data_list) > max_joints:
                data_list[:] = data_list[:max_joints]
            while len(data_list) < max_joints:
                data_list.append(0.0)

        return np.array(positions + velocities + efforts, dtype=np.float32)

    def extract_gripper_data(self, msg: JointState):
        pos = msg.position[0] if msg.position else 0.0
        vel = msg.velocity[0] if msg.velocity else 0.0
        effort = msg.effort[0] if msg.effort else 0.0
        return np.array([pos, vel, effort], dtype=np.float32)

    def extract_chassis_data(self, msg: JointState):
        n_motors = 3
        positions = list(msg.position)[:n_motors] if msg.position else []
        velocities = list(msg.velocity)[:n_motors] if msg.velocity else []
        efforts = list(msg.effort)[:n_motors] if msg.effort else []
        for data_list in [positions, velocities, efforts]:
            while len(data_list) < n_motors:
                data_list.append(0.0)
        return np.array(positions + velocities + efforts, dtype=np.float32)

    def extract_ee_pose_data(self, msg: Pose):
        return np.array([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ], dtype=np.float32)

    def _update_state_msg(self, topicName: str, msg: JointState):
        if topicName == '/left_arm/joint_states':
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

    def _update_action_msg(self, topicName: str, msg: JointState):
        if topicName == '/left_arm/joint_cmd':
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

    def _update_chassis_msg(self, topicName: str, msg: JointState):
        chassis_data = self.extract_chassis_data(msg)
        if topicName == self.chassis_state_topic:
            self._cur_chassis_state_msg[:] = chassis_data
        elif topicName == self.chassis_action_topic:
            self._cur_chassis_action_msg[:] = chassis_data

    def update_messages(self, topicName, data):
        """Update state/video/action caches from one message.

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

    def add_state_and_video_packet(self, frame_data: dict):
        """Record current state into frame_data (PNGs are injected at save time)."""
        self._materialize_state()
        frame_data['observation.state'] = self._cur_state_msg.copy()
        frame_data['observation.chassis_state'] = self._cur_chassis_state_msg.copy()

    def add_action(self, frame_data: dict):
        self._materialize_action()
        frame_data['action'] = self._cur_action_msg.copy()
        frame_data['chassis_action'] = self._cur_chassis_action_msg.copy()

    def _build_episode_camera_metadata(self):
        cameras_meta = {}
        for camera_key in self.video_topics.keys():
            info = self._camera_info_cache.get(camera_key)
            if info is None:
                cameras_meta[camera_key] = {
                    'found': False,
                    'K': np.zeros((9,), dtype=np.float64),
                    'D': [],
                    'P': np.zeros((12,), dtype=np.float64),
                    'width': np.int32(0),
                    'height': np.int32(0),
                    'distortion_model': '',
                }
            else:
                cameras_meta[camera_key] = {
                    'found': True,
                    'K': np.array(info['K'], dtype=np.float64),
                    'D': [np.float64(v) for v in np.array(info['D'], dtype=np.float64).tolist()],
                    'P': np.array(info['P'], dtype=np.float64),
                    'width': np.int32(int(info['width'])),
                    'height': np.int32(int(info['height'])),
                    'distortion_model': str(info['distortion_model']),
                }
        return cameras_meta

    def _build_rlds_episode_example(self, episode_index: int, episode_frames: list,
                                    task_description: str, source_bag: str,
                                    recording_time: str):
        n_steps = len(episode_frames)
        aligned_timestamps = [i / self.fps for i in range(n_steps)]

        steps = []

        for step_idx, (frame, ts) in enumerate(zip(episode_frames, aligned_timestamps)):
            is_first = step_idx == 0
            is_last = step_idx == (n_steps - 1)
            is_terminal = is_last
            reward = 1.0 if is_last else 0.0
            discount = 0.0 if is_terminal else 1.0

            observation_images = {}

            for camera_key in self.video_topics.keys():
                png_bytes = frame.get(f'observation.{camera_key}')
                observation_images[camera_key] = (
                    bytes(png_bytes) if png_bytes else self._empty_png_bytes
                )
            step = {
                'observation': {
                    'state': frame['observation.state'].astype(np.float32),
                    'chassis_state': frame['observation.chassis_state'].astype(np.float32),
                    **observation_images,
                },
                'action': frame['action'].astype(np.float32),
                'chassis_action': frame['chassis_action'].astype(np.float32),
                'timestamp': np.float64(ts),
                'discount': np.float32(discount),
                'reward': np.float32(reward),
                'is_first': bool(is_first),
                'is_last': bool(is_last),
                'is_terminal': bool(is_terminal),
                'language_instruction': task_description,
            }
            steps.append(step)

        episode_example = {
            'steps': steps,
            'episode_metadata': {
                'episode_idx': np.int64(episode_index),
                'n_steps': np.int32(n_steps),
                'fps': np.int32(self.fps),
                'task': task_description,
                'source_bag': source_bag,
                'recording_time': recording_time,
                'image_height': np.int32(self._target_image_h),
                'image_width': np.int32(self._target_image_w),
                'feature_names_state_action': self.setup_features(),
                'feature_names_chassis': self.setup_chassis_features(),
                'camera_keys': list(self.video_topics.keys()),
                'camera_topics': [self.video_topics[k] for k in self.video_topics.keys()],
                'cameras': self._build_episode_camera_metadata(),
            },
        }
        return episode_example, n_steps

    def save_episode_rlds(self, episode_index: int, episode_frames: list, task_description: str,
                          source_bag: str = '', recording_time: str = ''):
        n_steps = len(episode_frames)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        episode_name = f'episode_{episode_index:06d}'
        out_tfrecord = self.output_directory / f'{episode_name}.tfrecord'

        episode_example, n_steps = self._build_rlds_episode_example(
            episode_index=episode_index,
            episode_frames=episode_frames,
            task_description=task_description,
            source_bag=source_bag,
            recording_time=recording_time,
        )

        try:
            encoded = self.rlds_episode_features.encode_example(episode_example)
            serialized = self._serializer.serialize_example(encoded)
        except Exception as e:
            error_msg = f'Failed to encode RLDS episode {episode_index}: {e}\n'
            error_msg += 'Traceback (most recent call last):\n'
            error_msg += traceback.format_exc()
            self.logger.error(error_msg)
            return False

        with tf.io.TFRecordWriter(str(out_tfrecord)) as writer:
            writer.write(serialized)

        # Release temporary large objects promptly.
        del episode_example
        del encoded
        del serialized
        gc.collect()

        self._stats_rlds_total_frames += n_steps
        self.logger.info(f'Saved episode {episode_index} with {n_steps} frames ({n_steps / self.fps:.1f}s)')
        return True

    def convert_single_bag(self, rosbag, task_description: str, ENFORCE_ALL_VIDEO_TOPICS_FLAG: bool):
        self.logger.info(f"\n=== Processing {rosbag['name']} ===")

        bag_path = Path(rosbag['path'])
        bag_size_bytes = sum(f.stat().st_size for f in bag_path.rglob('*') if f.is_file())
        self._stats_rosbag_size_bytes += bag_size_bytes
        self.logger.info(f"   Bag size: {bag_size_bytes / 1e9:.2f} GB")

        try:
            bag_file = rosbag['bag_file']
            storage_id = 'mcap' if bag_file.endswith('.mcap') else 'sqlite3'
            storage_options = StorageOptions(uri=rosbag['path'], storage_id=storage_id)
            converter_options = ConverterOptions('', '')
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
        except Exception as e:
            error_msg = f"Failed to open {rosbag['bag_file']} in {rosbag['name']}: {e}\n"
            error_msg += "Traceback (most recent call last):\n"
            error_msg += traceback.format_exc()
            self.logger.error(error_msg)
            return

        self.topic_types_dict = {}
        topic_types = reader.get_all_topics_and_types()
        for topic_metadata in topic_types:
            if topic_metadata.name in self.all_topics_set:
                self.topic_types_dict[topic_metadata.name] = topic_metadata.type

        if ENFORCE_ALL_VIDEO_TOPICS_FLAG is True:
            for video_topic_name in self.video_topics_set:
                if video_topic_name not in self.topic_types_dict:
                    self.logger.warning(
                        f"skip {rosbag['name']}: missing video topic {video_topic_name}"
                    )
                    del reader
                    return

        self._camera_info_cache = self._collect_camera_info(rosbag)
        self._reset_video_runtime_state()

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

        x_button, y_button = 2, 3
        is_recording = False
        previous_buttons = None
        frame_data = dict()
        episode_frames = []
        episode_state_target_t = None
        episode_action_target_t = None
        is_in_adding_phase = False
        episode_prev_frame_target_t = None
        episode_n_time_gaps = 0
        episode_max_time_gap_s = 0.0

        _bag_first_ts = None
        _bag_last_ts = None
        _bag_xy_pairs_before = self.num_xy_pairs
        _bag_episodes_before = self.total_episodes_saved

        while reader.has_next():
            try:
                (topic, data, timestamp) = reader.read_next()
            except RuntimeError as e:
                error_msg = f"Error reading message from {rosbag['bag_file']}: {e}\n"
                error_msg += "Traceback (most recent call last):\n"
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
                            if (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and not is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"Start #Episode marker pair: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = (
                                    episode_state_target_t + ACTION_OFFSET_S
                                )

                                is_in_adding_phase = False
                                episode_prev_frame_target_t = None
                                episode_n_time_gaps = 0
                                episode_max_time_gap_s = 0.0
                                frame_data.clear()
                                self._clear_episode_frames(episode_frames)
                                episode_frames = []
                                self._reset_message_caches()
                                self._start_episode_packets(start_time)

                            elif (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"Re-record #Episode marker pair: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = (
                                    episode_state_target_t + ACTION_OFFSET_S
                                )

                                is_in_adding_phase = False
                                episode_prev_frame_target_t = None
                                episode_n_time_gaps = 0
                                episode_max_time_gap_s = 0.0
                                frame_data.clear()
                                self._clear_episode_frames(episode_frames)
                                episode_frames = []
                                self._reset_message_caches()
                                self._start_episode_packets(start_time)

                            if (previous_buttons[y_button] == 0 and buttons[y_button] == 1 and is_recording):
                                is_recording = False
                                end_time = timestamp
                                duration = end_time - start_time
                                self.logger.info(
                                    f"Stop TimeRange: {start_time:.3f} to {end_time:.3f} seconds. "
                                    f"Duration: {duration:.3f} seconds"
                                )

                                self.num_xy_pairs += 1

                                if episode_n_time_gaps > 0:
                                    self.logger.warning(
                                        f"⚠️  Episode candidate contains {episode_n_time_gaps} time gap(s), "
                                        f"max {episode_max_time_gap_s:.2f}s — data may contain hidden jumps"
                                    )
                                if len(episode_frames) >= MIN_EPISODE_LENGTH:
                                    self._inject_episode_images(episode_frames, start_time)
                                    save_ok = self.save_episode_rlds(
                                        episode_index=self.total_episodes_saved,
                                        episode_frames=episode_frames,
                                        task_description=task_description,
                                        source_bag=rosbag['name'],
                                        recording_time=datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                                    )
                                    if save_ok:
                                        self.total_episodes_saved += 1

                                self._clear_episode_frames(episode_frames)
                                episode_frames = []
                                self._ep_packets = None
                                episode_state_target_t = None
                                episode_action_target_t = None

                        previous_buttons = list(buttons)

                except Exception as e:
                    error_msg = f"Error processing Joy message: {e}\n"
                    error_msg += "Traceback (most recent call last):\n"
                    error_msg += traceback.format_exc()
                    self.logger.error(error_msg)

            if is_recording:
                if timestamp < episode_state_target_t:
                    continue

                elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                    self.add_state_and_video_packet(frame_data)
                    # Grid tick index of this row (tick 0 = start_time + 1/fps);
                    # used to pick the matching video frame at save time
                    frame_data['tick_index'] = int(round(
                        (episode_state_target_t - start_time) * self.fps
                    )) - 1
                    is_in_adding_phase = True

                elif timestamp > episode_action_target_t and is_in_adding_phase:
                    if episode_prev_frame_target_t is not None:
                        frame_gap_s = episode_state_target_t - episode_prev_frame_target_t
                        if frame_gap_s >= TIME_GAP_WARN_RATIO * self.frame_duration:
                            episode_n_time_gaps += 1
                            episode_max_time_gap_s = max(episode_max_time_gap_s, frame_gap_s)
                            self.logger.warning(
                                f"⚠️  TIME GAP: {frame_gap_s:.2f}s "
                                f"(~{frame_gap_s / self.frame_duration:.0f} frame periods) of source "
                                f"messages missing before sampled frame {len(episode_frames)} — the "
                                f"synthetic step timestamps hide this gap"
                            )
                    episode_prev_frame_target_t = episode_state_target_t
                    self.add_action(frame_data)
                    episode_frames.append(frame_data.copy())
                    frame_data.clear()

                    time_gap = timestamp - episode_action_target_t
                    skipped_frames = time_gap // self.frame_duration

                    episode_state_target_t += (skipped_frames + 1) * self.frame_duration
                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_S
                    is_in_adding_phase = False

                elif timestamp > episode_action_target_t and not is_in_adding_phase:
                    time_gap = timestamp - episode_action_target_t
                    skipped_frames = time_gap // self.frame_duration + 1

                    episode_state_target_t += skipped_frames * self.frame_duration
                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_S
                    frame_data.clear()

        if _bag_first_ts is not None and _bag_last_ts is not None:
            bag_duration = _bag_last_ts - _bag_first_ts
            self._stats_rosbag_duration_s += bag_duration
        else:
            bag_duration = 0.0

        bag_xy_pairs = self.num_xy_pairs - _bag_xy_pairs_before
        bag_ep_saved = self.total_episodes_saved - _bag_episodes_before
        self._stats_num_xy_pairs_total += bag_xy_pairs
        self.logger.info(
            f"   Bag duration: {bag_duration:.1f}s | "
            f"Marker pairs: {bag_xy_pairs} | "
            f"Episodes saved: {bag_ep_saved}"
        )

        self._clear_episode_frames(episode_frames)
        del reader

    def convert_all(self, task_description: str, MULTIBAG_FLAG: bool, ENFORCE_ALL_VIDEO_TOPICS_FLAG: bool):
        self.logger.info(f"Starting multi-bag conversion: {self.input_directory}")
        _convert_start = time.time()

        rosbags = self.discover_rosbags(MULTIBAG_FLAG)
        total_rosbags = len(rosbags)
        if not rosbags:
            self.logger.error('No rosbag found!')
            return False

        processed_rosbags = 0
        for rosbag in rosbags:
            self.convert_single_bag(rosbag, task_description, ENFORCE_ALL_VIDEO_TOPICS_FLAG)
            processed_rosbags += 1
            self.logger.info(f"[{processed_rosbags}/{total_rosbags}] Finished processing rosbag: {rosbag.get('name')}")

        _convert_elapsed = time.time() - _convert_start

        rlds_size_bytes = sum(
            f.stat().st_size
            for f in self.output_directory.rglob('*.tfrecord')
            if f.is_file()
        )

        rlds_duration_s = self._stats_rlds_total_frames / self.fps if self.fps > 0 else 0.0
        valid_ratio = (self.total_episodes_saved / self._stats_num_xy_pairs_total * 100
                       if self._stats_num_xy_pairs_total > 0 else 0.0)
        rosbag_size_gb = self._stats_rosbag_size_bytes / 1e9
        rlds_size_gb = rlds_size_bytes / 1e9

        def _fmt_duration(secs: float) -> str:
            secs = int(secs)
            m, s = divmod(secs, 60)
            return f"{m}m {s}s" if m else f"{s}s"

        sep = '=' * 52
        self.logger.info(f"\n{sep}")
        self.logger.info('  转换完成摘要 / Conversion Summary')
        self.logger.info(sep)
        self.logger.info(f"  任务               : {task_description}")
        self.logger.info(f"  原始 rosbag 大小   : {rosbag_size_gb:.2f} GB")
        self.logger.info(f"  原始 rosbag 时长   : {_fmt_duration(self._stats_rosbag_duration_s)}  ({self._stats_rosbag_duration_s:.1f}s)")
        self.logger.info(f"  RLDS 数据片段数    : {self.total_episodes_saved} 段")
        self.logger.info(f"  RLDS 数据总时长    : {_fmt_duration(rlds_duration_s)}  ({rlds_duration_s:.1f}s)")
        self.logger.info(f"  RLDS 数据大小      : {rlds_size_gb:.2f} GB")
        self.logger.info(f"  有效 episode 占比  : {self.total_episodes_saved}/{self._stats_num_xy_pairs_total}  ({valid_ratio:.1f}%)")
        self.logger.info(f"  转换耗时           : {_fmt_duration(_convert_elapsed)}  ({_convert_elapsed:.1f}s)")
        self.logger.info(f"  输出目录           : {self.output_directory}")
        self.logger.info(sep)
        return True


def setup_logging_to_file(log_file_path):
    pid = os.getpid()
    print(f"Process ID: {pid}", file=sys.__stderr__)
    sys.__stderr__.flush()

    log_file = open(log_file_path, 'a', encoding='utf-8')

    log_file.write(f"\n{'=' * 80}\n")
    log_file.write(f"Process started - Process ID: {pid} - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'=' * 80}\n")
    log_file.flush()

    class FileOutput:
        def __init__(self, file):
            self.file = file

        def write(self, text):
            self.file.write(text)
            self.file.flush()

        def flush(self):
            self.file.flush()

    sys.stdout = FileOutput(log_file)
    sys.stderr = FileOutput(log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    return log_file


def main():
    parser = argparse.ArgumentParser(description='Convert multiple ROS2 bags to RLDS-like TFRecord dataset')
    parser.add_argument('--multibag', action='store_true',
                        help='Whether input_directory contains multiple rosbags, pass True if yes, False if no')
    parser.add_argument('--enforce_all_video_topics', action='store_true',
                        help='Enforce that rosbag must have all configured video topics')
    parser.add_argument('--input_directory', default='./data/rosbags',
                        help='Directory containing ROS2 bag segments')
    parser.add_argument('--output_directory', '-o', default='./data/rlds_output',
                        help='Output directory for RLDS episode files')
    parser.add_argument('--fps', type=int, default=45, help='Target FPS for dataset')
    parser.add_argument('--task', default='task description', help='Task description')
    parser.add_argument('--log_file', default=None,
                        help='Log file path (default: convert_rosbag_to_rlds_YYYYMMDD_HHMMSS.log in script directory)')
    args = parser.parse_args()

    if args.log_file is None:
        script_dir = Path(__file__).parent
        logs_dir = script_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = logs_dir / f'convert_rosbag_to_rlds_{timestamp}.log'
    else:
        log_file_path = Path(args.log_file)

    log_file = setup_logging_to_file(log_file_path)

    try:
        if not os.path.exists(args.input_directory):
            print(f'Error: Input directory {args.input_directory} not found!')
            sys.exit(1)

        converter = MultiVideoRosBag2RLDSConverter(args.input_directory, args.output_directory, args.fps)
        converter.convert_all(args.task, args.multibag, args.enforce_all_video_topics)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()
