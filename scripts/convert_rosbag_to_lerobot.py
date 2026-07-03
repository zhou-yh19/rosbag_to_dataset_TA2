#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Convert ROS2 bag files with X/Y episode markers to LeRobot dataset format v2.1.

This script processes rosbag files that contain multiple episodes marked by operator
button presses (X button to start, Y button to end recording).

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
        --fps 30 \
        --task "task description"
    ```

    Process a single rosbag file:
    ```bash
    rm -rf ~/.cache/huggingface/lerobot/username/dataset_name
    python scripts/convert_rosbag_to_lerobot.py \
        --input_directory ./data/rosbags \
        --output username/dataset_name \
        --fps 30 \
        --task "task description"
    ```

    Enforce requirement for all four video topics:
    ```bash
    rm -rf ~/.cache/huggingface/lerobot/username/dataset_name
    python scripts/convert_rosbag_to_lerobot.py \
        --multibag \
        --enforce_four_video_topics \
        --input_directory ./data/rosbags \
        --output username/dataset_name \
        --task "task description"
    ```
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import logging
import shutil
import traceback
from datetime import datetime

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
    from lerobot.datasets.video_packet_writer import VideoPacketBuffer
except ImportError as e:
    print(f"LeRobot dependencies not found: {e}")
    sys.exit(1)

# Constants
STATE_ACTION_DIM = 62
MIN_EPISODE_LENGTH = 30
ACTION_OFFSET_RATIO = 1.0 / 3.0

class MultiVideoRosBag2LeRobotConverter:
    """Enhanced converter for multiple ROS2 bags to single LeRobot dataset with multiple episodes."""

    def __init__(self, input_directory: str, output_repo_id: str, fps: int = 30):
        self.input_directory = Path(input_directory)
        self.output_repo_id = output_repo_id
        self.fps = fps
        self.frame_duration = 1.0 / self.fps

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Video topics mapping - all 7 cameras from your robot
        self.video_topics = {
            'left_color':  '/left/color/image_raw/ffmpeg',
            'right_color': '/right/color/image_raw/ffmpeg',
            'head_camera': '/xr_video_topic/ffmpeg',
            'chest_camera':'/head/color/image_raw/ffmpeg',
        }
        # Video topics set
        self.video_topics_set = set(self.video_topics.values())
        # State topics set
        self.state_topics_set = {
            '/left_arm/joint_states', '/left_arm/current_ee_pose', '/left_gripper/joint_states',
            '/right_arm/joint_states','/right_arm/current_ee_pose','/right_gripper/joint_states',
        }
        # Action topics set
        self.action_topics_set = {
            '/left_arm/joint_cmd', '/left_arm/target_ee_pose', '/left_gripper/joint_cmd',
            '/right_arm/joint_cmd','/right_arm/target_ee_pose','/right_gripper/joint_cmd',
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

    def _update_video_frame(self, topicName:str, msg:FFMPEGPacket):
        camera_key = self._get_camera_key_from_topic(topicName)
        if camera_key:
            height, width, _ = self.get_camera_resolution(camera_key)
            self._cur_video_packet[camera_key] = {
                'data': bytes(msg.data),
                'pts': msg.pts,
                'width': width,
                'height': height,
                'encoding': msg.encoding,
            }
        
    def update_messages(self, topicName, data):
        """Use a single interface to update state, video_frame, and action"""
        msg_type = self.topic_types_dict.get(topicName)
        if not msg_type:
            pass
        else:
            msg_class = get_message(msg_type)
            msg = deserialize_message(data, msg_class)

            if topicName in self.state_topics_set:
                self._update_state_msg(topicName, msg)

            elif topicName in self.action_topics_set:
                self._update_action_msg(topicName, msg)
            
            elif topicName in self.video_topics_set:
                self._update_video_frame(topicName, msg)
    
    def add_state_and_video_packet(self, frame_data:dict, packet_buffer:VideoPacketBuffer):
        """Create a frame with state data and video at the specified time."""
        frame_data["observation.state"] = self._cur_state_msg.copy()

        for camera_key in self.video_topics.keys():
            height, width, channels = self.get_camera_resolution(camera_key)
            black_frame = np.zeros((height, width, channels), dtype=np.uint8)
            frame_data[f"observation.images.{camera_key}"] = black_frame

        for camera_key, packet in self._cur_video_packet.items():
            if packet is not None:
                packet_buffer.add_packet(
                    camera_name=camera_key,
                    packet_data=packet['data'],
                    width=packet['width'],
                    height=packet['height'],
                    codec=packet['encoding'],
                )

    def add_action(self, frame_data:dict):
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


    def convert_single_bag(self, rosbag, task_description: str, ENFORCE_FOUR_VIDEO_TOPICS_FLAG: bool):
        """Convert a single bag to one or multiple episodes in the dataset."""
        self.logger.info(f"\n=== Processing {rosbag['name']} ===")

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

        if ENFORCE_FOUR_VIDEO_TOPICS_FLAG is True:
            for video_topic_name in self.video_topics_set:
                if video_topic_name not in self.topic_types_dict:
                    del reader
                    shutil.rmtree(self.dataset.root/'images', ignore_errors=True)
                    return

        non_recording_filter = StorageFilter(topics=['/xr/left_hand_inputs'])
        reader.set_filter(non_recording_filter)

        x_button, y_button = 2, 3
        is_recording = False
        previous_buttons = None
        frame_data = dict()
        packet_buffer = None
        episode_state_target_t = None
        episode_action_target_t = None
        is_in_adding_phase = False

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            timestamp = timestamp / 1e9

            self.update_messages(topic, data)

            if topic == '/xr/left_hand_inputs':
                try:
                    buttons = deserialize_message(data, Joy).buttons

                    if len(buttons) > max(x_button, y_button):
                        if previous_buttons is not None:
                            if   (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and not is_recording):
                                is_recording = True
                                reader.reset_filter()
                                start_time = timestamp
                                self.logger.info(f"🔴 Start #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                is_in_adding_phase = False
                                frame_data.clear()

                                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                                packet_buffer = VideoPacketBuffer(root_dir=self.dataset.root, fps=self.dataset.fps)
                                
                            elif (previous_buttons[x_button] == 0 and buttons[x_button] == 1 and is_recording):
                                is_recording = True
                                start_time = timestamp
                                self.logger.info(f"🔴 Re-record #Episode: {self.num_xy_pairs}")

                                episode_state_target_t = start_time + self.frame_duration
                                episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration

                                self.dataset.clear_episode_buffer()
                                packet_buffer.clear()

                                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                                packet_buffer = VideoPacketBuffer(root_dir=self.dataset.root, fps=self.dataset.fps)

                                is_in_adding_phase = False
                                frame_data.clear()

                            if (previous_buttons[y_button] == 0 and buttons[y_button] == 1 and is_recording):
                                is_recording = False
                                reader.set_filter(non_recording_filter)
                                end_time = timestamp
                                duration = end_time - start_time
                                self.logger.info(f"⏹️ Stop TimeRange: {start_time:.3f} to {end_time:.3f} seconds. Duration: {duration:.3f} seconds")

                                self.num_xy_pairs += 1

                                if packet_buffer.episodeLength < MIN_EPISODE_LENGTH or self.dataset.episode_buffer["size"] < MIN_EPISODE_LENGTH:
                                    self.dataset.clear_episode_buffer()
                                    packet_buffer.clear()

                                else:
                                    if   packet_buffer.episodeLength > self.dataset.episode_buffer["size"]:
                                        for camera_key in self.video_topics.keys():
                                            packet_buffer.delete_final_packet(camera_key)

                                    elif packet_buffer.episodeLength < self.dataset.episode_buffer["size"]:
                                        self.dataset.delete_final_frame()

                                    self.realign_timestamps()

                                    episode_length = packet_buffer.episodeLength
                                    packet_buffer.save_episode(
                                        episode_index= self.dataset.num_episodes,
                                        dataset_meta = self.dataset.meta,
                                    )
                                    packet_buffer.clear()

                                    for camera_key in self.video_topics.keys():
                                        video_key = f"observation.images.{camera_key}"

                                        height, width, _ = self.get_camera_resolution(camera_key)

                                        video_path = self.dataset.root / self.dataset.meta.get_video_file_path(
                                            self.dataset.num_episodes, video_key
                                        )

                                        if not video_path.exists():
                                            logging.warning(f"Video not found: {video_path}, skipping stats for {camera_key}")
                                            continue
                                        try:
                                            sampled_frames = sample_frames_from_video(
                                                video_path=video_path,
                                                episode_length=episode_length,
                                                fps=self.dataset.fps,
                                                width=width,
                                                height=height,
                                            )
                                            self.dataset.episode_buffer[video_key] = sampled_frames

                                            logging.info(f"Replaced {camera_key} PNG paths with {len(sampled_frames)} sampled frames")
                                        except Exception as e:
                                            error_msg = f"Failed to sample frames from {camera_key}: {e}\n"
                                            error_msg += f"Traceback (most recent call last):\n"
                                            error_msg += traceback.format_exc()
                                            logging.error(error_msg)

                                    self.dataset.save_episode()
                                    logging.info(f"✅ Saved episode {self.dataset.num_episodes} with {episode_length} frames")

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
                    self.add_state_and_video_packet(frame_data, packet_buffer)
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

        del reader
        shutil.rmtree(self.dataset.root/'images', ignore_errors=True)


    def convert_all(self, task_description: str, MULTIBAG_FLAG: bool, ENFORCE_FOUR_VIDEO_TOPICS_FLAG: bool):
        """Convert all discovered rosbags to a multi-episode dataset."""
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
            self.convert_single_bag(rosbag, task_description, ENFORCE_FOUR_VIDEO_TOPICS_FLAG)
            processed_rosbags += 1
            self.logger.info(f"[{processed_rosbags}/{total_rosbags}] Finished processing rosbag: {rosbag.get('name')}")

        # Final summary
        self.logger.info(f"\n🎉 Multi-bag conversion complete!")
        self.logger.info(f"   Dataset: {self.output_repo_id}")
        self.logger.info(f"   Episodes: {self.dataset.num_episodes} successful")


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
    parser = argparse.ArgumentParser(description="Convert multiple ROS2 bags to LeRobot dataset with video support")
    parser.add_argument("--multibag",
                        action="store_true", # If not in command, defaults to false
                        help="Whether input_directory contains multiple rosbags, pass True if yes, False if no")
    parser.add_argument("--enforce_four_video_topics",
                        action="store_true", # If not in command, defaults to false
                        help="Enforce that rosbag must have four video topics, if you don't want to enforce this, you can comment out unwanted topics in self.video_topics")
    parser.add_argument("--input_directory",
                        default="./data/rosbags",
                        help="Directory containing ROS2 bag segments")
    parser.add_argument("--output", "-o",
                        default="output/dataset",
                        help="Output dataset repo ID")
    parser.add_argument("--fps", type=int,
                        default=30,
                        help="Target FPS for dataset")
    parser.add_argument("--task",
                        default="task description",
                        help="Task description")
    parser.add_argument("--log_file",
                        default=None,
                        help="Log file path (default: convert_rosbag_to_lerobot_YYYYMMDD_HHMMSS.log in script directory)")
    args = parser.parse_args()

    # Set log file path
    if args.log_file is None:
        script_dir = Path(__file__).parent
        logs_dir = script_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = logs_dir / f"convert_rosbag_to_lerobot_{timestamp}.log"
    else:
        log_file_path = Path(args.log_file)

    # Set up log redirection
    log_file = setup_logging_to_file(log_file_path)

    try:
        if not os.path.exists(args.input_directory):
            print(f"Error: Input directory {args.input_directory} not found!")
            sys.exit(1)

        converter = MultiVideoRosBag2LeRobotConverter(args.input_directory, args.output, args.fps)
        converter.convert_all(args.task, args.multibag, args.enforce_four_video_topics)
    finally:
        # Restore original stdout and stderr
        sys.stdout.flush()
        sys.stderr.flush()
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
    