# ROS2 Bag to LeRobot / HDF5 / RLDS Dataset Converter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![CUDA 12.6+](https://img.shields.io/badge/CUDA-12.6+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![LeRobot 0.3.3](https://img.shields.io/badge/LeRobot-0.3.3-orange.svg)](https://github.com/huggingface/lerobot)
[![Dataset v2.1](https://img.shields.io/badge/Dataset%20Format-v2.1-orange.svg)](https://github.com/huggingface/lerobot)

A toolkit for converting ROS2 bag files (`.db3` / `.mcap`) into LeRobot v2.1 datasets, per-episode HDF5 files, or TFDS RLDS datasets for OpenVLA-style training — plus a web visualizer with a synchronized 3D URDF robot view.

## 📑 Table of Contents

- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
  - [1. Install Conda](#1-install-conda)
  - [2. Create Conda Environment](#2-create-conda-environment)
  - [3. Install Project-Specific LeRobot](#3-install-project-specific-lerobot)
- [Input Bag Format (db3 / mcap)](#-input-bag-format-db3--mcap)
- [Usage](#-usage)
  - [Scenario 1: Convert to LeRobot Format](#scenario-1-convert-to-lerobot-format)
  - [Scenario 2: Convert to HDF5 Format](#scenario-2-convert-to-hdf5-format)
  - [Scenario 3: Build TFDS RLDS Dataset for OpenVLA](#scenario-3-build-tfds-rlds-dataset-for-openvla)
- [Dataset Visualization with URDF](#-dataset-visualization-with-urdf)
- [Additional Tools](#️-additional-tools)
  - [Merge Multiple Datasets](#merge-multiple-datasets)
  - [Delete Wrong Episodes](#delete-wrong-episodes)
  - [Split Train/Test Subsets](#split-traintest-subsets)
  - [Unify Dataset Tasks](#unify-dataset-tasks)
  - [Rosbag Data Validation](#rosbag-data-validation)
- [Important Notes](#️-important-notes)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 📋 System Requirements

- **OS**: Ubuntu 22.04
- **ROS2 Version**: Humble
- **Python Version**: 3.11
- **CUDA Version**: 12.6+ (tested with 12.8)
- **LeRobot Version**: 0.3.3 (Modified)
- **Dataset Format**: v2.1
- **Conda/Anaconda**: For environment management

## 🚀 Quick Start

### 1. Install Conda

Download and install Miniconda or Anaconda:

```bash
# Install Anaconda (replace with your actual installer name)
chmod +x Anaconda3-2025.06-0-Linux-x86_64.sh
./Anaconda3-2025.06-0-Linux-x86_64.sh
```

After installation, open a new terminal. You should see `(base)` before your username.

### 2. Create Conda Environment

```bash
# Navigate to project directory
cd rosbag_to_lerobot

# Create environment from environment.yml
conda env create -f environment.yml -n rosbag2lerobot

# Activate the environment
conda activate rosbag2lerobot
```

### 3. Install Project-Specific LeRobot

```bash
# Fetch the modified lerobot (git submodule; or clone with --recurse-submodules)
git submodule update --init lerobot

# Install modified lerobot version from the project (editable install required)
cd lerobot
pip install -e .
cd ..
```

> The URDF-enabled visualizer resolves the `urdf-loaders/` assets relative to this repository checkout, so lerobot must be installed in editable mode (`pip install -e .`).

## 📦 Input Bag Format (db3 / mcap)

All converters accept both rosbag2 storage formats and pick the storage plugin automatically from the file extension — no flag is needed:

- **`.db3`** (sqlite3) and **`.mcap`** are both supported. MCAP support comes from `ros-humble-rosbag2-storage-mcap`, which is included in `environment.yml`.
- A bag folder must contain the rosbag2 `metadata.yaml` plus at least one `.db3` or `.mcap` file. A raw `.mcap` file copied without its `metadata.yaml` will not open.
- With `--multibag`, the input directory is walked recursively; every folder containing a bag file and a `.yaml`/`.yml` is treated as one bag. Without `--multibag`, the bag file(s) must sit directly in `--input_directory`.
- If a folder contains both `.db3` and `.mcap` files, the `.db3` files win and the `.mcap` files are ignored.

**Episode markers**: the converters segment episodes with the joystick topic `/xr/left_hand_inputs` (`sensor_msgs/Joy`). A rising edge on `buttons[2]` (**X**) starts an episode (pressing X again while recording discards the buffer and restarts), and a rising edge on `buttons[3]` (**Y**) stops and saves it. Episodes shorter than 30 frames, and episodes still open when the bag ends, are discarded.

**Expected topics** (shared by all converters):

| Group | Topics |
|---|---|
| Markers | `/xr/left_hand_inputs` (`sensor_msgs/Joy`) |
| Video (HEVC `FFMPEGPacket`) | `/left/color/image_raw/ffmpeg`, `/right/color/image_raw/ffmpeg`, `/xr_video_topic/ffmpeg` (head) |
| Camera info | `/left/color/camera_info`, `/right/color/camera_info`, `/head/color/camera_info` |
| State (62 dims) | `/left_arm/joint_states`, `/right_arm/joint_states`, `/left_gripper/joint_states`, `/right_gripper/joint_states`, `/left_arm/current_ee_pose`, `/right_arm/current_ee_pose` |
| Action (62 dims) | `/left_arm/joint_cmd`, `/right_arm/joint_cmd`, `/left_gripper/joint_cmd`, `/right_gripper/joint_cmd`, `/left_arm/target_ee_pose`, `/right_arm/target_ee_pose` |
| Chassis (9 dims) | `/chassis/joint_states` (state), `/chassis/joint_cmd` (action) |

## 📖 Usage

This project provides conversion workflows for LeRobot, HDF5, and TFDS RLDS datasets.

### Scenario 1: Convert to LeRobot Format

**Use case**: Rosbags contain operator start (X key) and end (Y key) markers; each bag may contain multiple episodes.

**Script**: `scripts/convert_rosbag_to_lerobot.py`

```bash
# Activate conda environment (ROS2 is already included)
conda activate rosbag2lerobot
unset PYTHONPATH

# Clear previous cache
rm -rf ~/.cache/huggingface/lerobot/username/dataset_name

# Convert dataset
python scripts/convert_rosbag_to_lerobot.py \
  --input_directory ./data/rosbags/multiepisode_rosbag \
  --output username/dataset_name \
  --fps 30 \
  --task "task description"
```

**Parameters**:
- `--input_directory`: Directory containing ROS2 bag files (default: `./data/rosbags`)
- `--output`, `-o`: Output dataset name (format: `username/dataset_name`)
- `--fps`: Target frame rate (default: 30)
- `--task`: Task description
- `--multibag`: (Optional) Use if the directory contains multiple rosbag folders
- `--enforce_all_video_topics`: (Optional) Skip bags missing any of the three video topics
- `--log_file`: (Optional) Log file path (default: `scripts/logs/convert_rosbag_to_lerobot_<timestamp>.log`)

### Scenario 2: Convert to HDF5 Format

**Use case**: Convert rosbags with X/Y markers directly to HDF5. Each episode is saved as an independent `episode_XXXXXX.hdf5` file. HEVC video is decoded in real time (PyAV) and stored as per-frame JPEG bytes — decode with `cv2.imdecode`.

**Script**: `scripts/convert_rosbag_to_hdf5.py`

**Output structure**:
```
episode_XXXXXX.hdf5
├── /meta                          attrs: task, fps, image_encoding="jpeg", jpeg_quality,
│   │                                     episode_idx, n_frames, source_bag, recording_time,
│   │                                     n_time_gaps, max_time_gap_s (episodes with real-time
│   │                                     gaps in the source bag have n_time_gaps > 0)
│   └── cameras/<key>              camera intrinsics per camera (attrs: K, D, P,
│                                  width, height, distortion_model)
└── /data
    ├── timestamp                  (N,)    float64 — synthetic, starts at 0, step = 1/fps
    ├── observation/
    │   ├── state                  (N, 62) float32  (attrs: names)
    │   ├── chassis_state          (N, 9)  float32  (attrs: names)
    │   └── images/
    │       ├── left_color         (N,)    vlen uint8 — JPEG bytes per frame
    │       ├── right_color        (N,)    vlen uint8
    │       └── head_camera        (N,)    vlen uint8
    ├── action                     (N, 62) float32  (attrs: names)
    └── chassis_action             (N, 9)  float32  (attrs: names)
```

```bash
conda activate rosbag2lerobot
unset PYTHONPATH

python scripts/convert_rosbag_to_hdf5.py \
  --multibag \
  --input_directory ./data/rosbags \
  --output_directory ./data/hdf5_output \
  --fps 30 \
  --task "task description"
```

**Parameters**:
- `--input_directory`: Directory containing ROS2 bag files (default: `./data/rosbags`)
- `--output_directory`, `-o`: Output directory for HDF5 episode files (default: `./data/hdf5_output`)
- `--fps`: Target frame rate (default: 30)
- `--task`: Task description
- `--multibag`: (Optional) Use if the directory contains multiple rosbag folders
- `--enforce_all_video_topics`: (Optional) Skip bags missing any of the three video topics
- `--start_episode_idx`: Starting index for output episode numbering (default: 0). Use this to append to an output directory that already contains episodes.
- `--overwrite`: Allow replacing existing `episode_*.hdf5` files. Without it, the script refuses to start if new episode indices would collide with existing files.
- `--jpeg_quality`: JPEG encode quality 1–100 (default: 85)
- `--jpeg_workers`: Number of JPEG encoding threads; 0 = synchronous (default: 0)
- `--log_file`: (Optional) Log file path (default: `scripts/logs/convert_rosbag_to_hdf5_<timestamp>.log`)

### Scenario 3: Build TFDS RLDS Dataset for OpenVLA

**Use case**: Build an OpenVLA-style TFDS RLDS dataset directly from ROS2 bags with X/Y episode markers. The TFDS builder discovers `.db3`/`.mcap` bags, segments episodes, samples frames at the target FPS, decodes HEVC camera packets (the head camera's side-by-side stereo image is cropped to its left half), resizes images to 224x224, and writes TFDS train shards (a single `train` split).

**Builder**: `rosbag_rlds_tfds/rosbag_rlds_tfds_dataset_builder.py` — it dynamically loads the conversion logic from `scripts/convert_rosbag_to_rlds.py`, so both files must stay at their current locations.

**Output structure**:
```text
data/tfds_output/
└── rosbag_rlds_tfds/1.0.0/
    ├── dataset_info.json
    ├── features.json
    └── rosbag_rlds_tfds-train.tfrecord-00000-of-000xx
```

**TFRecord unit**:
Each serialized TFDS example is one complete RLDS episode.
```text
RLDS episode example
├── steps                                      (N,) sampled frames
│   ├── observation/
│   │   ├── state                              (62,) float32
│   │   ├── chassis_state                      (9,)  float32
│   │   ├── left_color                         (224, 224, 3) uint8 PNG
│   │   ├── right_color                        (224, 224, 3) uint8 PNG
│   │   └── head_camera                        (224, 224, 3) uint8 PNG
│   ├── action                                 (62,) float32
│   ├── chassis_action                         (9,)  float32
│   ├── timestamp                              float64, starts at 0
│   ├── reward / discount                      float32
│   ├── is_first / is_last / is_terminal       bool
│   └── language_instruction                   text
└── episode_metadata/
    ├── episode_idx                            int64
    ├── n_steps                                int32
    ├── fps                                    int32
    ├── task / source_bag / recording_time     text
    ├── image_height / image_width             int32
    ├── feature_names_state_action             sequence text
    ├── feature_names_chassis                  sequence text
    ├── camera_keys / camera_topics            sequence text
    └── cameras                                calibration metadata
```

**Run**:
```bash
conda activate rosbag2lerobot
unset PYTHONPATH

# Edit the defaults in the User Config block, or export any ROSBAG_* variable
# before sourcing — pre-exported values are preserved.
source rosbag_rlds_tfds/setup_rosbag_rlds_env.sh

cd "$ROSBAG_RLDS_BUILDER_DIR"
tfds build --overwrite
```

**Configuration**:
Edit the `CFG_*` defaults in [setup_rosbag_rlds_env.sh](rosbag_rlds_tfds/setup_rosbag_rlds_env.sh), or export the corresponding `ROSBAG_*` variables before sourcing it. The script exports the `ROSBAG_*` variables read by the TFDS builder.

- `CFG_ROSBAG_ROOT`: Rosbag file, single rosbag directory, or root directory containing multiple rosbag folders
- `CFG_ROSBAG_MULTIBAG`: `1` to recursively discover multiple bags; `0` for one bag directory/file
- `CFG_ROSBAG_TASK`: Language instruction stored in each RLDS step
- `CFG_ROSBAG_FPS`: Target sampling rate (default: 30)
- `CFG_ROSBAG_ENFORCE_VIDEO_TOPICS`: `1` to skip bags missing required camera topics
- `CFG_ROSBAG_TFDS_N_WORKERS` / `CFG_ROSBAG_TFDS_MAX_PATHS_IN_MEMORY`: Chunking of bag paths during generation; lower these for safer memory usage (generation itself currently runs single-threaded)
- `CFG_ROSBAG_TFDS_DISABLE_SHUFFLING`: `1` (default) writes episodes in deterministic order
- `CFG_TFDS_DATA_DIR`: TFDS output root

**Standalone script**: `scripts/convert_rosbag_to_rlds.py` can also run on its own (same marker/sampling logic), writing one self-contained `episode_XXXXXX.tfrecord` per episode instead of TFDS shards:

```bash
python scripts/convert_rosbag_to_rlds.py \
  --multibag \
  --input_directory ./data/rosbags \
  --output_directory ./data/rlds_output \
  --fps 30 \
  --task "task description"
```

## 🤖 Dataset Visualization with URDF

`lerobot/src/lerobot/scripts/visualize_dataset_html_urdf.py` extends the stock LeRobot HTML visualizer with a 3D URDF robot panel (rendered with the bundled [urdf-loaders](https://github.com/gkjohnson/urdf-loaders) library) that animates in sync with video playback and the timeline slider.

```bash
conda activate rosbag2lerobot

python -m lerobot.scripts.visualize_dataset_html_urdf \
  --repo-id username/dataset_name \
  --root /path/to/local/dataset

# then open http://127.0.0.1:9090
```

**Parameters** (same as the stock visualizer):
- `--repo-id`: Dataset name (`username/dataset_name`)
- `--root`: Path to the local dataset (default: HuggingFace cache)
- `--episodes`: (Optional) Episode indices to visualize (default: all)
- `--host` / `--port`: Server address (default: `127.0.0.1:9090`)
- `--load-from-hf-hub`: Stream data from the HuggingFace Hub instead of local files (default: 0)

**How it works**:
- The robot model is currently fixed to the bundled robot: `urdf-loaders/urdf/tele/urdf/robot.urdf` with STL meshes from `urdf-loaders/urdf/tele/meshes/`. To visualize a different robot, edit the URDF path and joint mapping in `visualize_dataset_html_urdf.py` and the template.
- The first 16 dimensions of `observation.state` are mapped in order to URDF joints `l_joint1..l_joint8` and `r_joint1..r_joint8` (values in radians; gripper dimensions use a built-in calibration).
- The `urdf-loaders/` library is served directly as ES modules — no `npm install` or build step is needed. The browser does need internet access for the CDN-hosted dependencies (three.js, Alpine.js, dygraphs).

## 🛠️ Additional Tools

### Merge Multiple Datasets

Merge multiple LeRobot datasets into a single dataset:

```bash
python scripts/merge_datasets.py \
  --root ./data/source_datasets \
  --output ./output/merged_dataset
```

**Parameters**:
- `--root`: Root directory containing source dataset folders (default: `./data/source_datasets`)
- `--output`: Output directory for the merged dataset (default: `./output/merged_dataset`)

### Delete Wrong Episodes

Remove specific episodes from a dataset and reindex:

```bash
python scripts/delete_wrong_episodes.py \
  --dataset /path/to/dataset \
  --episode-indices 2 5 10
```

**Parameters**:
- `--dataset`: Path to the dataset directory
- `--episode-indices`: Space-separated list of episode indices to delete

### Split Train/Test Subsets

Split a LeRobot dataset sequentially into `<dataset>_train` (first N episodes) and `<dataset>_test` (remaining episodes), created next to the source dataset, which is left untouched:

```bash
python scripts/split_train_test.py \
  --dataset /path/to/dataset \
  --num-train-episodes 80
```

**Parameters**:
- `--dataset`: Path to the source dataset directory
- `--num-train-episodes`: Number of episodes for the train split (must be > 0)

### Unify Dataset Tasks

Copy a dataset with all episodes rewritten to a single task description:

```bash
python scripts/unify_dataset_tasks.py \
  --input /path/to/dataset \
  --output /path/to/unified_dataset \
  --task "task description"
```

**Parameters**:
- `--input`: Path to the source dataset
- `--output`: Output directory (must not exist or be empty)
- `--task`: The single task description applied to all episodes

### Rosbag Data Validation

Scan rosbags and report per-bag statistics (episode segment counts, durations, sizes) before converting:

```bash
python scripts/validate_rosbags.py \
  --multibag \
  --input_directory ./data/rosbags
```

## ⚠️ Important Notes

1. **Critical**: Always delete the previous dataset cache before re-running LeRobot conversion scripts
2. Ensure the conda environment is activated (`conda activate rosbag2lerobot`) before running scripts. ROS2 is included in the environment.
3. Run `unset PYTHONPATH` to avoid conflicts with system Python packages
4. First run may take longer due to dependency downloads
5. Converter output is automatically written to `.log` files under `scripts/logs/`; the terminal shows only the process ID and ROS2 system output
6. Episodes shorter than 30 frames are silently discarded by all converters
7. Bags recorded with file splitting (multiple `.db3`/`.mcap` segments under one `metadata.yaml`) should be converted as a single bag folder

## 📁 Project Structure

```
rosbag_to_lerobot/
├── scripts/
│   ├── convert_rosbag_to_lerobot.py    # Rosbags with X/Y markers → LeRobot dataset
│   ├── convert_rosbag_to_hdf5.py       # Rosbags with X/Y markers → per-episode HDF5 (JPEG frames)
│   ├── convert_rosbag_to_rlds.py       # Rosbag → RLDS episodes (used by the TFDS builder)
│   ├── validate_rosbags.py             # Pre-conversion bag statistics
│   ├── merge_datasets.py               # Merge multiple LeRobot datasets
│   ├── delete_wrong_episodes.py        # Delete specific episodes from a dataset
│   ├── split_train_test.py             # Split a dataset into train/test subsets
│   └── unify_dataset_tasks.py          # Rewrite all episodes to a single task
├── rosbag_rlds_tfds/                                   # TFDS builder for OpenVLA-style RLDS datasets
├── lerobot/                                            # Modified LeRobot library (git submodule, URDF-enabled visualizer)
├── urdf-loaders/                                       # Vendored URDF loader library + robot URDF/meshes
├── environment.yml                                     # Conda environment specification
├── README.md                                           # This file
└── LICENSE                                             # MIT License
```

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is based on [LeRobot v0.3.3](https://github.com/huggingface/lerobot) (Apache 2.0 License) and [urdf-loaders](https://github.com/gkjohnson/urdf-loaders) (Apache 2.0 License).
