# rosbag_rlds_tfds

Direct rosbag -> TFDS RLDS builder.

This builder converts ROS2 bag directories directly during `tfds build`.
It reuses the same conversion logic as `scripts/convert_rosbag_to_rlds.py`
(X/Y episode markers, state/action layout, camera image processing).

## Build Workflow

First source the environment config script:

```bash
source rosbag_rlds_tfds/setup_rosbag_rlds_env.sh
```

Then run TFDS build:

```bash
cd "${ROSBAG_RLDS_BUILDER_DIR}"
tfds build --overwrite --data_dir "${TFDS_DATA_DIR}"
```

You can override any variable before `source`, for example:

```bash
export ROSBAG_ROOT=/abs/path/to/rosbags
export ROSBAG_MULTIBAG=1
export ROSBAG_TASK="your task description"
export ROSBAG_FPS=45
export ROSBAG_ENFORCE_VIDEO_TOPICS=0
export ROSBAG_TFDS_N_WORKERS=4
export ROSBAG_TFDS_MAX_PATHS_IN_MEMORY=8
export ROSBAG_TFDS_DISABLE_SHUFFLING=1
source rosbag_rlds_tfds/setup_rosbag_rlds_env.sh
```

## Environment variables

- `ROSBAG_ROOT` default: `<repo>/data/rosbags` (via the setup script; the builder itself falls back to `./rosbag`)
- `ROSBAG_MULTIBAG` default: `1`
- `ROSBAG_TASK` default: `task description`
- `ROSBAG_FPS` default: `45`
- `ROSBAG_ENFORCE_VIDEO_TOPICS` default: `0`
- `ROSBAG_TFDS_N_WORKERS` default: `4` (lower for memory safety)
- `ROSBAG_TFDS_MAX_PATHS_IN_MEMORY` default: `8` (lower for memory safety)
- `ROSBAG_TFDS_DISABLE_SHUFFLING` default: `1` (recommended for large-scale conversion)
- `ROSBAG_RLDS_BUILDER_DIR` default: `<repo>/rosbag_rlds_tfds`
- `TFDS_DATA_DIR` default: `<repo>/data/tfds_output`

## Notes

- Bag discovery requires each bag directory to contain metadata (`.yaml`/`.yml`) and at least one `.db3` or `.mcap` file.
- If `tfds build` reports `No module named apache_beam`, install it:

```bash
pip install apache-beam
```
