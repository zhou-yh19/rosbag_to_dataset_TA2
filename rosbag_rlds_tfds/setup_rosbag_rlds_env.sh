#!/usr/bin/env bash
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# Configure environment variables for rosbag_rlds_tfds dataset building.
#
# Usage (bash or zsh):
#   source rosbag_rlds_tfds/setup_rosbag_rlds_env.sh
#
# Either edit the defaults in the config block below, or export any ROSBAG_* /
# TFDS_DATA_DIR variable before sourcing — pre-exported values are preserved.

# Resolve this script's path in both zsh and bash (zsh has no BASH_SOURCE).
if [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in
    *:file*) ;;
    *) echo "Please source this script instead of executing it:"
       echo "  source rosbag_rlds_tfds/setup_rosbag_rlds_env.sh"
       exit 1;;
  esac
  eval '_SCRIPT_PATH="${(%):-%x}"'
else
  if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
    echo "Please source this script instead of executing it:"
    echo "  source rosbag_rlds_tfds/setup_rosbag_rlds_env.sh"
    exit 1
  fi
  _SCRIPT_PATH="${BASH_SOURCE[0]}"
fi

_SCRIPT_DIR="$(cd "$(dirname "${_SCRIPT_PATH}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

# =========================
# User Config (edit defaults here; pre-exported variables win)
# =========================
# Builder input/behavior variables used by rosbag_rlds_tfds_dataset_builder.py
CFG_ROSBAG_ROOT="${ROSBAG_ROOT:-${_REPO_ROOT}/data/rosbags}"
CFG_ROSBAG_MULTIBAG="${ROSBAG_MULTIBAG:-1}"
CFG_ROSBAG_TASK="${ROSBAG_TASK:-task description}"
CFG_ROSBAG_FPS="${ROSBAG_FPS:-30}"
CFG_ROSBAG_ENFORCE_VIDEO_TOPICS="${ROSBAG_ENFORCE_VIDEO_TOPICS:-0}"
CFG_ROSBAG_TFDS_N_WORKERS="${ROSBAG_TFDS_N_WORKERS:-4}"
CFG_ROSBAG_TFDS_MAX_PATHS_IN_MEMORY="${ROSBAG_TFDS_MAX_PATHS_IN_MEMORY:-8}"
CFG_ROSBAG_TFDS_DISABLE_SHUFFLING="${ROSBAG_TFDS_DISABLE_SHUFFLING:-1}"

# Convenience vars for running tfds build command
CFG_ROSBAG_RLDS_BUILDER_DIR="${ROSBAG_RLDS_BUILDER_DIR:-${_REPO_ROOT}/rosbag_rlds_tfds}"
CFG_TFDS_DATA_DIR="${TFDS_DATA_DIR:-${_REPO_ROOT}/data/tfds_output}"

# Export environment variables from the config above.
export ROSBAG_ROOT="${CFG_ROSBAG_ROOT}"
export ROSBAG_MULTIBAG="${CFG_ROSBAG_MULTIBAG}"
export ROSBAG_TASK="${CFG_ROSBAG_TASK}"
export ROSBAG_FPS="${CFG_ROSBAG_FPS}"
export ROSBAG_ENFORCE_VIDEO_TOPICS="${CFG_ROSBAG_ENFORCE_VIDEO_TOPICS}"
export ROSBAG_TFDS_N_WORKERS="${CFG_ROSBAG_TFDS_N_WORKERS}"
export ROSBAG_TFDS_MAX_PATHS_IN_MEMORY="${CFG_ROSBAG_TFDS_MAX_PATHS_IN_MEMORY}"
export ROSBAG_TFDS_DISABLE_SHUFFLING="${CFG_ROSBAG_TFDS_DISABLE_SHUFFLING}"
export ROSBAG_RLDS_BUILDER_DIR="${CFG_ROSBAG_RLDS_BUILDER_DIR}"
export TFDS_DATA_DIR="${CFG_TFDS_DATA_DIR}"

echo "[rosbag_rlds_tfds env configured]"
echo "  ROSBAG_ROOT=${ROSBAG_ROOT}"
echo "  ROSBAG_MULTIBAG=${ROSBAG_MULTIBAG}"
echo "  ROSBAG_TASK=${ROSBAG_TASK}"
echo "  ROSBAG_FPS=${ROSBAG_FPS}"
echo "  ROSBAG_ENFORCE_VIDEO_TOPICS=${ROSBAG_ENFORCE_VIDEO_TOPICS}"
echo "  ROSBAG_TFDS_N_WORKERS=${ROSBAG_TFDS_N_WORKERS}"
echo "  ROSBAG_TFDS_MAX_PATHS_IN_MEMORY=${ROSBAG_TFDS_MAX_PATHS_IN_MEMORY}"
echo "  ROSBAG_TFDS_DISABLE_SHUFFLING=${ROSBAG_TFDS_DISABLE_SHUFFLING}"
echo "  ROSBAG_RLDS_BUILDER_DIR=${ROSBAG_RLDS_BUILDER_DIR}"
echo "  TFDS_DATA_DIR=${TFDS_DATA_DIR}"
