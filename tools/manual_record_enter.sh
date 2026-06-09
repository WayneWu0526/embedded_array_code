#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-noetic}"
TOPIC="${TOPIC:-/maggrad_manual_record/record_trigger}"

usage() {
  cat <<'EOF'
Usage: tools/manual_record_enter.sh

Interactive Enter-only controller for the MagGrad manual recorder.

Environment:
  TOPIC            Trigger topic to publish. Default:
                   /maggrad_manual_record/record_trigger
  ROS_DISTRO       ROS distribution name. Default: noetic
  ZLAB_ROBOTS_WS   Dependency workspace. Default: $HOME/zlab_robots

Run this in a second terminal after stm32_manual.launch is running.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
if [[ -f "${ROS_SETUP}" ]]; then
  source "${ROS_SETUP}"
fi

ZLAB_ROBOTS_WS="${ZLAB_ROBOTS_WS:-${HOME}/zlab_robots}"
if [[ -f "${ZLAB_ROBOTS_WS}/devel/setup.bash" ]]; then
  source "${ZLAB_ROBOTS_WS}/devel/setup.bash"
fi

if [[ -f "${WS_ROOT}/devel/setup.bash" ]]; then
  source "${WS_ROOT}/devel/setup.bash"
fi

if ! command -v rostopic >/dev/null 2>&1; then
  echo "ERROR: rostopic not found. Source ROS or build/source this workspace first." >&2
  echo "Tip: run tools/run_stm32_manual_migels.sh in another terminal first." >&2
  exit 1
fi

recording=0

publish_trigger() {
  local value="$1"
  rostopic pub "${TOPIC}" std_msgs/Bool "data: ${value}" -1 >/dev/null
}

cleanup() {
  if [[ "${recording}" == "1" ]]; then
    echo
    echo "Stopping active recording..."
    publish_trigger false || true
  fi
}
trap cleanup EXIT INT TERM

echo "Manual recorder control"
echo "Topic: ${TOPIC}"
echo "Press Enter to start, Enter again to stop. Press Ctrl-C to exit."
echo

run_index=1
while true; do
  read -r -p "[${run_index}] Press Enter to START recording..."
  publish_trigger true
  recording=1
  echo "[${run_index}] Recording."

  read -r -p "[${run_index}] Press Enter to STOP recording..."
  publish_trigger false
  recording=0
  echo "[${run_index}] Stopped."
  echo

  run_index=$((run_index + 1))
done
