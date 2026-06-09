#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PKG_DIR="${WS_ROOT}/src/sensor_data_collection"
LAUNCH_NAME="stm32_manual.launch"
ROS_DISTRO_NAME="${ROS_DISTRO:-noetic}"

usage() {
  cat <<EOF
Usage: tools/run_stm32_manual_migels.sh [roslaunch args...]

Launch sensor_data_collection ${LAUNCH_NAME} from this Mi-Gels workspace,
even when other workspaces also contain a sensor_data_collection package.

Environment:
  ROS_DISTRO       ROS distribution name. Default: noetic
  ZLAB_ROBOTS_WS   Dependency workspace. Default: \$HOME/zlab_robots

Examples:
  tools/run_stm32_manual_migels.sh
  tools/run_stm32_manual_migels.sh array_config:=qmc6309_12ch_v1 output_dir:="${WS_ROOT}/data/manual_calibration"
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${PKG_DIR}/package.xml" ]]; then
  echo "ERROR: sensor_data_collection package not found under ${WS_ROOT}/src" >&2
  exit 1
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ERROR: ROS setup not found: ${ROS_SETUP}" >&2
  exit 1
fi
source "${ROS_SETUP}"

# Source dependency workspace first, then this Mi-Gels workspace.
ZLAB_ROBOTS_WS="${ZLAB_ROBOTS_WS:-${HOME}/zlab_robots}"
if [[ -f "${ZLAB_ROBOTS_WS}/devel/setup.bash" ]]; then
  source "${ZLAB_ROBOTS_WS}/devel/setup.bash"
else
  echo "WARN: zlab_robots setup not found: ${ZLAB_ROBOTS_WS}/devel/setup.bash" >&2
fi

if [[ ! -f "${WS_ROOT}/devel/setup.bash" ]]; then
  echo "ERROR: Mi-Gels workspace has not been built or sourced yet: ${WS_ROOT}/devel/setup.bash" >&2
  echo "Run: cd ${WS_ROOT} && catkin build" >&2
  exit 2
fi
source "${WS_ROOT}/devel/setup.bash"

# Make the intended source tree win even if the user's shell sourced another
# workspace earlier.
export ROS_PACKAGE_PATH="${WS_ROOT}/src${ROS_PACKAGE_PATH:+:${ROS_PACKAGE_PATH}}"

if command -v rospack >/dev/null 2>&1; then
  rospack profile >/dev/null
  RESOLVED_PKG="$(rospack find sensor_data_collection)"
  if [[ "${RESOLVED_PKG}" != "${PKG_DIR}" ]]; then
    echo "ERROR: sensor_data_collection resolves to the wrong package:" >&2
    echo "  resolved: ${RESOLVED_PKG}" >&2
    echo "  expected: ${PKG_DIR}" >&2
    echo "ROS_PACKAGE_PATH:" >&2
    echo "${ROS_PACKAGE_PATH}" | tr ':' '\n' >&2
    exit 3
  fi
else
  echo "ERROR: rospack not found after sourcing ROS." >&2
  exit 1
fi

echo "Using sensor_data_collection: ${RESOLVED_PKG}"
echo "Launching: roslaunch sensor_data_collection ${LAUNCH_NAME} $*"
exec roslaunch sensor_data_collection "${LAUNCH_NAME}" "$@"
