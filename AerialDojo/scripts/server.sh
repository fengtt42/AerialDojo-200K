#!/usr/bin/env bash
set -euo pipefail

AERIAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AERIAL_PROJECT_DIR="$(cd "${AERIAL_SCRIPT_DIR}/.." && pwd)"
AERIAL_SERVER_CONFIG="${AERIAL_PROJECT_DIR}/config/server_config.yaml"

cd "${AERIAL_PROJECT_DIR}"

exec env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  MALLOC_ARENA_MAX=2 \
  python -m AerialDojo.projectairsim_plugin.ProjectAirSimSimulatorServerTool \
    --config "${AERIAL_SERVER_CONFIG}" \
    --low_graphics \
    --low_graphics_width 320 \
    --low_graphics_height 240 \
    --texture_pool_size_mb 256 \
    "$@"
