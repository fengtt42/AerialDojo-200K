#!/usr/bin/env bash
set -euo pipefail
AERIAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AERIAL_PROJECT_DIR="$(cd "${AERIAL_SCRIPT_DIR}/.." && pwd)"
AERIAL_JOB_FILE="${AERIAL_JOB_FILE:-${AERIAL_PROJECT_DIR}/config/record_jobs.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${AERIAL_PROJECT_DIR}"
exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  "${PYTHON_BIN}" -m AerialDojo.projectairsim_plugin.ProjectAirSimSimulatorServerTool \
  --config "${AERIAL_JOB_FILE}" "$@"
