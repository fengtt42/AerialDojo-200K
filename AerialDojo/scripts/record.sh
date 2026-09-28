#!/usr/bin/env bash
set -euo pipefail
AERIAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AERIAL_PROJECT_DIR="$(cd "${AERIAL_SCRIPT_DIR}/.." && pwd)"
AERIAL_JOB_FILE="${AERIAL_JOB_FILE:-${AERIAL_PROJECT_DIR}/config/record_jobs.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${AERIAL_PROJECT_DIR}"
exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  "${PYTHON_BIN}" -m AerialDojo.trajectory_recording.multi_scene_astar_record \
  --job_file "${AERIAL_JOB_FILE}" \
  --resume --interval 0.05 --image_retries 5 --manifest_interval 30 \
  --local_stage_root "${AERIAL_LOCAL_STAGE_ROOT:-/tmp/aerialdojo_record_stage}" \
  --async_nas_queue_frames 32 "$@"
