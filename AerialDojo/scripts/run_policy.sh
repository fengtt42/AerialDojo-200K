#!/usr/bin/env bash
set -euo pipefail

AERIAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AERIAL_PROJECT_DIR="$(cd "${AERIAL_SCRIPT_DIR}/.." && pwd)"

cd "${AERIAL_PROJECT_DIR}"

exec env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  python AerialDojo/eval_policy.py \
    "$@"
