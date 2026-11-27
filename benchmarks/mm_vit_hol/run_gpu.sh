#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Portable launcher for Issue #69844 cross-GPU benchmarks.
#
# Usage:
#   bash benchmarks/mm_vit_hol/run_gpu.sh <gpu_tag> <gpu_ids> [preset]
#
# Examples:
#   # Single idle A6000 GPU (GPU 0), all 10 models:
#   bash benchmarks/mm_vit_hol/run_gpu.sh a6000_48g 0 all
#
#   # Two idle A6000 GPUs (GPU 0,1), sharded automatically:
#   bash benchmarks/mm_vit_hol/run_gpu.sh a6000_48g 0,1 all
#
#   # Core 6-model fast subset on GPU 0:
#   bash benchmarks/mm_vit_hol/run_gpu.sh h100_80g 0 core6
#
# Environment overrides:
#   PYTHON_BIN=.venv/bin/python   (auto-detected if .venv exists)
#   SKIP_GPU_CHECK=1              (skip idle-GPU safety guard)
#   STAGES=partA,partA_cg,partB_img,partB_vid

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu_tag> <gpu_ids_csv> [all|core6|local6|downloaded4|comma_list]" >&2
  exit 1
fi

GPU_TAG="$1"
GPU_IDS_CSV="$2"
PRESET="${3:-all}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/results/$GPU_TAG"
STAGES="${STAGES:-partA,partA_cg,partB_img,partB_vid}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

case "$PRESET" in
  all)
    MODELS=(qwen2_5_vl_7b qwen2_5_vl_3b qwen3_vl_8b qwen3_vl_2b internvl3_5_8b minicpm_v_4_5 pixtral_12b internvl3_5_14b gemma3_12b kimi_vl_a3b)
    ;;
  core6)
    MODELS=(qwen2_5_vl_7b qwen2_5_vl_3b qwen3_vl_8b internvl3_5_8b internvl3_5_14b gemma3_12b kimi_vl_a3b)
    ;;
  local6)
    MODELS=(qwen2_5_vl_7b qwen2_5_vl_3b qwen3_vl_8b qwen3_vl_2b internvl3_5_8b minicpm_v_4_5)
    ;;
  downloaded4)
    MODELS=(pixtral_12b internvl3_5_14b gemma3_12b kimi_vl_a3b)
    ;;
  *)
    IFS=',' read -r -a MODELS <<< "$PRESET"
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"

# Guard against running on GPUs with active external processes (causes noisy ITL).
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  echo "=== Checking exclusivity of GPU(s): ${GPUS[*]} ==="
  for g in "${GPUS[@]}"; do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')"
    util_pct="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | tr -d ' ')"
    gname="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$g")"
    echo "  GPU $g ($gname): used=${used_mib} MiB, util=${util_pct}%"
    if [[ "$used_mib" -gt 1024 || "$util_pct" -gt 5 ]]; then
      echo "ERROR: GPU $g appears busy (used=${used_mib} MiB, util=${util_pct}%)." >&2
      echo "Pick idle GPUs or set SKIP_GPU_CHECK=1 to override." >&2
      exit 2
    fi
  done
fi

mkdir -p "$OUT_DIR"

# Round-robin shard MODELS across GPUS
NUM_GPUS="${#GPUS[@]}"
declare -a SHARDS=()
for ((i = 0; i < NUM_GPUS; i++)); do
  SHARDS[$i]=""
done
for idx in "${!MODELS[@]}"; do
  slot=$((idx % NUM_GPUS))
  if [[ -z "${SHARDS[$slot]}" ]]; then
    SHARDS[$slot]="${MODELS[$idx]}"
  else
    SHARDS[$slot]="${SHARDS[$slot]},${MODELS[$idx]}"
  fi
done

echo "=== Launching Issue #69844 suite -> $OUT_DIR ==="
PIDS=()
for ((i = 0; i < NUM_GPUS; i++)); do
  if [[ -z "${SHARDS[$i]}" ]]; then
    continue
  fi
  g="${GPUS[$i]}"
  log_file="$OUT_DIR/gpu_${g}.log"
  echo "  GPU $g -> models=${SHARDS[$i]} (log: $log_file)"
  CUDA_VISIBLE_DEVICES="$g" "$PY" "$SCRIPT_DIR/run_suite.py" \
    --models "${SHARDS[$i]}" \
    --stages "$STAGES" \
    --model-root /share/shared_weights \
    --model-root /share/xbh/model \
    --out-dir "$OUT_DIR" \
    --skip-existing \
    > "$log_file" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

echo "=== Generating summaries ==="
"$PY" "$SCRIPT_DIR/summarize.py" "$OUT_DIR" --out "$OUT_DIR/summary.md"

ALL_DIRS=()
for d in "$SCRIPT_DIR"/results/*; do
  if [[ -d "$d" && -f "$d/env_metadata.json" ]]; then
    ALL_DIRS+=("$d")
  fi
done
if [[ "${#ALL_DIRS[@]}" -ge 2 ]]; then
  "$PY" "$SCRIPT_DIR/summarize.py" "${ALL_DIRS[@]}" --out "$SCRIPT_DIR/results/cross_gpu_summary.md"
fi

echo "=== Done (exit=$FAIL). Summary: $OUT_DIR/summary.md ==="
exit "$FAIL"
