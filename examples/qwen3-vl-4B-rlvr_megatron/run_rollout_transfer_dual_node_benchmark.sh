#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONFIG_PATH=${CONFIG_PATH:-examples/qwen3-vl-4B-rlvr_megatron}
CONFIG_NAME=${CONFIG_NAME:-rlvr_dual_node_transfer_benchmark}
ROLLOUT_TRANSFER_BACKEND=${ROLLOUT_TRANSFER_BACKEND:-mooncake}
ROLLOUT_TRANSFER_PROTOCOL=${ROLLOUT_TRANSFER_PROTOCOL:-v1}
BENCHMARK_MAX_STEPS=${BENCHMARK_MAX_STEPS:-10}
BENCHMARK_LOGGING_STEPS=${BENCHMARK_LOGGING_STEPS:-1}
BENCHMARK_EVAL_STEPS=${BENCHMARK_EVAL_STEPS:-1000}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-4}
INFER_DEVICE_MAPPING=${INFER_DEVICE_MAPPING:-list(range(0,4))}
TRAIN_DEVICE_MAPPING=${TRAIN_DEVICE_MAPPING:-list(range(4,8))}
REFERENCE_DEVICE_MAPPING=${REFERENCE_DEVICE_MAPPING:-list(range(4,8))}

export WORLD_SIZE=${WORLD_SIZE:-2}
export MASTER_PORT=${MASTER_PORT:-6379}

require_env() {
  local key=$1
  if [[ -z "${!key:-}" ]]; then
    echo "Missing required environment variable: ${key}" >&2
    exit 1
  fi
}

resolve_local_hostname() {
  if [[ -n "${MOONCAKE_LOCAL_HOSTNAME:-}" ]]; then
    printf '%s\n' "${MOONCAKE_LOCAL_HOSTNAME}"
    return
  fi

  local detected
  detected=$(hostname -I 2>/dev/null | cut -d' ' -f1)
  if [[ -n "${detected}" ]]; then
    printf '%s\n' "${detected}"
    return
  fi

  hostname
}

require_env MASTER_ADDR
require_env RANK

if [[ "${ROLLOUT_TRANSFER_BACKEND}" == "mooncake" ]]; then
  export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-rdma}
  export MOONCAKE_MASTER=${MOONCAKE_MASTER:-${MASTER_ADDR}:50051}
  export MOONCAKE_TE_META_DATA_SERVER=${MOONCAKE_TE_META_DATA_SERVER:-P2PHANDSHAKE}
  export MOONCAKE_LOCAL_HOSTNAME=$(resolve_local_hostname)
fi

echo "[rollout-transfer-benchmark] rank=${RANK} world_size=${WORLD_SIZE} backend=${ROLLOUT_TRANSFER_BACKEND} protocol=${ROLLOUT_TRANSFER_PROTOCOL}"
echo "[rollout-transfer-benchmark] config=${CONFIG_PATH}/${CONFIG_NAME} num_gpus_per_node=${NUM_GPUS_PER_NODE}"
echo "[rollout-transfer-benchmark] actor_infer.device_mapping=${INFER_DEVICE_MAPPING}"
echo "[rollout-transfer-benchmark] actor_train.device_mapping=${TRAIN_DEVICE_MAPPING}"
echo "[rollout-transfer-benchmark] reference.device_mapping=${REFERENCE_DEVICE_MAPPING}"

if [[ "${ROLLOUT_TRANSFER_BACKEND}" == "mooncake" ]]; then
  echo "[rollout-transfer-benchmark] mooncake_protocol=${MOONCAKE_PROTOCOL} mooncake_master=${MOONCAKE_MASTER} mooncake_local_hostname=${MOONCAKE_LOCAL_HOSTNAME}"
fi

python "${REPO_ROOT}/examples/start_rollout_transfer_benchmark.py" \
  --config_path "${CONFIG_PATH}" \
  --config_name "${CONFIG_NAME}" \
  num_gpus_per_node=${NUM_GPUS_PER_NODE} \
  max_steps=${BENCHMARK_MAX_STEPS} \
  logging_steps=${BENCHMARK_LOGGING_STEPS} \
  eval_steps=${BENCHMARK_EVAL_STEPS} \
  save_steps=${BENCHMARK_EVAL_STEPS} \
  rollout_transfer_backend=${ROLLOUT_TRANSFER_BACKEND} \
  rollout_transfer_protocol=${ROLLOUT_TRANSFER_PROTOCOL} \
  rollout_transfer_metrics_enabled=true \
  rollout_transfer_profiling_enabled=true \
  rollout_transfer_enable_mm_dedup=true \
  rollout_transfer_enable_mm_strip=true \
  actor_infer.device_mapping="${INFER_DEVICE_MAPPING}" \
  actor_train.device_mapping="${TRAIN_DEVICE_MAPPING}" \
  reference.device_mapping="${REFERENCE_DEVICE_MAPPING}" \
  "$@"
