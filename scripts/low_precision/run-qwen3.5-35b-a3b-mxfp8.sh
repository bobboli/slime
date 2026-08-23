#!/bin/bash

set -euo pipefail

if [[ -z "${BASE_FOLDER:-}" ]]; then
  echo "BASE_FOLDER is not set. Point it at the directory containing the model checkpoints and datasets." >&2
  exit 1
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "MASTER_ADDR is not set. Set it to the address of the Ray head node." >&2
  exit 1
fi

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
HOSTFILE="${HOSTFILE:-}"
SSH_USER="${SSH_USER:-root}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

HF_CHECKPOINT="${HF_CHECKPOINT:-${BASE_FOLDER}/Qwen3.5-35B-A3B-MXFP8}"
REF_LOAD="${REF_LOAD:-${BASE_FOLDER}/Qwen3.5-35B-A3B_torch_dist}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${REF_LOAD}}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-${BASE_FOLDER}/Qwen3.5-35B-A3B_mxfp8_slime}"
PROMPT_DATA="${PROMPT_DATA:-${BASE_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-${BASE_FOLDER}/aime-2024-boxed/aime-2024.jsonl}"

NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-2048}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8704}"
SEQ_LENGTH="${SEQ_LENGTH:-34816}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.5}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-35b-a3b-mxfp8}"

for value in \
  "ACTOR_NUM_NODES=${ACTOR_NUM_NODES}" \
  "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}" \
  "NUM_ROLLOUT=${NUM_ROLLOUT}" \
  "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}" \
  "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}" \
  "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" \
  "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}" \
  "ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}"; do
  if [[ ! "${value#*=}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value%%=*} must be a positive integer: ${value#*=}" >&2
    exit 1
  fi
done

ACTOR_WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
if ((ACTOR_WORLD_SIZE % (2 * 2 * 4) != 0)); then
  echo "Actor world size ${ACTOR_WORLD_SIZE} is incompatible with trainer TP2/PP2/CP4." >&2
  exit 1
fi
if ((ACTOR_WORLD_SIZE % (2 * 4) != 0)); then
  echo "Actor world size ${ACTOR_WORLD_SIZE} is incompatible with trainer TP2/EP4." >&2
  exit 1
fi
if ((GLOBAL_BATCH_SIZE % (ACTOR_WORLD_SIZE / (2 * 2 * 4)) != 0)); then
  echo "GLOBAL_BATCH_SIZE must be divisible by the trainer data-parallel size." >&2
  exit 1
fi
if ((GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT)); then
  echo "GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT." >&2
  exit 1
fi
if ((ROLLOUT_NUM_GPUS > ACTOR_WORLD_SIZE)); then
  echo "ROLLOUT_NUM_GPUS cannot exceed the colocated actor world size (${ACTOR_WORLD_SIZE})." >&2
  exit 1
fi
if ((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0)); then
  echo "ROLLOUT_NUM_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE." >&2
  exit 1
fi
if [[ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -ne 4 ]]; then
  echo "This recipe requires four GPUs per rollout engine for DP4/EP4." >&2
  exit 1
fi

WORKER_HOSTS=()
if ((ACTOR_NUM_NODES > 1)); then
  if [[ -z "${HOSTFILE}" || ! -f "${HOSTFILE}" ]]; then
    echo "HOSTFILE must name an existing hostfile when ACTOR_NUM_NODES is greater than one." >&2
    exit 1
  fi
  mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}" | sort -u)
  for HOST in "${HOSTS[@]}"; do
    if [[ "${HOST}" != "${MASTER_ADDR}" ]]; then
      WORKER_HOSTS+=("${HOST}")
    fi
  done
  if [[ "${#WORKER_HOSTS[@]}" -ne $((ACTOR_NUM_NODES - 1)) ]]; then
    echo "HOSTFILE must list exactly $((ACTOR_NUM_NODES - 1)) workers, optionally plus MASTER_ADDR." >&2
    exit 1
  fi
fi

for path in \
  "${HF_CHECKPOINT}/config.json" \
  "${HF_CHECKPOINT}/model.safetensors.index.json" \
  "${REF_LOAD}" \
  "${LOAD_CHECKPOINT}" \
  "${PROMPT_DATA}" \
  "${EVAL_DATA}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required input not found: ${path}" >&2
    exit 1
  fi
done

python3 - "${HF_CHECKPOINT}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)
quantization_config = config.get("quantization_config", {})
if quantization_config.get("quant_method") != "mxfp8":
    raise ValueError(f"Expected a serialized MXFP8 checkpoint: {sys.argv[1]}")
PY

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=../models/qwen3.5-35B-A3B.sh
source "${SCRIPT_DIR}/../models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_CHECKPOINT}"
  --save "${SAVE_CHECKPOINT}"
  --save-interval "${SAVE_INTERVAL}"
  --dist-ckpt-workers 1
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rollout-seed 42
  --rm-type deepscaler
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1
  --num-steps-per-rollout 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

EVAL_ARGS=(
  --eval-interval "${EVAL_INTERVAL}"
  --eval-prompt-data aime "${EVAL_DATA}"
  --n-samples-per-eval-prompt 8
  --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --eval-temperature 1
  --eval-top-p 1
  --log-passrate
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 2
  --context-parallel-size 4
  --expert-model-parallel-size 4
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --seq-length "${SEQ_LENGTH}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --moe-token-dispatcher-type flex
  --moe-flex-dispatcher-backend deepep
  --transformer-impl transformer_engine
  --bf16
  --fp8-format e4m3
  --fp8-recipe mxfp8
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-distributed-optimizer
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-dtype bfloat16
  --sglang-kv-cache-dtype auto
  --sglang-enable-dp-attention
  --sglang-dp-size 4
  --sglang-ep-size 4
  --sglang-enable-dp-lm-head
  --sglang-moe-dense-tp-size 1
  --sglang-fp8-gemm-backend flashinfer_trtllm
  --sglang-moe-runner-backend flashinfer_trtllm_routed
  --sglang-moe-a2a-backend flashinfer
  --sglang-cuda-graph-bs-decode 1 2 4 8 16 32
  --sglang-max-running-requests 256
  --sglang-mamba-radix-cache-strategy extra_buffer
  --sglang-attention-backend triton
)

WEIGHT_SYNC_ARGS=(
  --update-weight-mode full
  --update-weight-transport nccl
  --update-weight-buffer-size 2147483648
)

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
  )
fi

MISC_ARGS=(
  --seed 1234
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --loss-mask-type qwen3_5
  --get-mismatch-metrics
  --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
  --custom-config-path "${SCRIPT_DIR}/../../examples/train_infer_mismatch_helper/metrics_only.yaml"
)

ray stop --force || true
pkill -9 sglang || true
ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --port "${RAY_PORT}" \
  --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

if ((${#WORKER_HOSTS[@]} > 0)); then
  for WORKER_HOST in "${WORKER_HOSTS[@]}"; do
    echo "Starting Ray worker on ${WORKER_HOST}"
    ssh "${SSH_USER}@${WORKER_HOST}" \
      "pkill -9 sglang || true; ray stop --force || true; ray start --address=${MASTER_ADDR}:${RAY_PORT} --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_HOST} --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "${no_proxy}",
    "NO_PROXY": "${NO_PROXY}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "${MEGATRON_PATH}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NVSHMEM_DISABLE_NCCL": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_TIMEOUT_MS": "36000000"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  --no-offload-train \
  --offload-rollout \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${WEIGHT_SYNC_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${MISC_ARGS[@]}"
