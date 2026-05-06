#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/sglang-venv/bin:$PATH
export RAY_ADDRESS="${RAY_ADDRESS:-192.168.22.70:6391}"
export MASTER_ADDR="${MASTER_ADDR:-192.168.22.70}"
export MASTER_PORT="${MASTER_PORT:-6391}"
export MULTI_TENANT="${MULTI_TENANT:-1}"
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-/root/sglang-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/include}"
export PYTHONPATH="${PYTHONPATH:-/root/sglang-venv/lib/python3.12/site-packages:/root/ROLL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export ROLLOUT_TRANSFER_BACKEND="${ROLLOUT_TRANSFER_BACKEND:-ray_optimized}"
export ROLLOUT_TRANSFER_PROTOCOL="${ROLLOUT_TRANSFER_PROTOCOL:-v1}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-16GB}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-1GB}"

cd /root/ROLL

echo "Using RAY_ADDRESS=${RAY_ADDRESS}"
echo "Using MASTER_ADDR=${MASTER_ADDR}"
echo "Using MASTER_PORT=${MASTER_PORT}"
echo "Using NVTE_CUDA_INCLUDE_DIR=${NVTE_CUDA_INCLUDE_DIR}"
echo "Using PYTHONPATH=${PYTHONPATH}"
echo "Using VLLM_USE_V1=${VLLM_USE_V1}"
echo "Using ROLLOUT_TRANSFER_BACKEND=${ROLLOUT_TRANSFER_BACKEND}"
echo "Using ROLLOUT_TRANSFER_PROTOCOL=${ROLLOUT_TRANSFER_PROTOCOL}"
echo "Using MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE}"
echo "Using MOONCAKE_LOCAL_BUFFER_SIZE=${MOONCAKE_LOCAL_BUFFER_SIZE}"

/root/sglang-venv/bin/python - <<'PY'
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
from dacite import from_dict

from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline

overrides = [
    'max_steps=1',
    'logging_steps=1',
    'save_steps=1000',
    'eval_steps=1000',
    'pretrain=Qwen/Qwen2.5-0.5B-Instruct',
    'reward_pretrain=Qwen/Qwen2.5-0.5B-Instruct',
    'rollout_batch_size=256',
    'num_return_sequences_in_group=8',
    '+is_num_return_sequences_expand=true',
    f'+rollout_transfer_backend={__import__("os").environ["ROLLOUT_TRANSFER_BACKEND"]}',
    f'+rollout_transfer_protocol={__import__("os").environ["ROLLOUT_TRANSFER_PROTOCOL"]}',
    '+rollout_transfer_metrics_enabled=true',
    '+rollout_transfer_profiling_enabled=true',
    "actor_infer.device_mapping='[0,1,2,3]'",
    "actor_train.device_mapping='[4,5,6,7]'",
    'actor_train.strategy_args.strategy_name=deepspeed_train',
    '+actor_train.strategy_args.strategy_config.train_micro_batch_size_per_gpu=auto',
    '+actor_train.strategy_args.strategy_config.bf16.enabled=true',
    '+actor_train.strategy_args.strategy_config.fp16.enabled=false',
    '+actor_train.strategy_args.strategy_config.fp16.loss_scale=0',
    '+actor_train.strategy_args.strategy_config.fp16.initial_scale_power=16',
    '+actor_train.strategy_args.strategy_config.fp16.loss_scale_window=1000',
    '+actor_train.strategy_args.strategy_config.fp16.hysteresis=2',
    '+actor_train.strategy_args.strategy_config.fp16.min_loss_scale=1',
    '+actor_train.strategy_args.strategy_config.zero_optimization.stage=3',
    '+actor_train.strategy_args.strategy_config.zero_optimization.offload_optimizer.device=cpu',
    '+actor_train.strategy_args.strategy_config.zero_optimization.offload_optimizer.pin_memory=true',
    '+actor_train.strategy_args.strategy_config.zero_optimization.overlap_comm=true',
    '+actor_train.strategy_args.strategy_config.zero_optimization.contiguous_gradients=true',
    '+actor_train.strategy_args.strategy_config.zero_optimization.sub_group_size=1000000000',
    '+actor_train.strategy_args.strategy_config.zero_optimization.reduce_bucket_size=auto',
    '+actor_train.strategy_args.strategy_config.zero_optimization.stage3_prefetch_bucket_size=auto',
    '+actor_train.strategy_args.strategy_config.zero_optimization.stage3_param_persistence_threshold=auto',
    '+actor_train.strategy_args.strategy_config.zero_optimization.stage3_max_live_parameters=1000000000',
    '+actor_train.strategy_args.strategy_config.zero_optimization.stage3_max_reuse_distance=1000000000',
    '+actor_train.strategy_args.strategy_config.zero_optimization.stage3_gather_16bit_weights_on_model_save=true',
    "reference.device_mapping='[4,5,6,7]'",
    'reference.strategy_args.strategy_name=hf_infer',
    'reference.strategy_args.strategy_config=null',
    'reference.infer_batch_size=1',
    'rewards.llm_judge.model_args.model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct',
    'rewards.llm_judge.model_args.attn_implementation=auto',
    'rewards.llm_judge.strategy_args.strategy_name=hf_infer',
    'rewards.llm_judge.strategy_args.strategy_config=null',
    "rewards.llm_judge.device_mapping='[7]'",
    '+actor_infer.strategy_args.strategy_config.VLLM_USE_V1=0',
]

initialize(config_path='examples/qwen2.5-7B-rlvr_megatron', job_name='app')
cfg = compose(config_name='rlvr_config_8gpus', overrides=overrides)
print(OmegaConf.to_yaml(cfg, resolve=True))
ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))
init()
pipeline = RLVRPipeline(pipeline_config=ppo_config)
pipeline.run()
PY
