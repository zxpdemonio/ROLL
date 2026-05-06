# Mooncake rollout-transfer troubleshooting notes

This note records the practical pitfalls hit while making ROLL's `mooncake` rollout-transfer backend run through a real RLVR smoke workload, plus the working setup and prevention checklist.

## Known-good smoke result

A single-node RLVR smoke run completed successfully with:

- backend: `rollout_transfer_backend=mooncake`
- protocol: `rollout_transfer_protocol=v1`
- transfer: Mooncake Store over RDMA
- expanded rollout samples: `rollout_batch_size=32`, `num_return_sequences_in_group=2`, `is_num_return_sequences_expand=true`
- result: pipeline reached `pipeline step 0 finished` and `pipeline complete!`

## Working runtime environment

Use the CUDA-capable runtime for ROLL, and put the rebuilt Mooncake wheel ahead of older installed bindings:

```bash
export PATH=/root/sglang-venv/bin:$PATH
export PYTHONPATH=/tmp/roll-mooncake-wheel:/root/sglang-venv/lib/python3.12/site-packages:/root/ROLL
export RAY_ADDRESS=192.168.22.70:6391
export MASTER_ADDR=192.168.22.70
export MASTER_PORT=6391
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=0
export MOONCAKE_MASTER=192.168.22.70:15062
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE
export MOONCAKE_LOCAL_HOSTNAME=192.168.22.70
export MOONCAKE_GLOBAL_SEGMENT_SIZE=1GB
export MOONCAKE_LOCAL_BUFFER_SIZE=128MB
```

The important detail is the `PYTHONPATH` order. `/tmp/roll-mooncake-wheel` must come before the venv site-packages path so Python imports the rebuilt `mooncake/store.so` instead of any stale extension module already installed in the venv.

## Mooncake source and build path

Use `/root/Mooncake-ROLL` as the canonical Mooncake source for ROLL testing. For clean upstream-main builds, use a separate worktree such as `/root/Mooncake-ROLL-upstream-main` instead of modifying dirty working files.

The working wheel was built from upstream main where the Python binding exposes `ReplicateConfig.with_hard_pin`. The resulting wheel was extracted to `/tmp/roll-mooncake-wheel` so it can override stale installed bindings without mutating the runtime venv.

## Pitfalls and fixes

### 1. `RPC_FAIL`, `invalid rpc arg`, `ret=-900`

Symptom:

```text
Mooncake store put failed ... RPC_FAIL ... invalid rpc arg ... ret=-900
```

Root cause:

- Mooncake Python binding and `mooncake_master` were ABI/reflection-incompatible.
- The old Python binding's `ReplicateConfig` lacked `with_hard_pin`, while the master/source expected it.

Fix:

- Build Mooncake from current upstream main under `/root/Mooncake-ROLL-upstream-main`.
- Run the matching `mooncake_master` from that wheel/build.
- Verify the imported binding includes `with_hard_pin` before running ROLL.

Prevention:

```python
from mooncake.store import ReplicateConfig
assert hasattr(ReplicateConfig(), "with_hard_pin")
```

Also verify `mooncake.store.__file__` points to the intended rebuilt wheel path, not a stale `.so` in another environment.

### 2. Stale extension module shadows the rebuilt wheel

Symptom:

- Reinstalling the rebuilt wheel appears successful, but Python still imports an older module.
- `ReplicateConfig` still lacks `with_hard_pin`.

Root cause:

- An existing file such as `mooncake/store.cpython-312-x86_64-linux-gnu.so` can take precedence over the newly installed `mooncake/store.so` in the same package directory.

Fix:

- Extract the rebuilt wheel to a clean directory, for example `/tmp/roll-mooncake-wheel`.
- Put that directory first in `PYTHONPATH`.
- Keep the venv site-packages path after it so Ray workers still find dependencies such as `pybase64`.

Prevention:

```bash
python - <<'PY'
import mooncake.store as store
from mooncake.store import ReplicateConfig
print(store.__file__)
print(hasattr(ReplicateConfig(), "with_hard_pin"))
PY
```

The path should be under `/tmp/roll-mooncake-wheel`, and the final line should be `True`.

### 3. CUDA-unavailable venv makes ROLL select `CpuPlatform`

Symptom:

- ROLL reports or behaves as `CpuPlatform` even on a GPU node.
- Resource mapping becomes wrong.
- Ray 2.54 can fail with:

```text
ValueError: Use the 'num_cpus' and 'num_gpus' keyword instead of 'CPU' and 'GPU' in 'resources' keyword
```

Root cause:

- ROLL chooses `current_platform` based on `torch.cuda.is_available()`.
- `/root/ray254-roll-bench-venv` had the right Ray/Mooncake pieces but CUDA was unavailable due to PyTorch/driver compatibility, so ROLL selected CPU.

Fix:

- Run ROLL with `/root/sglang-venv`, where CUDA and Ray 2.54 are available.
- Use the rebuilt Mooncake wheel via `PYTHONPATH`, not by switching to a CUDA-broken venv.

Prevention:

```bash
python - <<'PY'
import torch
from roll.platforms import current_platform
print(torch.cuda.is_available())
print(type(current_platform).__name__, current_platform.ray_device_key)
PY
```

Expected output includes `True` and `CudaPlatform GPU`.

### 4. Ray cluster version mismatch

Symptom:

```text
RuntimeError: Version mismatch: cluster Ray: 2.48.0, process Ray: 2.54.0
```

Root cause:

- The running Ray head was started with a different Ray version than the local process.

Fix:

- Restart or attach to a Ray cluster using the same Ray version as the ROLL runtime.
- For this benchmark path, the working process runtime is `/root/sglang-venv` with Ray 2.54.

Prevention:

- Check both local `ray.__version__` and the cluster startup environment before long runs.

### 5. Missing worker dependencies after overriding `PYTHONPATH`

Symptom:

```text
ModuleNotFoundError: No module named 'pybase64'
```

Root cause:

- Setting `PYTHONPATH=/tmp/roll-mooncake-wheel:/root/ROLL` hides the venv's site-packages from Ray workers.

Fix:

- Include the venv site-packages directory in `PYTHONPATH` after the extracted wheel:

```bash
export PYTHONPATH=/tmp/roll-mooncake-wheel:/root/sglang-venv/lib/python3.12/site-packages:/root/ROLL
```

Prevention:

- Any override that injects the rebuilt Mooncake wheel must still preserve the full runtime dependency path.

### 6. RDMA memory registration failure with large per-client segments

Symptom:

```text
register_local_memory_failed base=... size=17179869184, error=-202
Failed to mount segment: INVALID_PARAMS
Failed to register memory ... Input/output error [5]
```

Root cause:

- Multiple Ray scheduler/driver processes each created a Mooncake Store client and attempted to mount/register a 16GB segment.
- The RDMA driver rejected later 16GB memory registrations.

Fix:

- For smoke and benchmark startup, use smaller per-client segments:

```bash
export MOONCAKE_GLOBAL_SEGMENT_SIZE=1GB
export MOONCAKE_LOCAL_BUFFER_SIZE=128MB
```

Prevention:

- Do not use large segment defaults blindly when many Ray actors may initialize their own Mooncake clients.
- Scale segment size with expected payload size and client count.
- Treat 16GB as a high-capacity setting that needs RDMA registration capacity validation, not as a safe default.

### 7. Hydra config path depends on current working directory

Symptom:

```text
hydra.errors.MissingConfigException: Primary config directory not found: /root/examples/...
```

Root cause:

- A relative Hydra `config_path` was evaluated from the current working directory, not from `/root/ROLL`.

Fix:

- Use the project launcher from the repo root, or use `initialize_config_dir(config_dir="/root/ROLL/examples/...")` in ad-hoc smoke scripts.

Prevention:

- Prefer absolute config dirs in one-off benchmark probes.
- Prefer checked-in launcher scripts for repeatable benchmark runs.

### 8. Local ad-hoc launcher drifted from repo API

Symptoms:

```text
AttributeError: type object 'RLVRConfig' has no attribute 'from_dict'
TypeError: init() takes 0 positional arguments but 1 was given
```

Root cause:

- The ad-hoc script used APIs that were not present in this repository revision.

Fix:

- Follow `examples/start_rlvr_pipeline.py`:
  - parse with `dacite.from_dict(data_class=RLVRConfig, ...)`
  - call `init()` with no positional config argument
  - construct `RLVRPipeline(pipeline_config=ppo_config)`

Prevention:

- Reuse existing launcher patterns instead of inventing a parallel entrypoint.

### 9. Smoke batch too small for training mini-batch

Symptom:

```text
AssertionError: 16 % 32 != 0
```

Root cause:

- The generated training batch size was not divisible by `per_device_train_batch_size * gradient_accumulation_steps`.
- The tested config had `per_device_train_batch_size=1` and `gradient_accumulation_steps=32`, so the train mini-batch requirement was 32.

Fix:

- Use `rollout_batch_size=32` with `num_return_sequences_in_group=2`, producing 64 samples for the successful smoke run.

Prevention:

- Before reducing smoke workload size, check the final train batch divisibility requirement.
- Keep smoke batch sizes small but divisible by the configured train mini-batch.

### 10. Validation runs at global step 0

Symptom:

- A smoke run still performs large validation generation even with `eval_steps=1000`.

Root cause:

- Validation is gated by `global_step % eval_steps == 0`; at `global_step=0`, this is true.

Fix:

- Account for validation cost in smoke timings, or explicitly use a config path that disables/removes validation data when the goal is train-only transfer testing.

Prevention:

- Do not assume large `eval_steps` skips validation at startup.

### 11. Ray session disk is nearly full

Symptom:

```text
/tmp/ray/session_... is over 95% full, available space: 0 GB; Object creation will fail if spilling is required.
```

Root cause:

- Ray's session/spill directory is full or nearly full.

Fix:

- For non-destructive runs, reduce payload sizes and avoid spilling.
- If cleanup is needed, coordinate before deleting Ray session data because it can affect shared/running jobs.

Prevention:

- Check Ray temp/spill capacity before large transfer benchmarks.
- Prefer a configured spill directory with enough capacity for large benchmark runs.

## Preflight checklist

Before running Mooncake + ROLL benchmarks:

1. Verify CUDA platform:

   ```bash
   python - <<'PY'
   import torch
   from roll.platforms import current_platform
   print(torch.cuda.is_available())
   print(type(current_platform).__name__, current_platform.ray_device_key)
   PY
   ```

2. Verify Mooncake binding path and ABI:

   ```bash
   python - <<'PY'
   import mooncake.store as store
   from mooncake.store import ReplicateConfig
   print(store.__file__)
   print(hasattr(ReplicateConfig(), "with_hard_pin"))
   PY
   ```

3. Verify Ray version alignment between local runtime and cluster.
4. Verify `PYTHONPATH` keeps both the rebuilt wheel and venv dependencies.
5. Use RDMA settings only with a reachable `MOONCAKE_MASTER` and correct `MOONCAKE_LOCAL_HOSTNAME`.
6. Start with `MOONCAKE_GLOBAL_SEGMENT_SIZE=1GB` and `MOONCAKE_LOCAL_BUFFER_SIZE=128MB` unless the benchmark requires larger payloads.
7. Ensure rollout/train batch divisibility before launching a long run.
8. Confirm `/tmp/ray` or Ray spill storage has free capacity.
9. For repeatable comparisons, keep model, batch size, validation behavior, and device mapping identical between `ray_optimized` and `mooncake` runs.

## Minimal known-good smoke parameters

These are the important knobs from the successful smoke run:

```text
pretrain=Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain=Qwen/Qwen2.5-0.5B-Instruct
rollout_batch_size=32
num_return_sequences_in_group=2
is_num_return_sequences_expand=true
rollout_transfer_backend=mooncake
rollout_transfer_protocol=v1
rollout_transfer_metrics_enabled=true
rollout_transfer_profiling_enabled=true
actor_infer.device_mapping=[0,1]
actor_train.device_mapping=[2,3]
reference.device_mapping=[4]
rewards.llm_judge.device_mapping=[5]
```

Expected success signal:

```text
pipeline step 0 finished
pipeline complete!
```

## Benchmark interpretation notes

- A successful direct Mooncake Store probe only proves Store put/get works; it does not prove ROLL's full scheduler, reward, reference, and train path works.
- A successful ROLL smoke proves integration correctness for the tested topology and payload size; it is not yet a final performance comparison.
- Mooncake-vs-Ray comparisons are only meaningful when:
  - both use the same generated workload shape,
  - both complete through the same pipeline stage,
  - Mooncake reports real Store/RDMA mode rather than a fallback path,
  - validation cost is either included for both or excluded for both.
