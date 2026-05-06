# Mooncake rollout transfer design and results

## Background

ROLL moves rollout results as `DataProto` objects from generation workers back to the driver/training side. The optimized rollout-transfer path keeps the existing `DataProto` semantics while allowing the transfer backend to be selected by config:

- `legacy`
- `ray_optimized`
- `mooncake`

The Mooncake backend is intended to reduce backend put/get overhead for large rollout payloads, especially in disaggregated or multi-node runs.

## Design

### Shared transfer payload

`ray_optimized` and `mooncake` use the same protocol v1 payload layout:

1. trim generation-only fields for the target transfer stage;
2. encode tensors and non-tensor fields into buffer specs plus bulk bytes;
3. keep metadata separate from bulk payload;
4. decode back into a normal local `DataProto` before downstream code calls `reorder`, `chunk`, `concat`, or other `DataProto` APIs.

This keeps the transfer layer backend-specific while avoiding special compatibility logic for normal `DataProto` operations.

### Mooncake store path

The Mooncake backend stores one rollout payload under one key and reads it back through pre-registered buffers:

1. producer encodes metadata and bulk chunks;
2. producer writes the object through Mooncake store;
3. consumer reads metadata first;
4. consumer uses `get_into_ranges` to scatter multiple bulk ranges into the destination buffer;
5. consumer decodes the buffer into `DataProto`.

ROLL only uses Mooncake `get_into_ranges` for range reads. The temporary `batch_get_buffer_ranges` experiment is not part of the production path.

### Metrics

When enabled, transfer metrics are written into `data.meta_info["metrics"]`, including:

- `transfer/time/serialize`
- `transfer/time/put`
- `transfer/time/get`
- `transfer/time/deserialize`
- `transfer/profile/time_seconds/backend_get`
- `transfer/profile/time_seconds/from_transfer_payload`
- `transfer/profile/time_seconds/decode_tensors`
- `transfer/profile/time_seconds/decode_non_tensors`
- `transfer/throughput/e2e_mbps`

Enable with:

```bash
rollout_transfer_metrics_enabled=true
rollout_transfer_profiling_enabled=true
```

## Validation summary

### Range-read comparison

All paths were measured after warmup. Generic `get_into_ranges` was sufficient, so no extra ROLL-facing API is needed.

| path | avg backend_get_s | avg get_s | avg metadata_range_get_s | avg bulk_range_get_s |
|---|---:|---:|---:|---:|
| batch_get_buffer_ranges | 0.043881 | 0.012624 | 0.003989 | 0.008499 |
| generic_get_into_ranges | 0.042767 | 0.011940 | 0.003316 | 0.008496 |
| fast_get_into_ranges | 0.044175 | 0.013357 | 0.004698 | 0.008535 |

### Dual-node synthetic benchmark

The benchmark used realistic `DataProto` shapes and compared warmed measured runs.

| backend | avg serialize_s | avg put_s | avg get_s | avg deserialize_s | avg backend_get_s | avg e2e_mbps |
|---|---:|---:|---:|---:|---:|---:|
| ray_optimized | 1.080286 | 0.095956 | 0.086608 | 0.030341 | 0.116949 | 531.9 |
| mooncake | 1.045302 | 0.016749 | 0.014197 | 0.030632 | 0.044829 | 1355.7 |

Mooncake improved e2e transfer throughput by about `2.55x` in this benchmark.

### Full ROLL RL pipeline

A full `RLVRPipeline` run was used to validate transfer metrics in a real training step. This run used the text RLVR configuration as a complete pipeline smoke/measurement path:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- config: `examples/qwen2.5-7B-rlvr_megatron/rlvr_config_8gpus`
- `rollout_batch_size=256`
- `num_return_sequences_in_group=8`

| backend | avg payload_mb | avg serialize_s | avg put_s | avg get_s | avg deserialize_s | avg backend_get_s | avg e2e_mbps |
|---|---:|---:|---:|---:|---:|---:|---:|
| ray_optimized | 78.552069 | 0.122854 | 0.102384 | 0.045547 | 0.028945 | 0.074492 | 1047.291221 |
| mooncake_generic | 78.554517 | 0.091561 | 0.070208 | 0.028652 | 0.028110 | 0.056762 | 1703.730487 |

Mooncake improved full-pipeline rollout transfer throughput by about `1.63x`. Overall step time is still dominated by generation, log-prob, reward, and training compute, so transfer improvements do not translate linearly to total step-time reduction.

A separate Qwen2.5-VL-7B detection validation was started with the intended train/validation parquet files, but it did not produce transfer metrics in this run because worker initialization failed before reaching rollout transfer.

## Running focused tests

```bash
pytest tests/distributed/scheduler/test_protocol.py \
  tests/distributed/scheduler/test_rollout_transfer_backend.py \
  tests/distributed/scheduler/test_rollout_transfer_benchmark.py
```

## Notes

- Use `rollout_transfer_backend=mooncake` and `rollout_transfer_protocol=v1` for Mooncake rollout transfer.
- For multi-client full-pipeline runs, avoid over-large Mooncake RDMA segment settings; start with smaller segment sizes and scale based on available RDMA registration resources.
- Keep Ray and Mooncake comparisons on the same transfer payload format so backend put/get is the main variable.
