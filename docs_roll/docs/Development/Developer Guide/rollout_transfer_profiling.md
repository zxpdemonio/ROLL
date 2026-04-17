# Rollout Transfer Profiling Guide

## Purpose

This guide describes the fine-grained profiling and benchmark support for rollout transfer optimization.

## Config switches

Enable these flags when you want transfer profiling metrics:

- `rollout_transfer_metrics_enabled=true`
- `rollout_transfer_profiling_enabled=true`
- optionally `rollout_transfer_debug_validate=true` in dev/test

## Emitted profiling metrics

The optimized path emits transfer metrics under the existing `transfer/*` namespace and fine-grained profiling metrics under `transfer/profile/*`.

Representative profiling keys:

- `transfer/profile/time_seconds/expand_requests`
- `transfer/profile/mm_ref_count/expand_requests`
- `transfer/profile/mm_dup_object_count/expand_requests`
- `transfer/profile/time_seconds/mm_ref_resolve`
- `transfer/profile/time_seconds/trim_for_stage`
- `transfer/profile/dropped_key_count/trim_for_stage`
- `transfer/profile/time_seconds/to_transfer_payload`
- `transfer/profile/temp_buffer_count/to_transfer_payload`
- `transfer/profile/peak_rss_gb/to_transfer_payload`
- `transfer/profile/time_seconds/from_transfer_payload`
- `transfer/profile/temp_buffer_count/from_transfer_payload`
- `transfer/profile/peak_rss_gb/from_transfer_payload`
- `transfer/profile/time_seconds/backend_get`
- `transfer/profile/peak_rss_gb/backend_get`
- `transfer/profile/time_seconds/materialize_rollout_transfer`
- `transfer/profile/time_seconds/materialize_concat`
- `transfer/profile/input_count/materialize_concat`

## Benchmarks

Current benchmark-style coverage lives in:

- `tests/distributed/scheduler/test_rollout_transfer_benchmark.py`

It covers:

1. protocol-layer v1 encode/decode metrics
2. multimodal dedup vs inline payload duplication during request expansion
3. optimized Ray backend round-trip profiling metrics

## Running

Run the focused benchmark-style tests with:

```bash
pytest tests/distributed/scheduler/test_rollout_transfer_benchmark.py -q
```

Run the core transfer regression suite with:

```bash
pytest tests/distributed/scheduler/test_protocol.py tests/distributed/scheduler/test_rollout_transfer_backend.py -q
```

## Notes

- Profiling is opt-in and should stay disabled in normal training runs.
- RSS metrics are process-level snapshots, useful for regression tracking rather than exact allocator attribution.
- `mooncake` remains a future competing backend over the same transfer payload format.
