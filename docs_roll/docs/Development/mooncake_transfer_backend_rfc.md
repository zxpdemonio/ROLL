# RFC: Mooncake Transfer Backend for ROLL DataProto

## Status

Draft.

This document describes the current Mooncake transfer backend proposal for ROLL, including goals, semantics, integration points, lifecycle ownership, expected invasiveness, risks, and testing plan.

## Background

ROLL already has a remote batch abstraction:

- `DataProto.to_remote()` converts a local `DataProto` into a remote batch.
- `RemoteBatch.materialize()` fetches selected fields from the remote batch.
- `DataProto.drop()` / `RemoteBatch.drop()` release remote resources.
- `transfer_backend` switches between transfer implementations.

When TransferQueue was introduced into ROLL, it was an architecture-level change. It added rollout transfer configuration, transfer payload encoding/decoding, scheduler producer changes, and multiple pipeline consumer changes. The Mooncake integration should not repeat that kind of architectural rewrite. Instead, it should be implemented as a peer backend under the existing transfer backend abstraction.

Mooncake PR2050 provides DataProto support in `mooncake.structured_object_store`. The relevant APIs are:

- `MooncakeBundleTransfer.put_dataproto()`
- `MooncakeBundleTransfer.get_dataproto()`
- `MooncakeBundleTransfer.append_dataproto_fields()`
- `MooncakeBundleTransfer.cleanup_dataproto()`
- `export_dataproto_ref()` / `import_dataproto_ref()`

Therefore ROLL should not reimplement tensor serialization or a minimal KV-row adapter. The integration should directly use Mooncake's DataProto / structured object APIs.

## Goals

1. Add Mooncake as a peer transfer backend alongside TransferQueue.
2. Avoid changes to RL pipelines, schedulers, and training loops.
3. Preserve the external semantics of `DataProto.to_remote()`.
4. Use Mooncake native tensor payloads for tensor fields.
5. Avoid `torch.save`, byte fallback paths, `cpu().numpy().tobytes()`, or other extra tensor serialization in ROLL.
6. Preserve remote tensor structure so consumers can fetch only the fields and rows they need.
7. Support existing ROLL remote batch operations:
   - `materialize`
   - `select`
   - `slice`
   - `repeat`
   - `cat`
   - `union` / append fields
   - `drop` cleanup
8. Preserve the existing `ref_data` semantics: it only supplies row ids and does not mean append to `ref_data`'s remote storage.

## Why Mooncake

Mooncake is useful here because rollout transfer is not just control-plane metadata. It moves large tensor batches between producer and consumer workers. TransferQueue already gives ROLL a row-id KV abstraction, but it still treats stored fields as `TensorDict` values behind the TransferQueue API. Mooncake can provide a lower-level data movement path that is better aligned with large tensor payloads and cross-node transfer.

The expected advantages are:

1. **Native tensor transport**

   Mooncake PR2050 can store tensor fields as tensor payloads through structured object metadata. ROLL does not need to serialize tensors into Python bytes, pickle blobs, or `torch.save` payloads. This is the main reason the current design uses `put_dataproto()` instead of the older bytes-based Mooncake store prototype.

2. **Less redundant memory copying**

   A bytes-based path requires turning tensors into contiguous CPU bytes and then reconstructing tensors on the consumer side. Mooncake's structured object path keeps tensor metadata and payload references separate, allowing Mooncake to use its native `put_tensor` / `get_tensor` path internally. This reduces avoidable CPU serialization overhead and memory pressure for large rollout batches.

3. **Partial materialization**

   ROLL frequently needs only a subset of fields at different stages. `MooncakeBundleTransfer.get_dataproto()` supports field selection and row selection. `MooncakeRemoteBatch.materialize(fields=...)`, `select_idxs()`, and `slice()` map directly to those capabilities, so consumers do not have to pull the entire batch when they only need selected fields or rows.

4. **Append without rewriting existing payloads**

   ROLL often creates a remote batch first and appends more fields later, for example generated outputs followed by logprobs, rewards, or other derived tensors. Mooncake PR2050 supports `append_dataproto_fields()`. For same-stage append, it merges manifests while keeping old payloads in place. This matches ROLL's "same rows, more fields" semantics without rewriting the existing tensor payloads.

5. **Cross-node data movement backend**

   Mooncake is designed as a data transfer/storage backend for distributed LLM systems. It can use protocols such as TCP/RDMA depending on runtime configuration, making it a better long-term target for multi-node rollout transfer than a Ray object-store-centered path.

6. **Lifecycle-aware remote handles**

   Mooncake DataProto refs contain stage refs, field indexes, manifest information, and cleanup relationships. ROLL can model those as `MooncakeRemoteBatch` segments, preserving ownership and row selection without exposing Mooncake details to pipeline code.

These advantages are only realized if ROLL uses Mooncake's structured DataProto API directly. A fallback design that packs a whole DataProto into bytes and stores it in Mooncake would use Mooncake only as a byte KV store, losing most of the benefits above.

## Non-goals

1. Do not redesign ROLL's transfer backend abstraction in this change.
2. Do not add Mooncake-specific branches in pipeline or scheduler code.
3. Do not add new training script CLI flags.
4. Do not implement deployment orchestration for Mooncake master or metadata servers.
5. Do not change TransferQueue's row-id KV semantics.
6. Do not support appending to a Mooncake batch that has already been concatenated into multiple ref segments. Multi-segment batches can still be materialized and dropped, but append requires a single DataProto ref segment.

## Current Design

### Configuration

The design reuses the existing `TransferBackendArguments` config:

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    namespace: roll
    require_native_tensors: true
```

Current `backend_config` fields include:

- `namespace`
- `key_prefix`
- `default_chunk_bytes`
- `buffer_pool`
- `require_native_tensors`
- `policy.max_inflight_put`
- `policy.put_mode`
- `policy.copy_mode`
- Mooncake store setup overrides:
  - `local_hostname`
  - `metadata_server`
  - `global_segment_size`
  - `local_buffer_size`
  - `protocol`
  - `rdma_devices`
  - `master_server_addr`
  - `enable_ssd_offload`
  - `ssd_offload_path`

### Initialization

`transfer_backend.init_transfer_backend()` adds a Mooncake branch:

```python
elif backend_name == "Mooncake":
    _check_mooncake_available()
```

`transfer_backend.init_client()` creates a Mooncake client:

```python
elif config.backend_name == "Mooncake":
    _client = MooncakeClient(config.backend_config)
```

The TransferQueue branch remains unchanged.

### `DataProto.to_remote()`

The previous logic merged tensor and non-tensor fields into one dictionary:

```python
data_dict = data._batch.to_dict() if data._batch is not None else {}
data_dict.update(data._non_tensor_batch)
```

The current implementation keeps that merged dictionary for backward compatibility, while also passing structured fields to the backend:

```python
batch_fields = data._batch.to_dict() if data._batch is not None else {}
non_tensor_fields = data._non_tensor_batch
remote_batch = transfer_backend.put(
    partition,
    row_ids,
    data_dict,
    batch_size,
    batch_fields=batch_fields,
    non_tensor_fields=non_tensor_fields,
    ref_remote_batch=data._remote_batch,
)
```

Important semantics:

- `data._remote_batch is not None` means the current `DataProto` already has a remote batch, so newly local fields should be appended to the same rows.
- `ref_data` only provides row ids. It is not passed to the backend as an append target.

This matches the TransferQueue behavior: `ref_data` is a row alignment reference, not a resource owner and not an append target.

### `MooncakeClient.put()`

Initial remote conversion uses:

```python
ref = transfer.put_dataproto(
    data,
    namespace=self.namespace,
    partition=partition,
    stage=stage,
    policy=self.policy,
)
```

Appending fields to an already remote `DataProto` uses:

```python
ref = import_dataproto_ref(source_segment["handle"])
stage = self._append_stage(ref)
ref = transfer.append_dataproto_fields(ref, data, stage=stage, policy=self.policy)
source_segment["owns_ref"] = False
```

The current `_append_stage()` picks the first existing stage in the ref. ROLL's append semantics are "add more fields to the same rows", not "create a new logical stage". Mooncake PR2050 supports new-stage append, same-stage append, and overwrite, but ROLL currently does not pass logical stage information through `DataProto.to_remote()`. Therefore same-stage append is the closest match to TransferQueue's "write more fields for the same row ids" semantics.

### `MooncakeRemoteBatch`

The implementation adds `MooncakeRemoteBatch(RowRemoteBatch)`. It stores Mooncake DataProto ref handles and row selection information:

```python
segments = [
    {
        "handle": export_dataproto_ref(ref),
        "rows": None | list[int],
        "row_ids": list[str],
        "owns_ref": bool,
    }
]
```

`materialize()` calls:

```python
transfer_backend.get(..., segments=self.segments)
```

which eventually calls:

```python
transfer.get_dataproto(ref, fields=fields, rows=segment["rows"])
```

`select_idxs()` / `slice()` / `repeat()` do not copy Mooncake payloads. They only construct new row selections. Derived batches do not own the ref.

`cat()` concatenates segments from the input batches. The concatenated batch does not own the refs.

`drop()` calls:

```python
transfer.cleanup_dataproto(import_dataproto_ref(segment["handle"]))
```

only for segments where `segment["owns_ref"] == True`.

### Append and Cleanup Ownership

Mooncake PR2050 same-stage append creates a new merged manifest. Existing payloads are not copied or deleted. The new ref records cleanup keys for old manifest/meta cleanup.

Therefore ROLL must transfer ownership after append:

- The new `MooncakeRemoteBatch` owns cleanup.
- The old source segment becomes `owns_ref = False`.

Otherwise an older `DataProto.drop()` could delete payloads that are still referenced by the new merged ref.

`MooncakeRemoteBatch.union()` handles the append-then-union path:

```python
if len(self.segments) == len(rhs.segments) == 1:
    rhs.segments[0]["owns_ref"] = False
super().union(rhs)
```

This keeps the new handle while merging old fields and cache metadata.

## TransferQueue Introduction vs. Current Mooncake Integration

### TransferQueue / Rollout Transfer Backend Introduction

Relevant historical ROLL commits include:

- `9dd6e3ac add rollout transfer config scaffold`
- `2bb04099 add rollout transfer backend abstraction`
- `00e51cab add transfer metrics and validation hooks`
- `e267a167 add rollout transfer profiling and benchmarks`

The core commit `2bb04099` changed:

```text
8 files changed, 521 insertions(+), 25 deletions(-)
```

It touched:

- `roll/distributed/scheduler/generate_scheduler.py`
- `roll/distributed/scheduler/protocol.py`
- `roll/pipeline/rlvr/rlvr_pipeline.py`
- `roll/pipeline/rlvr/rlvr_rollout_pipeline.py`
- `roll/pipeline/rlvr/rlvr_vlm_pipeline.py`
- tests and docs

That change introduced a new architecture:

1. Transfer payload encode/decode in `DataProto`.
2. Backend selection in the scheduler producer path.
3. Transfer handle materialization in multiple RLVR pipeline consumer paths.
4. New legacy/ray_optimized paths and later metrics/profiling additions.

### Current Mooncake Integration

The current production-code footprint is:

```text
roll/distributed/scheduler/protocol.py         15 insertions, 3 deletions
roll/distributed/scheduler/remote_protocol.py  233 insertions
roll/distributed/scheduler/transfer_backend.py 185 insertions, 9 deletions
```

Characteristics:

1. No change to `generate_scheduler.py`.
2. No change to RLVR pipeline code.
3. No change to training loops.
4. No new CLI flags.
5. No change to TransferQueue behavior.
6. Only adds a Mooncake backend and a Mooncake remote batch type under the existing abstraction.

### Invasiveness Assessment

The TransferQueue introduction was high-invasiveness because it established the transfer abstraction itself. The current Mooncake integration is medium-to-low invasiveness because it reuses that abstraction and adds a peer backend.

| Dimension | TransferQueue / rollout transfer introduction | Current Mooncake integration |
|---|---:|---:|
| Scheduler changes | Yes | No |
| Pipeline changes | Yes | No |
| Training loop changes | No | No |
| Core DataProto protocol changes | Yes, payload encoding/decoding | Minimal field splitting |
| New backend abstraction | Yes | No |
| New backend implementation | Yes | Yes |
| Default path impact | High, new legacy/optimized branches | Low, default backend unchanged |
| Main complexity location | scheduler/protocol/pipeline | backend/remote batch |
| Invasiveness | High | Medium-low |

The largest invasive part in the current PR is the 233-line `MooncakeRemoteBatch`. However, it is isolated as a remote batch type and does not spread Mooncake-specific logic into scheduler or pipeline code.

## Why Not Use the Old Mooncake Bytes Fallback Design

ROLL history includes `0a3c60f4 add Mooncake backend and dual-node transfer benchmark`. That design pickled `DataProto.to_transfer_payload()` into bytes and put/get those bytes through Mooncake store.

Problems with that approach:

1. Tensor fields go through CPU byte serialization.
2. Tensor and non-tensor data are packed into one encoded payload.
3. It does not use Mooncake PR2050's native tensor paths.
4. It is not aligned with the requirement to avoid extra memcpy and serialization.

The current design uses PR2050's structured object DataProto API directly, so tensor fields are handled by Mooncake tensor payloads.

## Compatibility

### TransferQueue

The common backend facade now accepts `**kwargs`:

```python
def put(..., **kwargs):
    ...
```

`TransferQueueClient` ignores these kwargs. Its row-id KV behavior remains unchanged.

### Dummy Backend / RayMemoryStore Backend

These clients also accept and ignore `**kwargs` for compatibility.

### DataProto Callers

Callers still use:

```python
DataProto.to_remote(data, partition=..., ref_data=...)
DataProto.drop(data)
data.batch
```

They do not need to know whether the backend is Mooncake.

## Risks

### 1. Incorrect Cleanup Ownership

This is the highest-risk area. After append, the new ref reuses old payloads. If the old ref still owns cleanup, dropping the old batch may corrupt the new batch.

Mitigations:

- Set `source_segment["owns_ref"] = False` after successful append.
- Set the old RHS segment to `owns_ref = False` during union.
- Derived batches from select/slice/repeat/cat do not own refs.
- Add append/drop ordering tests.

### 2. Multi-segment Append

A concatenated Mooncake batch may contain multiple segments. The current append implementation requires a single segment, because PR2050 append operates on one DataProto ref.

Mitigations:

- `MooncakeClient.put()` validates `len(ref_remote_batch.segments) == 1` for append.
- Multi-segment batches can still be materialized and dropped.

### 3. Non-tensor Field Support Boundaries

Mooncake PR2050 supports structured non-tensor codecs, but exact object support depends on Mooncake implementation.

Mitigations:

- Delegate handling to `put_dataproto()`.
- Do not add ROLL-side pickle fallback.
- Cover common object array, string, and numeric fields in e2e tests.

### 4. Native Tensor Path Regression

If Mooncake stores tensor fields through bytes fallback, the integration would violate its main performance requirement.

Mitigations:

- `require_native_tensors=True` validates manifest payload kind for tensor fields.
- Unit tests cover API usage paths.
- Real Mooncake e2e should validate native tensor payloads.

### 5. Import / Circular Dependency Complexity

`MooncakeRemoteBatch` currently lives in `remote_protocol.py`, consistent with existing `RemoteBatch` types.

If file size becomes a concern, it can be moved to a separate module, but that may require careful circular import handling.

## Test Plan

### Unit Tests

`tests/distributed/scheduler/test_rollout_transfer_backend.py` currently covers:

1. `MooncakeClient.put()` uses `put_dataproto()`.
2. `MooncakeClient.get()` uses `get_dataproto()`.
3. `select_idxs()` passes row selection to `get_dataproto(rows=...)`.
4. `DataProto.to_remote()` passes sectioned fields.
5. `ref_data` only reuses row ids and does not trigger append.
6. Existing remote batch triggers `append_dataproto_fields()`.
7. Append uses an existing stage.
8. Drop only cleans owned refs.

Current result:

```text
7 passed
```

### Real Mooncake Smoke Test

Use the locally built Mooncake wheel:

```text
../Mooncake-RL/mooncake-wheel/dist-roll/mooncake_transfer_engine-0.3.11.post1-cp312-cp312-manylinux_2_39_x86_64.whl
```

Basic import check:

```python
from mooncake.structured_object_store import MooncakeBundleTransfer
from mooncake.store import MooncakeDistributedStore
```

### ROLL Wheel Test

Current ROLL wheel:

```text
dist-roll/roll-0.3.0-py3-none-any.whl
```

Recommended test setup: install both Mooncake and ROLL wheels into a clean venv, then run the target tests.

### Pipeline Smoke Tests

Use a small-model or lightweight agentic config, and add:

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    namespace: roll
    require_native_tensors: true
```

Suggested order:

1. `tests/distributed/scheduler/test_rollout_transfer_backend.py`
2. Real Mooncake `put_dataproto/get_dataproto/append_dataproto_fields/cleanup_dataproto` smoke
3. Small-model agentic rollout smoke
4. RLVR `max_steps=1` smoke

## Alternatives

### Alternative A: Move `MooncakeRemoteBatch` out of `remote_protocol.py`

Pros:

- Keeps `remote_protocol.py` smaller.
- Makes Mooncake-specific logic more isolated.

Cons:

- Requires careful import cycle handling.
- Current project pattern keeps `RemoteBatch` types in `remote_protocol.py`.

### Alternative B: Make backend-specific put arguments explicit

Instead of `**kwargs`:

```python
def put(..., batch_fields=None, non_tensor_fields=None, ref_remote_batch=None):
```

Pros:

- More explicit typing.

Cons:

- All backend client signatures need to expose Mooncake-specific optional parameters.
- Higher visible invasiveness.

### Alternative C: Add Mooncake-specific pipeline logic

Not recommended. It would repeat the high-invasiveness pattern from the original TransferQueue abstraction introduction and make business logic aware of Mooncake.

## Recommendation

Keep the current direction:

1. Implement Mooncake as backend-local logic.
2. Do not modify scheduler, pipeline, or training loop code.
3. Keep complexity inside `MooncakeClient` and `MooncakeRemoteBatch`.
4. Focus testing on real Mooncake e2e and cleanup lifecycle correctness.
5. Do not use the old bytes fallback design.

From an invasiveness perspective, the current implementation is substantially smaller than the historical TransferQueue introduction. TransferQueue introduced the abstraction; Mooncake reuses it. The main risk is not architectural invasiveness, but lifecycle correctness and ensuring the real Mooncake runtime uses native tensor payloads as expected.
