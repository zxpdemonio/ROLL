# ROLL Rollout Transfer Optimization Design

## Background

This document proposes a rollout data transfer optimization design for ROLL, targeting a real RLVR-VLM workload with large rollout batches, long responses, and long-video multimodal inputs. The immediate goal is to migrate the useful ideas from `THUDM/slime#1709` into ROLL, while adapting them to ROLL's current `DataProto`-based architecture and the much heavier VLM rollout workload.

The target workload characteristics are:

- `rollout_batch_size: 256`
- `num_return_sequences_in_group: 8`
- `is_num_return_sequences_expand: true`
- effective rollout sample count per step: `256 * 8 = 2048`
- `prompt_length: 2048`
- `response_length: 4096`
- VLM with long-video multimodal inputs

This workload changes the optimization priority: for ROLL, serialization improvements matter, but multimodal duplication and post-generate payload hygiene matter even more.

---

## Goals

1. Reduce rollout transfer overhead for RLVR-VLM workloads.
2. Prevent avoidable duplication of multimodal prompt-side payloads.
3. Introduce an optimized `DataProto` transfer protocol that is independent of the transport backend.
4. Support a phased backend strategy:
   - legacy direct Ray object transfer
   - optimized Ray transfer backend
   - Mooncake backend in a later phase
5. Add a fine-grained profiling plan after implementation to support follow-up performance tuning.

---

## Non-goals

The first version does not aim to:

- change RL algorithms or training math
- change reward logic semantics
- change vLLM/SGLang engine internals
- fully redesign the scheduler/router stack in one step
- require Mooncake in the first implementation

---

## Current State in ROLL

### Unified rollout container

ROLL already has a unified transport object:

- `roll/distributed/scheduler/protocol.py`
- `DataProto`

`DataProto` contains:

- `batch: TensorDict`
- `non_tensor_batch: Dict[str, np.ndarray(dtype=object) | other arrays]`
- `meta_info: Dict`

This is a strong starting point because the optimization can focus on transfer protocol and transport backend rather than inventing a new rollout schema.

### Current serialization path

`DataProto.__getstate__` / `__setstate__` still rely on generic torch serialization for the tensor payload:

- `torch.save(self.batch, buffer)`
- `torch.load(...)`

This is simple and correct, but it becomes costly for large rollout batches with many fields.

### Current transport style

ROLL frequently passes `DataProto` directly through Ray actor RPC / object transport, for example:

- RLVR scheduler and reward paths
- agentic env manager to queue paths

This couples transport mechanics to the business path and makes backend substitution difficult.

---

## Problem Statement

For the target workload, the rollout transfer bottleneck is not only serialization overhead.

There are three distinct bottlenecks:

### 1. Multimodal duplication under `num_return_sequences_expand=true`

For VLM rollout, if prompt-side multimodal payloads are duplicated when requests are expanded from 256 prompts to 2048 samples, transfer cost grows roughly with:

- `O(prompt_count * num_return_sequences * multimodal_payload)`

Instead, multimodal payload should remain prompt-scoped and samples should reference it.

This is the largest theoretical optimization space for long-video workloads.

### 2. Generic `TensorDict` + object-array serialization

Even after removing multimodal duplication, rollout batches remain large. The current `torch.save(TensorDict)` plus heavy `np.object`-based non-tensor fields costs CPU time and makes transport layouts opaque.

### 3. Generated rollout batches may carry fields that are no longer needed

Fields that are necessary for request generation, especially multimodal prompt-side data, should not necessarily remain attached to post-generate rollout batches used by reward, scheduler, and training.

---

## Workload-aware Bottleneck Model

For the target workload, the post-generate text-side rollout tensors alone can already be very large.

Assume an expanded batch size of 2048 and maximum total sequence length near 6144 tokens.

Typical retained fields may include:

- `input_ids`
- `attention_mask`
- `position_ids`
- `responses`
- `response_mask`
- `prompt_mask`
- `infer_logprobs`

Even before multimodal data is considered, these fields can push the rollout batch into the hundreds of megabytes. If multimodal prompt-side payload is duplicated across return sequences, the effective payload can explode far beyond that.

Therefore, optimization priority should be:

1. avoid multimodal duplication
2. strip multimodal fields after generation if no longer needed
3. optimize `DataProto` serialization/deserialization
4. optimize transport backend

---

## Design Overview

The new design is split into three layers:

1. **Semantic payload optimization**
2. **Transfer protocol optimization**
3. **Transport backend abstraction**

### Layer 1: Semantic payload optimization

This layer reduces the amount of data that needs to be transferred at all.

### Layer 2: Transfer protocol optimization

This layer defines how a `DataProto` is serialized into a backend-agnostic payload.

### Layer 3: Transport backend abstraction

This layer defines where the payload goes:

- legacy Ray object transport
- optimized Ray backend
- Mooncake backend later

---

## Semantic Payload Optimization

### A. Prompt-level multimodal deduplication

#### Problem

Under `num_return_sequences_expand=true`, request expansion risks repeating prompt-side multimodal payload eight times per prompt.

#### Design

Multimodal payload must be represented as prompt-scoped data rather than sample-scoped data.

Instead of carrying duplicated `multi_modal_data` in each expanded sample, expanded requests should carry a lightweight reference such as:

- `mm_ref_id`
- `origin_prompt_id`

A separate prompt-level multimodal store or context table should hold the actual payload.

#### Expected effect

This changes multimodal transfer cost from approximately:

- `O(B * N * M)`

into:

- `O(B * M + B * N * text_part)`

for prompt batch size `B`, return sequences `N`, and multimodal payload size `M`.

This is the biggest optimization opportunity for long-video VLM rollout.

### B. Post-generate multimodal stripping

#### Problem

After generation is complete, the training path usually no longer needs prompt-side multimodal payload.

#### Design

Immediately after response postprocessing, remove multimodal generation-only fields from the rollout batch unless a downstream component explicitly requires them.

This should be implemented as a schema-level operation, not a scattered manual `pop` convention.

Examples of fields that should be considered generation-only:

- `multi_modal_data`
- processor intermediate multimodal structures
- prompt-only temporary fields

#### Expected effect

This prevents large multimodal objects from entering reward, scheduler, concat, and training transfer paths.

### C. Stage-aware payload trimming

Add stage-aware trimming semantics to `DataProto` transfer.

Suggested stages:

- `generate_request`
- `post_generate`
- `train_batch`

Each stage should define which fields are preserved.

This makes the transfer protocol workload-aware and avoids unnecessary carry-over.

---

## Transfer Protocol Optimization

### New principle

Do not serialize the whole `TensorDict` as one opaque object. Instead:

1. serialize tensor fields individually as raw typed buffers
2. explicitly encode high-frequency non-tensor fields
3. keep schema and small metadata in a compact metadata block

### New `DataProto` transfer API

Add explicit transfer helpers to `DataProto`:

- `to_transfer_payload(stage: str)`
- `from_transfer_payload(payload)`

The optimized transfer path should use these methods, while the legacy path can continue to rely on current behavior.

### Transfer payload structure

Introduce a backend-agnostic payload format with:

- `meta_bytes`
- `bulk_buffer`
- `buffer_specs`

`meta_bytes` stores schema and small metadata.
`bulk_buffer` stores raw tensor and encoded non-tensor bytes.
`buffer_specs` describe where each field lives.

### Tensor encoding

Each tensor field in `batch` should be encoded individually as:

- CPU contiguous buffer
- dtype
- shape
- offset and size into the bulk buffer

This avoids whole-`TensorDict` `torch.save` costs and makes transport layout explicit.

### Non-tensor encoding

Non-tensor fields should be encoded by category.

#### Numeric arrays

Encode as raw contiguous bytes with dtype and shape.

#### String arrays

Encode as:

- UTF-8 blob
- offsets array

This should be the default for common rollout fields like:

- `domain`
- `tag`
- `tags`
- `traj_id`
- `traj_group_id`
- `rollout_id`

#### Fallback object fields

Only truly irregular fields should use pickle-based fallback.

### `meta_info`

For the first implementation, `meta_info` can remain in `meta_bytes`, serialized with `pickle.dumps(..., protocol=5)`.

If large structured metadata becomes important later, it can be externalized into dedicated buffers.

### Stage-aware encoding

The transfer encoder must be stage-aware.

For example:

- `generate_request` may preserve multimodal references
- `post_generate` should drop generation-only payload
- `train_batch` should contain only fields required by reward/training/metrics

---

## Transport Backend Abstraction

### Backend interface

Add a rollout transfer backend abstraction with a common API:

- `put(data, stage)`
- `get(handle)`
- `cleanup(handle)`

### Backend modes

#### 1. `legacy`

Keep current behavior for compatibility:

- direct Ray transport of `DataProto`

#### 2. `ray_optimized`

Use the new transfer protocol but still store/fetch via Ray.

Flow:

1. `DataProto -> TransferPayload`
2. `TransferPayload -> ray.put(single object)`
3. `ray.get(...) -> TransferPayload -> DataProto`

This allows protocol gains to be measured independently of Mooncake.

#### 3. `mooncake`

Future backend.

Use the same `TransferPayload` but store it as a single bulk value in Mooncake.

This aligns with the useful design direction from `THUDM/slime#1709`:

- single-key bulk transfer
- explicit tensor buffer layout
- better cross-node get-side performance

---

## Integration Plan for the Target RLVR-VLM Workload

Unlike a generic rollout optimization plan, the first integration priority for this workload is **RLVR VLM rollout**, not agentic rollout.

### Phase 1 integration priority

Focus on:

- `roll/distributed/scheduler/generate_scheduler.py`
- `roll/distributed/scheduler/user_defined_rollout_loop.py`
- `roll/distributed/scheduler/router.py`
- `roll/pipeline/rlvr/rlvr_pipeline.py`

### Integration item 1: request expansion without multimodal duplication

Replace full-object request duplication with prompt-aware lightweight expansion.

Expanded requests should share multimodal references rather than duplicate multimodal payload.

In the first implementation, each expanded request should keep only request-local mutable prompt progression state such as `prompt_token_ids`, while the heavy immutable multimodal payload is moved behind a shared `mm_ref_id` stored in prompt-level context.

### Integration item 2: router preprocessing should support multimodal references

`RouterClient._preprocess_generate()` should support:

- old style direct `multi_modal_data`
- new style `mm_ref_id` lookup from a prompt-level multimodal context store

This keeps multimodal payload prompt-scoped.

### Integration item 3: strip multimodal data after generation

After generation and response postprocessing, remove multimodal generation-only payload before the rollout batch enters reward, concat, and training.

### Integration item 4: use optimized transfer protocol for post-generate batches

Once the rollout batch is text-dominant and multimodal fields have been stripped, use the optimized `DataProto` transfer protocol for transport-heavy boundaries.

---

## Further Optimization Space Beyond the Slime PR Direction

The target workload has additional optimization opportunities beyond generic protocol/backend improvements.

### 1. Multimodal content-addressed cache

If the same video or prompt-side multimodal structure appears repeatedly across retries or repeated sampling, store it by content hash and pass references.

### 2. Processor-output caching

For VLM, the expensive multimodal representation may be the processed prompt-side structure rather than the original raw media reference. If processor outputs are stable for repeated prompts, cache them.

### 3. Mask compression

For transfer, the following fields should be compacted:

- `attention_mask`
- `response_mask`
- `prompt_mask`

Suggested first step:

- `attention_mask -> uint8`
- `response_mask/prompt_mask -> bool`

A later phase may bit-pack boolean masks.

### 4. Optional `infer_logprobs` precision compression

If numerically acceptable, transfer-time compression from fp32 to fp16/bf16 could further reduce payload size. This should not be part of the first implementation.

---

## Benchmark and Test Plan

The benchmark plan must be derived from the real workload rather than a small synthetic text-only batch.

### Benchmark A: protocol-layer microbenchmark

Purpose:

- measure serialize/deserialize time and payload size only

Synthetic batch parameters:

- prompt batch size: 256
- return sequences: 8
- expanded batch size: 2048
- prompt length: 2048
- response length: 4096

Construct batch fields similar to the real rollout batch:

- `input_ids`
- `attention_mask`
- `position_ids`
- `responses`
- `response_mask`
- `prompt_mask`
- `infer_logprobs`
- non-tensor fields like `domain`, `tags`, `sample_uuid`

Compare:

- legacy `torch.save(TensorDict)` path
- optimized transfer protocol

Metrics:

- serialize time
- deserialize time
- payload bytes
- peak RSS

### Benchmark B: long-video multimodal duplication benchmark

Purpose:

- isolate the effect of multimodal duplication vs multimodal sharing

Construct three cases:

1. baseline duplicated multimodal payload
2. shared multimodal reference (`mm_ref_id`)
3. shared multimodal reference plus post-generate stripping

Multimodal payload should be synthetic but heavy enough to reflect real long-video behavior. Suggested prompt-side payload sizes:

- 4 MB per prompt
- 16 MB per prompt
- 64 MB per prompt

Metrics:

- request expansion time
- total payload size
- transfer time
- duplication factor

This benchmark should answer whether multimodal deduplication is a larger win than transfer protocol optimization for the target workload.

### Benchmark C: RLVR-VLM path benchmark

Purpose:

- simulate the critical rollout path without requiring a full inference engine

Simulated stages:

1. collator output
2. request expansion
3. response postprocess
4. reward union
5. scheduler-side concat

Compare:

1. legacy
2. optimized protocol only
3. multimodal dedup only
4. multimodal dedup + optimized protocol
5. multimodal dedup + optimized protocol + optimized Ray backend
6. future: Mooncake backend

Metrics:

- end-to-end latency
- bytes transferred
- peak memory
- stage breakdown

---

## Correctness Tests

### Unit tests

Add tests for:

1. tensor-only `DataProto` round-trip
2. string-heavy `non_tensor_batch` round-trip
3. VLM-style multimodal reference round-trip
4. post-generate stripped batch round-trip
5. expanded 2048-sample batch round-trip

### Semantic tests

Add tests to verify:

1. multiple samples can share the same multimodal reference
2. post-generate stripping does not remove training-required fields
3. reward/scheduler paths do not depend on raw prompt-side multimodal payload once generation is complete

---

## Configuration Plan

Add transfer-related configuration options:

- `rollout_transfer_backend: legacy | ray_optimized | mooncake`
- `rollout_transfer_protocol: legacy | v1`
- `rollout_transfer_enable_string_codec: true | false`
- `rollout_transfer_trim_stage: producer | consumer`
- `rollout_transfer_debug_validate: true | false`
- `rollout_transfer_enable_mm_dedup: true | false`
- `rollout_transfer_enable_mm_strip: true | false`
- `rollout_transfer_metrics_enabled: true | false`

Canonical stage names:

- `generate_request`
- `post_generate`
- `train_batch`

This allows gradual rollout, A/B comparisons, and reuse of the same stage vocabulary across semantic trimming, protocol encoding, and backend integration.

---

## Recommended Implementation Order

### Phase 1: highest-value semantic optimization

1. prevent multimodal duplication during request expansion
2. introduce prompt-level multimodal reference (`mm_ref_id`)
3. strip multimodal generation-only fields after generation
4. add long-video multimodal benchmark

### Phase 2: optimized `DataProto` transfer protocol

1. add `to_transfer_payload(stage=...)`
2. add `from_transfer_payload(...)`
3. add tensor codec
4. add string codec
5. add compact mask transfer representation

### Phase 3: optimized Ray backend

1. add `ray_optimized` transfer backend
2. wire the heavy rollout boundaries to use it
3. benchmark legacy vs optimized Ray

### Phase 4: Mooncake backend

1. reuse the same `TransferPayload`
2. implement single-key bulk transfer
3. add cross-node benchmark

---

## Risk Management

### Risk 1: tensor view lifetime

If tensors are reconstructed as views over a shared bulk buffer, the buffer lifetime must outlive the tensor usage.

Mitigation:

- first implementation may keep the bulk buffer strongly referenced by the reconstructed object
- or conservatively copy selected buffers first, then optimize further later

### Risk 2: schema drift

If stage-aware trimming is implemented ad hoc, fields may accidentally disappear from training batches.

Mitigation:

- define explicit stage schemas
- add validation tests

### Risk 3: premature full-stack migration

Changing protocol, backend, router, reward, and scheduler all at once will make debugging difficult.

Mitigation:

- first land the semantic VLM optimizations and protocol support
- then incrementally move heavy transfer boundaries

---

## Fine-grained Profiling Task After Development

After the implementation is complete, add a fine-grained profiling task for follow-up performance tuning.

### Objective

Measure where time and memory are spent in the optimized rollout path under the target RLVR-VLM long-video workload.

### Required profiling dimensions

At minimum, capture per-stage timing for:

1. request expansion
2. multimodal reference resolution
3. multimodal stripping
4. `DataProto.to_transfer_payload`
5. transfer backend `put`
6. transfer backend `get`
7. `DataProto.from_transfer_payload`
8. scheduler-side concat / materialization
9. reward-side union if the path is migrated later

### Required payload metrics

For each profiled transfer, record:

- total payload bytes
- tensor bytes
- string bytes
- fallback object bytes
- multimodal bytes before stripping
- multimodal bytes after stripping
- sample count
- sequence length summary

### Required memory profiling

Record:

- peak RSS
- number of temporary buffers allocated
- number of duplicated multimodal payload objects if applicable

### Suggested implementation

Add lightweight profiling hooks or timers in the transfer path and emit structured metrics under a dedicated namespace, for example:

- `transfer/time/serialize`
- `transfer/time/put`
- `transfer/time/get`
- `transfer/time/deserialize`
- `transfer/bytes/total`
- `transfer/bytes/tensor`
- `transfer/bytes/string`
- `transfer/bytes/object`
- `transfer/bytes/multimodal_before_strip`
- `transfer/bytes/multimodal_after_strip`

### Deliverable

A dedicated profiling mode or benchmark script that can be run after development and used later for iterative tuning.

This profiling task is part of the implementation plan and should be treated as a required follow-up task, not optional cleanup.

---

## Summary

For the target RLVR-VLM long-video workload, simply porting the serialization/backend pattern from `THUDM/slime#1709` is not enough.

The correct priority for ROLL is:

1. remove multimodal duplication
2. strip multimodal generation-only payload after generation
3. add an optimized stage-aware `DataProto` transfer protocol
4. add an optimized Ray backend first
5. add a Mooncake backend later
6. add fine-grained profiling after implementation for iterative tuning

This phased plan keeps the migration low-risk while targeting the largest performance wins for the real workload.
