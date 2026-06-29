# Mooncake DataProto Rollout Transfer

ROLL can use Mooncake as a transfer backend for large rollout `DataProto` objects. The goal is to keep Ray on the control path while Mooncake carries the heavy rollout payload.

## When to use it

Use Mooncake transfer when rollout batches are large enough that sending the full `DataProto` through Ray object serialization becomes expensive, especially for VLM or multimodal workloads where most bytes live in `non_tensor_batch`.

The Mooncake path preserves the normal ROLL `DataProto` structure:

- `batch`
- `non_tensor_batch`
- `meta_info`

Workers should continue to consume normal materialized `DataProto` fields. Mooncake is a backend transport detail, not a new business object model for reward or training code.

## How it works

At rollout handoff time, ROLL stores the heavy payload in Mooncake and passes lightweight remote references through Ray. Downstream workers materialize the fields and rows they need through the transfer backend.

The adapter uses Mooncake structured-object APIs:

| ROLL operation | Mooncake API |
|---|---|
| Store rollout payload | `put_dataproto` |
| Append derived fields | `append_dataproto_fields` |
| Read selected fields/rows | `get_dataproto` |
| Pass references through Ray | `export_dataproto_ref` / `import_dataproto_ref` |
| Cleanup owned remote payload | `cleanup_dataproto` |

ROLL does not construct Mooncake `BufferPool` objects. Buffer registration, reusable staging buffers, fallback copy behavior, and multi-buffer transfer policy are Mooncake-side responsibilities.

## Configuration

Enable the backend through `transfer_backend`:

```yaml
transfer_backend:
  backend_name: Mooncake
  backend_config:
    client_scope: node
    protocol: rdma
    device_name: erdma_1
    local_hostname: 192.168.22.70
    metadata_server: P2PHANDSHAKE
    master_server_address: 192.168.22.70:50053
    global_segment_size: 34359738368
    local_buffer_size: 42949672960
    default_chunk_bytes: 536870912
    namespace: roll_rl
    key_prefix: roll_rl
    policy:
      max_inflight_put: 1
      put_mode: auto
      copy_mode: auto
```

Important fields:

- `master_server_address` / `master_server_addr`: address of the already-running Mooncake master.
- `local_hostname`: local host address visible to Mooncake.
- `metadata_server`: Mooncake metadata mode, for example `P2PHANDSHAKE`.
- `protocol`: use `rdma` for performance validation.
- `device_name` / `rdma_devices`: required when `protocol: rdma`.
- `client_scope`: `node` creates one Mooncake client actor per Ray node; `process` creates a client in each process.
- `require_native_tensors`: validates that tensor fields use Mooncake native tensor payloads when possible.

When explicit master config is provided, ROLL passes the backend config directly to `MooncakeDistributedStore.setup`. Environment-based Mooncake config remains a fallback only when no explicit master address is configured.

## Deployment notes

Start Mooncake master before launching the ROLL job. Do not rely on ad-hoc Ray actor environment variables for production configuration; put the store parameters in `transfer_backend.backend_config`.

For RDMA performance runs, ensure:

- Mooncake master and client versions match.
- The configured RDMA device exists on the node.
- `local_buffer_size` is large enough for the measured payload to avoid fallback registration noise.
- Benchmarks report setup/registration cost separately from online put/get time when possible.

## Validation checklist

A Mooncake rollout transfer run should confirm:

1. Ray initializes normally.
2. The Mooncake transfer backend initializes from explicit `backend_config`.
3. Scheduler `DataProto.to_remote` succeeds.
4. `put_dataproto` stores `batch`, `non_tensor_batch`, and metadata.
5. Downstream workers materialize required fields through `get_dataproto`.
6. Derived fields can be appended without rewriting the original rollout payload.
7. Owning remote batches clean up Mooncake payloads exactly once.
