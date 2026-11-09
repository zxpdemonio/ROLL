"""ROLL-owned GPU export for vLLM's token-aligned auxiliary connector.

The original connector still owns acceptance, cursors, block hashes and tails.
Only payload operations and delivery are replaced for explicit direct schemas.
"""

import threading
import uuid
from collections import OrderedDict

import torch

from roll.distributed.scheduler.mooncake_sequence import (
    MooncakeRows,
    WorkerAuxRows,
    decode_rows_carrier,
)
from roll.third_party.vllm.aux_output_gpu import (
    GPUArrayOps,
    GPURows,
    current_writer,
    gpu_allocation,
)

CONFIG_KEY = "roll_mooncake_aux_output"
_installed = False


class _Writer:
    def __init__(self, config, vllm_config):
        from roll.distributed.scheduler.transfer_backend import (
            MooncakeClient,
            client_only_mooncake_config,
        )
        from roll.utils.execution_artifacts import AttentionSelectionProfile

        backend = client_only_mooncake_config(config["backend_config"])
        backend["transfer_policy"] = {**(backend.get("transfer_policy") or {}), "copy_mode": "auto"}
        self.client = MooncakeClient(backend)
        self.profile = (
            AttentionSelectionProfile(**config["attention_profile"]) if config.get("attention_profile") else None
        )
        self.lock = threading.RLock()
        self.exports = {}
        self.max_pending_batches = vllm_config.max_concurrent_batches
        self.max_step_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_request_tokens = vllm_config.model_config.max_model_len

    def validate_workspace(self, pool_bytes, row_bytes):
        # Bound the existing GPU publication queue, snapshots, combined rows,
        # and one prefix materialization/export in addition to the tail pool.
        transient = row_bytes * (
            (2 * self.max_pending_batches + 3) * self.max_step_tokens + 3 * self.max_request_tokens
        )
        free, _ = torch.cuda.mem_get_info()
        available = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if pool_bytes + transient > available:
            raise ValueError(
                f"GPU auxiliary workspace needs {pool_bytes + transient} bytes "
                f"(tail pool {pool_bytes}, transient reserve {transient}), available {available}. "
                "Reduce max_num_seqs/max_num_batched_tokens/max_model_len."
            )

    def put(self, fields, partition):
        size = len(next(iter(fields.values())))
        try:
            return self.client.put(partition, [uuid.uuid4().hex for _ in range(size)], fields, size)
        except BaseException as error:
            from roll.distributed.scheduler.mooncake_direct import add_cleanup_note

            # Registration may fail after the client's asynchronous private
            # clone, before Mooncake reaches its normal source stream fence.
            # Drain those reads before vLLM releases a snapshot in finally.
            for device in {tensor.device for tensor in fields.values() if tensor.device.type == "cuda"}:
                try:
                    torch.cuda.current_stream(device).synchronize()
                except Exception as fence_error:
                    add_cleanup_note(error, fence_error)
            raise

    def remove(self, batch):
        self.client.delete(batch.partition, list(batch.fields), list(batch.fields.values()))

    def export(self, output):
        fields = {}
        if output.routed_rows is not None:
            fields["routed_experts"] = output.routed_rows.tensor
        if output.indexer_rows is not None:
            if self.profile is None:
                raise ValueError("GPU indexer export requires the ROLL attention profile")
            self.profile.validate_vllm_profile(output.indexer_profile)
            selection = output.indexer_rows.tensor
            rows = len(selection)
            if self.profile.storage_row_width == self.profile.row_width:
                selection = selection.reshape(rows, self.profile.num_layers, self.profile.topk)
            else:
                padded = torch.full(
                    (rows, self.profile.storage_row_width),
                    self.profile.padding_value,
                    dtype=torch.int32,
                    device=selection.device,
                )
                padded[:, : self.profile.row_width].copy_(selection)
                selection = padded
            fields["attention_selection"] = selection
        for key in fields:
            schema = self.client.field_schemas.get(key)
            if schema is None or schema.direct_alignment is None:
                raise ValueError(f"GPU auxiliary export requires a direct schema for {key}")
        with self.lock:
            batch = self.put(fields, "train_eval")
            identity = batch._roll_aux_delivery_id = uuid.uuid4().hex
            self.exports[identity] = [batch, None]
        for name, key in (("routed_rows", "routed_experts"), ("indexer_rows", "attention_selection")):
            if key in fields:
                tensor = fields[key]
                setattr(
                    output,
                    name,
                    WorkerAuxRows(
                        [(batch, key, 0, len(tensor))], tuple(tensor.shape), str(tensor.dtype).removeprefix("torch.")
                    ),
                )
        return output

    def settle(self, *, claims=(), request_ids=(), discard=False):
        """Acknowledge frontend ownership, or delete undelivered outputs."""
        with self.lock:
            requests = {self.exports[key][1] for key in claims if key in self.exports}
            if discard:
                request_ids = set(request_ids) | (requests - {None})
            if claims and not discard:
                accepted = set(claims)
                self._remove_exports(
                    [
                        key
                        for key, (_, request) in self.exports.items()
                        if request is not None and request in requests and key not in accepted
                    ]
                )
            selected = [
                key for key, (_, request_id) in self.exports.items() if key in claims or request_id in request_ids
            ]
            if discard:
                self._remove_exports(selected)
            else:
                for key in selected:
                    del self.exports[key]

    def _remove_exports(self, keys):
        first = None
        for key in keys:
            try:
                self.remove(self.exports[key][0])
                del self.exports[key]
            except Exception as error:
                first = first or error
        if first is not None:
            raise first


class _Arena:
    """Keep the original fixed-slot LRU; slot payloads live in Mooncake."""

    def __init__(self, writer, object_nbytes):
        self.writer, self.object_nbytes, self.slots = writer, object_nbytes, {}

    def __setitem__(self, span, payload):
        if not isinstance(payload, GPURows) or span.stop - span.start != self.object_nbytes:
            raise TypeError("GPU auxiliary store requires a fixed-size owned GPU block")
        if payload.ready is not None:
            payload.ready.synchronize()
        slot = span.start // self.object_nbytes
        with self.writer.lock:
            previous = self.slots.get(slot)
            if previous is not None:
                # Client deletion retains failed refs in its existing retry queue.
                self.writer.remove(previous)
                del self.slots[slot]
            batch = self.writer.put(
                {"_roll_aux_bytes": payload.tensor.view(torch.uint8).reshape(1, -1)}, "vllm_aux_prefix"
            )
            self.slots[slot] = batch

    def read(self, slots):
        from roll.distributed.scheduler.mooncake_direct import read_direct_columns

        with self.writer.lock:
            sources = [{"_roll_aux_bytes": self.slots[slot].fields["_roll_aux_bytes"]} for slot in slots]
            result = read_direct_columns(
                self.writer.client.backend,
                ["_roll_aux_bytes"],
                sources,
                [(index, 0) for index in range(len(sources))],
                device="cuda",
            )
            return GPURows(result["_roll_aux_bytes"].reshape(-1))

    def close(self):
        with self.writer.lock:
            first = None
            for slot, batch in list(self.slots.items()):
                try:
                    self.writer.remove(batch)
                    del self.slots[slot]
                except Exception as exc:
                    first = first or exc
            if first is not None:
                raise first


def _block_store_type(original):
    class GPUBlockObjectStore(original):
        def __init__(self, *, max_bytes, object_nbytes):
            if object_nbytes <= 0 or max_bytes < object_nbytes:
                raise ValueError("Auxiliary prefix store must fit a positive fixed-size block")
            # Same metadata used by the inherited LRU/ref algorithms. Avoid
            # reserving the original host mmap for a remote payload arena.
            self.object_nbytes = object_nbytes
            self.num_slots = max_bytes // object_nbytes
            self._slots = {}
            self._lru = OrderedDict()
            self._references = {}
            self._free_slots = []
            self._next_slot = 0
            self._arena = _Arena(current_writer(), object_nbytes)

        def close(self):
            if self._arena is not None:
                self._arena.close()
                self._arena = None

        def get_concatenated(self, keys):
            from vllm.distributed.aux_output_connector.store import (
                BlockObjectStoreError,
            )

            try:
                slots = [self._slots[key] for key in keys]
            except KeyError as exc:
                raise BlockObjectStoreError("Auxiliary prefix block was evicted or never published") from exc
            result = self._arena.read(slots)
            for key in keys:
                if key in self._lru:
                    self._lru.move_to_end(key)
            return result

    return GPUBlockObjectStore


def install_aux_output_mooncake(*, required=False):
    """Install in frontend and worker processes; defaults remain NumPy/SHM."""
    global _installed
    if _installed:
        return True
    try:
        from vllm.distributed.aux_output_connector import artifacts, worker
        from vllm.v1.engine import output_processor
    except ModuleNotFoundError as exc:
        if not required and exc.name.startswith("vllm.distributed.aux_output_connector"):
            return False
        raise RuntimeError("This vLLM build lacks the token-aligned auxiliary connector") from exc
    if not hasattr(worker.PendingAuxOutput, "enqueue_cpu_copy"):
        if required:
            raise RuntimeError("Unsupported vLLM auxiliary connector API")
        return False

    init = worker.AuxOutputWorkerConnector.__init__
    enqueue = worker.PendingAuxOutput.enqueue_cpu_copy
    process = worker.AuxOutputWorkerConnector.process_output
    make_output = worker.AuxOutputWorkerConnector._make_output
    close = worker.AuxOutputWorkerConnector.close
    store = worker.BlockObjectStore
    background_store = worker.BackgroundBlockObjectStore
    gpu_store = _block_store_type(store)
    validate_host_budget = worker.validate_shm_budget

    def store_factory(*args, **kwargs):
        return (gpu_store if current_writer() is not None else store)(*args, **kwargs)

    def background_factory(store, *, max_pending_batches):
        writer = current_writer()
        if writer is not None:
            max_pending_batches = min(max_pending_batches, writer.max_pending_batches)
        return background_store(store, max_pending_batches=max_pending_batches)

    def prefix_budget(**kwargs):
        if current_writer() is None:
            return validate_host_budget(**kwargs)
        required = kwargs["num_blocks"] * kwargs["scheduler_block_size"] * kwargs["bytes_per_token"]
        block_bytes = kwargs["hash_block_size"] * kwargs["bytes_per_token"]
        capacity = kwargs["max_bytes"] if kwargs["max_bytes"] is not None else required
        capacity = capacity // block_bytes * block_bytes
        if capacity < required:
            raise ValueError(f"Mooncake auxiliary prefix capacity {capacity} bytes cannot cover KV needs {required}")
        # The arena is remote; the actual local GPU workspace is checked when
        # AuxOutputBuffer allocates. Do not charge nonexistent SHM/staging here.
        return capacity

    def initialize(self, *, vllm_config, **kwargs):
        config = (vllm_config.additional_config or {}).get(CONFIG_KEY)
        self._roll_writer = None
        if config is not None and worker.get_tp_group().is_first_rank:
            config = {**config, "backend_config": {**config["backend_config"]}}
            config["backend_config"]["field_schemas"] = {
                **(config["backend_config"].get("field_schemas") or {}),
                "_roll_aux_bytes": {"codec": "torch_tensor", "direct_alignment": 1},
            }
            self._roll_writer = _Writer(config, vllm_config)
        with gpu_allocation(self._roll_writer):
            init(self, vllm_config=vllm_config, **kwargs)
        if self._roll_writer is not None:
            # process_output uses this same lock, including deferred teardown.
            self._lock = threading.RLock()

    def enqueue_rows(self, num_sampled, num_rejected):
        if self.connector._roll_writer is None:
            return enqueue(self, num_sampled, num_rejected)
        self.num_sampled, self.num_rejected = num_sampled, num_rejected
        for gpu, host in (("routed_experts_gpu", "routed_experts"), ("indexer_gpu", "indexer")):
            tensor = getattr(self, gpu)
            if tensor is not None:
                if not tensor.is_cuda:
                    raise TypeError("GPU auxiliary export requires CUDA snapshots")
                setattr(self, host, GPURows(tensor))
        tensor = self.indexer_gpu if self.indexer_gpu is not None else self.routed_experts_gpu
        self._roll_ready = torch.cuda.Event()
        self._roll_ready.record(torch.cuda.current_stream(tensor.device))

    def commit(self, pending):
        writer = self._roll_writer
        if writer is None:
            return process(self, pending)
        pending._roll_ready.synchronize()
        with self._lock:
            before = set(writer.exports)
            discard = {rid for rid in pending.request_ids if getattr(self._requests[rid], "_roll_discard", False)}
            try:
                outputs = process(self, pending)
                for rid, output in outputs.items():
                    for rows in (output.routed_rows, output.indexer_rows):
                        if isinstance(rows, MooncakeRows):
                            for identity in rows.delivery_ids:
                                writer.exports[identity][1] = rid
                writer.settle(request_ids=discard, discard=True)
                return outputs
            except BaseException as error:
                from roll.distributed.scheduler.mooncake_direct import add_cleanup_note

                try:
                    writer.settle(claims=set(writer.exports) - before, discard=True)
                except Exception as cleanup_error:
                    add_cleanup_note(error, cleanup_error)
                raise

    def export(self, token_start, rows):
        output = make_output(self, token_start, rows)
        return self._roll_writer.export(output) if self._roll_writer is not None else output

    def shutdown(self):
        failure = None
        try:
            return close(self)
        except BaseException as error:
            failure = error
            raise
        finally:
            if self._roll_writer is not None:
                try:
                    self._roll_writer.settle(claims=list(self._roll_writer.exports), discard=True)
                except Exception as cleanup_error:
                    if failure is None:
                        raise
                    from roll.distributed.scheduler.mooncake_direct import (
                        add_cleanup_note,
                    )

                    add_cleanup_note(failure, cleanup_error)

    worker.np = artifacts.np = GPUArrayOps()
    worker.BlockObjectStore = store_factory
    worker.BackgroundBlockObjectStore = background_factory
    worker.validate_shm_budget = prefix_budget
    worker.AuxOutputWorkerConnector.__init__ = initialize
    worker.PendingAuxOutput.enqueue_cpu_copy = enqueue_rows
    worker.AuxOutputWorkerConnector.process_output = commit
    worker.AuxOutputWorkerConnector._make_output = export
    worker.AuxOutputWorkerConnector.close = shutdown

    original_process = output_processor.OutputProcessor.process_outputs
    original_completion = output_processor.RequestState._new_completion_output

    def process_frontend(self, outputs, *args, **kwargs):
        for output in outputs:
            for field in ("routed_experts", "indexer_topk"):
                setattr(output, field, decode_rows_carrier(getattr(output, field)))
        return original_process(self, outputs, *args, **kwargs)

    def completion(self, *args, **kwargs):
        saved = {}
        for name in ("routed_experts_chunks", "indexer_chunks"):
            values = getattr(self, name)
            if values and isinstance(values[0], MooncakeRows):
                saved[name] = values
                setattr(self, name, [])
        try:
            output = original_completion(self, *args, **kwargs)
        finally:
            for name, values in saved.items():
                setattr(self, name, values)
        if output.finish_reason is not None:
            for name, field in (("routed_experts_chunks", "routed_experts"), ("indexer_chunks", "indexer_topk")):
                if name in saved:
                    setattr(output, field, MooncakeRows.concatenate(saved[name]))
        return output

    output_processor.OutputProcessor.process_outputs = process_frontend
    output_processor.RequestState._new_completion_output = completion
    _installed = True
    return True


def settle_worker_aux_outputs(worker, *, claims=(), request_ids=(), discard=False):
    connector = getattr(worker.model_runner, "aux_output_connector", None)
    writer = getattr(connector, "_roll_writer", None)
    if writer is None:
        return
    with connector._lock:
        if discard:
            for request_id in request_ids:
                state = connector._requests.get(request_id)
                if state is not None:
                    state._roll_discard = True
        writer.settle(claims=claims, request_ids=request_ids, discard=discard)


def drain_worker_aux_outputs(worker):
    """Use the existing sleep allocator only after publication/DMA is drained."""
    from roll.distributed.scheduler.mooncake_direct import retry_direct_registrations

    connector = getattr(worker.model_runner, "aux_output_connector", None)
    writer = getattr(connector, "_roll_writer", None)
    if writer is None:
        return
    with connector._lock:
        if connector._pending_outputs:
            raise RuntimeError("Cannot offload with unconsumed auxiliary snapshots")
        # Background writer needs writer.lock; do not hold it while joining.
        connector._store._queue.join()
        connector._store._raise_if_failed()
        with writer.lock:
            if writer.exports:
                raise RuntimeError("Cannot offload before auxiliary delivery ownership is settled")
            writer.client._retry_pending_put_cleanups()
            retry_direct_registrations(writer.client.backend)
