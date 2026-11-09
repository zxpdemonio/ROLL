"""Materialize Mooncake column rows into consumer-process tensor buffers."""

import math
from contextlib import contextmanager
from typing import NamedTuple

import torch


class _Registration(NamedTuple):
    pointer: int
    size: int
    tensor: torch.Tensor


def add_cleanup_note(error, detail):
    # Python 3.10 has no BaseException.add_note.
    note = getattr(error, "add_note", None)
    if callable(note):
        note(str(detail))


def retry_direct_registrations(backend):
    pending = getattr(backend, "_roll_pending_direct_registrations", [])
    backend._roll_pending_direct_registrations = []
    for entry in pending:
        try:
            status = backend.store.unregister_buffer(entry.pointer)
            if status != 0:
                raise RuntimeError(f"Mooncake pending deregistration failed: {status}")
        except Exception:
            backend._roll_pending_direct_registrations.append(entry)
    if backend._roll_pending_direct_registrations:
        raise RuntimeError("Mooncake has pending direct buffer deregistrations")


@contextmanager
def registered_direct_tensors(backend):
    """Own registrations through physically drained I/O, then retry failures.

    Sources must not be modified/resized during I/O. Save the registered
    address separately from mutable tensor metadata for cleanup and retry.
    """
    retry_direct_registrations(backend)
    registrations = []
    failure = None

    def register(tensor):
        entry = _Registration(tensor.data_ptr(), tensor.nbytes, tensor)
        if not entry.size:
            return
        status = backend.store.register_buffer(entry.pointer, entry.size)
        if status != 0:
            raise RuntimeError(f"Mooncake direct buffer registration failed: {status}")
        registrations.append(entry)

    try:
        yield register
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_error = None
        for entry in reversed(registrations):
            try:
                status = backend.store.unregister_buffer(entry.pointer)
                if status != 0:
                    raise RuntimeError(f"Mooncake direct buffer deregistration failed: {status}")
            except Exception as exc:
                backend._roll_pending_direct_registrations.append(entry)
                cleanup_error = cleanup_error or exc
                if failure is not None:
                    add_cleanup_note(failure, exc)
        if failure is None and cleanup_error is not None:
            raise cleanup_error


def read_direct_columns(backend, keys, sources, row_plan, *, device):
    """Execute row routing with metadata only; DMA writes the final tensor.

    ``sources`` are column reference dictionaries. ``row_plan`` contains
    (source index, source row) pairs, after existing RemoteBatch plan operators.
    Registration is borrowed only by synchronous, physically drained reads.
    The returned PyTorch tensors retain their allocations for replay consumers.
    """
    from mooncake.structured_object_store import tensor_destination

    output = {}
    with registered_direct_tensors(backend) as register:
        for key in keys:
            used = {source for source, _ in row_plan}
            if not used:
                used = {
                    next(
                        i
                        for i, source in enumerate(sources)
                        if source[key].get("schema", {}).get("direct_alignment") is not None
                    )
                }
            spec = sources[min(used)][key]
            dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
            alignments = [sources[index][key].get("schema", {}).get("direct_alignment") for index in used]
            if any(
                isinstance(alignment, bool)
                or not isinstance(alignment, int)
                or alignment <= 0
                or alignment & (alignment - 1)
                for alignment in alignments
            ):
                raise ValueError("direct sources require schemas with a positive power-of-two direct_alignment")
            alignment = max(alignments)
            if not isinstance(dtype, torch.dtype) or torch.device(device).type not in {"cpu", "cuda"}:
                raise ValueError("direct destination requires a CPU/CUDA tensor dtype")
            if not spec["shape"] or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in spec["shape"]
            ):
                raise ValueError("direct tensor shape must contain non-negative dimensions")
            shape = (len(row_plan), *spec["shape"][1:])
            nbytes = math.prod(shape) * torch.empty([], dtype=dtype).element_size()
            backing = torch.empty(nbytes + alignment - 1, dtype=torch.uint8, device=device)
            offset = -backing.data_ptr() % alignment
            target = backing[offset : offset + nbytes].view(dtype).view(shape)
            output[key] = target
            if not nbytes:
                continue
            register(target)
            if any(sources[index][key].get("kind") == "token_sequence" for index in used):
                from roll.distributed.scheduler.mooncake_sequence import read_sequence_rows

                for output_row, (source_index, source_row) in enumerate(row_plan):
                    field = sources[source_index][key]
                    if field["shape"][1:] != spec["shape"][1:] or field["dtype"] != spec["dtype"]:
                        raise ValueError("Mooncake token-sequence layouts differ")
                    if field.get("kind") == "token_sequence":
                        read_sequence_rows(backend, field, source_row, target[output_row], target)
                    else:
                        destination = target[output_row : output_row + 1]
                        result = backend.get(
                            field["ref"], type="dataproto", batch_fields=[key], non_tensor_fields=[],
                            meta_info_keys=[], data_cls=dict, rows=[source_row],
                            destinations={key: tensor_destination(destination, alignment=alignment,
                                                                  pre_registered=True, allocation=target)},
                        )
                        if result["batch"][key] is not destination:
                            raise RuntimeError("Mooncake replaced the consumer destination")
                continue
            start = 0
            while start < len(row_plan):
                source_index = row_plan[start][0]
                stop = start + 1
                while stop < len(row_plan) and row_plan[stop][0] == source_index:
                    stop += 1
                field = sources[source_index][key]
                if field["shape"][1:] != spec["shape"][1:] or field["dtype"] != spec["dtype"]:
                    raise ValueError("Mooncake direct sources have incompatible tensor schemas")
                destination = target[start:stop]
                result = backend.get(
                    field["ref"],
                    type="dataproto",
                    batch_fields=[key],
                    non_tensor_fields=[],
                    meta_info_keys=[],
                    data_cls=dict,
                    rows=[row for _, row in row_plan[start:stop]],
                    destinations={
                        # The allocation base satisfies the consumer alignment.
                        # A run can begin at an interior row offset; it only
                        # needs the dtype's natural alignment for the DMA write.
                        key: tensor_destination(
                            destination, alignment=alignment, pre_registered=True, allocation=target
                        )
                    },
                )
                if result["batch"][key] is not destination:
                    raise RuntimeError("Mooncake direct read replaced the consumer destination")
                start = stop
    return output
