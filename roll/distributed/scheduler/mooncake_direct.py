"""Materialize Mooncake column rows into consumer-process tensor buffers."""

import math

import torch


def read_direct_columns(backend, keys, sources, row_plan, *, device):
    """Execute row routing with metadata only; DMA writes the final tensor.

    ``sources`` are column reference dictionaries. ``row_plan`` contains
    (source index, source row) pairs, after existing RemoteBatch plan operators.
    Registration is borrowed only by synchronous, physically drained reads.
    The returned PyTorch tensors retain their allocations for replay consumers.
    """
    from mooncake.structured_object_store import tensor_destination

    output = {}
    registrations = []
    failure = None
    try:
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
            status = backend.store.register_buffer(target.data_ptr(), nbytes)
            if status != 0:
                raise RuntimeError(f"Mooncake direct destination registration failed: {status}")
            registrations.append(target)
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
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_error = None
        for target in reversed(registrations):
            try:
                status = backend.store.unregister_buffer(target.data_ptr())
                if status != 0:
                    raise RuntimeError(f"Mooncake direct destination deregistration failed: {status}")
            except Exception as exc:
                # Keep the registered allocation reachable for an explicit retry.
                backend._roll_pending_direct_registrations.append(target)
                cleanup_error = cleanup_error or exc
                if failure is not None:
                    failure.add_note(str(exc))
        if failure is None and cleanup_error is not None:
            raise cleanup_error
    return output
