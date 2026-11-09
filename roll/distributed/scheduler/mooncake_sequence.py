"""Token-range references carried from vLLM to trajectory-shaped RemoteBatch."""

import json
import math
import uuid
import weakref

import numpy as np

_MAGIC = b"ROLL_MOONCAKE_AUX_V1\0"
AUX_OWNER_KEY = "_roll_aux_remote_owner"
_owners = weakref.WeakValueDictionary()


class MooncakeRows:
    """A token-axis view. Slicing and concatenation move metadata only."""

    def __init__(self, parts, shape, dtype):
        self.parts = tuple(parts)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        if not self.parts:
            raise ValueError("Mooncake rows require a source owner, including empty token views")
        if not self.shape or any(type(n) is not int or n < 0 for n in self.shape):
            raise ValueError("Mooncake rows require a non-negative token shape")
        count = 0
        for owner, key, start, stop in self.parts:
            field = owner.fields[key]
            if not 0 <= start <= stop <= field["shape"][0]:
                raise ValueError("Mooncake token range exceeds its source")
            if math.prod(field["shape"][1:]) != math.prod(self.shape[1:]):
                raise ValueError("Mooncake token width changed")
            if np.dtype(field["dtype"].removeprefix("torch.")) != self.dtype:
                raise TypeError("Mooncake token dtype changed")
            count += stop - start
        if count != len(self):
            raise ValueError("Mooncake token ranges do not cover the declared rows")

    def __len__(self):
        return self.shape[0]

    def __reduce__(self):
        identities = [(owner, owner._roll_aux_delivery_id) for owner, _, _, _ in self.parts]
        return (_restore_rows, (type(self), self.parts, self.shape, self.dtype.str, identities))

    def __array__(self, *args, **kwargs):
        raise TypeError("MooncakeRows contains references; materialize through RemoteBatch")

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("MooncakeRows supports token slices only")
        start, stop, step = item.indices(len(self))
        if step != 1:
            raise ValueError("Mooncake token slices require step=1")
        stop = max(start, stop)
        parts, offset = [], 0
        for owner, key, first, last in self.parts:
            lo, hi = max(start - offset, 0), min(stop - offset, last - first)
            if lo < hi:
                parts.append((owner, key, first + lo, first + hi))
            else:
                # Retain ownership of fully trimmed deliveries for ack/drop.
                parts.append((owner, key, first, first))
            offset += last - first
        return type(self)(parts, (stop - start, *self.shape[1:]), self.dtype)

    @classmethod
    def concatenate(cls, rows):
        if not rows or not all(isinstance(row, cls) for row in rows):
            raise TypeError("Cannot mix Mooncake references and local artifact arrays")
        first = rows[0]
        if any(row.shape[1:] != first.shape[1:] or row.dtype != first.dtype for row in rows):
            raise ValueError("Mooncake artifact profiles differ between chunks")
        return cls(
            [part for row in rows for part in row.parts],
            (sum(map(len, rows)), *first.shape[1:]),
            first.dtype,
        )

    @property
    def delivery_ids(self):
        return {owner._roll_aux_delivery_id for owner, _, _, _ in self.parts}

    def to_carrier(self):
        """Use the existing ndarray IPC codec for a small JSON reference packet."""
        from mooncake.structured_object_store import export_dataproto_ref

        owners, parts = {}, []
        for owner, key, start, stop in self.parts:
            identity = owner._roll_aux_delivery_id
            if identity not in owners:
                owners[identity] = {
                    "partition": owner.partition,
                    "batch_size": len(owner),
                    "fields": {
                        name: {**field, "ref": export_dataproto_ref(field["ref"])}
                        for name, field in owner.fields.items()
                    },
                }
            parts.append([identity, key, start, stop])
        metadata = {"owners": owners, "parts": parts, "shape": self.shape, "dtype": self.dtype.str}
        return np.frombuffer(_MAGIC + json.dumps(metadata, separators=(",", ":")).encode(), dtype=np.uint8)


class WorkerAuxRows(MooncakeRows):
    """The existing scheduler's accepted-token slice emits a normal ndarray."""

    def __getitem__(self, item):
        return super().__getitem__(item).to_carrier()


def _restore_rows(cls, parts, shape, dtype, identities):
    for owner, identity in identities:
        owner._roll_aux_delivery_id = identity
    return cls(parts, shape, dtype)


def decode_rows_carrier(value):
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != np.uint8:
        return value
    if value.size < len(_MAGIC) or value[: len(_MAGIC)].tobytes() != _MAGIC:
        return value
    from mooncake.structured_object_store import import_dataproto_ref

    from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch

    metadata = json.loads(value[len(_MAGIC) :].tobytes())
    decoded = {}
    for identity, spec in metadata["owners"].items():
        owner = _owners.get(identity)
        if owner is None:
            owner = ColumnRemoteBatch(
                partition=spec["partition"],
                device=None,
                cache=None,
                fields={
                    key: {**field, "ref": import_dataproto_ref(field["ref"])} for key, field in spec["fields"].items()
                },
                batch_size=spec["batch_size"],
            )
            owner._roll_aux_delivery_id = identity
            owner._roll_aux_wire_spec = spec
            _owners[identity] = owner
        elif owner._roll_aux_wire_spec != spec:
            raise ValueError("Mooncake delivery identity was reused for another reference")
        decoded[identity] = owner
    return MooncakeRows(
        [(decoded[identity], key, start, stop) for identity, key, start, stop in metadata["parts"]],
        metadata["shape"],
        metadata["dtype"],
    )


def rows_owner_batch(values, batch_size):
    """Own response refs before token postprocessing has produced field layouts."""
    from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch

    owners = {
        id(owner): owner for value in values if isinstance(value, MooncakeRows) for owner, _, _, _ in value.parts
    }
    if not owners:
        return None
    lifetime = ColumnRemoteBatch._cat(list(owners.values()))._lifetime
    return ColumnRemoteBatch(
        partition=next(iter(owners.values())).partition,
        device=None,
        fields={},
        cache=None,
        batch_size=batch_size,
        lifetime=lifetime,
        row_ids=[uuid.uuid4().hex for _ in range(batch_size)],
    )


def artifact_rows(metadata):
    return [
        value
        for key in ("routed_experts", "attention_selection", "pre_routed_experts", "pre_attention_selection")
        for value in (metadata.get(key) if isinstance(metadata.get(key), (list, tuple)) else ())
        if isinstance(value, MooncakeRows)
    ]


def release_artifact_rows(metadata, *, error=None):
    owners = {id(owner): owner for value in artifact_rows(metadata) for owner, _, _, _ in value.parts}
    first = None
    for owner in owners.values():
        try:
            owner.drop()
        except Exception as cleanup_error:
            if error is not None:
                from roll.distributed.scheduler.mooncake_direct import add_cleanup_note

                add_cleanup_note(error, cleanup_error)
            first = first or cleanup_error
    if error is None and first is not None:
        raise first


def merge_resumed_rows(previous, current):
    """Keep previously executed choices; resumed capture includes the full prefix."""
    if previous is None:
        return current
    if len(previous) != len(current):
        raise ValueError("Resumed artifact sample counts differ")
    result = []
    for old, new in zip(previous, current):
        if not isinstance(old, MooncakeRows) or not isinstance(new, MooncakeRows) or len(new) < len(old):
            raise ValueError("Resumed artifacts must contain a complete remote token prefix")
        result.append(MooncakeRows.concatenate([old, new[len(old) :]]))
    return result


def sequence_remote_batch(fields, valid_counts, sequence_length, profile=None):
    """Reuse token owners while describing a padded trajectory layout."""
    from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch

    owners, descriptors = {}, {}
    batch_size = len(valid_counts)
    for key, values in fields.items():
        if len(values) != batch_size or not all(isinstance(value, MooncakeRows) for value in values):
            raise TypeError("Remote artifact rows must cover every output sample")
        first = values[0]
        if any(value.dtype != first.dtype or value.shape[1:] != first.shape[1:] for value in values):
            raise ValueError("Remote artifact sample profiles differ")
        sequences, alignment = [], 1
        for value, valid in zip(values, valid_counts):
            if len(value) < valid or not 0 <= valid < sequence_length:
                raise ValueError("Remote artifact valid rows exceed their trajectory")
            parts = []
            for owner, source_key, start, stop in value[:valid].parts:
                owners[id(owner)] = owner
                source = owner.fields[source_key]
                required = source.get("schema", {}).get("direct_alignment")
                if type(required) is not int or required <= 0 or required & (required - 1):
                    raise ValueError("Remote GPU artifacts require an explicit direct schema")
                alignment = max(alignment, required)
                parts.append({"source": source, "key": source_key, "start": start, "stop": stop})
            sequences.append(parts)
        is_selection = key == "attention_selection"
        if is_selection:
            if profile is None:
                raise ValueError("Remote selection requires its AttentionSelectionProfile")
            expected = (
                (profile.num_layers, profile.topk)
                if profile.storage_row_width == profile.row_width
                else (profile.storage_row_width,)
            )
            if first.shape[1:] != expected or first.dtype != np.int32:
                raise ValueError("Remote selection does not match its storage profile")
        descriptors[key] = {
            "kind": "token_sequence",
            "shape": [batch_size, sequence_length - int(is_selection), *first.shape[1:]],
            "dtype": "torch." + first.dtype.name,
            "schema": {"codec": "torch_tensor", "direct_alignment": alignment},
            "sequences": sequences,
            "padding": profile.padding_value if is_selection else "distinct_experts",
        }
    if not descriptors:
        return None
    # _cat connects the existing lifetime domains; its token row plan is unused.
    lifetime = ColumnRemoteBatch._cat(list(owners.values()))._lifetime
    return ColumnRemoteBatch(
        partition=next(iter(owners.values())).partition,
        device=None,
        cache=None,
        fields=descriptors,
        batch_size=batch_size,
        lifetime=lifetime,
        row_ids=[uuid.uuid4().hex for _ in range(batch_size)],
    )


def read_sequence_rows(backend, field, source_row, target, allocation):
    """DMA token fragments into one final trajectory; no gather/staging tensor."""
    from mooncake.structured_object_store import tensor_destination

    if field["padding"] == "distinct_experts":
        import torch

        target.copy_(torch.arange(target.shape[-1], device=target.device).to(target.dtype).expand_as(target))
    else:
        target.fill_(field["padding"])
    offset = 0
    for part in field["sequences"][source_row]:
        count = part["stop"] - part["start"]
        if not count:
            continue
        source = part["source"]
        destination = target[offset : offset + count].view(count, *source["shape"][1:])
        result = backend.get(
            source["ref"],
            type="dataproto",
            batch_fields=[part["key"]],
            non_tensor_fields=[],
            meta_info_keys=[],
            data_cls=dict,
            rows=slice(part["start"], part["stop"]),
            destinations={
                part["key"]: tensor_destination(
                    destination,
                    alignment=field["schema"]["direct_alignment"],
                    pre_registered=True,
                    allocation=allocation,
                )
            },
        )
        if result["batch"][part["key"]] is not destination:
            raise RuntimeError("Mooncake replaced the final token-sequence destination")
        offset += count
