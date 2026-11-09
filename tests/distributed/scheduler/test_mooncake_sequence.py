"""Token refs must preserve data, row plans and source ownership without GET."""

import multiprocessing
import pickle
import uuid

import numpy as np
import pytest
import torch

from roll.distributed.scheduler import transfer_backend as tb
from roll.distributed.scheduler.mooncake_direct import read_direct_columns
from roll.distributed.scheduler.mooncake_sequence import (
    MooncakeRows,
    WorkerAuxRows,
    decode_rows_carrier,
    merge_resumed_rows,
    sequence_remote_batch,
)
from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch
from roll.utils.execution_artifacts import AttentionSelectionProfile

mc = pytest.importorskip("mooncake.structured_object_store")


def owner(shape=(4, 2, 3), *, key="routed_experts", dtype="uint16", alignment=64):
    identity = uuid.uuid4().hex
    ref = mc.MooncakeDataProtoRef(
        batch_size=shape[0],
        stage_refs={"capture": mc.RemoteBundleRef(identity, {})},
        field_index={key: mc.StructuredFieldLocation("capture", key, "batch")},
        meta_info={},
    )
    batch = ColumnRemoteBatch(
        "train_eval",
        None,
        fields={
            key: {
                "ref": ref,
                "kind": "batch",
                "shape": list(shape),
                "dtype": "torch." + dtype,
                "schema": {"codec": "torch_tensor", "direct_alignment": alignment},
            }
        },
        cache=None,
        batch_size=shape[0],
    )
    batch._roll_aux_delivery_id = identity
    return batch


def rows(batch, cls=MooncakeRows):
    key, field = next(iter(batch.fields.items()))
    return cls([(batch, key, 0, len(batch))], field["shape"], field["dtype"].removeprefix("torch."))


def _ipc_scheduler(connection):
    value = connection.recv()
    # The original scheduler uses len() then slices accepted rows.
    connection.send((len(value), value[1:-1]))
    connection.close()


def test_worker_pickle_ipc_then_small_ndarray_carrier():
    source = owner((100_000, 61, 2048), dtype="int32")
    value = rows(source, WorkerAuxRows)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_ipc_scheduler, args=(child,))
    process.start()
    child.close()
    try:
        parent.send(value)
        assert parent.poll(30), "scheduler IPC did not respond"
        count, carrier = parent.recv()
        assert count == 100_000
        assert isinstance(carrier, np.ndarray) and carrier.dtype == np.uint8
        assert carrier.nbytes < 10_000  # logical payload is about 50 GB
        decoded = decode_rows_carrier(carrier)
        assert decoded.shape == (99_998, 61, 2048)
        assert decoded.parts[0][2:] == (1, 99_999)
        assert decoded.delivery_ids == value.delivery_ids
    finally:
        parent.close()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def test_trim_keeps_all_delivery_owners_including_zero_rows():
    first, second = owner((3, 2, 3)), owner((2, 2, 3))
    value = MooncakeRows.concatenate([rows(first), rows(second)])[:2]
    assert len(value) == 2
    assert value.delivery_ids == {first._roll_aux_delivery_id, second._roll_aux_delivery_id}
    empty = pickle.loads(pickle.dumps(value[:0]))
    assert len(empty) == 0 and empty.delivery_ids == value.delivery_ids
    with pytest.raises(TypeError, match="references"):
        np.asarray(value)


@pytest.mark.parametrize("item", [slice(None, None, 2), 0])
def test_unsupported_token_index_is_rejected(item):
    with pytest.raises((ValueError, TypeError)):
        rows(owner())[item]


class Backend:
    def __init__(self, values):
        self.values = values
        self.store = self
        self.registered = {}
        self._roll_pending_direct_registrations = []
        self.targets = []

    def register_buffer(self, ptr, size):
        self.registered[ptr] = size
        return 0

    def unregister_buffer(self, ptr):
        del self.registered[ptr]
        return 0

    def get(self, ref, *, rows, destinations, **kwargs):
        output = {}
        identity = ref.stage_refs["capture"].manifest_key
        for key, destination in destinations.items():
            target = destination.tensor
            root = destination.allocation
            assert root.data_ptr() in self.registered
            assert root.data_ptr() % destination.alignment == 0
            target.copy_(self.values[identity][rows])
            self.targets.append(target)
            output[key] = target
        return {"batch": output}


@pytest.mark.parametrize("operation", ["select", "repeat", "cat", "pickle", "pop_union"])
def test_trajectory_plans_scatter_tokens_and_preserve_lifetime(monkeypatch, operation):
    first, second = owner((3, 2, 3)), owner((2, 2, 3))
    values = {
        first._roll_aux_delivery_id: torch.arange(18, dtype=torch.int32).view(3, 2, 3).to(torch.uint16),
        second._roll_aux_delivery_id: torch.empty(0),
    }
    # uint16 arithmetic is not available on every supported Torch version.
    values[second._roll_aux_delivery_id] = (torch.arange(12, dtype=torch.int32).view(2, 2, 3) + 50).to(torch.uint16)
    a = MooncakeRows.concatenate([rows(first)[:2], rows(second)[1:]])
    b = rows(first)[:2]
    remote = sequence_remote_batch({"routed_experts": [a, b]}, [3, 2], 5)
    backend = Backend(values)
    monkeypatch.setattr(
        tb,
        "get_direct",
        lambda partition, keys, sources, plan, **kw: tb.create_tensordict(
            read_direct_columns(backend, keys, sources, plan, device="cpu")
        ),
    )
    deleted = []
    monkeypatch.setattr(tb, "delete", lambda partition, keys, fields: deleted.extend(fields))
    expected = torch.arange(3, dtype=torch.int32).view(1, 1, 3).expand(2, 5, 2, 3).clone()
    expected[0, :2] = values[first._roll_aux_delivery_id][:2].to(torch.int32)
    expected[0, 2] = values[second._roll_aux_delivery_id][1].to(torch.int32)
    expected[1, :2] = values[first._roll_aux_delivery_id][:2].to(torch.int32)
    if operation == "select":
        result = remote.select_idxs([1, 0, 0])
        expected = expected[[1, 0, 0]]
    elif operation == "repeat":
        result = remote.repeat(2, interleave=True)
        expected = expected.repeat_interleave(2, dim=0)
    elif operation == "cat":
        result = ColumnRemoteBatch._cat([remote, remote.select_idxs([0])])
        expected = torch.cat([expected, expected[:1]])
    elif operation == "pickle":
        result = pickle.loads(pickle.dumps(remote))
    else:
        result = remote.pop(["routed_experts"])
        result = remote.union(result)
    data = result.materialize()["routed_experts"]
    assert torch.equal(data.to(torch.int32), expected)
    assert data.data_ptr() % 64 == 0
    assert all(data.untyped_storage().data_ptr() == target.untyped_storage().data_ptr() for target in backend.targets)
    assert not backend.registered
    result.drop()
    assert len(deleted) == 2
    if operation != "pickle":
        with pytest.raises(RuntimeError, match="dropped"):
            first.materialize()


def test_selection_row_padding_is_kept_without_a_payload_get(monkeypatch):
    profile = AttentionSelectionProfile("test", "deepseek_v3_2", "vllm", 2, 3, row_alignment_bytes=64)
    source = owner((3, profile.storage_row_width), key="attention_selection", dtype="int32", alignment=4096)
    remote = sequence_remote_batch({"attention_selection": [rows(source)]}, [3], 6, profile)
    assert remote.fields["attention_selection"]["shape"] == [1, 5, 16]
    assert remote.fields["attention_selection"]["padding"] == -1
    assert remote.cache is None


def test_resume_keeps_original_prefix_choices_and_only_appends_new_tail():
    previous, current = owner((2, 1, 2)), owner((4, 1, 2))
    merged = merge_resumed_rows([rows(previous)], [rows(current)])[0]
    assert merged.shape == (4, 1, 2)
    assert [(part[0], part[2], part[3]) for part in merged.parts] == [(previous, 0, 2), (current, 2, 4)]


@pytest.mark.parametrize("consumer_device", [None, "cpu", "cuda:1"])
def test_dataproto_packs_refs_without_get_or_second_payload_put(monkeypatch, consumer_device):
    from tensordict import TensorDict

    from roll.distributed.scheduler.protocol import DataProto
    from roll.utils.functionals import postprocess_generate

    source = owner((3, 1, 2))
    prompts = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.int64),
                "position_ids": torch.tensor([[0, 1]]),
            },
            batch_size=[1],
        )
    )
    packed = postprocess_generate(
        prompts, torch.tensor([[1, 2, 3, 4]]), 1, 4, -2, -1, routed_experts=[rows(source)], num_moe_experts=512
    )
    assert "routed_experts" not in packed._batch
    assert packed._remote_batch.cache is None
    if consumer_device is not None:
        packed._remote_batch.to(consumer_device)
    actual = []

    def put(partition, row_ids, fields, batch_size):
        assert "routed_experts" not in fields
        actual.extend(fields)
        return ColumnRemoteBatch(
            partition,
            torch.device("cpu"),
            {key: {"ref": "tokens", "kind": "batch"} for key in fields},
            cache=tb.create_tensordict(fields) if consumer_device != "cuda:1" else None,
            batch_size=batch_size,
            row_ids=row_ids,
        )

    monkeypatch.setattr(tb, "put", put)
    remote = DataProto.to_remote(packed)
    assert "routed_experts" in remote._remote_batch.fields
    assert "input_ids" in actual
    assert len(remote._remote_batch._states) == 2
    if consumer_device is None:
        assert remote._remote_batch.device is None
    else:
        assert torch.device(remote._remote_batch.device) == torch.device(consumer_device)


@pytest.mark.parametrize("change", ["length", "width", "dtype", "schema"])
def test_bad_sequence_reference_fails_before_transfer(change):
    source = owner()
    value = rows(source)
    if change == "length":
        with pytest.raises(ValueError, match="valid rows"):
            sequence_remote_batch({"routed_experts": [value]}, [5], 6)
    elif change == "width":
        with pytest.raises(ValueError, match="width"):
            MooncakeRows(value.parts, (4, 5), value.dtype)
    elif change == "dtype":
        with pytest.raises(TypeError, match="dtype"):
            MooncakeRows(value.parts, value.shape, "uint8")
    else:
        source.fields["routed_experts"]["schema"] = {"codec": "torch_tensor"}
        with pytest.raises(ValueError, match="schema"):
            sequence_remote_batch({"routed_experts": [value]}, [4], 6)
