import sys
from types import SimpleNamespace

import pytest
import torch

from roll.distributed.scheduler import transfer_backend
from roll.distributed.scheduler.mooncake_direct import read_direct_columns
from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch


def descriptor(ref, shape=(3, 2, 16)):
    return {
        "ref": ref,
        "kind": "batch",
        "shape": list(shape),
        "dtype": "torch.int32",
        "direct": True,
    }


class FakeDirectBackend:
    def __init__(self):
        self.store = self
        self.registered = set()
        self.destinations = []
        self._roll_pending_direct_registrations = []
        self.fail_get = False
        self.fail_unregister = False

    def register_buffer(self, pointer, nbytes):
        self.registered.add(pointer)
        return 0

    def unregister_buffer(self, pointer):
        if self.fail_unregister:
            return -1
        self.registered.remove(pointer)
        return 0

    def get(self, ref, *, rows, destinations, **kwargs):
        if self.fail_get:
            raise RuntimeError("read failed")
        targets = {}
        for key, destination in destinations.items():
            target = destination.tensor
            self.destinations.append(target)
            assert target.data_ptr() % target.element_size() == 0
            for index, row in enumerate(rows):
                target[index].fill_(ref * 10 + row)
            targets[key] = target
        return {"batch": targets}


@pytest.fixture
def tensor_destination_stub(monkeypatch):
    module = SimpleNamespace(tensor_destination=lambda tensor, **kwargs: SimpleNamespace(tensor=tensor))
    monkeypatch.setitem(sys.modules, "mooncake.structured_object_store", module)


def test_direct_columns_gather_into_final_allocation(tensor_destination_stub):
    backend = FakeDirectBackend()
    sources = [{"attention_selection": descriptor(1)}, {"attention_selection": descriptor(2)}]
    output = read_direct_columns(
        backend,
        ["attention_selection"],
        sources,
        [(1, 2), (0, 1), (0, 1), (1, 0)],
        alignments={"attention_selection": 64},
        device="cpu",
    )["attention_selection"]
    assert output[:, 0, 0].tolist() == [22, 11, 11, 20]
    assert output.data_ptr() == backend.destinations[0].data_ptr()
    assert not backend.registered


def test_direct_columns_deregister_after_synchronous_read_failure(tensor_destination_stub):
    backend = FakeDirectBackend()
    backend.fail_get = True
    with pytest.raises(RuntimeError, match="read failed"):
        read_direct_columns(
            backend,
            ["routed_experts"],
            [{"routed_experts": descriptor(1)}],
            [(0, 0)],
            alignments={"routed_experts": 64},
            device="cpu",
        )
    assert not backend.registered


def test_direct_columns_retain_failed_deregistrations(tensor_destination_stub):
    backend = FakeDirectBackend()
    backend.fail_unregister = True
    with pytest.raises(RuntimeError, match="deregistration"):
        read_direct_columns(
            backend,
            ["routed_experts"],
            [{"routed_experts": descriptor(1)}],
            [(0, 0)],
            alignments={"routed_experts": 64},
            device="cpu",
        )
    assert len(backend._roll_pending_direct_registrations) == 1
    assert backend._roll_pending_direct_registrations[0].data_ptr() in backend.registered


def test_direct_remote_batch_pushes_row_operations_to_metadata(monkeypatch):
    first = ColumnRemoteBatch(
        partition="rollout",
        device=None,
        fields={"attention_selection": descriptor(1)},
        cache=None,
        batch_size=3,
        row_ids=["a", "b", "c"],
    )
    second = ColumnRemoteBatch(
        partition="rollout",
        device=None,
        fields={"attention_selection": descriptor(2)},
        cache=None,
        batch_size=3,
        row_ids=["d", "e", "f"],
    )
    selected = (
        ColumnRemoteBatch._cat([first.slice(1, 3), second])
        .select_idxs(torch.tensor([4, 0, 3, 3]))
        .repeat(2, interleave=True)
    )
    calls = []

    def direct(partition, keys, sources, rows):
        calls.append(rows)
        values = [sources[source][keys[0]]["ref"] * 10 + row for source, row in rows]
        return transfer_backend.create_tensordict({keys[0]: torch.tensor(values).view(-1, 1)})

    monkeypatch.setattr(transfer_backend, "get_direct", direct)
    monkeypatch.setattr(transfer_backend, "get", lambda **kwargs: pytest.fail("payload staging"))
    result = selected.materialize()["attention_selection"]
    assert result.flatten().tolist() == [22, 22, 11, 11, 21, 21, 21, 21]
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["select", "slice", "repeat", "clone", "cat", "device"])
def test_direct_cached_transforms_read_again_without_payload_copy(monkeypatch, operation):
    batch = ColumnRemoteBatch(
        partition="rollout",
        device=None,
        fields={"routed_experts": descriptor(1, (3, 1, 8))},
        cache=None,
        batch_size=3,
        row_ids=["a", "b", "c"],
    )
    calls = []

    def direct(partition, keys, sources, rows, **kwargs):
        calls.append(kwargs)
        return transfer_backend.create_tensordict(
            {"routed_experts": torch.zeros((len(rows), 1, 8), dtype=torch.uint8)}
        )

    monkeypatch.setattr(transfer_backend, "get_direct", direct)
    batch.materialize()
    if operation == "select":
        transformed = batch.select_idxs([2, 0])
    elif operation == "slice":
        transformed = batch.slice(1, 3)
    elif operation == "repeat":
        transformed = batch.repeat(2, interleave=False)
    elif operation == "clone":
        transformed = batch.clone()
    elif operation == "cat":
        transformed = ColumnRemoteBatch._cat([batch, batch])
    else:
        transformed = batch.to("cuda:1")
    assert transformed.cache is None
    transformed.materialize()
    assert len(calls) == 2
    if operation == "device":
        assert calls[-1]["device"] == "cuda:1"


def test_direct_small_r3_rows_keep_root_alignment(tensor_destination_stub):
    backend = FakeDirectBackend()
    sources = [{"routed_experts": descriptor(1, (3, 1, 2))}, {"routed_experts": descriptor(2, (3, 1, 2))}]
    output = read_direct_columns(
        backend,
        ["routed_experts"],
        sources,
        [(0, 0), (1, 1)],
        alignments={"routed_experts": 64},
        device="cpu",
    )["routed_experts"]
    assert output.data_ptr() % 64 == 0
    assert output[:, 0, 0].tolist() == [10, 21]
