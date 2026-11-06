import sys
from types import SimpleNamespace

import pytest
import torch

from roll.distributed.scheduler import transfer_backend
from roll.distributed.scheduler.mooncake_direct import read_direct_columns
from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch


def descriptor(ref, shape=(3, 2, 16), alignment=64):
    return {
        "ref": ref,
        "kind": "batch",
        "shape": list(shape),
        "dtype": "torch.int32",
        "schema": {"codec": "torch_tensor", "direct_alignment": alignment},
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
            assert destination.allocation.data_ptr() % destination.alignment == 0
            for index, row in enumerate(rows):
                target[index].fill_(ref * 10 + row)
            targets[key] = target
        return {"batch": targets}


@pytest.fixture
def tensor_destination_stub(monkeypatch):
    module = SimpleNamespace(tensor_destination=lambda tensor, **kwargs: SimpleNamespace(tensor=tensor, **kwargs))
    monkeypatch.setitem(sys.modules, "mooncake.structured_object_store", module)


def test_direct_columns_gather_into_final_allocation(tensor_destination_stub):
    backend = FakeDirectBackend()
    sources = [{"attention_selection": descriptor(1)}, {"attention_selection": descriptor(2)}]
    output = read_direct_columns(
        backend,
        ["attention_selection"],
        sources,
        [(1, 2), (0, 1), (0, 1), (1, 0)],
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
        device="cpu",
    )["routed_experts"]
    assert output.data_ptr() % 64 == 0
    assert output[:, 0, 0].tolist() == [10, 21]


def test_schema_controls_directness_independently_of_field_name():
    backend = transfer_backend.MooncakeClient.__new__(transfer_backend.MooncakeClient)
    backend._pending_release_results = []
    backend.transfer_policy = None
    backend.backend = SimpleNamespace(put=lambda data, **kwargs: "ref")
    backend.field_schemas = {
        "custom_replay": SimpleNamespace(codec="torch_tensor", direct_alignment=64),
    }
    remote = backend.put(
        "rollout",
        ["a", "b"],
        {
            "attention_selection": torch.zeros(2, 16, dtype=torch.int32),
            "routed_experts": torch.zeros(2, 8, dtype=torch.uint8),
            "custom_replay": torch.zeros(2, 16, dtype=torch.int32),
        },
        2,
    )
    assert not transfer_backend.requires_direct(remote.fields["attention_selection"])
    assert not transfer_backend.requires_direct(remote.fields["routed_experts"])
    assert transfer_backend.requires_direct(remote.fields["custom_replay"])
    assert "direct" not in remote.fields["custom_replay"]


def test_direct_columns_reject_missing_schema(tensor_destination_stub):
    spec = descriptor(1)
    del spec["schema"]
    with pytest.raises(ValueError, match="schemas"):
        read_direct_columns(FakeDirectBackend(), ["value"], [{"value": spec}], [(0, 0)], device="cpu")


@pytest.mark.parametrize("rows,kind", [([0], "direct"), ([2], "plain"), ([0, 2], "mixed")])
def test_selected_rows_determine_schema_contract(monkeypatch, rows, kind):
    direct_field = descriptor(1)
    plain_field = {"ref": SimpleNamespace(batch_size=2), "kind": "batch"}
    direct = ColumnRemoteBatch(
        partition="replay",
        device=None,
        fields={"value": direct_field},
        cache=None,
        batch_size=3,
        row_ids=["a", "b", "c"],
    )
    plain = ColumnRemoteBatch(
        partition="replay", device=None, fields={"value": plain_field}, cache=None, batch_size=2, row_ids=["d", "e"]
    )
    # First source contributes two rows; index 2 belongs to the plain source.
    selected = ColumnRemoteBatch._cat([direct.slice(0, 2), plain]).select_idxs(rows)
    calls = []

    def get_direct(partition, keys, sources, plan, **kwargs):
        calls.append("direct")
        assert all(sources[index]["value"].get("schema") for index, row in plan)
        return transfer_backend.create_tensordict({"value": torch.ones(len(plan), 1)})

    def get_plain(partition, keys, fields):
        calls.append("plain")
        assert fields == [plain_field]
        return transfer_backend.create_tensordict({"value": torch.full((2, 1), 2.0)})

    monkeypatch.setattr(transfer_backend, "get_direct", get_direct)
    monkeypatch.setattr(transfer_backend, "get", get_plain)
    if kind == "mixed":
        with pytest.raises(ValueError, match="selected rows"):
            selected.materialize()
        assert calls == []
    else:
        result = selected.materialize()["value"]
        assert calls == [kind]
        assert result.item() == (1 if kind == "direct" else 2)


def test_fields_are_grouped_by_their_own_source_schemas(monkeypatch):
    def source(index, direct_key):
        fields = {key: descriptor(index) for key in ("a", "b")}
        fields["b" if direct_key == "a" else "a"].pop("schema")
        return ColumnRemoteBatch(
            partition="replay",
            device=None,
            fields=fields,
            cache=None,
            batch_size=3,
            row_ids=[f"{index}-{row}" for row in range(3)],
        )

    batch = ColumnRemoteBatch._cat([source(1, "a"), source(2, "b")]).select_idxs([0])
    calls = []

    def direct(partition, keys, sources, rows, **kwargs):
        assert keys == ["a"]
        calls.append("direct")
        return transfer_backend.create_tensordict({"a": torch.ones(len(rows), 1)})

    def plain(partition, keys, fields):
        assert keys == ["b"]
        calls.append("plain")
        return transfer_backend.create_tensordict({"b": torch.full((3, 1), 2.0)})

    monkeypatch.setattr(transfer_backend, "get_direct", direct)
    monkeypatch.setattr(transfer_backend, "get", plain)
    result = batch.materialize()
    assert result["a"].item() == 1
    assert result["b"].item() == 2
    assert sorted(calls) == ["direct", "plain"]


def _boundary_batch(key="x", ref=1, row_ids=None):
    return ColumnRemoteBatch(
        partition="boundary",
        device=None,
        fields={key: descriptor(ref, (3, 1))},
        cache=None,
        batch_size=3,
        row_ids=row_ids,
    )


def _install_boundary_reader(monkeypatch, calls):
    def direct(partition, keys, sources, rows, **kwargs):
        calls.append(kwargs)
        return transfer_backend.create_tensordict(
            {
                key: torch.tensor(
                    [sources[source][key]["ref"] * 10 + row for source, row in rows], dtype=torch.int32
                ).view(-1, 1)
                for key in keys
            }
        )

    monkeypatch.setattr(transfer_backend, "get_direct", direct)


def test_nested_cat_uses_each_fields_child_row_plan(monkeypatch):
    x = _boundary_batch("x", 1, ["a", "b", "c"]).select_idxs([2, 0])
    y = _boundary_batch("y", 2, ["c", "b", "a"]).select_idxs([0, 2])
    x.union(y)
    _install_boundary_reader(monkeypatch, [])
    result = ColumnRemoteBatch._cat([x, x]).materialize()
    assert result["x"].flatten().tolist() == [12, 10, 12, 10]
    assert result["y"].flatten().tolist() == [20, 22, 20, 22]


@pytest.mark.parametrize("operation", ["slice", "rows", "fields", "clone", "pickle", "cat"])
def test_consumer_device_survives_remote_view_reconstruction(monkeypatch, operation):
    import pickle

    batch = _boundary_batch().to("cpu")
    views = {
        "slice": lambda: batch.slice(1, 3),
        "rows": lambda: batch.select_idxs([2, 0]),
        "fields": lambda: batch.select(["x"]),
        "clone": batch.clone,
        "pickle": lambda: pickle.loads(pickle.dumps(batch)),
        "cat": lambda: ColumnRemoteBatch._cat([batch, batch]),
    }
    calls = []
    _install_boundary_reader(monkeypatch, calls)
    view = views[operation]()
    view.materialize()
    assert calls[0]["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU index metadata")
@pytest.mark.parametrize("boolean", [False, True])
def test_direct_row_plan_accepts_cuda_indices(monkeypatch, boolean):
    calls = []
    _install_boundary_reader(monkeypatch, calls)
    index = torch.tensor([True, False, True] if boolean else [2, 0], device="cuda")
    result = _boundary_batch().select_idxs(index).materialize()["x"]
    assert result.flatten().tolist() == ([10, 12] if boolean else [12, 10])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for a mismatched direct cache")
def test_union_rematerializes_direct_cache_on_consumer_device(monkeypatch):
    left = _boundary_batch("plain").to("cpu")
    left.fields["plain"].pop("schema")
    right = _boundary_batch().to("cuda")
    right.cache = transfer_backend.create_tensordict({"x": torch.zeros(3, 1, device="cuda")})
    calls = []
    _install_boundary_reader(monkeypatch, calls)
    left.union(right)
    assert left.cache is None or "x" not in left.cache
    result = left.materialize(["x"])
    assert result["x"].device.type == "cpu"
    assert calls[0]["device"] == "cpu"
