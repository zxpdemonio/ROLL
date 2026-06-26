import sys
import types

import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler import transfer_backend
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.remote_protocol import MooncakeRemoteBatch


class FakeRef:
    def __init__(self, ref_id: int, data: dict, stage_refs):
        self.ref_id = ref_id
        self.data = data
        if isinstance(stage_refs, str):
            stage_refs = {stage_refs: object()}
        self.stage_refs = stage_refs


class FakeStore:
    def setup(self, config: dict) -> int:
        return 0


class FakeConfig:
    local_hostname = "localhost"
    metadata_server = "P2PHANDSHAKE"
    global_segment_size = 1024
    local_buffer_size = 1024
    protocol = "tcp"
    device_name = ""
    master_server_address = "localhost:50051"
    enable_ssd_offload = False
    ssd_offload_path = ""

    @classmethod
    def load_from_env(cls):
        return cls()


class FakePolicy:
    def __init__(self, max_inflight_put=1, put_mode="auto", copy_mode="auto"):
        self.max_inflight_put = max_inflight_put
        self.put_mode = put_mode
        self.copy_mode = copy_mode


class FakeTransfer:
    refs: dict[int, FakeRef] = {}
    put_calls: list[dict] = []
    append_calls: list[dict] = []
    get_calls: list[dict] = []
    cleanup_calls: list[int] = []
    next_id = 1

    def __init__(self, store, key_prefix="roll", default_chunk_bytes=1, buffer_pool=None):
        self.store = store
        self.key_prefix = key_prefix

    @classmethod
    def reset(cls) -> None:
        cls.refs = {}
        cls.put_calls = []
        cls.append_calls = []
        cls.get_calls = []
        cls.cleanup_calls = []
        cls.next_id = 1

    def put_dataproto(self, data, namespace="roll", partition="test", stage="stage", policy=None):
        FakeTransfer.put_calls.append({"data": data, "stage": stage})
        ref = FakeRef(FakeTransfer.next_id, data, stage)
        FakeTransfer.next_id += 1
        FakeTransfer.refs[ref.ref_id] = ref
        return ref

    def append_dataproto_fields(self, ref, data, stage="append", policy=None):
        FakeTransfer.append_calls.append({"data": data, "stage": stage})
        merged = {
            "batch": {**ref.data.get("batch", {}), **data.get("batch", {})},
            "non_tensor_batch": {**ref.data.get("non_tensor_batch", {}), **data.get("non_tensor_batch", {})},
            "meta_info": {**ref.data.get("meta_info", {}), **data.get("meta_info", {})},
        }
        stage_refs = dict(ref.stage_refs)
        stage_refs[stage] = object()
        new_ref = FakeRef(FakeTransfer.next_id, merged, stage_refs)
        FakeTransfer.next_id += 1
        FakeTransfer.refs[new_ref.ref_id] = new_ref
        return new_ref

    def get_dataproto(self, ref, fields=None, rows=None):
        FakeTransfer.get_calls.append({"fields": fields, "rows": rows})
        batch = {}
        non_tensor_batch = {}
        row_index = rows if rows is not None else slice(None)
        for field in fields:
            if field in ref.data.get("batch", {}):
                batch[field] = ref.data["batch"][field][row_index]
            else:
                non_tensor_batch[field] = ref.data["non_tensor_batch"][field][row_index]
        return {"batch": batch, "non_tensor_batch": non_tensor_batch, "meta_info": ref.data.get("meta_info", {})}

    def cleanup_dataproto(self, ref) -> None:
        FakeTransfer.cleanup_calls.append(ref.ref_id)

    def dataproto_manifest_view(self, ref) -> dict:
        return {"batch_fields": {}, "non_tensor_fields": {}}


def install_fake_mooncake(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeTransfer.reset()
    mooncake_module = types.ModuleType("mooncake")
    config_module = types.ModuleType("mooncake.mooncake_config")
    store_module = types.ModuleType("mooncake.store")
    structured_module = types.ModuleType("mooncake.structured_object_store")

    config_module.MooncakeConfig = FakeConfig
    store_module.MooncakeDistributedStore = FakeStore
    structured_module.BundleTransferPolicy = FakePolicy
    structured_module.MooncakeBundleTransfer = FakeTransfer
    structured_module.export_dataproto_ref = lambda ref: {"ref_id": ref.ref_id}
    structured_module.import_dataproto_ref = lambda handle: FakeTransfer.refs[handle["ref_id"]]

    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.mooncake_config", config_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.structured_object_store", structured_module)


def make_client(monkeypatch: pytest.MonkeyPatch) -> transfer_backend.MooncakeClient:
    install_fake_mooncake(monkeypatch)
    return transfer_backend.MooncakeClient({"skip_store_setup": True, "require_native_tensors": False})


def test_mooncake_client_uses_dataproto_put_and_get(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    values = torch.arange(6).reshape(3, 2)

    remote_batch = client.put(
        partition="test",
        row_ids=["a", "b", "c"],
        fields={"values": values},
        batch_size=3,
        batch_fields={"values": values},
        non_tensor_fields={},
    )

    assert isinstance(remote_batch, MooncakeRemoteBatch)
    assert FakeTransfer.put_calls[0]["data"]["batch"]["values"] is values

    materialized = client.get(
        partition="test",
        keys=["a", "b", "c"],
        fields=["values"],
        segments=remote_batch.segments,
    )

    torch.testing.assert_close(materialized["values"], values)
    assert FakeTransfer.get_calls[-1] == {"fields": ["values"], "rows": None}


def test_mooncake_remote_batch_select_pushes_rows_to_get(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    values = torch.arange(12).reshape(4, 3)
    remote_batch = client.put(
        partition="test",
        row_ids=["a", "b", "c", "d"],
        fields={"values": values},
        batch_size=4,
        batch_fields={"values": values},
        non_tensor_fields={},
    )

    selected = remote_batch.select_idxs([3, 1])
    materialized = client.get(
        partition="test",
        keys=["d", "b"],
        fields=["values"],
        segments=selected.segments,
    )

    torch.testing.assert_close(materialized["values"], values[[3, 1]])
    assert FakeTransfer.get_calls[-1] == {"fields": ["values"], "rows": [3, 1]}


def test_dataproto_to_remote_passes_sectioned_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    values = torch.arange(6).reshape(3, 2)
    data = DataProto(batch=TensorDict({"values": values}, batch_size=[3]), non_tensor_batch={})

    remote_data = DataProto.to_remote(data, partition="test")

    assert isinstance(remote_data._remote_batch, MooncakeRemoteBatch)
    assert FakeTransfer.put_calls[0]["data"]["batch"]["values"] is values
    torch.testing.assert_close(remote_data.batch["values"], values)


def test_dataproto_to_remote_ref_data_only_reuses_rows_without_append(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    row_source = client.put(
        partition="test",
        row_ids=["a", "b", "c"],
        fields={"values": torch.arange(3)},
        batch_size=3,
        batch_fields={"values": torch.arange(3)},
        non_tensor_fields={},
    )
    ref_data = DataProto(batch=None, non_tensor_batch={}, remote_batch=row_source)
    logits = torch.arange(3, dtype=torch.float32)
    data = DataProto(batch=TensorDict({"logits": logits}, batch_size=[3]), non_tensor_batch={})

    remote_data = DataProto.to_remote(data, partition="test", ref_data=ref_data)

    assert isinstance(remote_data._remote_batch, MooncakeRemoteBatch)
    assert FakeTransfer.append_calls == []
    assert FakeTransfer.put_calls[-1]["data"]["batch"]["logits"] is logits
    assert remote_data._remote_batch.row_ids() == ["a", "b", "c"]


def test_mooncake_append_uses_append_dataproto_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    values = torch.arange(3)
    remote_batch = client.put(
        partition="test",
        row_ids=["a", "b", "c"],
        fields={"values": values},
        batch_size=3,
        batch_fields={"values": values},
        non_tensor_fields={},
    )
    logits = torch.arange(3, dtype=torch.float32)

    appended = client.put(
        partition="test",
        row_ids=["a", "b", "c"],
        fields={"logits": logits},
        batch_size=3,
        batch_fields={"logits": logits},
        non_tensor_fields={},
        ref_remote_batch=remote_batch,
    )

    assert isinstance(appended, MooncakeRemoteBatch)
    assert FakeTransfer.append_calls[0]["stage"] == "stage_1"
    assert FakeTransfer.append_calls[0]["data"]["batch"]["logits"] is logits
    assert appended.segments[0]["owns_ref"] is True
    assert remote_batch.segments[0]["owns_ref"] is False


def test_mooncake_append_uses_existing_stage_for_multi_stage_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    values = torch.arange(3)
    ref = FakeRef(
        100,
        {"batch": {"values": values}, "non_tensor_batch": {}, "meta_info": {}},
        {"rollout": object(), "critic": object()},
    )
    FakeTransfer.refs[ref.ref_id] = ref
    remote_batch = MooncakeRemoteBatch(
        partition="test",
        device=None,
        fields={"values"},
        row_ids=["a", "b", "c"],
        handle={"ref_id": ref.ref_id},
        rows=None,
        owns_ref=True,
        cache=None,
    )
    logits = torch.arange(3, dtype=torch.float32)

    appended = client.put(
        partition="test",
        row_ids=["a", "b", "c"],
        fields={"logits": logits},
        batch_size=3,
        batch_fields={"logits": logits},
        non_tensor_fields={},
        ref_remote_batch=remote_batch,
    )

    assert isinstance(appended, MooncakeRemoteBatch)
    assert FakeTransfer.append_calls[0]["stage"] == "rollout"
    assert appended.segments[0]["owns_ref"] is True
    assert remote_batch.segments[0]["owns_ref"] is False


def test_mooncake_drop_cleans_owned_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    remote_batch = client.put(
        partition="test",
        row_ids=["a"],
        fields={"values": torch.arange(1)},
        batch_size=1,
        batch_fields={"values": torch.arange(1)},
        non_tensor_fields={},
    )

    remote_batch.drop()

    assert FakeTransfer.cleanup_calls == [remote_batch.segments[0]["handle"]["ref_id"]]


def test_mooncake_clone_does_not_own_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    remote_batch = client.put(
        partition="test",
        row_ids=["a"],
        fields={"values": torch.arange(1)},
        batch_size=1,
        batch_fields={"values": torch.arange(1)},
        non_tensor_fields={},
    )

    cloned = remote_batch.clone()
    remote_batch.drop()
    cloned.drop()

    assert cloned.owns_ref is False
    assert cloned.segments[0]["owns_ref"] is False
    assert FakeTransfer.cleanup_calls == [remote_batch.segments[0]["handle"]["ref_id"]]


def test_dataproto_clone_does_not_double_cleanup_mooncake_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    remote_batch = client.put(
        partition="test",
        row_ids=["a"],
        fields={"values": torch.arange(1)},
        batch_size=1,
        batch_fields={"values": torch.arange(1)},
        non_tensor_fields={},
    )
    data = DataProto(batch=None, non_tensor_batch={}, remote_batch=remote_batch)

    cloned = data.clone()
    DataProto.drop(data)
    DataProto.drop(cloned)

    assert cloned._remote_batch.owns_ref is False
    assert FakeTransfer.cleanup_calls == [remote_batch.segments[0]["handle"]["ref_id"]]


def test_mooncake_single_batch_cat_returns_non_owning_view(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    remote_batch = client.put(
        partition="test",
        row_ids=["a"],
        fields={"values": torch.arange(1)},
        batch_size=1,
        batch_fields={"values": torch.arange(1)},
        non_tensor_fields={},
    )

    cat_batch = MooncakeRemoteBatch._cat([remote_batch])
    remote_batch.drop()
    cat_batch.drop()

    assert cat_batch.owns_ref is False
    assert cat_batch.segments[0]["owns_ref"] is False
    assert FakeTransfer.cleanup_calls == [remote_batch.segments[0]["handle"]["ref_id"]]
