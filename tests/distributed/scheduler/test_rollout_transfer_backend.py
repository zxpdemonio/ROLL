import sys
import types

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from roll.distributed.scheduler import transfer_backend
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.remote_protocol import MooncakeRemoteBatch


class FakeRef:
    def __init__(self, ref_id: int, data: dict, stage: str):
        self.ref_id = ref_id
        self.data = data
        self.stage_refs = {stage: object()}


class FakeStore:
    setup_calls: list[dict] = []

    def setup(self, config: dict) -> int:
        FakeStore.setup_calls.append(config)
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

    def __init__(self, store, key_prefix="roll", default_chunk_bytes=1):
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
        new_ref = FakeRef(FakeTransfer.next_id, merged, stage)
        FakeTransfer.next_id += 1
        FakeTransfer.refs[new_ref.ref_id] = new_ref
        return new_ref

    def get_dataproto(self, ref, fields=None, rows=None):
        FakeTransfer.get_calls.append({"fields": fields, "rows": rows})
        row_index = rows if rows is not None else slice(None)
        batch = {}
        non_tensor_batch = {}
        for field in fields:
            if field in ref.data.get("batch", {}):
                batch[field] = ref.data["batch"][field][row_index]
            else:
                non_tensor_batch[field] = ref.data["non_tensor_batch"][field][row_index]
        return {"batch": batch, "non_tensor_batch": non_tensor_batch, "meta_info": ref.data.get("meta_info", {})}

    def cleanup_dataproto(self, ref) -> None:
        FakeTransfer.cleanup_calls.append(ref.ref_id)


def install_fake_mooncake(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeTransfer.reset()
    FakeStore.setup_calls = []
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
    return transfer_backend.MooncakeClient({"skip_store_setup": True})


def test_mooncake_client_uses_explicit_backend_config_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_mooncake(monkeypatch)
    monkeypatch.setattr(FakeConfig, "load_from_env", classmethod(lambda cls: pytest.fail("load_from_env should not run")))

    transfer_backend.MooncakeClient(
        {
            "local_hostname": "192.168.22.70",
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": 32 * 1024**3,
            "local_buffer_size": 40 * 1024**3,
            "protocol": "rdma",
            "device_name": "erdma_1",
            "master_server_address": "192.168.22.70:50053",
        }
    )

    assert FakeStore.setup_calls == [
        {
            "local_hostname": "192.168.22.70",
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": 32 * 1024**3,
            "local_buffer_size": 40 * 1024**3,
            "protocol": "rdma",
            "rdma_devices": "erdma_1",
            "master_server_addr": "192.168.22.70:50053",
            "enable_ssd_offload": False,
            "ssd_offload_path": "",
        }
    ]
    with pytest.raises(ValueError, match="device_name"):
        transfer_backend.MooncakeClient(
            {
                "local_hostname": "192.168.22.70",
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "rdma",
                "master_server_address": "192.168.22.70:50053",
            }
        )


def test_mooncake_client_put_get_rows_and_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    values = torch.arange(12).reshape(4, 3)
    tags = np.array(["a", "b", "c", "d"], dtype=object)

    remote_batch = client.put(
        partition="test",
        row_ids=["a", "b", "c", "d"],
        fields={"values": values, "tags": tags},
        batch_size=4,
        batch_fields={"values": values},
        non_tensor_fields={"tags": tags},
    )
    selected = remote_batch.select_idxs([3, 1])
    materialized = client.get("test", ["d", "b"], ["values", "tags"], segments=selected.segments)

    assert isinstance(remote_batch, MooncakeRemoteBatch)
    torch.testing.assert_close(materialized["values"], values[[3, 1]])
    assert list(materialized["tags"]) == ["d", "b"]
    assert FakeTransfer.get_calls[-1] == {"fields": ["values", "tags"], "rows": [3, 1]}


def test_dataproto_to_remote_uses_transfer_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    values = torch.arange(6).reshape(3, 2)
    data = DataProto(
        batch=TensorDict({"values": values}, batch_size=[3]),
        non_tensor_batch={"tags": np.array(["a", "b", "c"], dtype=object)},
    )

    remote_data = DataProto.to_remote(data, partition="test")

    assert isinstance(remote_data._remote_batch, MooncakeRemoteBatch)
    assert FakeTransfer.put_calls[0]["data"]["batch"]["values"] is values
    assert list(remote_data.non_tensor_batch["tags"]) == ["a", "b", "c"]


def test_mooncake_append_and_drop_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(transfer_backend, "init_client", lambda: None)
    monkeypatch.setattr(transfer_backend, "_client", client)
    base = client.put(
        "test",
        ["a", "b"],
        {"values": torch.arange(2)},
        2,
        batch_fields={"values": torch.arange(2)},
        non_tensor_fields={},
    )
    appended = client.put(
        "test",
        ["a", "b"],
        {"logits": torch.arange(2, dtype=torch.float32)},
        2,
        batch_fields={"logits": torch.arange(2, dtype=torch.float32)},
        non_tensor_fields={},
        ref_remote_batch=base,
    )

    appended.drop()
    base.drop()

    assert FakeTransfer.append_calls[0]["stage"] == "stage_1"
    assert base.segments[0]["owns_ref"] is False
    assert FakeTransfer.cleanup_calls == [appended.segments[0]["handle"]["ref_id"]]
