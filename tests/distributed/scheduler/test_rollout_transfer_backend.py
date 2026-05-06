import ctypes
import os
import sys
import types

import numpy as np
import pytest
import ray
import torch

from roll.distributed.scheduler import protocol
from roll.distributed.scheduler.protocol import (
    DataProto,
    RolloutTransferHandle,
    get_rollout_transfer_backend,
    materialize_rollout_transfer,
)


def test_legacy_rollout_transfer_backend_round_trip() -> None:
    proto = DataProto.from_dict(
        tensors={"input_ids": torch.arange(6, dtype=torch.long).reshape(2, 3)},
        non_tensors={"domain": ["math", "code"]},
        meta_info={"seed": 7},
    )

    backend = get_rollout_transfer_backend("legacy", "legacy")
    handle = backend.put(proto, stage="post_generate")
    restored = backend.get(handle)

    assert isinstance(handle, RolloutTransferHandle)
    assert handle.backend == "legacy"
    assert torch.equal(restored.batch["input_ids"], proto.batch["input_ids"])
    np.testing.assert_array_equal(restored.non_tensor_batch["domain"], proto.non_tensor_batch["domain"])
    assert restored.meta_info == proto.meta_info


def test_materialize_rollout_transfer_passthrough_for_dataproto() -> None:
    proto = DataProto.from_dict(tensors={"input_ids": torch.ones((1, 2), dtype=torch.long)})

    restored = materialize_rollout_transfer(proto, backend_name="legacy", protocol="legacy")

    assert restored is proto


def test_ray_optimized_rollout_transfer_backend_round_trip() -> None:
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        proto = DataProto.from_dict(
            tensors={
                "input_ids": torch.arange(8, dtype=torch.long).reshape(2, 4),
                "attention_mask": torch.ones((2, 4), dtype=torch.long),
            },
            non_tensors={"domain": ["math", "code"], "score": [1.0, 2.0]},
            meta_info={"metrics": {"acc": [0.5, 1.0]}},
        )

        backend = get_rollout_transfer_backend("ray_optimized", "v1")
        handle = backend.put(proto, stage="post_generate")
        restored = backend.get(handle)

        assert isinstance(handle, RolloutTransferHandle)
        assert handle.backend == "ray_optimized"
        assert handle.obj_ref is not None
        assert torch.equal(restored.batch["input_ids"], proto.batch["input_ids"])
        assert torch.equal(restored.batch["attention_mask"], proto.batch["attention_mask"])
        np.testing.assert_array_equal(restored.non_tensor_batch["domain"], proto.non_tensor_batch["domain"])
        np.testing.assert_array_equal(restored.non_tensor_batch["score"], proto.non_tensor_batch["score"])
        assert restored.meta_info == proto.meta_info
    finally:
        ray.shutdown()


def test_ray_optimized_rollout_transfer_backend_emits_metrics() -> None:
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        proto = DataProto.from_dict(
            tensors={"input_ids": torch.arange(6, dtype=torch.long).reshape(2, 3)},
            non_tensors={"domain": ["math", "code"]},
            meta_info={"rollout_transfer_metrics_enabled": True},
        )

        backend = get_rollout_transfer_backend("ray_optimized", "v1")
        handle = backend.put(proto, stage="post_generate")
        restored = backend.get(handle)

        metrics = restored.meta_info["metrics"]
        assert metrics["transfer/backend/ray_optimized"] == 1.0
        assert metrics["transfer/backend/mooncake"] == 0.0
        assert metrics["transfer/time/serialize"] >= 0
        assert metrics["transfer/time/put"] >= 0
        assert metrics["transfer/time/get"] >= 0
        assert metrics["transfer/time/deserialize"] >= 0
        assert metrics["transfer/bytes/total"] > 0
        assert metrics["transfer/bytes/wire"] > 0
        assert metrics["transfer/throughput/serialize_mbps"] >= 0
        assert metrics["transfer/throughput/put_mbps"] >= 0
        assert metrics["transfer/throughput/get_mbps"] >= 0
        assert metrics["transfer/throughput/deserialize_mbps"] >= 0
    finally:
        ray.shutdown()


def test_mooncake_rollout_transfer_backend_round_trip_uses_fallback_when_runtime_unavailable() -> None:
    os.environ.pop("ROLL_MOONCAKE_STRICT", None)
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        proto = DataProto.from_dict(
            tensors={"input_ids": torch.arange(8, dtype=torch.long).reshape(2, 4)},
            non_tensors={"domain": ["math", "code"], "score": [1.0, 2.0]},
            meta_info={"metrics": {"acc": [0.5, 1.0]}, "rollout_transfer_metrics_enabled": True},
        )

        backend = get_rollout_transfer_backend("mooncake", "v1")
        handle = backend.put(proto, stage="post_generate")
        restored = backend.get(handle)

        assert isinstance(handle, RolloutTransferHandle)
        assert handle.backend == "mooncake"
        assert handle.transport_info["mode"] in {"store", "ray_bytes_fallback"}
        assert torch.equal(restored.batch["input_ids"], proto.batch["input_ids"])
        np.testing.assert_array_equal(restored.non_tensor_batch["domain"], proto.non_tensor_batch["domain"])
        np.testing.assert_array_equal(restored.non_tensor_batch["score"], proto.non_tensor_batch["score"])
        metrics = restored.meta_info["metrics"]
        assert metrics["transfer/backend/mooncake"] == 1.0
        assert metrics["transfer/backend/ray_optimized"] == 0.0
        assert metrics["transfer/mooncake_transport/store"] in {0.0, 1.0}
        assert metrics["transfer/mooncake_transport/ray_bytes_fallback"] in {0.0, 1.0}
        assert metrics["transfer/mooncake_rdma_requested"] in {0.0, 1.0}
        assert metrics["transfer/bytes/wire"] > 0
        assert metrics["transfer/throughput/put_mbps"] >= 0
        assert metrics["transfer/throughput/get_mbps"] >= 0
    finally:
        ray.shutdown()


def test_mooncake_store_adapter_reuses_registered_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConfig:
        local_hostname = "127.0.0.1"
        metadata_server = "P2PHANDSHAKE"
        global_segment_size = 1024 * 1024
        local_buffer_size = 1024 * 1024
        protocol = "rdma"
        device_name = "erdma_0"
        master_server_address = "127.0.0.1:50051"

        @classmethod
        def load_from_env(cls) -> "FakeConfig":
            return cls()

    class FakeStore:
        def __init__(self) -> None:
            self.register_calls: list[tuple[int, int]] = []
            self.unregister_calls: list[int] = []
            self.objects: dict[str, bytes] = {}

        def setup(self, config: dict[str, str]) -> int:
            return 0

        def register_buffer(self, buffer_ptr: int, size: int) -> int:
            self.register_calls.append((buffer_ptr, size))
            return 0

        def unregister_buffer(self, buffer_ptr: int) -> int:
            self.unregister_calls.append(buffer_ptr)
            return 0

        def put_from(self, key: str, buffer_ptr: int, size: int) -> int:
            self.objects[key] = ctypes.string_at(buffer_ptr, size)
            return 0

        def get_into(self, key: str, buffer_ptr: int, size: int) -> int:
            payload = self.objects[key]
            ctypes.memmove(buffer_ptr, payload, len(payload))
            return len(payload)

        def batch_put_from_multi_buffers(
            self, keys: list[str], ptrs_list: list[list[int]], sizes_list: list[list[int]]
        ) -> list[int]:
            for key, ptrs, sizes in zip(keys, ptrs_list, sizes_list):
                self.objects[key] = b"".join(ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes))
            return [0] * len(keys)

        def batch_get_into(self, keys: list[str], ptrs: list[int], sizes: list[int]) -> list[int]:
            results = []
            for key, ptr, size in zip(keys, ptrs, sizes):
                payload = self.objects[key]
                ctypes.memmove(ptr, payload, min(len(payload), size))
                results.append(len(payload))
            return results

        def get_into_ranges(
            self,
            buffer_ptrs: list[int],
            all_keys: list[list[str]],
            all_dst_offsets: list[list[list[int]]],
            all_src_offsets: list[list[list[int]]],
            all_sizes: list[list[list[int]]],
        ) -> list[list[list[int]]]:
            results = []
            for buffer_ptr, keys, dst_offsets, src_offsets, sizes in zip(
                buffer_ptrs, all_keys, all_dst_offsets, all_src_offsets, all_sizes
            ):
                buffer_results = []
                for key, dst_group, src_group, size_group in zip(keys, dst_offsets, src_offsets, sizes):
                    payload = self.objects[key]
                    group_results = []
                    for dst_offset, src_offset, size in zip(dst_group, src_group, size_group):
                        ctypes.memmove(buffer_ptr + dst_offset, payload[src_offset:src_offset + size], size)
                        group_results.append(size)
                    buffer_results.append(group_results)
                results.append(buffer_results)
            return results

    fake_store = FakeStore()
    fake_config_module = types.ModuleType("mooncake.mooncake_config")
    fake_config_module.MooncakeConfig = FakeConfig
    fake_store_module = types.ModuleType("mooncake.store")
    fake_store_module.MooncakeDistributedStore = lambda: fake_store
    monkeypatch.setenv("ROLL_MOONCAKE_ZEROCOPY_BUFFER_SIZE", "32")
    monkeypatch.setitem(sys.modules, "mooncake.mooncake_config", fake_config_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store_module)

    adapter = protocol._MooncakeStoreTransportAdapter()
    handle_a = adapter.put_bytes("a", b"abc")
    handle_b = adapter.put_bytes("b", b"defgh")

    assert adapter.get_bytes(handle_a) == b"abc"
    assert adapter.get_bytes(handle_b) == b"defgh"
    assert len(fake_store.register_calls) == 2
    assert fake_store.unregister_calls == []

    adapter.put_bytes("big", b"x" * 64)
    assert len(fake_store.register_calls) == 3
    assert len(fake_store.unregister_calls) == 1

    put_register_count_before_payload = adapter._put_register_count
    payload = {
        "protocol": "v1",
        "meta_bytes": b"meta",
        "bulk_buffer": None,
        "bulk_chunks": [b"abcdef", b"abc" + np.array([0, 1, 3], dtype=np.int64).tobytes(order="C")],
        "buffer_specs": [
            {"section": "batch", "key": "x", "dtype": "uint8", "shape": [3], "offset": 0, "nbytes": 3},
            {
                "section": "non_tensor_batch",
                "key": "s",
                "codec": "string_array",
                "dtype": "str",
                "shape": [2],
                "offset": 6,
                "nbytes": 3,
                "offsets_offset": 9,
                "offsets_nbytes": 24,
            },
        ],
        "transfer_stats": {},
    }
    handle = adapter.put_payload("payload", payload)
    restored = adapter.get_payload(handle)
    assert restored["meta_bytes"] == b"meta"
    assert bytes(restored["bulk_buffer"][:3]) == b"abc"
    assert protocol._decode_non_tensor_field(restored["bulk_buffer"], restored["buffer_specs"][1]).tolist() == ["a", "bc"]
    assert adapter._put_register_count == put_register_count_before_payload + 1

    adapter.close()
    assert len(fake_store.unregister_calls) == len(fake_store.register_calls)



def test_mooncake_transfer_payload_auto_scatter_respects_size_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    proto = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(8, dtype=torch.long).reshape(2, 4),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
        },
        meta_info={"rollout_transfer_profiling_enabled": True},
    )

    monkeypatch.setenv("ROLL_MOONCAKE_SCATTER_MAX_BYTES", "1000000")
    scatter_payload = proto.to_transfer_payload(stage="post_generate", protocol="v1", coalesce_bulk=None)
    assert scatter_payload["bulk_buffer"] is None
    assert scatter_payload["bulk_chunks"] is not None
    assert scatter_payload["transfer_stats"]["transfer/profile/bulk_coalesced/to_transfer_payload"] == 0.0

    monkeypatch.setenv("ROLL_MOONCAKE_SCATTER_MAX_BYTES", "1")
    coalesced_payload = proto.to_transfer_payload(stage="post_generate", protocol="v1", coalesce_bulk=None)
    assert coalesced_payload["bulk_buffer"] is not None
    assert coalesced_payload["bulk_chunks"] is None
    assert coalesced_payload["transfer_stats"]["transfer/profile/bulk_coalesced/to_transfer_payload"] == 1.0



def test_mooncake_rollout_transfer_backend_requires_v1_protocol() -> None:
    with pytest.raises(ValueError, match="mooncake transfer backend requires"):
        get_rollout_transfer_backend("mooncake", "legacy")
