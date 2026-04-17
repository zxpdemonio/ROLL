import ray
import torch
import numpy as np

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
        assert metrics["transfer/backend"] == "ray_optimized"
        assert metrics["transfer/protocol"] == "v1"
        assert metrics["transfer/stage"] == "post_generate"
        assert metrics["transfer/time/serialize"] >= 0
        assert metrics["transfer/time/put"] >= 0
        assert metrics["transfer/time/get"] >= 0
        assert metrics["transfer/time/deserialize"] >= 0
        assert metrics["transfer/bytes/total"] > 0
    finally:
        ray.shutdown()
