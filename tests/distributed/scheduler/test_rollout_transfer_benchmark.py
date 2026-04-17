import time

import numpy as np
import pytest
import ray
import torch

from roll.distributed.scheduler.generate_scheduler import expand_requests
from roll.distributed.scheduler.protocol import DataProto, get_rollout_transfer_backend, materialize_rollout_transfer


def _make_protocol_benchmark_proto(batch_size: int = 32, prompt_length: int = 128, response_length: int = 256) -> DataProto:
    total_length = prompt_length + response_length
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(batch_size * total_length, dtype=torch.long).reshape(batch_size, total_length),
            "attention_mask": torch.ones((batch_size, total_length), dtype=torch.long),
            "responses": torch.arange(batch_size * response_length, dtype=torch.long).reshape(batch_size, response_length),
            "response_mask": torch.ones((batch_size, response_length), dtype=torch.long),
            "prompt_mask": torch.ones((batch_size, total_length), dtype=torch.long),
            "infer_logprobs": torch.randn(batch_size, response_length, dtype=torch.float32),
        },
        non_tensors={
            "domain": ["math"] * batch_size,
            "tag": ["benchmark"] * batch_size,
            "sample_uuid": [f"sample-{idx}" for idx in range(batch_size)],
        },
        meta_info={
            "rollout_transfer_metrics_enabled": True,
            "rollout_transfer_profiling_enabled": True,
        },
    )


def _make_multimodal_request(payload_bytes: int = 4096) -> DataProto:
    payload = {"image": [b"x" * payload_bytes]}
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((1, 16), dtype=torch.long),
            "attention_mask": torch.ones((1, 16), dtype=torch.long),
        },
        non_tensors={
            "multi_modal_data": [{
                "prompt_token_ids": list(range(16)),
                "multi_modal_data": payload,
            }],
        },
        meta_info={
            "generation_config": {"num_return_sequences": 8, "max_new_tokens": 32},
            "rollout_transfer_profiling_enabled": True,
        },
    )


def test_protocol_benchmark_reports_v1_metrics() -> None:
    proto = _make_protocol_benchmark_proto()

    start = time.time()
    payload = proto.to_transfer_payload(stage="post_generate", protocol="v1")
    restored = DataProto.from_transfer_payload(payload)
    elapsed = time.time() - start

    assert elapsed >= 0
    assert restored.batch.batch_size == proto.batch.batch_size
    stats = payload["transfer_stats"]
    assert stats["transfer/bytes/total"] > 0
    assert stats["transfer/bytes/tensor"] > 0
    assert stats["transfer/sample_count"] == len(proto)
    assert stats["transfer/sequence_length/max"] == proto.batch["input_ids"].shape[-1]


def test_multimodal_dedup_benchmark_reduces_inline_payload_objects() -> None:
    request = _make_multimodal_request()

    expanded_legacy = expand_requests(
        data=request.clone(),
        num_return_sequences=8,
        is_num_return_sequences_expand=True,
        enable_mm_dedup=False,
    )
    expanded_dedup = expand_requests(
        data=request.clone(),
        num_return_sequences=8,
        is_num_return_sequences_expand=True,
        enable_mm_dedup=True,
    )

    legacy_inline = sum(
        1
        for req in expanded_legacy
        if req.non_tensor_batch["multi_modal_data"][0].get("multi_modal_data") is not None
    )
    dedup_inline = sum(
        1
        for req in expanded_dedup
        if req.non_tensor_batch["multi_modal_data"][0].get("multi_modal_data") is not None
    )
    ref_ids = {req.non_tensor_batch["multi_modal_data"][0]["mm_ref_id"] for req in expanded_dedup}

    assert legacy_inline == 8
    assert dedup_inline == 0
    assert len(ref_ids) == 1
    metrics = expanded_dedup[0].meta_info["metrics"]
    assert metrics["transfer/profile/mm_ref_count/expand_requests"] == 1.0
    assert metrics["transfer/profile/mm_dup_object_count/expand_requests"] == 0.0


@pytest.mark.skipif(not ray.is_initialized() and False, reason="placeholder to keep pytest marker local")
def test_ray_backend_benchmark_reports_round_trip_profile_metrics() -> None:
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        proto = _make_protocol_benchmark_proto(batch_size=16)
        backend = get_rollout_transfer_backend("ray_optimized", "v1")
        handle = backend.put(proto, stage="post_generate")
        restored = materialize_rollout_transfer(handle, backend_name="ray_optimized", protocol="v1")

        metrics = restored.meta_info["metrics"]
        assert metrics["transfer/time/serialize"] >= 0
        assert metrics["transfer/time/put"] >= 0
        assert metrics["transfer/time/get"] >= 0
        assert metrics["transfer/time/deserialize"] >= 0
        assert metrics["transfer/profile/temp_buffer_count/to_transfer_payload"] >= 1.0
        assert metrics["transfer/profile/peak_rss_gb/to_transfer_payload"] > 0
        assert metrics["transfer/profile/peak_rss_gb/backend_get"] > 0
    finally:
        ray.shutdown()
