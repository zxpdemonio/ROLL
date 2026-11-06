import base64

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from roll.configs import GeneratingArguments
from roll.configs.base_config import PPOConfig
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.execution_artifacts import (
    AttentionSelectionProfile,
    decode_attention_selection,
    pack_attention_selection,
)
from roll.utils.functionals import hold_aside_actor_replay_artifacts, postprocess_generate


@pytest.fixture
def profile():
    return AttentionSelectionProfile(
        profile_id="deepseek-v3.2-dsa-v1",
        model_family="deepseek_v3_2",
        backend_family="sglang",
        num_layers=2,
        topk=4,
    )


def make_vllm_raw_profile(profile, sources=None):
    if sources is None:
        sources = [
            {
                "layer_id": layer,
                "shape": [profile.topk],
                "semantic": "logical_token_id",
                "codec": "int32",
            }
            for layer in range(profile.num_layers)
        ]
    return {
        "kind": "indexer",
        "codec_version": 1,
        "model_identity": "test-model-identity",
        "storage_dtype": "int32",
        "sources": sources,
    }


def test_profile_estimates_top2048_payload_bytes():
    top2048 = AttentionSelectionProfile(
        profile_id="deepseek-v3.2-dsa-top2048-v1",
        model_family="deepseek_v3_2",
        backend_family="sglang",
        num_layers=6,
        topk=2048,
    )

    assert top2048.row_width == 12_288
    assert top2048.estimate_payload_bytes(num_tokens=128, batch_size=2) == 12_582_912
    assert top2048.estimate_packed_bytes(num_tokens=128, batch_size=2) == 12_582_920
    assert top2048.max_concurrent_requests(num_tokens=128, max_bytes=25_165_824) == 4


def test_profile_rejects_non_positive_capacity(profile):
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        profile.max_concurrent_requests(num_tokens=128, max_bytes=0)


def test_worker_rejects_non_positive_vllm_store_budget(profile):
    with pytest.raises(ValueError, match="attention_selection_max_bytes"):
        WorkerConfig(
            strategy_args=StrategyArguments(strategy_name="vllm"),
            attention_selection_profile=profile,
            attention_selection_max_bytes=0,
        )


def test_ppo_config_limits_request_concurrency_by_bytes(profile):
    tokens = 4
    per_request_bytes = profile.estimate_payload_bytes(tokens)
    actor_infer = WorkerConfig(
        strategy_args=StrategyArguments(strategy_name="sglang"),
        generating_args=GeneratingArguments(num_return_sequences=1),
        attention_selection_profile=profile,
    )

    config = PPOConfig(
        sequence_length=tokens + 1,
        rollout_batch_size=2,
        max_running_requests=128,
        execution_artifact_max_inflight_bytes=per_request_bytes * 2,
        execution_artifact_max_batch_bytes=profile.estimate_packed_bytes(tokens, batch_size=2),
        actor_infer=actor_infer,
    )

    assert config.max_running_requests == 2
    assert config.actor_train.attention_selection_profile == profile


def test_ppo_config_rejects_batch_over_byte_budget(profile):
    tokens = 4
    per_request_bytes = profile.estimate_payload_bytes(tokens)
    actor_infer = WorkerConfig(
        strategy_args=StrategyArguments(strategy_name="sglang"),
        generating_args=GeneratingArguments(num_return_sequences=1),
        attention_selection_profile=profile,
    )

    with pytest.raises(ValueError, match="cannot fit one rollout batch"):
        PPOConfig(
            sequence_length=tokens + 1,
            rollout_batch_size=2,
            execution_artifact_max_inflight_bytes=per_request_bytes * 2,
            execution_artifact_max_batch_bytes=per_request_bytes,
            actor_infer=actor_infer,
        )


def test_ppo_config_requires_inflight_byte_budget(profile):
    tokens = 4
    actor_infer = WorkerConfig(
        strategy_args=StrategyArguments(strategy_name="sglang"),
        generating_args=GeneratingArguments(num_return_sequences=1),
        attention_selection_profile=profile,
    )

    with pytest.raises(ValueError, match="execution_artifact_max_inflight_bytes"):
        PPOConfig(
            sequence_length=tokens + 1,
            rollout_batch_size=1,
            execution_artifact_max_batch_bytes=profile.estimate_packed_bytes(tokens),
            actor_infer=actor_infer,
        )


def test_profile_rejects_wrong_producer_backend(profile):
    with pytest.raises(ValueError, match="does not match producer vllm"):
        profile.validate_producer("vllm")


def test_profile_validates_vllm_raw_sources(profile):
    profile.validate_vllm_profile(make_vllm_raw_profile(profile))


def test_profile_rejects_vllm_codec_version_mismatch(profile):
    producer_profile = make_vllm_raw_profile(profile)
    producer_profile["codec_version"] = 2

    with pytest.raises(ValueError, match="unsupported kind or codec version"):
        profile.validate_vllm_profile(producer_profile)


def test_profile_rejects_vllm_compact_source_without_adapter(profile):
    sources = make_vllm_raw_profile(profile)["sources"]
    for source in sources:
        source["shape"] = [profile.topk // 4 + 2]
        source["codec"] = "qsa_blocks"

    with pytest.raises(ValueError, match="source-aware adapter"):
        profile.validate_vllm_profile(make_vllm_raw_profile(profile, sources))


def test_profile_rejects_duplicate_vllm_source_layers(profile):
    sources = [
        {
            "layer_id": 0,
            "shape": [profile.topk],
            "semantic": "logical_token_id",
            "codec": "int32",
        }
        for _ in range(profile.num_layers)
    ]

    with pytest.raises(ValueError, match="layer_ids must be unique"):
        profile.validate_vllm_profile(make_vllm_raw_profile(profile, sources))


def test_profile_rejects_out_of_order_vllm_source_layers(profile):
    sources = make_vllm_raw_profile(profile)["sources"]
    sources.reverse()

    with pytest.raises(ValueError, match="one source per model layer in layer order"):
        profile.validate_vllm_profile(make_vllm_raw_profile(profile, sources))


def test_profile_rejects_shared_vllm_source_without_training_adapter(profile):
    sources = make_vllm_raw_profile(profile)["sources"]
    sources[0]["consumer_layer_ids"] = [0, 1]

    with pytest.raises(ValueError, match="source-aware training adapter"):
        profile.validate_vllm_profile(make_vllm_raw_profile(profile, sources))


@pytest.mark.parametrize("shape", [[], [0], [-1], [True], ["4"]])
def test_profile_rejects_invalid_vllm_source_shape(profile, shape):
    sources = make_vllm_raw_profile(profile)["sources"]
    sources[0]["shape"] = shape

    with pytest.raises(ValueError, match="does not match raw topk"):
        profile.validate_vllm_profile(make_vllm_raw_profile(profile, sources))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", 2, "protocol_version"),
        ("num_layers", 0, "num_layers"),
        ("topk", 0, "topk"),
        ("stored_dtype", "int64", "stored_dtype"),
        ("coordinate_space", "unknown", "coordinate_space"),
    ],
)
def test_profile_rejects_invalid_contract(field, value, message):
    kwargs = {
        "profile_id": "profile",
        "model_family": "model",
        "backend_family": "backend",
        "num_layers": 2,
        "topk": 4,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        AttentionSelectionProfile(**kwargs)


def test_decode_attention_selection_from_base64(profile):
    values = np.arange(3 * profile.row_width, dtype=np.int32)
    encoded = base64.b64encode(values.tobytes()).decode("ascii")

    decoded = decode_attention_selection(encoded, profile, expected_rows=3)

    assert decoded.shape == (3, profile.num_layers, profile.topk)
    assert decoded.dtype == torch.int32
    assert decoded.is_contiguous()
    torch.testing.assert_close(decoded.flatten(), torch.from_numpy(values))


def test_decode_attention_selection_reuses_writable_int32_numpy_storage(profile):
    values = np.arange(3 * profile.row_width, dtype=np.int32).reshape(
        3, profile.num_layers, profile.topk
    )

    decoded = decode_attention_selection(values, profile, expected_rows=3)

    assert np.shares_memory(decoded.numpy(), values)


def test_decode_attention_selection_rejects_shape_mismatch(profile):
    values = torch.arange(profile.row_width, dtype=torch.int32)

    with pytest.raises(ValueError, match="rows 1 != expected 2"):
        decode_attention_selection(values, profile, expected_rows=2)


def test_pack_attention_selection_adds_padding_and_valid_counts(profile):
    first = torch.arange(3 * profile.row_width, dtype=torch.int64).reshape(3, profile.num_layers, profile.topk)
    second = torch.arange(profile.row_width, dtype=torch.int32).reshape(1, profile.num_layers, profile.topk)

    packed, valid_counts = pack_attention_selection(
        [first, second],
        valid_counts=[3, 1],
        sequence_length=5,
        profile=profile,
    )

    assert packed.shape == (2, 4, profile.num_layers, profile.topk)
    assert packed.dtype == torch.int32
    assert packed.is_contiguous()
    torch.testing.assert_close(packed[0, :3], first.to(torch.int32))
    torch.testing.assert_close(packed[1, 0], second[0])
    assert torch.all(packed[0, 3] == profile.padding_value)
    assert torch.all(packed[1, 1:] == profile.padding_value)
    torch.testing.assert_close(valid_counts, torch.tensor([3, 1], dtype=torch.int32))


def test_postprocess_generate_attaches_selection_to_dataproto(profile):
    prompts = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[0, 10, 11], [20, 21, 22]]),
                "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
                "position_ids": torch.tensor([[0, 0, 1], [0, 1, 2]]),
            },
            batch_size=[2],
        )
    )
    output = torch.tensor([[0, 10, 11, 30, 2], [20, 21, 22, 40, 0]])
    selections = [
        torch.arange(3 * profile.row_width, dtype=torch.int32).reshape(3, profile.num_layers, profile.topk),
        torch.arange(3 * profile.row_width, dtype=torch.int32).reshape(3, profile.num_layers, profile.topk),
    ]

    result = postprocess_generate(
        prompts=prompts,
        output=output,
        num_return_sequences=1,
        sequence_length=5,
        eos_token_id=2,
        pad_token_id=0,
        attention_selection=selections,
        attention_selection_profile=profile,
    )

    assert result.batch["attention_selection"].shape == (2, 4, profile.num_layers, profile.topk)
    torch.testing.assert_close(
        result.batch["attention_selection_valid_count"], torch.tensor([3, 3], dtype=torch.int32)
    )
    assert torch.all(result.batch["attention_selection"][:, 3] == profile.padding_value)


def test_hold_aside_actor_replay_artifacts_realigns_local_tensors():
    batch = DataProto(
        batch=TensorDict(
            {
                "tokens": torch.tensor([[10], [20]]),
                "routed_experts": torch.tensor([[1], [2]]),
                "attention_selection": torch.tensor([[[[11]]], [[[22]]]], dtype=torch.int32),
                "attention_selection_valid_count": torch.tensor([1, 1], dtype=torch.int32),
            },
            batch_size=[2],
        )
    )

    with hold_aside_actor_replay_artifacts(batch, realign_indices=torch.tensor([1, 0])):
        assert "routed_experts" not in batch.batch
        assert "attention_selection" not in batch.batch
        assert "attention_selection_valid_count" not in batch.batch

    torch.testing.assert_close(batch.batch["routed_experts"], torch.tensor([[2], [1]]))
    torch.testing.assert_close(
        batch.batch["attention_selection"],
        torch.tensor([[[[22]]], [[[11]]]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        batch.batch["attention_selection_valid_count"],
        torch.tensor([1, 1], dtype=torch.int32),
    )


def test_hold_aside_actor_replay_artifacts_restores_after_exception():
    batch = DataProto(
        batch=TensorDict(
            {"attention_selection": torch.tensor([[[[11]]]], dtype=torch.int32)},
            batch_size=[1],
        )
    )

    with pytest.raises(RuntimeError, match="reference failed"):
        with hold_aside_actor_replay_artifacts(batch):
            raise RuntimeError("reference failed")

    torch.testing.assert_close(
        batch.batch["attention_selection"],
        torch.tensor([[[[11]]]], dtype=torch.int32),
    )


def test_hold_aside_actor_replay_artifacts_ignores_realign_key_when_empty():
    batch = DataProto(
        batch=TensorDict({"tokens": torch.tensor([[10]])}, batch_size=[1]),
    )

    with hold_aside_actor_replay_artifacts(batch, realign_key="missing"):
        pass

    torch.testing.assert_close(batch.batch["tokens"], torch.tensor([[10]]))
