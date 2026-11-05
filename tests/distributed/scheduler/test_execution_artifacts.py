import base64

import numpy as np
import pytest
import torch
from tensordict import TensorDict

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


def test_profile_rejects_wrong_producer_backend(profile):
    with pytest.raises(ValueError, match="does not match producer vllm"):
        profile.validate_producer("vllm")


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
