import sys
import types

import pytest
import torch

from roll.distributed.scheduler.generate_scheduler import expand_requests
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterClient


def test_expand_requests_with_mm_dedup_shares_payload_and_copies_prompt_state() -> None:
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        },
        non_tensors={
            "multi_modal_data": [{
                "prompt_token_ids": [11, 22, 33],
                "multi_modal_data": {"image": ["image-blob"]},
            }],
        },
        meta_info={"generation_config": {"num_return_sequences": 4, "max_new_tokens": 8}},
    )

    expanded = expand_requests(
        data=data,
        num_return_sequences=4,
        is_num_return_sequences_expand=True,
        enable_mm_dedup=True,
    )

    assert len(expanded) == 4
    ref_ids: list[str] = []
    for req in expanded:
        mm_entry = req.non_tensor_batch["multi_modal_data"][0]
        assert "multi_modal_data" not in mm_entry
        assert mm_entry["prompt_token_ids"] == [11, 22, 33]
        ref_ids.append(mm_entry["mm_ref_id"])
        assert req.meta_info["generation_config"]["num_return_sequences"] == 1
        assert req.meta_info["mm_context"][mm_entry["mm_ref_id"]] == {"image": ["image-blob"]}

    expanded[0].non_tensor_batch["multi_modal_data"][0]["prompt_token_ids"].append(44)

    assert len(set(ref_ids)) == 1
    assert expanded[1].non_tensor_batch["multi_modal_data"][0]["prompt_token_ids"] == [11, 22, 33]
    assert data.non_tensor_batch["multi_modal_data"][0]["multi_modal_data"] == {"image": ["image-blob"]}


def test_expand_requests_without_mm_dedup_keeps_legacy_multimodal_payload() -> None:
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        non_tensors={
            "multi_modal_data": [{
                "prompt_token_ids": [1, 2],
                "multi_modal_data": {"image": ["legacy-image"]},
            }],
        },
        meta_info={"generation_config": {"num_return_sequences": 2, "max_new_tokens": 4}},
    )

    expanded = expand_requests(
        data=data,
        num_return_sequences=2,
        is_num_return_sequences_expand=True,
        enable_mm_dedup=False,
    )

    assert len(expanded) == 2
    for req in expanded:
        assert req.non_tensor_batch["multi_modal_data"][0]["multi_modal_data"] == {"image": ["legacy-image"]}
        assert "mm_context" not in req.meta_info


def test_router_preprocess_generate_supports_mm_ref_resolution() -> None:
    vllm_strategy_stub = types.ModuleType("roll.distributed.strategy.vllm_strategy")
    vllm_strategy_stub.create_sampling_params_for_vllm = lambda *args, **kwargs: {"stub": True}
    sys.modules["roll.distributed.strategy.vllm_strategy"] = vllm_strategy_stub

    client = RouterClient(
        proxy=None,
        meta={"strategy_name": "vllm", "eos_token_id": 7, "pad_token_id": 0},
    )
    req = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "multi_modal_data": [{
                "mm_ref_id": "mm-1",
                "prompt_token_ids": [1, 2, 3],
            }],
        },
        meta_info={
            "generation_config": {"num_return_sequences": 1, "max_new_tokens": 8},
            "mm_context": {"mm-1": {"image": ["shared-image"]}},
        },
    )

    payload, request_id = client._preprocess_generate(req, request_id="rid-1")

    assert request_id == "rid-1"
    assert payload["multi_modal_data"] == {
        "prompt_token_ids": [1, 2, 3],
        "multi_modal_data": {"image": ["shared-image"]},
    }
    assert payload["sampling_params"] == {"stub": True}


def test_router_preprocess_generate_rejects_missing_mm_ref_context() -> None:
    client = RouterClient(
        proxy=None,
        meta={"strategy_name": "vllm", "eos_token_id": 7, "pad_token_id": 0},
    )
    req = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        non_tensors={
            "multi_modal_data": [{
                "mm_ref_id": "missing",
                "prompt_token_ids": [1, 2],
            }],
        },
        meta_info={"generation_config": {"num_return_sequences": 1, "max_new_tokens": 4}},
    )

    with pytest.raises(ValueError, match="missing multimodal context"):
        client._preprocess_generate(req, request_id="rid-2")
