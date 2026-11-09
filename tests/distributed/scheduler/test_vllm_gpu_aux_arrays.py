"""Exercise the payload adapter on CPU without importing a GPU vLLM build."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_path = Path(__file__).parents[3] / "roll/third_party/vllm/aux_output_gpu.py"
_spec = importlib.util.spec_from_file_location("_roll_gpu_aux_arrays", _path)
gpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.int32])
def test_published_block_owns_storage_before_pool_reuse(dtype):
    original = torch.arange(24, dtype=torch.int32).reshape(3, 2, 4).to(dtype)
    wrapper = gpu.GPURows(original)
    payload = wrapper.tobytes()
    expected = original.clone()
    original.zero_()
    restored = gpu.GPUArrayOps().frombuffer(payload, np.dtype(str(dtype).removeprefix("torch."))).reshape(3, 2, 4)
    assert torch.equal(restored.tensor, expected)
    assert len(payload) == original.nbytes
    assert payload.tensor.data_ptr() != original.data_ptr()
    with pytest.raises(TypeError, match="host"):
        np.asarray(payload)


def test_byte_record_split_preserves_uint16_and_indexer_rows():
    ops = gpu.GPUArrayOps()
    experts = torch.tensor([[[0, 511]], [[256, 510]]], dtype=torch.uint16)
    indexer = torch.tensor([[0, -1, -1], [1, 0, -1]], dtype=torch.int32)
    combined = ops.concatenate(
        [
            gpu.GPURows(experts).view(np.uint8).reshape(2, -1),
            gpu.GPURows(indexer).view(np.uint8),
        ],
        axis=1,
    )
    routed = combined[:, :4].copy().view(np.uint16).reshape(2, 1, 2)
    selected = combined[:, 4:].copy().view(np.int32)
    assert torch.equal(routed.tensor, experts)
    assert torch.equal(selected.tensor, indexer)


def test_numpy_behavior_stays_default_and_mixed_payload_fails():
    ops = gpu.GPUArrayOps()
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert isinstance(ops.empty((2, 3), np.uint8), np.ndarray)
    np.testing.assert_array_equal(ops.concatenate([a, a]), np.concatenate([a, a]))
    with pytest.raises(TypeError, match="CPU and GPU"):
        ops.concatenate([a, gpu.GPURows(torch.from_numpy(a))])
