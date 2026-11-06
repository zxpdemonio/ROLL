import pytest
import torch

from roll.third_party.megatron.sparse_attention_replay import (
    prepare_sparse_attention_layer,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA replay kernels require a GPU")


@pytest.mark.parametrize("topk", [1, 3, 2048])
def test_fused_attention_matches_reference_with_strided_layer_views(topk):
    source = torch.full((2, 3, 2, topk), -1, dtype=torch.int32)
    for row in range(3):
        source[:, row, :, 0] = row
        source[:, row, :, 1::2] = 0
    counts = torch.tensor([3, 0, 2, 0], dtype=torch.int32)[::2]
    expected = prepare_sparse_attention_layer(source[:, :, 1, :], counts, query_length=4)
    device_counts = torch.tensor([3, 0, 2, 0], dtype=torch.int32, device="cuda")[::2]
    actual = prepare_sparse_attention_layer(source.cuda()[:, :, 1, :], device_counts, query_length=4)
    assert actual.dtype == torch.int64
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected)


def test_attention_large_strides_compile_without_32bit_pointer_offsets():
    from roll.third_party.megatron.replay_kernels import _attention_indices

    source = torch.zeros((1,), dtype=torch.int32, device="cuda")
    counts = torch.ones((1,), dtype=torch.int32, device="cuda")
    output = torch.empty((1,), dtype=torch.int64, device="cuda")
    # Compile only: never allocate or access the multi-billion-element span.
    compiled = _attention_indices.warmup(
        source,
        counts,
        output,
        1 << 34,
        1 << 30,
        1,
        1,
        8,
        8,
        8,
        8,
        grid=(8,),
    )
    assert "1073741824 : i64" in compiled.asm["ttir"]


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.int64])
def test_fused_router_matches_permute_cast_and_clamp(dtype):
    from roll.third_party.megatron.replay_kernels import prepare_router_indices

    source = torch.arange(2 * 3 * 4 * 16).view(2, 3, 4, 16).to(dtype)[..., ::2]
    expected = source.permute(1, 0, 2, 3).contiguous()
    expected = expected.reshape(-1, 4, 8).permute(1, 0, 2).to(torch.int64).clamp(0, 15)
    actual = prepare_router_indices(source.cuda(), 16)
    torch.testing.assert_close(actual.cpu(), expected)
