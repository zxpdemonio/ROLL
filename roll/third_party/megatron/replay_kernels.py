"""Fused replay layout conversion from compact, directly received GPU indices."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_indices(
    source,
    counts,
    output,
    stride_b: tl.constexpr,
    stride_t: tl.constexpr,
    stride_k: tl.constexpr,
    count_stride: tl.constexpr,
    stored_length: tl.constexpr,
    query_length: tl.constexpr,
    topk: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch, query = row // query_length, row % query_length
    slots = tl.arange(0, block)
    valid_row = (query < stored_length) & (query < tl.load(counts + batch * count_stride))
    values = tl.load(
        source + batch * stride_b + query * stride_t + slots * stride_k,
        valid_row & (slots < topk),
        other=-1,
    ).to(tl.int64)
    first = tl.min(tl.where((slots < topk) & (values >= 0), slots, block), 0)
    first_value = tl.sum(tl.where(slots == first, values, 0), 0)
    normalized = tl.where(values >= 0, values, first_value)
    normalized = tl.where(valid_row, normalized, 0)
    tl.store(output + row * topk + slots, normalized, slots < topk)


def prepare_attention_indices(selection, valid_counts, query_length):
    batch, stored_length, topk = selection.shape
    result = torch.empty((batch, query_length, topk), dtype=torch.int64, device=selection.device)
    if not result.numel():
        return result
    _attention_indices[(batch * query_length,)](
        selection,
        valid_counts,
        result,
        *selection.stride(),
        valid_counts.stride(0),
        stored_length,
        query_length,
        topk,
        triton.next_power_of_2(topk),
    )
    return result


@triton.jit
def _router_indices(
    source,
    output,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_l: tl.constexpr,
    stride_k: tl.constexpr,
    batch: tl.constexpr,
    sequence: tl.constexpr,
    layers: tl.constexpr,
    topk: tl.constexpr,
    experts: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    slot = index % topk
    token = index // topk % (sequence * batch)
    layer = index // (topk * sequence * batch)
    sample, position = token % batch, token // batch
    values = tl.load(
        source + sample * stride_b + position * stride_s + layer * stride_l + slot * stride_k,
        index < layers * sequence * batch * topk,
        other=0,
    ).to(tl.int64)
    values = tl.minimum(tl.maximum(values, 0), experts - 1)
    tl.store(output + index, values, index < layers * sequence * batch * topk)


def prepare_router_indices(selection, num_experts):
    batch, sequence, layers, topk = selection.shape
    result = torch.empty((layers, sequence * batch, topk), dtype=torch.int64, device=selection.device)
    if not result.numel():
        return result
    _router_indices[(triton.cdiv(result.numel(), 256),)](
        selection,
        result,
        *selection.stride(),
        batch,
        sequence,
        layers,
        topk,
        num_experts,
        256,
    )
    return result
