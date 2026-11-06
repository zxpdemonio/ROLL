"""DeepSeek sparse-attention replay support for Megatron Core.

ROLL owns the replay lifecycle and layout conversion.  Megatron continues to
compute indexer scores (and therefore its indexer loss), but the discrete top-k
choice is replaced with the indices captured during rollout.

This compatibility patch targets the DSA implementation shipped with MCore
0.16.  It is intentionally narrow and fails closed when the DSA call path no
longer goes through ``DSAIndexer.forward_with_scores``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import torch


class SparseAttentionReplayAction(Enum):
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


@dataclass(frozen=True)
class SparseAttentionReplayTarget:
    """One local DSA layer's int32 rollout view and its row contract."""

    selection: torch.Tensor
    valid_counts: torch.Tensor
    query_length: int

    def materialize(self) -> torch.Tensor:
        return prepare_sparse_attention_layer(
            self.selection,
            self.valid_counts,
            query_length=self.query_length,
        )


@dataclass
class SparseAttentionReplayState:
    """Per-indexer state that survives activation-checkpoint recomputation."""

    action: SparseAttentionReplayAction | None = None
    target: SparseAttentionReplayTarget | None = None
    backward_targets: list[SparseAttentionReplayTarget] = field(default_factory=list)

    def set_target(self, target: SparseAttentionReplayTarget) -> None:
        self.target = target
        self.backward_targets.append(target)

    def get_indices(self) -> torch.Tensor:
        if self.action == SparseAttentionReplayAction.REPLAY_FORWARD:
            if self.target is None:
                raise RuntimeError("sparse-attention replay target is not set")
            return self.target.materialize()
        if self.action == SparseAttentionReplayAction.REPLAY_BACKWARD:
            if not self.backward_targets:
                raise RuntimeError("sparse-attention replay backward queue is empty")
            return self.backward_targets.pop(0).materialize()
        raise RuntimeError("sparse-attention replay action is not set")

    def clear(self) -> None:
        self.action = None
        self.target = None
        self.backward_targets.clear()


def _compute_scores_before_topk(indexer, dsa_module, x, qr, mask, packed_seq_params):
    """Run the MCore DSA indexer up to, but not including, ``topk``."""

    if hasattr(indexer, "forward_before_topk"):
        q, k, weights = indexer.forward_before_topk(x, qr, packed_seq_params)
        score_fn = getattr(indexer, "_compute_index_scores", None)
        if score_fn is None:
            score_fn = dsa_module._compute_index_scores
        index_scores = score_fn(q, weights, k)
    else:
        if packed_seq_params is not None:
            raise NotImplementedError(
                "sparse-attention replay does not support Megatron packed sequences yet"
            )

        rotary_seq_len = indexer.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, indexer.config, packed_seq_params
        )
        if indexer.config.rope_type == "rope":
            rotary_pos_emb = indexer.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = indexer.rotary_pos_emb(
                rotary_seq_len, packed_seq=False
            )

        if indexer.config.sequence_parallel and indexer.pg_collection.tp.size() > 1:
            x = dsa_module.gather_from_sequence_parallel_region(
                x, group=indexer.pg_collection.tp
            )
            qr = dsa_module.gather_from_sequence_parallel_region(
                qr, group=indexer.pg_collection.tp
            )

        seqlen, batch_size, _ = x.size()
        q, _ = indexer.linear_wq_b(qr)
        q = q.reshape(
            seqlen, batch_size, indexer.index_n_heads, indexer.index_head_dim
        )
        q = indexer._apply_rope(q, rotary_pos_emb, mscale)

        k, _ = indexer.linear_wk(x)
        k = indexer.k_norm(k)
        k = k.reshape(seqlen, batch_size, 1, indexer.index_head_dim)
        k = indexer._apply_rope(k, rotary_pos_emb, mscale)
        k = k.reshape(seqlen, batch_size, indexer.index_head_dim)

        q = dsa_module.rotate_activation(q)
        k = dsa_module.rotate_activation(k)
        weights, _ = indexer.linear_weights_proj(x)
        weights = (
            weights
            * (indexer.index_n_heads**-0.5)
            * indexer.softmax_scale
        )
        index_scores = indexer._compute_index_scores(q, weights, k)

    if mask is not None:
        if mask.dtype != index_scores.dtype:
            raise TypeError("DSA replay mask dtype must match index scores dtype")
        index_scores = index_scores + mask
    return index_scores


def patch_megatron_dsa_replay(dsa_module=None) -> None:
    """Patch MCore DSA so replay mode bypasses its discrete ``topk`` call."""

    if dsa_module is None:
        from megatron.core.transformer.experimental_attention_variant import (
            dsa as dsa_module,
        )

    indexer_cls = dsa_module.DSAIndexer
    if getattr(indexer_cls, "_roll_sparse_attention_replay_patched", False):
        return

    attention_forward = dsa_module.DSAttention.forward
    if "forward_with_scores" not in attention_forward.__code__.co_names:
        raise RuntimeError(
            "unsupported Megatron DSA implementation: DSAttention.forward no longer "
            "calls DSAIndexer.forward_with_scores"
        )

    original_forward_with_scores = indexer_cls.forward_with_scores

    def replay_forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: torch.Tensor | None = None,
        packed_seq_params=None,
    ):
        state = getattr(self, "_roll_sparse_attention_replay", None)
        if state is None or state.action is None:
            return original_forward_with_scores(
                self, x, qr, mask=mask, packed_seq_params=packed_seq_params
            )

        index_scores = _compute_scores_before_topk(
            self, dsa_module, x, qr, mask, packed_seq_params
        )
        replay_indices = state.get_indices().to(device=index_scores.device)
        expected_prefix = tuple(index_scores.shape[:2])
        if replay_indices.ndim != 3 or tuple(replay_indices.shape[:2]) != expected_prefix:
            raise ValueError(
                "sparse-attention replay shape mismatch: "
                f"indices={tuple(replay_indices.shape)}, scores={tuple(index_scores.shape)}"
            )
        if replay_indices.dtype != torch.int64:
            replay_indices = replay_indices.to(torch.int64)
        return index_scores, replay_indices

    indexer_cls._roll_original_forward_with_scores = original_forward_with_scores
    indexer_cls.forward_with_scores = replay_forward_with_scores
    indexer_cls._roll_sparse_attention_replay_patched = True


def _iter_dsa_modules(model) -> Iterable:
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

    for module in model.modules():
        if isinstance(module, DSAttention):
            yield module


def initialize_sparse_attention_replay(model) -> list[int]:
    """Attach replay state to every DSA indexer and return global layer ids."""

    layer_ids = []
    for attention in _iter_dsa_modules(model):
        layer_id = attention.layer_number - 1
        if layer_id < 0:
            raise ValueError(f"invalid Megatron DSA layer number {attention.layer_number}")
        attention.indexer._roll_sparse_attention_replay = SparseAttentionReplayState()
        layer_ids.append(layer_id)
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError(f"duplicate Megatron DSA layer ids on one model chunk: {layer_ids}")
    return layer_ids


def _iter_replay_states(model) -> Iterable[tuple[int, SparseAttentionReplayState]]:
    for attention in _iter_dsa_modules(model):
        state = getattr(attention.indexer, "_roll_sparse_attention_replay", None)
        if state is not None:
            yield attention.layer_number - 1, state


def set_sparse_attention_replay_action(
    model, action: SparseAttentionReplayAction
) -> None:
    for _, state in _iter_replay_states(model):
        state.action = action


def set_sparse_attention_replay_data(
    model,
    selection: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    query_length: int,
) -> None:
    """Bind raw ``[B, T, global_layers, K]`` views to local DSA layers."""

    if selection.ndim != 4:
        raise ValueError(
            f"attention selection must have shape [B, S, L, K], got {selection.shape}"
        )
    for layer_id, state in _iter_replay_states(model):
        if layer_id >= selection.shape[2]:
            raise ValueError(
                f"DSA layer {layer_id} exceeds selection layer count {selection.shape[2]}"
            )
        state.set_target(
            SparseAttentionReplayTarget(
                selection=selection[:, :, layer_id, :],
                valid_counts=valid_counts,
                query_length=query_length,
            )
        )


def clear_sparse_attention_replay(model) -> None:
    for _, state in _iter_replay_states(model):
        state.clear()


def prepare_sparse_attention_layer(
    selection: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    query_length: int,
) -> torch.Tensor:
    """Materialize one local layer for a Megatron DSA forward.

    vLLM uses ``-1`` for unused top-k slots.  Megatron's scatter-based DSA
    kernel cannot consume negative indices, so each sentinel is replaced by a
    duplicate of the row's first valid key.  Duplicates do not add an attention
    edge and therefore preserve the sparse path.
    """

    batch_size, stored_length, topk = selection.shape

    if selection.is_cuda:
        from roll.third_party.megatron.replay_kernels import prepare_attention_indices

        if valid_counts.device != selection.device:
            raise ValueError("CUDA replay counts and selections must be on the same device")
        return prepare_attention_indices(selection, valid_counts, query_length)

    result = torch.zeros(
        (batch_size, query_length, topk),
        dtype=torch.int64,
        device=selection.device,
    )
    source = selection.to(torch.int64)
    valid_slots = source >= 0
    first_valid_slot = valid_slots.to(torch.int64).argmax(dim=-1, keepdim=True)
    first_valid = source.gather(-1, first_valid_slot)
    normalized = torch.where(valid_slots, source, first_valid)

    row_positions = torch.arange(stored_length, device=source.device)
    valid_rows = row_positions.unsqueeze(0) < valid_counts.to(source.device).unsqueeze(1)
    normalized = torch.where(
        valid_rows[:, :, None], normalized, torch.zeros_like(normalized)
    )
    result[:, :stored_length] = normalized

    return result.contiguous()


def validate_sparse_attention_replay(
    selection: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    query_length: int,
    num_layers: int,
    topk: int,
) -> None:
    """Validate one batch before entering Megatron's pipeline schedule."""

    if selection.ndim != 4:
        raise ValueError(
            f"attention selection must have shape [B, T, L, K], got {selection.shape}"
        )
    if selection.dtype != torch.int32:
        raise TypeError(
            f"attention selection must use int32 storage, got {selection.dtype}"
        )
    batch_size, stored_length, stored_layers, stored_topk = selection.shape
    if (stored_layers, stored_topk) != (num_layers, topk):
        raise ValueError(
            "attention selection profile mismatch: "
            f"tensor L/K={(stored_layers, stored_topk)}, profile L/K={(num_layers, topk)}"
        )
    if valid_counts.shape != (batch_size,):
        raise ValueError(
            f"attention selection valid counts must have shape {(batch_size,)}"
        )
    if valid_counts.dtype != torch.int32:
        raise TypeError(
            f"attention selection valid counts must use int32, got {valid_counts.dtype}"
        )
    if query_length <= 0 or stored_length > query_length:
        raise ValueError(
            f"invalid attention replay lengths: stored={stored_length}, query={query_length}"
        )

    counts = valid_counts.detach().to(device="cpu", dtype=torch.int64).tolist()
    source = selection.detach()
    for batch_index, valid_count in enumerate(counts):
        if valid_count < 0 or valid_count > stored_length or valid_count > query_length:
            raise ValueError(
                f"invalid attention selection valid count {valid_count} for batch {batch_index}"
            )
        if valid_count == 0:
            continue
        rows = source[batch_index, :valid_count]
        valid_slots = rows >= 0
        if not valid_slots.any(dim=-1).all().item():
            raise ValueError(
                f"attention selection contains an all-padding row in batch {batch_index}"
            )
        first_valid_slot = valid_slots.to(torch.int64).argmax(dim=-1, keepdim=True)
        first_valid = rows.gather(-1, first_valid_slot)
        rows = torch.where(valid_slots, rows, first_valid)
        positions = torch.arange(valid_count, device=rows.device).view(-1, 1, 1)
        if (rows > positions).any().item():
            raise ValueError(
                f"attention selection contains a non-causal key in batch {batch_index}"
            )
        if (rows >= query_length).any().item():
            raise ValueError(
                f"attention selection contains an out-of-range key in batch {batch_index}"
            )
