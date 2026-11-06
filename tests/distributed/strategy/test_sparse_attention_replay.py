from types import SimpleNamespace

import pytest
import torch

from roll.third_party.megatron.sparse_attention_replay import (
    SparseAttentionReplayAction,
    SparseAttentionReplayState,
    SparseAttentionReplayTarget,
    patch_megatron_dsa_replay,
    prepare_sparse_attention_layer,
    validate_sparse_attention_replay,
)


def test_prepare_replay_replaces_padding_with_duplicate_without_new_edge():
    selection = torch.tensor(
        [
            [
                [[0, -1, -1, -1]],
                [[1, 0, -1, -1]],
                [[2, 1, 0, -1]],
            ]
        ],
        dtype=torch.int32,
    )

    validate_sparse_attention_replay(
        selection,
        torch.tensor([3], dtype=torch.int32),
        query_length=4,
        num_layers=1,
        topk=4,
    )
    result = prepare_sparse_attention_layer(
        selection[:, :, 0, :],
        torch.tensor([3], dtype=torch.int32),
        query_length=4,
    )

    assert result.dtype == torch.int64
    assert result.shape == (1, 4, 4)
    assert result[0, 0].tolist() == [0, 0, 0, 0]
    assert result[0, 1].tolist() == [1, 0, 1, 1]
    assert result[0, 2].tolist() == [2, 1, 0, 2]
    assert result[0, 3].tolist() == [0, 0, 0, 0]


def test_prepare_replay_rejects_non_causal_selection():
    selection = torch.tensor([[[[1, -1]]]], dtype=torch.int32)

    with pytest.raises(ValueError, match="non-causal"):
        validate_sparse_attention_replay(
            selection,
            torch.tensor([1], dtype=torch.int32),
            query_length=2,
            num_layers=1,
            topk=2,
        )


def test_prepare_replay_rejects_all_padding_valid_row():
    selection = torch.full((1, 1, 1, 2), -1, dtype=torch.int32)

    with pytest.raises(ValueError, match="all-padding"):
        validate_sparse_attention_replay(
            selection,
            torch.tensor([1], dtype=torch.int32),
            query_length=2,
            num_layers=1,
            topk=2,
        )


def test_replay_validation_rejects_non_int32_storage():
    with pytest.raises(TypeError, match="must use int32 storage"):
        validate_sparse_attention_replay(
            torch.zeros((1, 1, 1, 1), dtype=torch.int64),
            torch.tensor([1], dtype=torch.int32),
            query_length=2,
            num_layers=1,
            topk=1,
        )

    with pytest.raises(TypeError, match="valid counts must use int32"):
        validate_sparse_attention_replay(
            torch.zeros((1, 1, 1, 1), dtype=torch.int32),
            torch.tensor([1], dtype=torch.int64),
            query_length=2,
            num_layers=1,
            topk=1,
        )


def test_megatron_patch_bypasses_original_topk_path_during_replay():
    class FakeDSAIndexer:
        original_calls = 0

        def forward_with_scores(self, x, qr, mask=None, packed_seq_params=None):
            type(self).original_calls += 1
            raise AssertionError("the original top-k path must not run during replay")

        def forward_before_topk(self, x, qr, packed_seq_params=None):
            return x, qr, x

        def _compute_index_scores(self, q, weights, k):
            return torch.zeros((1, 2, 2), dtype=torch.float32)

    class FakeDSAttention:
        def forward(self, x, qr):
            return self.indexer.forward_with_scores(x, qr)

    fake_dsa = SimpleNamespace(
        DSAIndexer=FakeDSAIndexer,
        DSAttention=FakeDSAttention,
    )
    patch_megatron_dsa_replay(fake_dsa)

    indexer = FakeDSAIndexer()
    indexer._roll_sparse_attention_replay = SparseAttentionReplayState(
        action=SparseAttentionReplayAction.REPLAY_FORWARD,
        target=SparseAttentionReplayTarget(
            selection=torch.tensor([[[0, 0], [1, 0]]], dtype=torch.int32),
            valid_counts=torch.tensor([2], dtype=torch.int32),
            query_length=2,
        ),
    )
    scores, indices = indexer.forward_with_scores(
        torch.zeros((2, 1, 1)),
        torch.zeros((2, 1, 1)),
    )

    assert FakeDSAIndexer.original_calls == 0
    assert scores.shape == (1, 2, 2)
    assert indices.dtype == torch.int64
    assert indices.tolist() == [[[0, 0], [1, 0]]]


def test_megatron_016_compat_path_computes_scores_without_topk():
    class FakeRotaryEmbedding:
        def get_rotary_seq_len(self, *args):
            return 2

        def __call__(self, rotary_seq_len, packed_seq=False):
            return torch.zeros((rotary_seq_len, 1, 1))

    class FakeDSAIndexer:
        original_calls = 0

        def __init__(self):
            self.config = SimpleNamespace(rope_type="rope", sequence_parallel=False)
            self.pg_collection = SimpleNamespace(tp=SimpleNamespace(size=lambda: 1))
            self.rotary_pos_emb = FakeRotaryEmbedding()
            self.index_n_heads = 1
            self.index_head_dim = 1
            self.softmax_scale = 1.0
            self.linear_wq_b = lambda value: (value, None)
            self.linear_wk = lambda value: (value, None)
            self.k_norm = lambda value: value
            self.linear_weights_proj = lambda value: (torch.ones_like(value), None)

        def _apply_rope(self, value, rotary_pos_emb, mscale):
            return value

        def _compute_index_scores(self, q, weights, k):
            return torch.tensor([[[2.0, 1.0], [1.0, 2.0]]])

        def forward_with_scores(self, x, qr, mask=None, packed_seq_params=None):
            type(self).original_calls += 1
            raise AssertionError("the MCore 0.16 top-k path must not run")

    class FakeDSAttention:
        def forward(self, x, qr):
            return self.indexer.forward_with_scores(x, qr)

    fake_dsa = SimpleNamespace(
        DSAIndexer=FakeDSAIndexer,
        DSAttention=FakeDSAttention,
        rotate_activation=lambda value: value,
    )
    patch_megatron_dsa_replay(fake_dsa)

    indexer = FakeDSAIndexer()
    indexer._roll_sparse_attention_replay = SparseAttentionReplayState(
        action=SparseAttentionReplayAction.REPLAY_FORWARD,
        target=SparseAttentionReplayTarget(
            selection=torch.tensor([[[0, -1], [1, 0]]], dtype=torch.int32),
            valid_counts=torch.tensor([2], dtype=torch.int32),
            query_length=2,
        ),
    )
    scores, indices = indexer.forward_with_scores(
        torch.ones((2, 1, 1)), torch.ones((2, 1, 1))
    )

    assert FakeDSAIndexer.original_calls == 0
    assert scores.tolist() == [[[2.0, 1.0], [1.0, 2.0]]]
    assert indices.tolist() == [[[0, 0], [1, 0]]]


def test_replay_backward_uses_forward_order():
    state = SparseAttentionReplayState(
        action=SparseAttentionReplayAction.REPLAY_FORWARD
    )
    counts = torch.tensor([1], dtype=torch.int32)
    first = SparseAttentionReplayTarget(
        torch.tensor([[[1]]], dtype=torch.int32), counts, 1
    )
    second = SparseAttentionReplayTarget(
        torch.tensor([[[2]]], dtype=torch.int32), counts, 1
    )
    state.set_target(first)
    state.set_target(second)

    state.action = SparseAttentionReplayAction.REPLAY_BACKWARD
    assert state.get_indices().item() == 1
    assert state.get_indices().item() == 2
