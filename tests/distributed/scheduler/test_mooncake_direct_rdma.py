"""Consumer-process direct reads against an explicitly configured RDMA store."""

import os

import pytest
import torch

from roll.distributed.scheduler import transfer_backend
from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch


@pytest.mark.parametrize("mode", ["copy", "auto"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_replay_columns_read_without_actor_payload(monkeypatch, mode, device):
    master = os.getenv("MOONCAKE_ALIGNMENT_MASTER")
    if not master:
        pytest.skip("Set MOONCAKE_ALIGNMENT_MASTER to an isolated RDMA master")
    config = {
        "local_hostname": os.environ["MOONCAKE_ALIGNMENT_HOST"],
        "metadata_server": "P2PHANDSHAKE",
        "master_server_addr": master,
        "protocol": "rdma",
        "rdma_devices": os.environ["MOONCAKE_ALIGNMENT_DEVICE"],
        "global_segment_size": 128 * 1024 * 1024,
        "local_buffer_size": 64 * 1024 * 1024,
        "key_prefix": "test-roll-direct-replay",
        "transfer_policy": {"copy_mode": mode},
        "field_schemas": {
            "attention_selection": {"codec": "torch_tensor", "direct_alignment": 4096},
            "attention_selection_valid_count": {"codec": "torch_tensor", "direct_alignment": 4},
            "routed_experts": {"codec": "torch_tensor", "direct_alignment": 64},
        },
        "direct_read_device": device,
    }
    writer = transfer_backend.MooncakeClient(config)
    # No actor can return a tensor: every direct read must stay in this process.
    monkeypatch.setattr(transfer_backend.MooncakeNodeClientProxy, "_get_node_actor", lambda self: None)
    proxy = transfer_backend.MooncakeNodeClientProxy(config)
    monkeypatch.setattr(transfer_backend, "_client", proxy)
    monkeypatch.setattr(transfer_backend, "get", lambda *a, **kw: pytest.fail("actor payload path used"))
    batches, fields = [], []
    try:
        for base in (0, 10):
            selection = torch.full((3, 2, 1, 1024), -1, dtype=torch.int32)
            selection[:, 0, :, :3] = 0
            selection[:, 1, :, :3] = 1
            routers = torch.arange(base, base + 6, dtype=torch.uint8).view(3, 1, 1, 2)
            values = {
                "attention_selection": selection,
                "attention_selection_valid_count": torch.full((3,), 2, dtype=torch.int32),
                "routed_experts": routers,
            }
            fields.append(values)
            batches.append(writer.put("replay", [f"{base}-{i}" for i in range(3)], values, 3))
        batch = ColumnRemoteBatch._cat([batches[0].slice(1, 3), batches[1]])
        # Alternating sources exercise scatter writes at interior R3 row offsets.
        batch = batch.select_idxs([4, 0, 3, 1]).repeat(2, interleave=True)
        result = batch.materialize()
        expected_rows = [(1, 2), (0, 1), (1, 1), (0, 2)]
        expected_rows = [pair for pair in expected_rows for _ in range(2)]
        for key, schema in config["field_schemas"].items():
            alignment = schema["direct_alignment"]
            target = result[key]
            assert target.device.type == device
            assert target.data_ptr() % alignment == 0
            expected = torch.stack([fields[source][key][row] for source, row in expected_rows])
            torch.testing.assert_close(target.cpu(), expected)
        if device == "cuda":
            from roll.third_party.megatron.replay_kernels import (
                prepare_attention_indices,
                prepare_router_indices,
            )

            selection = result["attention_selection"][:, :, 0, :3]
            actual = prepare_attention_indices(selection, result["attention_selection_valid_count"], 2)
            torch.testing.assert_close(actual.cpu(), selection.cpu().to(torch.int64))
            routers = result["routed_experts"]
            actual = prepare_router_indices(routers, 32)
            expected = routers.cpu().permute(2, 1, 0, 3).reshape(1, 8, 2).to(torch.int64)
            torch.testing.assert_close(actual.cpu(), expected)
        assert not proxy._direct_client.backend._roll_pending_direct_registrations
    finally:
        for batch in batches:
            writer.delete(batch.partition, list(batch.fields), list(batch.fields.values()))
