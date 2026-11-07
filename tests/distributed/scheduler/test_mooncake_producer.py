"""Schema-selected producer writes keep tensor payloads out of Ray."""

import pickle
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from roll.distributed.scheduler import transfer_backend as tb
from roll.distributed.scheduler.remote_protocol import ColumnRemoteBatch


@pytest.fixture
def transport(monkeypatch):
    events = []
    clients = []

    class Client:
        def __init__(self, config):
            self.config = config
            clients.append(self)
            events.append("client")

        def put(self, partition, row_ids, fields, batch_size):
            events.append(("put", fields))
            schemas = self.config.get("field_schemas") or {}
            refs = {}
            for key, value in fields.items():
                refs[key] = {"ref": "bundle", "kind": "batch"}
                if schemas.get(key, {}).get("direct_alignment") is not None:
                    refs[key].update(shape=list(value.shape), dtype=str(value.dtype), schema=schemas[key])
            return ColumnRemoteBatch(
                partition=partition,
                device=None,
                fields=refs,
                cache=None,
                batch_size=batch_size,
                row_ids=row_ids,
            )

    def ready():
        events.append("ready")
        return None

    def actor_put(*args):
        events.append("actor_put")
        return Client({}).put(*args)

    actor = SimpleNamespace(ready=SimpleNamespace(remote=ready), put=SimpleNamespace(remote=actor_put))
    monkeypatch.setattr(tb.MooncakeNodeClientProxy, "_get_node_actor", lambda self: actor)
    monkeypatch.setattr(tb, "MooncakeClient", Client)
    monkeypatch.setattr(tb.ray, "get", lambda value: value)
    return SimpleNamespace(events=events, clients=clients, actor=actor)


def config(key="custom", **kwargs):
    return {
        "global_segment_size": 1 << 30,
        "field_schemas": {key: {"codec": "torch_tensor", "direct_alignment": 64}},
        **kwargs,
    }


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.uint32, torch.int32])
def test_direct_producer_bypasses_actor_for_entire_mixed_bundle(transport, dtype):
    proxy = tb.MooncakeNodeClientProxy(config())
    values = {
        "custom": torch.tensor([[0, 255]], dtype=dtype),
        "tokens": torch.tensor([[42]]),
        "prompt": np.array(["hello"], dtype=object),
    }
    remote = proxy.put("rollout", ["r0"], values, 1)
    assert transport.events[:2] == ["ready", "client"]
    assert len(transport.events) == 3
    assert transport.events[2][0] == "put"
    assert transport.events[2][1] is values
    assert transport.clients[0].config["global_segment_size"] == 0
    assert set(remote.cache.keys()) == {"tokens", "prompt"}
    assert tb.requires_direct(remote.fields["custom"])
    assert not tb.requires_direct(remote.fields["tokens"])
    restored = pickle.loads(pickle.dumps(remote))
    assert restored.cache is None
    assert restored.fields == remote.fields
    assert restored.row_ids() == ["r0"]


@pytest.mark.parametrize(
    "schemas",
    [
        None,
        {},
        {"routed_experts": {"codec": "torch_tensor"}},
        {"absent": {"codec": "torch_tensor", "direct_alignment": 64}},
    ],
)
def test_plain_batches_keep_actor_put_even_with_replay_field_names(transport, schemas):
    proxy = tb.MooncakeNodeClientProxy({"field_schemas": schemas})
    remote = proxy.put("rollout", ["r0"], {"routed_experts": torch.zeros(1, 8, dtype=torch.uint8)}, 1)
    assert transport.events[0] == "actor_put"
    assert "ready" not in transport.events
    assert proxy._direct_client is None
    assert set(remote.cache.keys()) == {"routed_experts"}


@pytest.mark.parametrize("style", ["normal", "args", "kwargs", "args_config", "kwargs_config"])
def test_process_client_is_shared_and_never_donates_a_segment(transport, style):
    cfg = config()
    if style == "args":
        cfg["setup_args"] = ["host", "P2PHANDSHAKE", 12345, 1024, "tcp", "", "host:1"]
    elif style == "kwargs":
        cfg["setup_kwargs"] = {"global_segment_size": 12345, "local_buffer_size": 1024}
    elif style == "args_config":
        cfg["setup_args"] = [{"global_segment_size": 12345, "local_buffer_size": 1024}]
    elif style == "kwargs_config":
        cfg["setup_kwargs"] = {"config": {"global_segment_size": 12345, "local_buffer_size": 1024}}
    original = pickle.loads(pickle.dumps(cfg))
    proxy = tb.MooncakeNodeClientProxy(cfg)
    for _ in range(2):
        proxy.put("rollout", ["r0"], {"custom": torch.zeros(1, 2, dtype=torch.int32)}, 1)
    assert len(transport.clients) == 1
    assert transport.events.count("ready") == 1
    assert proxy._get_direct_client_locked() is transport.clients[0]
    actual = transport.clients[0].config
    assert actual["global_segment_size"] == 0
    if style == "args":
        assert actual["setup_args"][2] == 0
    elif style == "kwargs":
        assert actual["setup_kwargs"]["global_segment_size"] == 0
    elif style == "args_config":
        assert actual["setup_args"][0]["global_segment_size"] == 0
    elif style == "kwargs_config":
        assert actual["setup_kwargs"]["config"]["global_segment_size"] == 0
        assert set(actual["setup_kwargs"]) == {"config"}
    assert cfg == original


def test_producer_waits_for_segment_and_can_retry_failed_readiness(transport):
    proxy = tb.MooncakeNodeClientProxy(config())
    original_ready = transport.actor.ready.remote

    def fail():
        raise RuntimeError("segment unavailable")

    transport.actor.ready.remote = fail
    values = {"custom": torch.zeros(1, 2, dtype=torch.int32)}
    with pytest.raises(RuntimeError, match="segment unavailable"):
        proxy.put("rollout", ["r0"], values, 1)
    assert transport.clients == []
    assert transport.events == []
    transport.actor.ready.remote = original_ready
    proxy.put("rollout", ["r0"], values, 1)
    assert len(transport.clients) == 1


def test_failed_direct_put_does_not_fall_back_to_actor(transport, monkeypatch):
    proxy = tb.MooncakeNodeClientProxy(config())
    client = proxy._get_direct_client_locked()

    def fail(*args):
        raise RuntimeError("native PUT failed after rollback")

    monkeypatch.setattr(client, "put", fail)
    with pytest.raises(RuntimeError, match="native PUT failed"):
        proxy.put("rollout", ["r0"], {"custom": torch.zeros(1, 2, dtype=torch.int32)}, 1)
    assert "actor_put" not in transport.events


def test_concurrent_producers_serialize_access_to_shared_process_client(transport, monkeypatch):
    proxy = tb.MooncakeNodeClientProxy(config())
    client = proxy._get_direct_client_locked()
    original_put = client.put
    entered, release, second_started, second_entered = (threading.Event() for _ in range(4))
    errors = []

    def put(partition, row_ids, fields, size):
        if row_ids == ["first"]:
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return original_put(partition, row_ids, fields, size)

    def run(name):
        try:
            if name == "second":
                second_started.set()
            proxy.put("rollout", [name], {"custom": torch.zeros(1, 2, dtype=torch.int32)}, 1)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(client, "put", put)
    threads = [threading.Thread(target=run, args=(name,)) for name in ("first", "second")]
    threads[0].start()
    assert entered.wait(5)
    threads[1].start()
    try:
        assert second_started.wait(5)
        assert not second_entered.wait(0.1)
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert second_entered.is_set()
    assert len(transport.clients) == 1
