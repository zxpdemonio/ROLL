"""Exercise production hooks without importing vLLM's native GPU runtime."""

import ast
import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, Optional

import pytest
import torch

from roll.distributed.scheduler import transfer_backend as tb
from roll.distributed.scheduler.mooncake_sequence import AUX_OWNER_KEY
from roll.utils.execution_artifacts import (
    ATTENTION_SELECTION_KEY,
    ATTENTION_SELECTION_PROFILE_KEY,
    EXECUTION_ARTIFACT_PROFILE_ID_KEY,
)
from tests.distributed.scheduler.test_mooncake_sequence import owner, rows

ROOT = Path(__file__).parents[3]


def methods(path, class_name, names, globals_dict, base=None):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [
        node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    cls.body = selected
    cls.bases = [ast.Name(id="_TestBase", ctx=ast.Load())] if base is not None else []
    namespace = dict(globals_dict)
    namespace["_TestBase"] = base
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(tree, str(ROOT / path), "exec"), namespace)
    return namespace[class_name]


@pytest.fixture
def bridge(monkeypatch):
    package = ModuleType("roll.third_party.vllm")
    package.__path__ = [str(ROOT / "roll/third_party/vllm")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ("aux_output_gpu", "aux_output_mooncake"):
        full = package.__name__ + "." + name
        spec = importlib.util.spec_from_file_location(full, ROOT / "roll/third_party/vllm" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
    return sys.modules["roll.third_party.vllm.aux_output_mooncake"]


def test_one_slot_replacement_does_not_need_second_block_capacity(bridge):
    class Writer:
        lock = threading.RLock()
        occupied = 0
        next_id = 0

        def put(self, fields, partition):
            size = next(iter(fields.values())).nbytes
            assert self.occupied + size <= 16
            self.occupied += size
            self.next_id += 1
            return (self.next_id, size)

        def remove(self, batch):
            self.occupied -= batch[1]

    writer = Writer()
    arena = bridge._Arena(writer, 16)
    source = bridge.GPURows(torch.arange(16, dtype=torch.uint8)).tobytes()
    arena[slice(0, 16)] = source
    arena[slice(0, 16)] = source
    assert writer.occupied == 16 and writer.next_id == 2
    arena.close()
    assert writer.occupied == 0


def test_claim_discards_unpublished_siblings_before_transfer(bridge):
    writer = object.__new__(bridge._Writer)
    writer.lock = threading.RLock()
    writer.exports = {"claimed": ["a", "internal"], "unused": ["b", "internal"], "other": ["c", "other"]}

    def fail(batch):
        raise RuntimeError("delete failed")

    writer.remove = fail
    with pytest.raises(RuntimeError, match="delete failed"):
        writer.settle(claims=["claimed"])
    assert set(writer.exports) == {"claimed", "unused", "other"}
    removed = []
    writer.remove = removed.append
    writer.settle(claims=["claimed"])
    assert removed == ["b"] and set(writer.exports) == {"other"}


def test_failed_delete_is_owned_by_existing_client_retry_queue():
    client = object.__new__(tb.MooncakeClient)
    client._pending_put_cleanup_refs = []
    client._pending_release_results = []
    failure = [True]

    def cleanup(ref):
        if failure[0]:
            raise RuntimeError("remove failed")

    client.backend = SimpleNamespace(cleanup_dataproto=cleanup)
    with pytest.raises(RuntimeError, match="remove failed"):
        client.delete("p", ["k"], [{"ref": "ref"}])
    assert client._pending_put_cleanup_refs == ["ref"]
    failure[0] = False
    client._retry_pending_put_cleanups()
    assert not client._pending_put_cleanup_refs


def test_remote_prefix_close_failure_keeps_client_retry_owner(bridge):
    client = object.__new__(tb.MooncakeClient)
    client._pending_put_cleanup_refs, client._pending_release_results = [], []
    failure = [True]

    def cleanup(ref):
        if failure[0]:
            raise RuntimeError("remove failed")

    client.backend = SimpleNamespace(cleanup_dataproto=cleanup)
    writer = object.__new__(bridge._Writer)
    writer.client, writer.lock = client, threading.RLock()
    batch = owner((1, 1, 2))
    arena = bridge._Arena(writer, 4)
    arena.slots[0] = batch
    with pytest.raises(RuntimeError, match="remove failed"):
        arena.close()
    assert len(client._pending_put_cleanup_refs) == 1
    failure[0] = False
    client._retry_pending_put_cleanups()
    assert not client._pending_put_cleanup_refs


def test_gpu_workspace_budget_is_distinct_from_prefix_capacity(bridge, monkeypatch):
    writer = object.__new__(bridge._Writer)
    writer.max_pending_batches, writer.max_step_tokens, writer.max_request_tokens = 1, 4, 8
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (512, 4096))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    with pytest.raises(ValueError, match="needs 832 bytes"):
        writer.validate_workspace(128, 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4096, 4096))
    writer.validate_workspace(128, 16)


def test_offload_drains_queue_without_holding_writer_lock(bridge):
    events = []

    class Lock:
        held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *args):
            self.held = False

    lock = Lock()

    def join():
        assert not lock.held
        events.append("queue drained")

    backend = SimpleNamespace(_roll_pending_direct_registrations=[])
    writer = SimpleNamespace(
        lock=lock,
        exports={},
        client=SimpleNamespace(backend=backend, _retry_pending_put_cleanups=lambda: events.append("cleanup")),
    )
    connector = SimpleNamespace(
        _roll_writer=writer,
        _lock=threading.RLock(),
        _pending_outputs=[],
        _store=SimpleNamespace(_queue=SimpleNamespace(join=join), _raise_if_failed=lambda: None),
    )
    worker = SimpleNamespace(model_runner=SimpleNamespace(aux_output_connector=connector))
    bridge.drain_worker_aux_outputs(worker)
    assert events == ["queue drained", "cleanup"]
    connector._pending_outputs.append(object())
    with pytest.raises(RuntimeError, match="unconsumed"):
        bridge.drain_worker_aux_outputs(worker)


def test_registration_failure_fences_r3_clone_before_snapshot_release(bridge, monkeypatch):
    events = []

    class Field:
        device = torch.device("cuda:0")

        def __len__(self):
            return 4

    writer = object.__new__(bridge._Writer)

    def put(*args):
        events.extend(["clone submitted", "register failed"])
        raise RuntimeError("registration failed")

    writer.client = SimpleNamespace(put=put)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: events.append("source fence"))
    )
    try:
        with pytest.raises(RuntimeError, match="registration failed"):
            writer.put({"routed_experts": Field()}, "train_eval")
    finally:
        events.append("snapshot released")
    assert events == ["clone submitted", "register failed", "source fence", "snapshot released"]


def test_abort_retry_keeps_internal_ids_after_mapping_is_removed():
    exports, calls = {"internal"}, []

    class Base:
        async def abort(self, request_id, *args, **kwargs):
            self.output_processor.external_req_ids.pop(request_id, None)

    cls = methods("roll/third_party/vllm/async_llm.py", "CustomAsyncLLM", {"abort", "settle_aux_outputs"}, {}, Base)
    instance = cls()
    instance._roll_mooncake_aux_enabled = True
    instance.output_processor = SimpleNamespace(external_req_ids={"external": ["internal"]})

    async def rpc(*, method, kwargs):
        calls.append(set(kwargs["request_ids"]))
        if len(calls) == 1:
            raise RuntimeError("transient cleanup failure")
        exports.difference_update(kwargs["request_ids"])

    instance.engine_core = SimpleNamespace(collective_rpc_async=rpc)
    with pytest.raises(RuntimeError, match="transient cleanup"):
        asyncio.run(instance.abort("external"))
    asyncio.run(instance.abort("external"))
    assert not exports
    assert all("internal" in call for call in calls)
    assert not instance._roll_pending_aux_discards


def test_release_retries_inactive_source_after_delete_failure(monkeypatch):
    from roll.distributed.scheduler.mooncake_sequence import release_artifact_rows

    batch = owner((2, 1, 2))
    metadata = {"routed_experts": [rows(batch)]}
    attempts = []

    def delete(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("delete failed")

    monkeypatch.setattr(tb, "delete", delete)
    error = RuntimeError("original failure")
    release_artifact_rows(metadata, error=error)
    release_artifact_rows(metadata)
    assert len(attempts) == 2


def test_pause_merge_failure_keeps_new_refs_available_for_cleanup(monkeypatch):
    from tensordict import TensorDict

    from roll.distributed.scheduler.mooncake_sequence import (
        release_artifact_rows,
        rows_owner_batch,
    )
    from roll.distributed.scheduler.protocol import DataProto
    from roll.distributed.scheduler.user_defined_rollout_loop import (
        postprocess_paused_data,
    )

    previous, current = owner((4, 1, 2)), owner((2, 1, 2))
    pre = DataProto(
        batch=TensorDict(
            {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.int64)},
            batch_size=[1],
        ),
        meta_info={"pre_routed_experts": [rows(previous)]},
    )
    data = DataProto(
        remote_batch=rows_owner_batch([rows(current)], 1),
        meta_info={"output_token_ids": [[3]], "output_logprobs": [[]], "routed_experts": [rows(current)]},
    )
    deleted = []
    monkeypatch.setattr(tb, "delete", lambda partition, keys, fields: deleted.extend(fields))
    with pytest.raises(ValueError, match="complete remote token prefix"):
        postprocess_paused_data(pre, data, 10, 2)
    release_artifact_rows({**pre.meta_info, **data.meta_info})
    assert len(deleted) == 2


def strategy():
    cls = methods(
        "roll/distributed/strategy/vllm_strategy.py",
        "VllmStrategy",
        {"generate_request", "_generate_request", "_finish_request_output"},
        {
            "asyncio": asyncio,
            "torch": torch,
            "Dict": Dict,
            "Optional": Optional,
            "RequestOutput": object,
            "TokensPrompt": lambda **kw: kw,
            "SamplingParams": lambda **kw: kw,
            "ATTENTION_SELECTION_KEY": ATTENTION_SELECTION_KEY,
            "ATTENTION_SELECTION_PROFILE_KEY": ATTENTION_SELECTION_PROFILE_KEY,
            "EXECUTION_ARTIFACT_PROFILE_ID_KEY": EXECUTION_ARTIFACT_PROFILE_ID_KEY,
        },
    )
    instance = cls()
    instance.is_lora = False
    instance.attention_selection_profile = None
    instance.get_speculative_metrics = lambda: {}
    return instance


def test_claim_response_failure_drops_frontend_owner(monkeypatch):
    batch = owner((2, 1, 2))
    value = rows(batch)
    deleted, calls = [], []
    monkeypatch.setattr(tb, "delete", lambda partition, keys, fields: deleted.extend(fields))

    async def generate(**kwargs):
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[3], finish_reason="stop", logprobs=None, routed_experts=value)]
        )

    async def settle(**kwargs):
        calls.append(kwargs)
        if kwargs.get("claims") and not kwargs.get("discard"):
            raise RuntimeError("claim reply lost")

    async def abort(**kwargs):
        pass

    instance = strategy()
    instance.model = SimpleNamespace(
        generate=generate, settle_aux_outputs=settle, abort=abort, _roll_mooncake_aux_enabled=True
    )
    with pytest.raises(RuntimeError, match="claim reply lost"):
        asyncio.run(instance.generate_request({"rid": "external", "input_ids": [1, 2], "sampling_params": {}}))
    assert len(deleted) == 1
    assert any(call.get("discard") and call.get("claims") for call in calls)
    with pytest.raises(RuntimeError, match="dropped"):
        batch._ensure_active()


def test_partial_r3_cancel_discards_even_without_delivered_refs():
    calls = []

    async def generate(**kwargs):
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[3], finish_reason=None, logprobs=None, routed_experts=None)]
        )
        raise asyncio.CancelledError()

    async def settle(**kwargs):
        calls.append(kwargs)

    async def abort(**kwargs):
        calls.append({"abort": kwargs})

    instance = strategy()
    instance.model = SimpleNamespace(
        generate=generate, settle_aux_outputs=settle, abort=abort, _roll_mooncake_aux_enabled=True
    )
    result = asyncio.run(instance.generate_request({"rid": "external", "input_ids": [1, 2], "sampling_params": {}}))
    assert result["finish_reasons"] == ["abort"]
    assert {"request_ids": ["external"], "discard": True} in calls


def test_router_owns_refs_before_token_postprocess_and_drops_on_error(monkeypatch):
    from roll.distributed.scheduler.mooncake_sequence import rows_owner_batch
    from roll.distributed.scheduler.protocol import DataProto

    cls = methods(
        "roll/distributed/scheduler/router.py",
        "RouterClient",
        {"_postprocess_generate", "_postprocess_generate_impl"},
        {
            "DataProto": DataProto,
            "ATTENTION_SELECTION_KEY": ATTENTION_SELECTION_KEY,
            "ATTENTION_SELECTION_PROFILE_KEY": ATTENTION_SELECTION_PROFILE_KEY,
            "EXECUTION_ARTIFACT_PROFILE_ID_KEY": EXECUTION_ARTIFACT_PROFILE_ID_KEY,
        },
    )
    instance = cls()
    instance.strategy_name, instance.eos_token_id, instance.pad_token_id = "vllm", -1, -2
    batch = owner((2, 1, 2))
    frame = rows_owner_batch([rows(batch)], 1)
    deleted = []
    monkeypatch.setattr(tb, "delete", lambda partition, keys, fields: deleted.extend(fields))
    response = {
        AUX_OWNER_KEY: frame,
        "finish_reasons": ["abort"],
        "output_token_ids": [[3]],
        "routed_experts": [rows(batch)],
    }
    result = instance._postprocess_generate(DataProto(meta_info={}), response)
    assert result._remote_batch is frame
    assert len(result) == 1 and not result._remote_batch.fields
    DataProto.drop(result)
    assert len(deleted) == 1
    another = owner((2, 1, 2))
    with pytest.raises(KeyError):
        instance._postprocess_generate(DataProto(meta_info={}), {AUX_OWNER_KEY: rows_owner_batch([rows(another)], 1)})
    assert len(deleted) == 2
