import os
import threading
import uuid
from typing import Any

import ray
import torch
import numpy as np
import sys
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

if sys.version_info < (3, 13):
    import transfer_queue as tq
else:
    tq = None
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

from roll.configs.base_config import TransferBackendArguments
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.constants import STORAGE_NAME, RAY_NAMESPACE
from roll.utils.logging import get_logger

logger = get_logger()

MOONCAKE_BACKEND_NAME = "Mooncake"


# Global reference to keep SharedStorage actor alive
_shared_storage = None

MOONCAKE_CLIENT_SCOPE_NODE = "node"
MOONCAKE_CLIENT_SCOPE_PROCESS = "process"
MOONCAKE_NODE_ACTOR_NAME_PREFIX = "MooncakeNodeTransfer"


def _check_transfer_queue_available():
    if tq is None:
        raise ImportError(
            "TransferQueue is not available on Python 3.13+. "
            "Please use an alternative transfer backend or downgrade to Python <= 3.12."
        )


def _check_mooncake_available():
    try:
        from mooncake.mooncake_config import MooncakeConfig  # noqa: F401
        from mooncake.store import MooncakeDistributedStore  # noqa: F401
        from mooncake.structured_object_store import MooncakeBundleTransfer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Mooncake transfer backend requires mooncake with structured_object_store support."
        ) from exc


def _mooncake_client_scope(config: dict | None) -> str:
    return (config or {}).get("client_scope", MOONCAKE_CLIENT_SCOPE_NODE)


def _prepare_mooncake_backend_config(config: TransferBackendArguments) -> None:
    if config.backend_name != MOONCAKE_BACKEND_NAME:
        return
    if config.backend_config is None:
        config.backend_config = {}
    if _mooncake_client_scope(config.backend_config) != MOONCAKE_CLIENT_SCOPE_NODE:
        return
    config.backend_config.setdefault("node_actor_session_id", uuid.uuid4().hex)


def init_transfer_backend(config: TransferBackendArguments | None):
    global _shared_storage

    _shared_storage = SharedStorage.options(
        name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
    ).remote()

    if config is None:
        config = TransferBackendArguments()
    _prepare_mooncake_backend_config(config)
    ray.get(_shared_storage.put.remote(key="transfer_backend_config", data=config))

    backend_name = config.backend_name
    backend_config = config.backend_config
    if backend_name is None:
        logger.info(f"Initialized dummy transfer backend: {config}")
    elif backend_name == "TransferQueue":
        _check_transfer_queue_available()
        init_transfer_queue_server(backend_config)
        logger.info(f"Initialized TransferQueue transfer backend: {config}")
    elif backend_name == MOONCAKE_BACKEND_NAME:
        _check_mooncake_available()
        logger.info(f"Initialized Mooncake transfer backend: {config}")
    else:
        raise ValueError(f"Unsupported transfer backend: {backend_name}")


_client = None
_client_lock = threading.Lock()

def reinit_after_fork():
    global _client, _client_lock
    _client_lock = threading.Lock()
    _client = None

os.register_at_fork(after_in_child=reinit_after_fork)

def init_client():
    global _client
    if _client is not None:
        return
    with _client_lock:
        if _client is not None:
            return
        shared_storage = ray.get_actor(name=STORAGE_NAME, namespace=RAY_NAMESPACE)
        config = ray.get(shared_storage.get.remote(key="transfer_backend_config"))
        assert config is not None
        if config.backend_name is None:
            _client = DummyClient()
        elif config.backend_name == "TransferQueue":
            _client = TransferQueueClient()
        elif config.backend_name == MOONCAKE_BACKEND_NAME:
            if _mooncake_client_scope(config.backend_config) == MOONCAKE_CLIENT_SCOPE_NODE:
                _client = MooncakeNodeClientProxy(config.backend_config)
            else:
                _client = MooncakeClient(config.backend_config)
        else:
            raise ValueError(f"Unsupported transfer backend: {config.backend_name}")
        logger.info(f"Initialized transfer client: {_client.__class__.__name__}")


def put(partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int, **kwargs):
    init_client()
    return _client.put(partition, row_ids, fields, batch_size, **kwargs)

def get(partition, keys: list[str], fields: list[Any], **kwargs):
    init_client()
    return _client.get(partition, keys, fields, **kwargs)

def delete(partition, keys: list[str], fields: list[Any], **kwargs):
    init_client()
    return _client.delete(partition, keys, fields, **kwargs)


def create_tensordict(fields: dict[str, torch.Tensor | np.ndarray]) -> TensorDict:
    assert fields
    td_dict = {}
    batch_size = None
    for key, val in fields.items():
        if isinstance(val, torch.Tensor):
            td_dict[key] = val
        elif isinstance(val, np.ndarray):
            td_dict[key] = NonTensorStack(*val)
        else:
            raise TypeError(f"Unsupported type: {type(val)}")
        if batch_size is None:
            batch_size = val.shape[0]
        elif batch_size != val.shape[0]:
            raise ValueError("Batch size mismatch")
    return TensorDict(td_dict, batch_size=[batch_size])


class DummyClient:

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int, **kwargs):
        return None

    def get(self, partition, keys: list[str], fields: list[Any]):
        raise RuntimeError("unexpected code path")

    def delete(self, partition, keys: list[str], fields: list[Any]):
        raise RuntimeError("unexpected code path")


def init_transfer_queue_server(config):
    # Must create enough storage units or may encounter:
    # EncodeError: Can't encode Ext objects with data longer than 2**32 - 1.
    # But also cannot set too many storage units that exceed the number of cores of ray cluster.
    config = OmegaConf.create(config)
    tq.init(config)


class TransferQueueClient:
    def __init__(self):
        _check_transfer_queue_available()
        tq.init()

    def put(self, partition, row_ids: list[str], fields: dict[str, torch.Tensor | np.ndarray], batch_size: int, **kwargs):
        # TODO move TransferQueueClient to another file
        from roll.distributed.scheduler.remote_protocol import RowRemoteBatch

        data = create_tensordict(fields)
        assert len(data) == batch_size
        tq.kv_batch_put(
            keys=row_ids,
            fields=data,
            partition_id=partition,
        )
        return RowRemoteBatch(
            partition=partition,
            device=data.device,
            fields=list(fields.keys()),
            row_ids=row_ids,
            cache=data,
        )

    def get(self, partition, keys: list[str], fields: list[Any]):
        return tq.kv_batch_get(keys=keys, select_fields=fields, partition_id=partition)

    def delete(self, partition, keys: list[str], fields: list[Any]):
        return tq.kv_clear(keys=keys, partition_id=partition)


@ray.remote
class MooncakeNodeTransferActor:
    def __init__(self, config: dict | None = None):
        self.client = MooncakeClient(config)

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
        **kwargs,
    ):
        return self.client.put(partition, row_ids, fields, batch_size, **kwargs)

    def get(self, partition, keys: list[str], fields: list[Any], **kwargs):
        return self.client.get(partition, keys, fields, **kwargs)

    def delete(self, partition, keys: list[str], fields: list[Any], **kwargs):
        return self.client.delete(partition, keys, fields, **kwargs)


class MooncakeNodeClientProxy:
    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.actor = self._get_node_actor()

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
        **kwargs,
    ):
        return ray.get(self.actor.put.remote(partition, row_ids, fields, batch_size, **kwargs))

    def get(self, partition, keys: list[str], fields: list[Any], **kwargs):
        return ray.get(self.actor.get.remote(partition, keys, fields, **kwargs))

    def delete(self, partition, keys: list[str], fields: list[Any], **kwargs):
        return ray.get(self.actor.delete.remote(partition, keys, fields, **kwargs))

    def _get_node_actor(self):
        node_id = ray.get_runtime_context().get_node_id()
        session_id = self.config.get("node_actor_session_id", "default")
        actor_name = f"{MOONCAKE_NODE_ACTOR_NAME_PREFIX}-{session_id}-{node_id[:16]}"
        return MooncakeNodeTransferActor.options(
            name=actor_name,
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
            max_concurrency=1,
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(self.config)


class MooncakeClient:
    def __init__(self, config: dict | None = None):
        _check_mooncake_available()
        from mooncake.mooncake_config import MooncakeConfig
        from mooncake.store import MooncakeDistributedStore
        from mooncake.structured_object_store import BundleTransferPolicy, MooncakeBundleTransfer

        config = config or {}
        store_cls = config.get("store_cls", MooncakeDistributedStore)
        store = store_cls()
        if not config.get("skip_store_setup", False):
            store_config = self._create_store_config(config, MooncakeConfig)
            ret = store.setup(store_config)
            if ret != 0:
                raise RuntimeError(f"Mooncake store setup failed with code {ret}")
        policy_config = config.get("policy", {})
        self.policy = BundleTransferPolicy(
            max_inflight_put=policy_config.get("max_inflight_put", 1),
            put_mode=policy_config.get("put_mode", "auto"),
            copy_mode=policy_config.get("copy_mode", "auto"),
        )
        transfer_cls = config.get("transfer_cls", MooncakeBundleTransfer)
        self.transfer = transfer_cls(
            store,
            key_prefix=config.get("key_prefix", "roll"),
            default_chunk_bytes=config.get("default_chunk_bytes", 512 * 1024**2),
        )
        self.namespace = config.get("namespace", "roll")
        self._stage_id = 0

    def put(
        self,
        partition,
        row_ids: list[str],
        fields: dict[str, torch.Tensor | np.ndarray],
        batch_size: int,
        **kwargs,
    ):
        from roll.distributed.scheduler.remote_protocol import MooncakeRemoteBatch

        batch_fields = kwargs.get("batch_fields") or {}
        non_tensor_fields = kwargs.get("non_tensor_fields") or {}
        ref_remote_batch = kwargs.get("ref_remote_batch")
        if not batch_fields and not non_tensor_fields:
            batch_fields = fields
        self._validate_batch_size(batch_fields, non_tensor_fields, batch_size)
        data = {
            "batch": batch_fields,
            "non_tensor_batch": non_tensor_fields,
            "meta_info": {"roll_row_ids": row_ids},
        }
        is_append = isinstance(ref_remote_batch, MooncakeRemoteBatch)
        if is_append:
            from mooncake.structured_object_store import import_dataproto_ref

            if len(ref_remote_batch.segments) != 1:
                raise ValueError("Mooncake append requires a single DataProto ref segment")
            source_segment = ref_remote_batch.segments[0]
            ref = import_dataproto_ref(source_segment["handle"])
            stage = self._append_stage(ref)
            ref = self.transfer.append_dataproto_fields(ref, data, stage=stage, policy=self.policy)
            source_segment["owns_ref"] = False
        else:
            stage = self._next_stage_name("stage")
            ref = self.transfer.put_dataproto(
                data,
                namespace=self.namespace,
                partition=partition,
                stage=stage,
                policy=self.policy,
            )
        from mooncake.structured_object_store import export_dataproto_ref

        return MooncakeRemoteBatch(
            partition=partition,
            device=None,
            fields=set(fields.keys()),
            row_ids=row_ids,
            handle=export_dataproto_ref(ref),
            rows=None,
            owns_ref=True,
            cache=None,
        )

    def get(self, partition, keys: list[str], fields: list[Any], segments=None):
        if not segments:
            raise ValueError("Mooncake get requires DataProto ref segments from MooncakeRemoteBatch")
        from mooncake.structured_object_store import import_dataproto_ref

        chunks = []
        for segment in segments:
            ref = import_dataproto_ref(segment["handle"])
            result = self.transfer.get_dataproto(ref, fields=fields, rows=segment["rows"])
            data = result["batch"]
            data.update(result["non_tensor_batch"])
            chunks.append(create_tensordict(data))
        if len(chunks) == 1:
            return chunks[0]
        return TensorDict.cat(chunks, dim=0)

    def delete(self, partition, keys: list[str], fields: list[Any], segments=None, owns_ref: bool = False):
        if not segments:
            return
        from mooncake.structured_object_store import import_dataproto_ref

        for segment in segments:
            if segment.get("owns_ref", False):
                self.transfer.cleanup_dataproto(import_dataproto_ref(segment["handle"]))

    def _next_stage_name(self, prefix: str) -> str:
        self._stage_id += 1
        return f"{prefix}_{self._stage_id}"

    @staticmethod
    def _append_stage(ref) -> str:
        stages = list(ref.stage_refs.keys())
        if not stages:
            raise ValueError("Mooncake append requires at least one DataProto stage")
        return stages[0]

    @staticmethod
    def _create_store_config(config: dict[str, Any], mooncake_config_cls) -> dict[str, Any]:
        master_server_addr = config.get("master_server_addr") or config.get("master_server_address")
        if master_server_addr is not None:
            local_hostname = MooncakeClient._require_config(config, "local_hostname")
            metadata_server = MooncakeClient._require_config(config, "metadata_server")
            protocol = config.get("protocol", "tcp")
            rdma_devices = config.get("rdma_devices") or config.get("device_name", "")
            if protocol == "rdma" and not rdma_devices:
                raise ValueError("Mooncake RDMA backend_config requires 'device_name' or 'rdma_devices'")
            return {
                "local_hostname": local_hostname,
                "metadata_server": metadata_server,
                "global_segment_size": config.get("global_segment_size", 3355443200),
                "local_buffer_size": config.get("local_buffer_size", 1073741824),
                "protocol": protocol,
                "rdma_devices": rdma_devices,
                "master_server_addr": master_server_addr,
                "enable_ssd_offload": config.get("enable_ssd_offload", False),
                "ssd_offload_path": config.get("ssd_offload_path", ""),
            }

        mooncake_config = mooncake_config_cls.load_from_env()
        return {
            "local_hostname": config.get("local_hostname", mooncake_config.local_hostname),
            "metadata_server": config.get("metadata_server", mooncake_config.metadata_server),
            "global_segment_size": config.get("global_segment_size", mooncake_config.global_segment_size),
            "local_buffer_size": config.get("local_buffer_size", mooncake_config.local_buffer_size),
            "protocol": config.get("protocol", mooncake_config.protocol),
            "rdma_devices": config.get("rdma_devices") or config.get("device_name", mooncake_config.device_name or ""),
            "master_server_addr": config.get("master_server_addr", mooncake_config.master_server_address),
            "enable_ssd_offload": config.get(
                "enable_ssd_offload", getattr(mooncake_config, "enable_ssd_offload", False)
            ),
            "ssd_offload_path": config.get("ssd_offload_path", getattr(mooncake_config, "ssd_offload_path", "")),
        }

    @staticmethod
    def _require_config(config: dict[str, Any], key: str) -> Any:
        value = config.get(key)
        if value is None or value == "":
            raise ValueError(f"Mooncake backend_config requires {key!r} when master_server_address is set")
        return value

    @staticmethod
    def _validate_batch_size(batch_fields: dict[str, Any], non_tensor_fields: dict[str, Any], batch_size: int) -> None:
        for field, value in {**batch_fields, **non_tensor_fields}.items():
            if len(value) != batch_size:
                raise ValueError(f"Field {field!r} batch size {len(value)} does not match {batch_size}")


__all__ = ["init_transfer_backend", "put", "get", "delete"]
