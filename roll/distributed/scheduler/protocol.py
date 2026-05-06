"""
ref: https://github.com/volcengine/verl/blob/main/verl/protocol.py
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import copy
import ctypes
import os
import pickle
import struct
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set

import numpy as np
import ray
import tensordict
import torch
from codetiming import Timer
from tensordict import TensorDict
from torch.utils.data import DataLoader

from roll.utils.functionals import union_two_dict, divide_by_chunk_size
from roll.platforms import current_platform
from roll.utils.logging import get_logger

logger = get_logger()

POST_GENERATE_DROP_NON_TENSOR_KEYS: tuple[str, ...] = (
    "multi_modal_data",
    "mm_refs",
)

TRANSFER_TIME_SERIALIZE_KEY = "transfer/time/serialize"
TRANSFER_TIME_PUT_KEY = "transfer/time/put"
TRANSFER_TIME_GET_KEY = "transfer/time/get"
TRANSFER_TIME_DESERIALIZE_KEY = "transfer/time/deserialize"
TRANSFER_BYTES_TOTAL_KEY = "transfer/bytes/total"
TRANSFER_BYTES_TENSOR_KEY = "transfer/bytes/tensor"
TRANSFER_BYTES_STRING_KEY = "transfer/bytes/string"
TRANSFER_BYTES_OBJECT_KEY = "transfer/bytes/object"
TRANSFER_BYTES_MULTIMODAL_BEFORE_KEY = "transfer/bytes/multimodal_before_strip"
TRANSFER_BYTES_MULTIMODAL_AFTER_KEY = "transfer/bytes/multimodal_after_strip"
TRANSFER_BYTES_WIRE_KEY = "transfer/bytes/wire"
TRANSFER_SAMPLE_COUNT_KEY = "transfer/sample_count"
TRANSFER_SEQUENCE_LENGTH_MEAN_KEY = "transfer/sequence_length/mean"
TRANSFER_SEQUENCE_LENGTH_MAX_KEY = "transfer/sequence_length/max"
TRANSFER_STAGE_KEY = "transfer/stage"
TRANSFER_BACKEND_KEY = "transfer/backend"
TRANSFER_PROTOCOL_KEY = "transfer/protocol"
TRANSFER_MOONCAKE_TRANSPORT_MODE_KEY = "transfer/mooncake_transport_mode"
TRANSFER_BACKEND_RAY_OPTIMIZED_KEY = "transfer/backend/ray_optimized"
TRANSFER_BACKEND_MOONCAKE_KEY = "transfer/backend/mooncake"
TRANSFER_MOONCAKE_TRANSPORT_STORE_KEY = "transfer/mooncake_transport/store"
TRANSFER_MOONCAKE_TRANSPORT_FALLBACK_KEY = "transfer/mooncake_transport/ray_bytes_fallback"
TRANSFER_MOONCAKE_PROTOCOL_RDMA_KEY = "transfer/mooncake_protocol/rdma"
TRANSFER_MOONCAKE_RDMA_REQUESTED_KEY = "transfer/mooncake_rdma_requested"
TRANSFER_MOONCAKE_RDMA_DEVICE_CONFIGURED_KEY = "transfer/mooncake_rdma_device_configured"
TRANSFER_THROUGHPUT_SERIALIZE_MBPS_KEY = "transfer/throughput/serialize_mbps"
TRANSFER_THROUGHPUT_PUT_MBPS_KEY = "transfer/throughput/put_mbps"
TRANSFER_THROUGHPUT_GET_MBPS_KEY = "transfer/throughput/get_mbps"
TRANSFER_THROUGHPUT_DESERIALIZE_MBPS_KEY = "transfer/throughput/deserialize_mbps"
TRANSFER_THROUGHPUT_E2E_MBPS_KEY = "transfer/throughput/e2e_mbps"
TRANSFER_PROFILE_PREFIX = "transfer/profile"
MOONCAKE_BUFFER_HEADER_SIZE = 40
MOONCAKE_BUFFER_ALIGNMENT = 64
MOONCAKE_DEFAULT_SCATTER_MAX_BYTES = 256 * 1024 * 1024

TRANSFER_MASK_DTYPES: dict[str, torch.dtype] = {
    "attention_mask": torch.uint8,
    "response_mask": torch.bool,
    "prompt_mask": torch.bool,
    "final_response_mask": torch.bool,
}

try:
    tensordict.set_lazy_legacy(False).set()
except:
    pass


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert (
        tensor_dict1.batch_size == tensor_dict2.batch_size
    ), f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(
                tensor_dict2[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            assert np.all(
                tensor_dict2[key] == tensor_dict1[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    """
    Convert a list of dictionaries into a dictionary of lists.

    Example:
        Input:  [{"a": 1, "b": 2}, {"a": 3}, {"b": 4}]
        Output: {"a": [1, 3], "b": [2, 4]}

    Only keys present in each dictionary are aggregated.
    Missing keys in a dictionary are simply skipped.
    """
    if not list_of_dict:
        return {}

    output = {}
    for d in list_of_dict:
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, but got {type(d)}: {d}")
        for k, v in d.items():
            output.setdefault(k, []).append(v)

    return output


def collate_fn(x: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    meta_info = None
    for data in x:
        meta_info = data.meta_info
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.empty(len(val), dtype=object)
        non_tensor_batch[key][:] = val
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def move_tensors_to_device(data, device):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = move_tensors_to_device(val, device)
    elif isinstance(data, list):
        for index, val in enumerate(data):
            data[index] = move_tensors_to_device(val, device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def custom_np_concatenate(val):
    concatenated_list = []
    for array in val:
        concatenated_list.extend(array)
    concatenated_array = np.empty(len(concatenated_list), dtype=object)
    concatenated_array[:] = concatenated_list
    return concatenated_array


def _append_buffer_chunk(chunks: list[bytes], payload: bytes) -> tuple[int, int]:
    offset = sum(len(chunk) for chunk in chunks)
    chunks.append(payload)
    return offset, len(payload)


def _mooncake_scatter_max_bytes() -> int:
    value = os.getenv("ROLL_MOONCAKE_SCATTER_MAX_BYTES")
    if value is None:
        return MOONCAKE_DEFAULT_SCATTER_MAX_BYTES
    return max(0, int(value))


def _should_scatter_mooncake_payload(total_bytes: int, chunk_count: int) -> bool:
    return chunk_count > 1 and total_bytes <= _mooncake_scatter_max_bytes()


def _safe_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _profile_key(metric: str, stage: str) -> str:
    return f"{TRANSFER_PROFILE_PREFIX}/{metric}/{stage}"


def _mbps(num_bytes: int | float, seconds: float | None) -> float:
    seconds = _safe_float(seconds)
    if seconds <= 0:
        return 0.0
    return float(num_bytes) / 1024**2 / seconds


def _add_transfer_throughput_metrics(
    stats: dict[str, Any],
    *,
    logical_bytes: int | float,
    wire_bytes: int | float,
    serialize_seconds: float | None = None,
    put_seconds: float | None = None,
    get_seconds: float | None = None,
    deserialize_seconds: float | None = None,
    e2e_seconds: float | None = None,
) -> None:
    stats[TRANSFER_BYTES_WIRE_KEY] = int(wire_bytes)
    if serialize_seconds is not None:
        stats[TRANSFER_THROUGHPUT_SERIALIZE_MBPS_KEY] = _mbps(logical_bytes, serialize_seconds)
    if put_seconds is not None:
        stats[TRANSFER_THROUGHPUT_PUT_MBPS_KEY] = _mbps(wire_bytes, put_seconds)
    if get_seconds is not None:
        stats[TRANSFER_THROUGHPUT_GET_MBPS_KEY] = _mbps(wire_bytes, get_seconds)
    if deserialize_seconds is not None:
        stats[TRANSFER_THROUGHPUT_DESERIALIZE_MBPS_KEY] = _mbps(logical_bytes, deserialize_seconds)
    if e2e_seconds is not None:
        stats[TRANSFER_THROUGHPUT_E2E_MBPS_KEY] = _mbps(wire_bytes, e2e_seconds)


def _align_mooncake_buffer_size(size: int) -> int:
    return ((size + MOONCAKE_BUFFER_ALIGNMENT - 1) // MOONCAKE_BUFFER_ALIGNMENT) * MOONCAKE_BUFFER_ALIGNMENT


def _pack_mooncake_buffer_header(spec: dict[str, Any]) -> bytes:
    if spec["section"] == "batch":
        ndim = len(spec["shape"])
        dtype_code = _MOONCAKE_DTYPE_CODES.get(np.dtype(spec["dtype"]).name, _MOONCAKE_DTYPE_CODES["uint8"])
        shape = [int(dim) for dim in spec["shape"]]
    else:
        ndim = 1
        dtype_code = _MOONCAKE_DTYPE_CODES["uint8"]
        shape = [int(spec["nbytes"])]
    return struct.pack("iiqqqq", dtype_code, ndim, *shape, *([-1] * (4 - ndim)))


_MOONCAKE_DTYPE_CODES = {
    "float32": 0,
    "float64": 1,
    "int8": 2,
    "uint8": 3,
    "int16": 4,
    "int32": 6,
    "int64": 8,
    "bool": 10,
    "float16": 11,
    "bfloat16": 12,
}


def set_rollout_transfer_meta_info(meta_info: dict[str, Any], pipeline_config: Any) -> None:
    meta_info["rollout_transfer_metrics_enabled"] = getattr(pipeline_config, "rollout_transfer_metrics_enabled", False)
    meta_info["rollout_transfer_profiling_enabled"] = getattr(pipeline_config, "rollout_transfer_profiling_enabled", False)
    meta_info["rollout_transfer_debug_validate"] = getattr(pipeline_config, "rollout_transfer_debug_validate", False)


def inherit_rollout_transfer_meta_info(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "rollout_transfer_metrics_enabled",
        "rollout_transfer_profiling_enabled",
        "rollout_transfer_debug_validate",
    ):
        if key in source:
            target[key] = source[key]


def _cpu_rss_gb() -> float:
    from roll.utils.context_managers import cpu_memory_info
    return cpu_memory_info().rss / 1024**3


def _maybe_set_profiling_metric(metrics: dict[str, Any], data: "DataProto", key: str, value: Any) -> None:
    if not data.meta_info.get("rollout_transfer_profiling_enabled", False):
        return
    metrics[key] = value


def _estimate_multimodal_bytes(values: np.ndarray) -> int:
    total = 0
    for item in values.tolist():
        total += len(pickle.dumps(item, protocol=5))
    return total


def _get_sequence_length_metrics(batch: Optional[TensorDict]) -> dict[str, float]:
    if batch is None:
        return {
            TRANSFER_SEQUENCE_LENGTH_MEAN_KEY: 0.0,
            TRANSFER_SEQUENCE_LENGTH_MAX_KEY: 0.0,
        }
    for key in ("input_ids", "responses", "attention_mask"):
        if key in batch.keys() and batch[key].ndim >= 2:
            lengths = batch[key].shape[-1]
            return {
                TRANSFER_SEQUENCE_LENGTH_MEAN_KEY: float(lengths),
                TRANSFER_SEQUENCE_LENGTH_MAX_KEY: float(lengths),
            }
    return {
        TRANSFER_SEQUENCE_LENGTH_MEAN_KEY: 0.0,
        TRANSFER_SEQUENCE_LENGTH_MAX_KEY: 0.0,
    }


def _validate_transfer_stage(data: "DataProto", stage: str) -> None:
    if stage == "post_generate" and "multi_modal_data" in data.non_tensor_batch:
        raise ValueError("post_generate transfer payload must not include raw multi_modal_data")


def _validate_transfer_payload(payload: dict[str, Any]) -> None:
    if payload.get("protocol") != "v1":
        return
    required_keys = {"meta_bytes", "buffer_specs"}
    missing_keys = required_keys - set(payload.keys())
    if missing_keys:
        raise ValueError(f"Transfer payload missing keys: {sorted(missing_keys)}")
    if payload.get("bulk_buffer") is None and payload.get("bulk_chunks") is None:
        raise ValueError("Transfer payload missing keys: ['bulk_buffer']")


def _collect_transfer_stats(
    *,
    stage: str,
    protocol: str,
    backend: str,
    batch: Optional[TensorDict],
    non_tensor_batch: dict[str, np.ndarray],
    buffer_specs: Optional[list[dict[str, Any]]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    tensor_bytes = 0
    string_bytes = 0
    object_bytes = 0
    if buffer_specs is not None:
        for spec in buffer_specs:
            if spec["section"] == "batch":
                tensor_bytes += spec["nbytes"]
                continue
            codec = spec["codec"]
            if codec == "string_array":
                string_bytes += spec["nbytes"] + spec["offsets_nbytes"]
            elif codec == "pickle_object_array":
                object_bytes += spec["nbytes"]
            else:
                object_bytes += spec["nbytes"]
    total_bytes = 0
    if payload is not None:
        bulk_buffer = payload.get("bulk_buffer")
        bulk_chunks = payload.get("bulk_chunks")
        bulk_bytes = len(bulk_buffer) if bulk_buffer is not None else sum(len(chunk) for chunk in bulk_chunks or [])
        total_bytes = len(payload.get("meta_bytes", b"")) + bulk_bytes
    multimodal_after = _estimate_multimodal_bytes(non_tensor_batch["multi_modal_data"]) if "multi_modal_data" in non_tensor_batch else 0
    sequence_metrics = _get_sequence_length_metrics(batch)
    sample_count = int(batch.batch_size[0]) if batch is not None else int(len(next(iter(non_tensor_batch.values())))) if non_tensor_batch else 0
    return {
        TRANSFER_BYTES_TOTAL_KEY: total_bytes,
        TRANSFER_BYTES_TENSOR_KEY: tensor_bytes,
        TRANSFER_BYTES_STRING_KEY: string_bytes,
        TRANSFER_BYTES_OBJECT_KEY: object_bytes,
        TRANSFER_BYTES_MULTIMODAL_AFTER_KEY: multimodal_after,
        TRANSFER_SAMPLE_COUNT_KEY: sample_count,
        **sequence_metrics,
    }


def _collect_expansion_profile(data: "DataProto", expanded: list["DataProto"], enable_mm_dedup: bool) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    _maybe_set_profiling_metric(metrics, data, _profile_key("request_count", "expand_requests"), float(len(expanded)))
    if not data.meta_info.get("rollout_transfer_profiling_enabled", False):
        return metrics
    unique_mm_ref_ids = set()
    duplicated_multimodal_payloads = 0
    for req in expanded:
        multi_modal_data = req.non_tensor_batch.get("multi_modal_data")
        if multi_modal_data is None:
            continue
        for item in multi_modal_data.tolist():
            if not isinstance(item, dict):
                continue
            if item.get("mm_ref_id") is not None:
                unique_mm_ref_ids.add(item["mm_ref_id"])
            if item.get("multi_modal_data") is not None:
                duplicated_multimodal_payloads += 1
    metrics[_profile_key("mm_ref_count", "expand_requests")] = float(len(unique_mm_ref_ids))
    metrics[_profile_key("mm_dup_object_count", "expand_requests")] = float(duplicated_multimodal_payloads)
    metrics[_profile_key("mm_dedup_enabled", "expand_requests")] = float(enable_mm_dedup)
    return metrics


def _record_profile_metrics(data: "DataProto", stats: dict[str, Any]) -> None:
    if not stats or not data.meta_info.get("rollout_transfer_profiling_enabled", False):
        return
    metrics = data.meta_info.setdefault("metrics", {})
    metrics.update(stats)


def _append_profile_sample(metrics: dict[str, Any], name: str, value: Any) -> None:
    metrics.setdefault(f"transfer/profile_samples/{name}", []).append(value)


def _record_transfer_profile_sample(data: "DataProto", stats: dict[str, Any]) -> None:
    if not stats or not data.meta_info.get("rollout_transfer_profiling_enabled", False):
        return
    metrics = data.meta_info.setdefault("metrics", {})
    samples = {
        "bytes_total": stats.get(TRANSFER_BYTES_TOTAL_KEY),
        "bytes_wire": stats.get(TRANSFER_BYTES_WIRE_KEY),
        "sample_count": stats.get(TRANSFER_SAMPLE_COUNT_KEY),
        "serialize_s": stats.get(TRANSFER_TIME_SERIALIZE_KEY),
        "put_s": stats.get(TRANSFER_TIME_PUT_KEY),
        "get_s": stats.get(TRANSFER_TIME_GET_KEY),
        "deserialize_s": stats.get(TRANSFER_TIME_DESERIALIZE_KEY),
        "backend_mooncake": stats.get(TRANSFER_BACKEND_MOONCAKE_KEY),
        "backend_ray_optimized": stats.get(TRANSFER_BACKEND_RAY_OPTIMIZED_KEY),
        "mooncake_store": stats.get(TRANSFER_MOONCAKE_TRANSPORT_STORE_KEY),
        "mooncake_fallback": stats.get(TRANSFER_MOONCAKE_TRANSPORT_FALLBACK_KEY),
        "mooncake_rdma": stats.get(TRANSFER_MOONCAKE_PROTOCOL_RDMA_KEY),
        "bulk_coalesced": stats.get(_profile_key("bulk_coalesced", "to_transfer_payload")),
        "bulk_chunk_count": stats.get(_profile_key("bulk_chunk_count", "to_transfer_payload")),
        "buffer_count": stats.get(_profile_key("buffer_count", "backend_payload_multi_buffer")),
        "mooncake_put_register_count": stats.get(_profile_key("put_register_count", "mooncake_buffer")),
        "mooncake_put_register_bytes": stats.get(_profile_key("put_register_bytes", "mooncake_buffer")),
        "mooncake_get_register_count": stats.get(_profile_key("get_register_count", "mooncake_buffer")),
        "mooncake_get_register_bytes": stats.get(_profile_key("get_register_bytes", "mooncake_buffer")),
        "mooncake_put_capacity_bytes": stats.get(_profile_key("put_capacity_bytes", "mooncake_buffer")),
        "mooncake_get_capacity_bytes": stats.get(_profile_key("get_capacity_bytes", "mooncake_buffer")),
        "mooncake_entry_count": stats.get(_profile_key("entry_count", "backend_payload_multi_buffer")),
    }
    for key, value in samples.items():
        if value is not None:
            _append_profile_sample(metrics, key, value)


def _encode_tensor_field(name: str, tensor: torch.Tensor, chunks: list[bytes]) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    original_dtype = str(tensor.dtype).replace("torch.", "")
    transfer_tensor = tensor.to(TRANSFER_MASK_DTYPES[name]) if name in TRANSFER_MASK_DTYPES else tensor
    payload = transfer_tensor.numpy().tobytes(order="C")
    offset, nbytes = _append_buffer_chunk(chunks, payload)
    return {
        "section": "batch",
        "key": name,
        "codec": "tensor",
        "dtype": str(transfer_tensor.dtype).replace("torch.", ""),
        "original_dtype": original_dtype,
        "shape": list(transfer_tensor.shape),
        "offset": offset,
        "nbytes": nbytes,
    }


def _encode_string_array(values: np.ndarray, chunks: list[bytes]) -> dict[str, Any]:
    encoded_items = [str(item).encode("utf-8") for item in values.tolist()]
    offsets = np.zeros(len(encoded_items) + 1, dtype=np.int64)
    total = 0
    for idx, item in enumerate(encoded_items, start=1):
        total += len(item)
        offsets[idx] = total
    payload = b"".join(encoded_items)
    data_offset, data_nbytes = _append_buffer_chunk(chunks, payload)
    offsets_offset, offsets_nbytes = _append_buffer_chunk(chunks, offsets.tobytes(order="C"))
    return {
        "codec": "string_array",
        "dtype": "str",
        "shape": [len(encoded_items)],
        "offset": data_offset,
        "nbytes": data_nbytes,
        "offsets_offset": offsets_offset,
        "offsets_nbytes": offsets_nbytes,
    }


def _encode_non_tensor_field(name: str, values: np.ndarray, chunks: list[bytes]) -> dict[str, Any]:
    if values.dtype != object:
        payload = np.ascontiguousarray(values).tobytes(order="C")
        offset, nbytes = _append_buffer_chunk(chunks, payload)
        return {
            "section": "non_tensor_batch",
            "key": name,
            "codec": "ndarray",
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "offset": offset,
            "nbytes": nbytes,
        }

    value_list = values.tolist()
    if all(isinstance(item, str) for item in value_list):
        spec = _encode_string_array(values, chunks)
        spec.update({"section": "non_tensor_batch", "key": name})
        return spec

    if all(isinstance(item, (bool, int, float, np.bool_, np.integer, np.floating)) for item in value_list):
        numeric = np.asarray(value_list)
        payload = np.ascontiguousarray(numeric).tobytes(order="C")
        offset, nbytes = _append_buffer_chunk(chunks, payload)
        return {
            "section": "non_tensor_batch",
            "key": name,
            "codec": "numeric_scalar_array",
            "dtype": str(numeric.dtype),
            "shape": list(numeric.shape),
            "offset": offset,
            "nbytes": nbytes,
        }

    payload = pickle.dumps(value_list, protocol=5)
    offset, nbytes = _append_buffer_chunk(chunks, payload)
    return {
        "section": "non_tensor_batch",
        "key": name,
        "codec": "pickle_object_array",
        "dtype": "object",
        "shape": list(values.shape),
        "offset": offset,
        "nbytes": nbytes,
    }


def _decode_tensor_field(buffer: memoryview, spec: dict[str, Any], copy_buffer: bool = True) -> torch.Tensor:
    np_dtype = np.dtype(spec["dtype"])
    data = np.frombuffer(buffer[spec["offset"]:spec["offset"] + spec["nbytes"]], dtype=np_dtype)
    if copy_buffer:
        data = data.copy()
    tensor = torch.from_numpy(data.reshape(spec["shape"]))
    original_dtype = spec.get("original_dtype")
    if original_dtype is not None and spec["dtype"] != original_dtype:
        tensor = tensor.to(getattr(torch, original_dtype))
    return tensor


def _decode_tensor_field_from_view(buffer: memoryview, spec: dict[str, Any]) -> torch.Tensor:
    return _decode_tensor_field(buffer, spec, copy_buffer=False)


def _decode_non_tensor_field(buffer: memoryview, spec: dict[str, Any]) -> np.ndarray:
    codec = spec["codec"]
    if codec == "ndarray":
        data = np.frombuffer(buffer[spec["offset"]:spec["offset"] + spec["nbytes"]], dtype=np.dtype(spec["dtype"])).copy()
        values = data.reshape(spec["shape"]).tolist()
        array = np.empty(np.prod(spec["shape"], dtype=int), dtype=object)
        array[:] = values
        return array.reshape(spec["shape"])
    if codec == "numeric_scalar_array":
        data = np.frombuffer(buffer[spec["offset"]:spec["offset"] + spec["nbytes"]], dtype=np.dtype(spec["dtype"])).copy()
        values = data.reshape(spec["shape"]).tolist()
        array = np.empty(np.prod(spec["shape"], dtype=int), dtype=object)
        array[:] = values
        return array.reshape(spec["shape"])
    if codec == "string_array":
        data = bytes(buffer[spec["offset"]:spec["offset"] + spec["nbytes"]])
        offsets = np.frombuffer(
            buffer[spec["offsets_offset"]:spec["offsets_offset"] + spec["offsets_nbytes"]], dtype=np.int64
        ).copy()
        values = [data[offsets[idx]:offsets[idx + 1]].decode("utf-8") for idx in range(len(offsets) - 1)]
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array
    if codec == "pickle_object_array":
        values = pickle.loads(bytes(buffer[spec["offset"]:spec["offset"] + spec["nbytes"]]))
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array.reshape(spec["shape"])
    raise ValueError(f"Unsupported non-tensor codec: {codec}")


@dataclass
class DataProtoItem:
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)
    _buffer_owner: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()
    
        if self.batch is not None and current_platform.is_npu():
            for key, val in self.batch.items():
                if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    self.batch[key] = val.to(torch.int32)

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        if self.non_tensor_batch is not None:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0

    def __getitem__(self, item):
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            DataProto: For all indexing types except single integers
            DataProtoItem: Only for single integer indices
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        elif isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item]
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        import io

        buffer = io.BytesIO()
        if tensordict.__version__ >= "0.5.0" and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        torch.save(self.batch, buffer)
        return buffer, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        batch_deserialized, non_tensor_batch, meta_info = data
        batch_deserialized.seek(0)
        batch = torch.load(
            batch_deserialized, weights_only=False, map_location="cpu" if not current_platform.is_available() else None
        )
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info
        self._buffer_owner = None

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert (
                    isinstance(val, np.ndarray) and val.dtype == object
                ), "data in the non_tensor_batch must be a numpy.array with dtype=object"
                assert (
                    val.shape[0] == batch_size
                ), f"key {key} length {len(val)} is not equal to batch size {batch_size}"

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if current_platform.is_npu() and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    val = val.to(torch.int32)
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        assert len(tensors) > 0, "tensors must not be empty"
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert (
                    batch_size == current_batch
                ), f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"

        for key, val in non_tensors.items():
            non_tensors[key] = np.empty(len(val), dtype=object)
            non_tensors[key][:] = val

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        if self.meta_info is not None:
            self.meta_info = move_tensors_to_device(self.meta_info, device)

        return self

    def clone(self) -> "DataProto":
        """
        Create a deep copy of this DataProto, including tensors,
        non-tensor data, and meta_info.

        The new DataProto will share no underlying storage with the original.

        Returns:
            DataProto: A new DataProto instance with the same content but
                       independent memory.
        """
        # Copy batch
        batch_copy = self.batch.clone() if self.batch is not None else None

        # Copy non-tensor objects (numpy arrays)
        non_tensor_copy = {k: np.copy(v) for k, v in self.non_tensor_batch.items()}

        # Deep copy meta_info to avoid shared mutable objects
        meta_copy = copy.deepcopy(self.meta_info)

        # Return new DataProto instance
        return DataProto(
            batch=batch_copy,
            non_tensor_batch=non_tensor_copy,
            meta_info=meta_copy
        )

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def trim_for_stage(self, stage: str) -> "DataProto":
        """Return a stage-trimmed copy of the current DataProto."""
        start_time = time.perf_counter()
        before_keys = set(self.non_tensor_batch.keys())
        trimmed = self.clone()
        if stage == "generate_request":
            _record_profile_metrics(
                trimmed,
                {
                    _profile_key("time_seconds", "trim_for_stage"): time.perf_counter() - start_time,
                    _profile_key("dropped_key_count", "trim_for_stage"): 0.0,
                },
            )
            return trimmed
        if stage == "post_generate":
            for key in POST_GENERATE_DROP_NON_TENSOR_KEYS:
                trimmed.non_tensor_batch.pop(key, None)
            _record_profile_metrics(
                trimmed,
                {
                    _profile_key("time_seconds", "trim_for_stage"): time.perf_counter() - start_time,
                    _profile_key("dropped_key_count", "trim_for_stage"): float(len(before_keys - set(trimmed.non_tensor_batch.keys()))),
                },
            )
            return trimmed
        if stage == "train_batch":
            for key in POST_GENERATE_DROP_NON_TENSOR_KEYS:
                trimmed.non_tensor_batch.pop(key, None)
            _record_profile_metrics(
                trimmed,
                {
                    _profile_key("time_seconds", "trim_for_stage"): time.perf_counter() - start_time,
                    _profile_key("dropped_key_count", "trim_for_stage"): float(len(before_keys - set(trimmed.non_tensor_batch.keys()))),
                },
            )
            return trimmed
        raise ValueError(f"Unsupported trim stage: {stage}")

    def to_transfer_payload(self, stage: str, protocol: str = "v1", coalesce_bulk: bool = True) -> dict[str, Any]:
        """Encode the DataProto into a backend-agnostic transfer payload."""
        if protocol == "legacy":
            return {
                "protocol": "legacy",
                "data": self.clone(),
            }
        if protocol != "v1":
            raise ValueError(f"Unsupported transfer protocol: {protocol}")

        rss_before = _cpu_rss_gb()
        start_time = time.perf_counter()
        multimodal_before = _estimate_multimodal_bytes(self.non_tensor_batch["multi_modal_data"]) if "multi_modal_data" in self.non_tensor_batch else 0
        trim_start = time.perf_counter()
        trimmed = self.trim_for_stage(stage)
        trim_seconds = time.perf_counter() - trim_start
        if self.meta_info.get("rollout_transfer_debug_validate", False):
            _validate_transfer_stage(trimmed, stage)
        chunks: list[bytes] = []
        buffer_specs: list[dict[str, Any]] = []

        tensor_field_count = 0
        tensor_encode_start = time.perf_counter()
        if trimmed.batch is not None:
            for key in sorted(trimmed.batch.keys()):
                buffer_specs.append(_encode_tensor_field(key, trimmed.batch[key], chunks))
                tensor_field_count += 1
        tensor_encode_seconds = time.perf_counter() - tensor_encode_start

        non_tensor_field_count = 0
        non_tensor_encode_start = time.perf_counter()
        for key in sorted(trimmed.non_tensor_batch.keys()):
            buffer_specs.append(_encode_non_tensor_field(key, trimmed.non_tensor_batch[key], chunks))
            non_tensor_field_count += 1
        non_tensor_encode_seconds = time.perf_counter() - non_tensor_encode_start

        meta_encode_start = time.perf_counter()
        meta_bytes = pickle.dumps(
            {
                "stage": stage,
                "meta_info": trimmed.meta_info,
                "batch_size": list(trimmed.batch.batch_size) if trimmed.batch is not None else None,
            },
            protocol=5,
        )
        meta_encode_seconds = time.perf_counter() - meta_encode_start
        bulk_bytes = sum(len(chunk) for chunk in chunks)
        if coalesce_bulk is None:
            coalesce_bulk = not _should_scatter_mooncake_payload(bulk_bytes + len(meta_bytes), len(chunks))
        bulk_join_start = time.perf_counter()
        bulk_buffer = b"".join(chunks) if coalesce_bulk else None
        bulk_join_seconds = time.perf_counter() - bulk_join_start
        payload = {
            "protocol": "v1",
            "meta_bytes": meta_bytes,
            "bulk_buffer": bulk_buffer,
            "bulk_chunks": chunks if not coalesce_bulk else None,
            "buffer_specs": buffer_specs,
        }
        transfer_stats: dict[str, Any] = {}
        if trimmed.meta_info.get("rollout_transfer_metrics_enabled", False):
            transfer_stats = {
                **_collect_transfer_stats(
                    stage=stage,
                    protocol=protocol,
                    backend="protocol",
                    batch=trimmed.batch,
                    non_tensor_batch=trimmed.non_tensor_batch,
                    buffer_specs=buffer_specs,
                    payload=payload,
                ),
                TRANSFER_BYTES_TOTAL_KEY: bulk_bytes + len(meta_bytes),
                TRANSFER_BYTES_MULTIMODAL_BEFORE_KEY: multimodal_before,
            }
        profile_metrics = {
            _profile_key("temp_buffer_count", "to_transfer_payload"): float(len(chunks) + len(buffer_specs) + 1),
            _profile_key("peak_rss_gb", "to_transfer_payload"): max(rss_before, _cpu_rss_gb()),
            _profile_key("time_seconds", "to_transfer_payload"): time.perf_counter() - start_time,
            _profile_key("time_seconds", "trim_for_stage"): trim_seconds,
            _profile_key("time_seconds", "encode_tensors"): tensor_encode_seconds,
            _profile_key("time_seconds", "encode_non_tensors"): non_tensor_encode_seconds,
            _profile_key("time_seconds", "encode_metadata"): meta_encode_seconds,
            _profile_key("time_seconds", "join_bulk_buffer"): bulk_join_seconds,
            _profile_key("bulk_coalesced", "to_transfer_payload"): float(coalesce_bulk),
            _profile_key("bulk_chunk_count", "to_transfer_payload"): float(len(chunks)),
            _profile_key("buffer_spec_count", "to_transfer_payload"): float(len(buffer_specs)),
            _profile_key("tensor_field_count", "to_transfer_payload"): float(tensor_field_count),
            _profile_key("non_tensor_field_count", "to_transfer_payload"): float(non_tensor_field_count),
            _profile_key("meta_bytes", "to_transfer_payload"): float(len(meta_bytes)),
            _profile_key("bulk_bytes", "to_transfer_payload"): float(bulk_bytes),
        }
        for key, value in profile_metrics.items():
            _maybe_set_profiling_metric(transfer_stats, trimmed, key, value)
        if transfer_stats:
            payload["transfer_stats"] = transfer_stats
        if trimmed.meta_info.get("rollout_transfer_debug_validate", False):
            _validate_transfer_payload(payload)
        return payload

    @classmethod
    def from_transfer_payload(cls, payload: dict[str, Any]) -> "DataProto":
        """Decode a backend-agnostic transfer payload into a DataProto."""
        return cls._from_transfer_payload(payload=payload, tensor_view=False)

    @classmethod
    def from_transfer_payload_view(cls, payload: dict[str, Any], buffer_owner: Any) -> "DataProto":
        """Decode a transfer payload while tensor fields view the provided backing buffer."""
        data = cls._from_transfer_payload(payload=payload, tensor_view=True)
        data._buffer_owner = buffer_owner
        return data

    @classmethod
    def _from_transfer_payload(cls, payload: dict[str, Any], tensor_view: bool) -> "DataProto":
        protocol = payload.get("protocol", "legacy")
        if protocol == "legacy":
            return payload["data"].clone()
        if protocol != "v1":
            raise ValueError(f"Unsupported transfer protocol: {protocol}")

        rss_before = _cpu_rss_gb()
        start_time = time.perf_counter()
        metadata_decode_start = time.perf_counter()
        metadata = pickle.loads(payload["meta_bytes"])
        metadata_decode_seconds = time.perf_counter() - metadata_decode_start
        if metadata["meta_info"].get("rollout_transfer_debug_validate", False):
            _validate_transfer_payload(payload)
        memoryview_start = time.perf_counter()
        bulk_buffer = payload.get("bulk_buffer")
        if bulk_buffer is None:
            bulk_buffer = b"".join(payload.get("bulk_chunks") or [])
        buffer = memoryview(bulk_buffer)
        memoryview_seconds = time.perf_counter() - memoryview_start

        tensors: dict[str, torch.Tensor] = {}
        non_tensors: dict[str, np.ndarray] = {}
        tensor_decode_seconds = 0.0
        non_tensor_decode_seconds = 0.0
        for spec in payload["buffer_specs"]:
            if spec["section"] == "batch":
                decode_start = time.perf_counter()
                if tensor_view:
                    tensors[spec["key"]] = _decode_tensor_field_from_view(buffer, spec)
                else:
                    tensors[spec["key"]] = _decode_tensor_field(buffer, spec)
                tensor_decode_seconds += time.perf_counter() - decode_start
            elif spec["section"] == "non_tensor_batch":
                decode_start = time.perf_counter()
                non_tensors[spec["key"]] = _decode_non_tensor_field(buffer, spec)
                non_tensor_decode_seconds += time.perf_counter() - decode_start
            else:
                raise ValueError(f"Unsupported payload section: {spec['section']}")

        construct_start = time.perf_counter()
        batch_size = metadata.get("batch_size")
        batch = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        data = cls(batch=batch, non_tensor_batch=non_tensors, meta_info=metadata["meta_info"])
        if payload.get("transfer_stats"):
            data.meta_info.setdefault("metrics", {}).update(payload["transfer_stats"])
        construct_seconds = time.perf_counter() - construct_start
        profile_stage = "from_transfer_payload_view" if tensor_view else "from_transfer_payload"
        _record_profile_metrics(
            data,
            {
                _profile_key("peak_rss_gb", profile_stage): max(rss_before, _cpu_rss_gb()),
                _profile_key("temp_buffer_count", profile_stage): float(len(payload["buffer_specs"]) + 2),
                _profile_key("time_seconds", profile_stage): time.perf_counter() - start_time,
                _profile_key("time_seconds", "decode_metadata"): metadata_decode_seconds,
                _profile_key("time_seconds", "create_memoryview"): memoryview_seconds,
                _profile_key("time_seconds", "decode_tensors"): tensor_decode_seconds,
                _profile_key("time_seconds", "decode_non_tensors"): non_tensor_decode_seconds,
                _profile_key("time_seconds", "construct_dataproto"): construct_seconds,
                _profile_key("tensor_field_count", profile_stage): float(len(tensors)),
                _profile_key("non_tensor_field_count", profile_stage): float(len(non_tensors)),
                _profile_key("tensor_view_enabled", profile_stage): float(tensor_view),
            },
        )
        return data

    def select_idxs(self, idxs):
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()}, batch_size=(batch_size,)
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        Slice the DataProto and return a new DataProto object.
        This is an improved version of direct slicing which returns a DataProtoItem.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index still returns DataProtoItem
            single_item = data_proto[5]
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []
        batch_keys = self.validate_input(batch_keys)
        non_tensor_batch_keys = self.validate_input(non_tensor_batch_keys)
        meta_info_keys = self.validate_input(meta_info_keys)

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @staticmethod
    def validate_input(keys):
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif isinstance(keys, list):
                pass
            else:
                raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
        return keys

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        Note that this function only rename the key in the batch
        """

        old_keys = self.validate_input(old_keys)
        new_keys = self.validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if
        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        metrics = {}
        if isinstance(self.meta_info.get("metrics"), dict):
            metrics.update(self.meta_info.pop("metrics"))
        if isinstance(other.meta_info.get("metrics"), dict):
            metrics.update(other.meta_info.pop("metrics"))
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        if metrics:
            self.meta_info["metrics"] = metrics
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.
        要求:
            batch_size > chunks，调用方保证，此处保证每个chunk会返回一个DataProto

        np.array_split(val, chunks) 和 self.batch.chunk(chunks=chunks, dim=0) 在不能均分时行为不同
        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        chunks_sizes = None
        if len(self) > 0:
            assert len(self) >= chunks, f"batch_size {self.batch.batch_size[0]} < chunks {chunks}"
            index_array = np.arange(len(self))
            chunks_sizes = [len(b) for b in np.array_split(index_array, chunks)]

        if self.batch is not None:
            batch_lst = divide_by_chunk_size(self.batch, chunk_sizes=chunks_sizes)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            non_tensor_lst = divide_by_chunk_size(val, chunk_sizes=chunks_sizes)
            assert len(non_tensor_lst) == chunks, f"len(non_tensor_lst) {len(non_tensor_lst)} != chunks {chunks}"
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                DataProto(
                    batch=batch_lst[i].clone() if batch_lst[i] is not None else batch_lst[i],
                    non_tensor_batch=non_tensor_batch_lst[i],
                    meta_info=self.meta_info,
                )
            )

        return output

    @staticmethod
    def concat(
            data: List["DataProto"],
            *,
            global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        """
        Concatenate a list of DataProto objects.

        Parameters
        ----------
        data : List[DataProto]
            List of DataProto instances to be concatenated.
        global_keys : Set[str], optional
            Keys in `meta_info` that should be **aggregated across ranks**.
            - If the value is a dict, each sub-key is concatenated across ranks.
            - Otherwise, values are collected into a list.
            Keys not listed retain only the value from rank 0.

        Returns
        -------
        DataProto
            A new DataProto with concatenated tensors, non-tensor data,
            and processed meta information.
        """
        global_keys = global_keys if global_keys is not None else {"metrics"}

        # ---------- 1. Concatenate tensor / non-tensor batches ----------
        batch_lst = [d.batch for d in data if d.batch is not None]
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst else None

        non_tensor_batch = list_of_dict_to_dict_of_list(
            [d.non_tensor_batch for d in data]
        )
        for k, v in non_tensor_batch.items():
            non_tensor_batch[k] = custom_np_concatenate(v)

        # ---------- 2. Aggregate meta information ----------
        merged_meta = dict(data[0].meta_info)  # start with rank-0 values

        for key in global_keys:
            if key not in merged_meta:
                continue

            values = [d.meta_info.get(key) for d in data]

            # Case 1: dict — aggregate each sub-key across ranks
            if isinstance(merged_meta[key], dict):
                sub_dict = list_of_dict_to_dict_of_list(values)
                for sub_key, sub_list in sub_dict.items():
                    try:
                        if np.isscalar(sub_list[0]):
                            sub_dict[sub_key] = np.array(sub_list).tolist()
                        else:
                            sub_dict[sub_key] = np.concatenate(sub_list, axis=0).tolist()
                    except Exception:
                        # fallback: keep as list
                        sub_dict[sub_key] = sub_list
                merged_meta[key] = sub_dict

            # Case 2: non-dict — collect into list
            else:
                merged_meta[key] = values

        return DataProto(
            batch=new_batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=merged_meta,
        )

    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        # Ensure that indices is at least a 1-D tensor.
        indices = indices.view(-1) if indices.dim() == 0 else indices
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def group_by(self, keys: Union[List[str], str]) -> Dict[str, "DataProto"]:
        """
        Group the data by specified keys. Supports grouping by both tensor and non-tensor fields.

        Args:
            keys: Field names to group by. Can be either in batch (tensors) or non_tensor_batch

        Returns:
            Dictionary mapping group keys to DataProto instances containing matching data

        Example:
            Given data with field "category" having values ["A", "B", "A"],
            returns {"A": DataProto(A_data), "B": DataProto(B_data)}
        """
        keys = self.validate_input(keys)
        assert len(keys) > 0, "Must provide at least one grouping key"

        # Collect grouping values across data types
        group_key_values = []
        for idx in range(len(self)):
            key_values = []
            for key in keys:
                # Check tensor data first
                if key in self.batch.keys():
                    key_values.append(str(self.batch[key][idx].numpy()))
                elif key in self.non_tensor_batch:
                    key_values.append(str(self.non_tensor_batch[key][idx]))
                else:
                    raise KeyError(f"Grouping key '{key}' not found in tensor or non-tensor data")

            # Create composite key for multi-field grouping
            group_key = "|".join(key_values) if len(key_values) > 1 else key_values[0]
            group_key_values.append(group_key)

        # Create index groups
        groups = defaultdict(list)
        for idx, group_key in enumerate(group_key_values):
            groups[group_key].append(idx)

        # Create grouped DataProtos
        grouped_data = {}
        for group_key, indices in groups.items():
            grouped_data[group_key] = collate_fn([self[idx] for idx in indices])

        return grouped_data

    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    @staticmethod
    def materialize_concat(
            data_refs: Union[List[ray.ObjectRef], ray.ObjectRef, List["ObjectRefWrap"]],
            *,
            global_keys: Optional[Set[str]] = None,
            transfer_backend: str = "legacy",
            transfer_protocol: str = "legacy",
    ) -> "DataProto":
        """
        Fetch a collection of DataProto objects from Ray ObjectRef(s) and concatenate
        them into a single DataProto instance.

        Parameters
        ----------
        data_refs : Union[List[ray.ObjectRef], ray.ObjectRef, List[ObjectRefWrap]]
            Ray object references (or ObjectRefWrap) pointing to DataProto objects.
        global_keys : Optional[Set[str]], optional
            Keys in ``meta_info`` that should be aggregated across all ranks when
            concatenating.  If None, only rank-0 values are kept for all keys.

        Returns
        -------
        DataProto
            The concatenated DataProto instance.
        """
        # Normalize input to List[<reference>]
        if isinstance(data_refs, DataProto):
            data_refs = [data_refs]

        timeout = None
        if "roll_RPC_TIMEOUT" in os.environ:
            timeout = int(os.environ["roll_RPC_TIMEOUT"])

        # Fetch objects from Ray
        if isinstance(data_refs[0], ObjectRefWrap):
            data_refs = [ref for ref in data_refs if ref.collected]
            if not data_refs:
                raise ValueError("No collected rollout transfer refs to materialize")
            obj_refs = [ref.obj_ref for ref in data_refs]
            fetched = ray.get(obj_refs, timeout=timeout)
            data = [
                materialize_rollout_transfer(
                    handle=item,
                    backend_name=transfer_backend,
                    protocol=transfer_protocol,
                )
                for item in fetched
            ]
        else:
            fetched = ray.get(data_refs, timeout=timeout)
            data = [
                materialize_rollout_transfer(
                    handle=item,
                    backend_name=transfer_backend,
                    protocol=transfer_protocol,
                )
                for item in fetched
            ]

        # Concatenate and apply global aggregation rules
        start_time = time.perf_counter()
        result = DataProto.concat(data, global_keys=global_keys)
        _record_profile_metrics(
            result,
            {
                _profile_key("input_count", "materialize_concat"): float(len(data)),
                _profile_key("time_seconds", "materialize_concat"): time.perf_counter() - start_time,
                _profile_key("peak_rss_gb", "materialize_concat"): _cpu_rss_gb(),
            },
        )
        return result


@dataclass
class RolloutTransferHandle:
    backend: str
    protocol: str
    stage: str
    payload: Optional[dict[str, Any]] = None
    obj_ref: Optional[ray.ObjectRef] = None
    key: Optional[str] = None
    transport_info: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


class RolloutTransferBackend:
    def __init__(self, protocol: str):
        self.protocol = protocol

    @staticmethod
    def _maybe_record_transfer_metrics(data: DataProto, stats: Optional[dict[str, Any]]) -> None:
        if not stats or not data.meta_info.get("rollout_transfer_metrics_enabled", False):
            return
        metrics = data.meta_info.setdefault("metrics", {})
        metrics.update(stats)

    def put(self, data: DataProto, stage: str) -> RolloutTransferHandle:
        raise NotImplementedError

    def get(self, handle: RolloutTransferHandle) -> DataProto:
        raise NotImplementedError

    def cleanup(self, handle: RolloutTransferHandle) -> None:
        return None


class _MooncakeTransportAdapter:
    def mode(self) -> str:
        raise NotImplementedError

    def metrics(self) -> dict[str, Any]:
        return {
            TRANSFER_MOONCAKE_TRANSPORT_STORE_KEY: 0.0,
            TRANSFER_MOONCAKE_TRANSPORT_FALLBACK_KEY: 1.0,
            TRANSFER_MOONCAKE_PROTOCOL_RDMA_KEY: 0.0,
            TRANSFER_MOONCAKE_RDMA_REQUESTED_KEY: 0.0,
            TRANSFER_MOONCAKE_RDMA_DEVICE_CONFIGURED_KEY: 0.0,
        }

    def put_bytes(self, key: str, payload: bytes) -> RolloutTransferHandle:
        raise NotImplementedError

    def get_bytes(self, handle: RolloutTransferHandle) -> bytes:
        raise NotImplementedError

    def cleanup(self, handle: RolloutTransferHandle) -> None:
        return None


class _MooncakeFallbackTransportAdapter(_MooncakeTransportAdapter):
    def mode(self) -> str:
        return "ray_bytes_fallback"

    def put_bytes(self, key: str, payload: bytes) -> RolloutTransferHandle:
        return RolloutTransferHandle(
            backend="mooncake",
            protocol="v1",
            stage="",
            key=key,
            obj_ref=ray.put(payload),
            transport_info={"mode": self.mode(), "size": len(payload)},
        )

    def get_bytes(self, handle: RolloutTransferHandle) -> bytes:
        assert handle.obj_ref is not None, "fallback mooncake handle requires object ref"
        return ray.get(handle.obj_ref)


class _MooncakeStoreTransportAdapter(_MooncakeTransportAdapter):
    def __init__(self):
        from mooncake.mooncake_config import MooncakeConfig
        from mooncake.store import MooncakeDistributedStore

        config = MooncakeConfig.load_from_env()
        store = MooncakeDistributedStore()
        ret = store.setup(
            {
                "local_hostname": config.local_hostname,
                "metadata_server": config.metadata_server,
                "global_segment_size": config.global_segment_size,
                "local_buffer_size": config.local_buffer_size,
                "protocol": config.protocol,
                "rdma_devices": config.device_name or "",
                "master_server_addr": config.master_server_address,
            }
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake store setup failed with code {ret}")
        self.store = store
        self.protocol = config.protocol or ""
        self.rdma_devices = config.device_name or ""
        self._put_buffers: dict[int, Any] = {}
        self._put_buffer_ptrs: dict[int, int] = {}
        self._put_buffer_capacities: dict[int, int] = {}
        self._get_buffer: Optional[Any] = None
        self._get_buffer_ptr = 0
        self._get_buffer_capacity = 0
        self._put_register_count = 0
        self._put_register_bytes = 0
        self._get_register_count = 0
        self._get_register_bytes = 0
        initial_buffer_size = int(os.getenv("ROLL_MOONCAKE_ZEROCOPY_BUFFER_SIZE", str(256 * 1024 * 1024)))
        self._reserve_put_buffer(0, initial_buffer_size)
        self._reserve_get_buffer(initial_buffer_size)

    def __del__(self) -> None:
        return None

    def close(self) -> None:
        if not hasattr(self, "store"):
            return
        for ptr in getattr(self, "_put_buffer_ptrs", {}).values():
            self._unregister_buffer(ptr)
        self._unregister_buffer(getattr(self, "_get_buffer_ptr", 0))
        self._put_buffers = {}
        self._put_buffer_ptrs = {}
        self._put_buffer_capacities = {}
        self._get_buffer = None
        self._get_buffer_ptr = 0
        self._get_buffer_capacity = 0

    def mode(self) -> str:
        return "store"

    def metrics(self) -> dict[str, Any]:
        is_rdma = float(self.protocol.lower() == "rdma")
        return {
            TRANSFER_MOONCAKE_TRANSPORT_STORE_KEY: 1.0,
            TRANSFER_MOONCAKE_TRANSPORT_FALLBACK_KEY: 0.0,
            TRANSFER_MOONCAKE_PROTOCOL_RDMA_KEY: is_rdma,
            TRANSFER_MOONCAKE_RDMA_REQUESTED_KEY: is_rdma,
            TRANSFER_MOONCAKE_RDMA_DEVICE_CONFIGURED_KEY: float(bool(self.rdma_devices)),
            _profile_key("put_register_count", "mooncake_buffer"): float(self._put_register_count),
            _profile_key("put_register_bytes", "mooncake_buffer"): float(self._put_register_bytes),
            _profile_key("get_register_count", "mooncake_buffer"): float(self._get_register_count),
            _profile_key("get_register_bytes", "mooncake_buffer"): float(self._get_register_bytes),
            _profile_key("put_capacity_bytes", "mooncake_buffer"): float(sum(self._put_buffer_capacities.values())),
            _profile_key("get_capacity_bytes", "mooncake_buffer"): float(self._get_buffer_capacity),
        }

    def _unregister_buffer(self, ptr: int) -> None:
        if ptr == 0:
            return
        ret = self.store.unregister_buffer(ptr)
        if ret != 0:
            logger.warning(f"Mooncake store unregister_buffer failed ptr={ptr} code={ret}")

    def _reserve_put_buffer(self, index: int, size: int) -> int:
        if self._put_buffers.get(index) is not None and self._put_buffer_capacities[index] >= size:
            return self._put_buffer_ptrs[index]
        self._unregister_buffer(self._put_buffer_ptrs.get(index, 0))
        capacity = max(size, self._put_buffer_capacities.get(index, 0) * 2, 1)
        buffer = ctypes.create_string_buffer(capacity)
        ptr = ctypes.addressof(buffer)
        ret = self.store.register_buffer(ptr, capacity)
        if ret != 0:
            raise RuntimeError(f"Mooncake store register_buffer failed ptr={ptr} size={capacity} code={ret}")
        self._put_buffers[index] = buffer
        self._put_buffer_ptrs[index] = ptr
        self._put_buffer_capacities[index] = capacity
        self._put_register_count += 1
        self._put_register_bytes += capacity
        logger.info(f"Mooncake put buffer registered index={index} requested={size} capacity={capacity}")
        return ptr

    def _reserve_get_buffer(self, size: int) -> int:
        if self._get_buffer is not None and self._get_buffer_capacity >= size:
            return self._get_buffer_ptr
        self._unregister_buffer(self._get_buffer_ptr)
        capacity = max(size, self._get_buffer_capacity * 2, 1)
        self._get_buffer = ctypes.create_string_buffer(capacity)
        self._get_buffer_ptr = ctypes.addressof(self._get_buffer)
        self._get_buffer_capacity = capacity
        ret = self.store.register_buffer(self._get_buffer_ptr, self._get_buffer_capacity)
        if ret != 0:
            failed_ptr = self._get_buffer_ptr
            failed_capacity = self._get_buffer_capacity
            self._get_buffer = None
            self._get_buffer_ptr = 0
            self._get_buffer_capacity = 0
            raise RuntimeError(
                f"Mooncake store register_buffer failed ptr={failed_ptr} size={failed_capacity} code={ret}"
            )
        self._get_register_count += 1
        self._get_register_bytes += self._get_buffer_capacity
        logger.info(f"Mooncake get buffer registered requested={size} capacity={self._get_buffer_capacity}")
        return self._get_buffer_ptr

    def put_payload(self, key: str, payload: dict[str, Any]) -> RolloutTransferHandle:
        meta_bytes = payload["meta_bytes"]
        bulk_buffer = payload.get("bulk_buffer")
        bulk_chunks = payload.get("bulk_chunks")
        buffer_specs = payload["buffer_specs"]
        layout_meta = pickle.dumps(
            {
                "meta_bytes": meta_bytes,
                "buffer_specs": buffer_specs,
                "transfer_stats": payload.get("transfer_stats", {}),
            },
            protocol=5,
        )
        ptrs: list[int] = []
        sizes: list[int] = []
        padded_sizes: list[int] = []
        actual_sizes: list[int] = []

        entries = [(layout_meta, {"section": "meta", "nbytes": len(layout_meta)})]
        if bulk_chunks is not None:
            for chunk_index, chunk in enumerate(bulk_chunks):
                if not chunk:
                    continue
                entries.append((chunk, {"section": "bulk", "nbytes": len(chunk), "chunk_index": chunk_index}))
        elif bulk_buffer is not None and buffer_specs:
            entries.append((bulk_buffer, {"section": "bulk", "nbytes": len(bulk_buffer), "chunk_offset": 0}))

        total_size = sum(
            MOONCAKE_BUFFER_HEADER_SIZE + _align_mooncake_buffer_size(len(chunk)) for chunk, _ in entries
        )
        base_ptr = self._reserve_put_buffer(0, total_size)
        offset = 0
        for chunk, spec in entries:
            size = len(chunk)
            padded_size = _align_mooncake_buffer_size(size)
            header_ptr = base_ptr + offset
            ctypes.memmove(header_ptr, _pack_mooncake_buffer_header(spec), MOONCAKE_BUFFER_HEADER_SIZE)
            offset += MOONCAKE_BUFFER_HEADER_SIZE

            payload_ptr = base_ptr + offset
            ctypes.memmove(payload_ptr, chunk, size)
            offset += padded_size

            ptrs.extend([header_ptr, payload_ptr])
            sizes.extend([MOONCAKE_BUFFER_HEADER_SIZE, padded_size])
            padded_sizes.append(padded_size)
            actual_sizes.append(size)

        ret = self.store.batch_put_from_multi_buffers([key], [ptrs], [sizes])
        if any(code != 0 for code in ret):
            raise RuntimeError(f"Mooncake store batch_put_from_multi_buffers failed for key={key} with code {ret}")
        return RolloutTransferHandle(
            backend="mooncake",
            protocol="v1",
            stage="",
            key=key,
            transport_info={
                "mode": self.mode(),
                "size": sum(MOONCAKE_BUFFER_HEADER_SIZE + size for size in padded_sizes),
                "meta_size": len(layout_meta),
                "padded_sizes": padded_sizes,
                "actual_sizes": actual_sizes,
            },
        )

    def put_bytes(self, key: str, payload: bytes) -> RolloutTransferHandle:
        size = len(payload)
        ptr = self._reserve_put_buffer(0, size)
        ctypes.memmove(ptr, payload, size)
        ret = self.store.put_from(key, ptr, size)
        if ret != 0:
            raise RuntimeError(f"Mooncake store put_from failed for key={key} with code {ret}")
        return RolloutTransferHandle(
            backend="mooncake",
            protocol="v1",
            stage="",
            key=key,
            transport_info={"mode": self.mode(), "size": size},
        )

    def get_payload(self, handle: RolloutTransferHandle) -> dict[str, Any]:
        assert handle.key is not None, "Mooncake store handle requires key"
        info = handle.transport_info or {}
        total_size = int(info.get("size", 0))
        padded_sizes = [int(size) for size in info["padded_sizes"]]
        actual_sizes = [int(size) for size in info["actual_sizes"]]
        if len(actual_sizes) > 2 and hasattr(self.store, "get_into_ranges"):
            return self._get_payload_with_ranges(handle.key, padded_sizes=padded_sizes, actual_sizes=actual_sizes)
        ptr = self._reserve_get_buffer(total_size)
        assert self._get_buffer is not None
        for attempt in range(5):
            ret = self.store.batch_get_into([handle.key], [ptr], [total_size])
            if ret and ret[0] >= 0:
                return self._decode_payload_from_get_buffer(padded_sizes=padded_sizes, actual_sizes=actual_sizes)
            time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"Mooncake store batch_get_into returned empty payload for key={handle.key} after retries")

    def get_bytes(self, handle: RolloutTransferHandle) -> bytes:
        assert handle.key is not None, "Mooncake store handle requires key"
        size = int((handle.transport_info or {}).get("size", 0))
        if size <= 0:
            return self._get_bytes_copy_fallback(handle)
        ptr = self._reserve_get_buffer(size)
        assert self._get_buffer is not None
        for attempt in range(5):
            read_size = self.store.get_into(handle.key, ptr, size)
            if read_size > 0:
                return self._get_buffer.raw[:read_size]
            time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"Mooncake store get_into returned empty payload for key={handle.key} after retries")

    def _get_payload_with_ranges(
        self, key: str, padded_sizes: list[int], actual_sizes: list[int]
    ) -> dict[str, Any]:
        meta_size = actual_sizes[0]
        meta_total_size = MOONCAKE_BUFFER_HEADER_SIZE + padded_sizes[0]
        meta_ptr = self._reserve_get_buffer(meta_total_size)
        assert self._get_buffer is not None
        metadata_range_seconds = 0.0
        for attempt in range(5):
            range_start = time.perf_counter()
            ret = self._get_ranges(
                meta_ptr, [[key]], [[[MOONCAKE_BUFFER_HEADER_SIZE]]], [[[MOONCAKE_BUFFER_HEADER_SIZE]]], [[[meta_size]]]
            )
            metadata_range_seconds += time.perf_counter() - range_start
            if ret and ret[0][0][0] >= 0:
                break
            time.sleep(0.1 * (attempt + 1))
        else:
            raise RuntimeError(f"Mooncake store range get returned empty metadata for key={key}")

        meta_view = memoryview(self._get_buffer)
        metadata = pickle.loads(bytes(meta_view[MOONCAKE_BUFFER_HEADER_SIZE:MOONCAKE_BUFFER_HEADER_SIZE + meta_size]))
        bulk_size = sum(actual_sizes[1:])
        ptr = self._reserve_get_buffer(bulk_size)
        assert self._get_buffer is not None
        dest_offsets: list[int] = []
        src_offsets: list[int] = []
        sizes: list[int] = []
        source_offset = meta_total_size
        dest_offset = 0
        for index in range(1, len(actual_sizes)):
            src_offsets.append(source_offset + MOONCAKE_BUFFER_HEADER_SIZE)
            dest_offsets.append(dest_offset)
            sizes.append(actual_sizes[index])
            dest_offset += actual_sizes[index]
            source_offset += MOONCAKE_BUFFER_HEADER_SIZE + padded_sizes[index]

        bulk_range_seconds = 0.0
        for attempt in range(5):
            range_start = time.perf_counter()
            ret = self._get_ranges(ptr, [[key]], [[dest_offsets]], [[src_offsets]], [[sizes]])
            bulk_range_seconds += time.perf_counter() - range_start
            if ret and all(code >= 0 for code in ret[0][0]):
                buffer_specs = []
                for spec in metadata["buffer_specs"]:
                    restored_spec = dict(spec)
                    restored_spec.pop("chunk_offset", None)
                    restored_spec.pop("chunk_index", None)
                    buffer_specs.append(restored_spec)
                transfer_stats = dict(metadata.get("transfer_stats", {}))
                transfer_stats[_profile_key("time_seconds", "mooncake_metadata_range_get")] = metadata_range_seconds
                transfer_stats[_profile_key("time_seconds", "mooncake_bulk_range_get")] = bulk_range_seconds
                transfer_stats[_profile_key("range_fragment_count", "mooncake_bulk_range_get")] = float(len(sizes))
                return {
                    "protocol": "v1",
                    "meta_bytes": metadata["meta_bytes"],
                    "bulk_buffer": memoryview(self._get_buffer)[:bulk_size],
                    "buffer_specs": buffer_specs,
                    "transfer_stats": transfer_stats,
                    "buffer_owner": self._get_buffer,
                }
            time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"Mooncake store range get returned empty payload for key={key} after retries")

    def _get_ranges(
        self,
        ptr: int,
        keys: list[list[str]],
        dest_offsets: list[list[list[int]]],
        src_offsets: list[list[list[int]]],
        sizes: list[list[list[int]]],
    ) -> list[list[list[int]]]:
        return self.store.get_into_ranges([ptr], keys, dest_offsets, src_offsets, sizes)

    def _decode_payload_from_get_buffer(self, padded_sizes: list[int], actual_sizes: list[int]) -> dict[str, Any]:
        assert self._get_buffer is not None
        buffer = memoryview(self._get_buffer)
        offset = MOONCAKE_BUFFER_HEADER_SIZE
        meta_bytes = bytes(buffer[offset:offset + actual_sizes[0]])
        offset += padded_sizes[0]
        buffer_specs: list[dict[str, Any]] = []
        metadata = pickle.loads(meta_bytes)
        bulk_buffer: Any = b""
        if len(actual_sizes) == 2:
            view_start = offset + MOONCAKE_BUFFER_HEADER_SIZE
            offset = view_start + padded_sizes[1]
            bulk_buffer = buffer[view_start:offset]
        elif len(actual_sizes) > 2:
            bulk_chunks = []
            for index in range(1, len(actual_sizes)):
                offset += MOONCAKE_BUFFER_HEADER_SIZE
                bulk_chunks.append(bytes(buffer[offset:offset + actual_sizes[index]]))
                offset += padded_sizes[index]
            bulk_buffer = b"".join(bulk_chunks)
        if len(actual_sizes) > 1:
            for spec in metadata["buffer_specs"]:
                restored_spec = dict(spec)
                restored_spec.pop("chunk_offset", None)
                restored_spec.pop("chunk_index", None)
                buffer_specs.append(restored_spec)
        decoded_payload = {
            "protocol": "v1",
            "meta_bytes": metadata["meta_bytes"],
            "bulk_buffer": bulk_buffer,
            "buffer_specs": buffer_specs,
            "transfer_stats": metadata.get("transfer_stats", {}),
        }
        if isinstance(bulk_buffer, memoryview):
            decoded_payload["buffer_owner"] = self._get_buffer
        return decoded_payload

    def _get_bytes_copy_fallback(self, handle: RolloutTransferHandle) -> bytes:
        assert handle.key is not None, "Mooncake store handle requires key"
        last_payload = b""
        for attempt in range(5):
            payload = self.store.get(handle.key)
            if payload:
                return payload
            last_payload = payload if payload is not None else b""
            time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(
            f"Mooncake store returned empty payload for key={handle.key} after retries; "
            f"last_payload_len={len(last_payload)}"
        )

    def cleanup(self, handle: RolloutTransferHandle) -> None:
        if handle.key is None:
            return
        self.store.remove(handle.key)


class MooncakeRolloutTransferBackend(RolloutTransferBackend):
    def __init__(self, protocol: str):
        if protocol != "v1":
            raise ValueError("mooncake transfer backend requires rollout_transfer_protocol='v1'")
        super().__init__(protocol=protocol)
        self._adapter: Optional[_MooncakeTransportAdapter] = None

    def _get_adapter(self) -> _MooncakeTransportAdapter:
        if self._adapter is not None:
            return self._adapter
        strict_mode = os.getenv("ROLL_MOONCAKE_STRICT", "0") == "1"
        try:
            self._adapter = _MooncakeStoreTransportAdapter()
        except Exception:
            if strict_mode:
                raise
            self._adapter = _MooncakeFallbackTransportAdapter()
        return self._adapter

    def put(self, data: DataProto, stage: str) -> RolloutTransferHandle:
        adapter = self._get_adapter()
        with Timer(logger=None) as serialize_timer:
            payload = data.to_transfer_payload(
                stage=stage,
                protocol=self.protocol,
                coalesce_bulk=None if isinstance(adapter, _MooncakeStoreTransportAdapter) else True,
            )
        key = f"rollout_{stage}_{uuid.uuid4().hex}"
        encoded_payload: Optional[bytes] = None
        try:
            with Timer(logger=None) as put_timer:
                if isinstance(adapter, _MooncakeStoreTransportAdapter):
                    handle = adapter.put_payload(key=key, payload=payload)
                else:
                    with Timer(logger=None) as backend_pickle_timer:
                        encoded_payload = pickle.dumps(payload, protocol=5)
                    handle = adapter.put_bytes(key=key, payload=encoded_payload)
        except Exception as exc:
            payload_bytes = int(payload.get("transfer_stats", {}).get(TRANSFER_BYTES_TOTAL_KEY, 0))
            raise RuntimeError(
                f"Mooncake rollout transfer put failed for stage={stage}, key={key}, payload_bytes={payload_bytes}"
            ) from exc
        stats = dict(payload.get("transfer_stats", {}))
        if data.meta_info.get("rollout_transfer_metrics_enabled", False):
            logical_bytes = int(stats.get(TRANSFER_BYTES_TOTAL_KEY, 0))
            wire_bytes = int((handle.transport_info or {}).get("size", logical_bytes))
            stats.update(
                {
                    TRANSFER_BACKEND_MOONCAKE_KEY: 1.0,
                    TRANSFER_BACKEND_RAY_OPTIMIZED_KEY: 0.0,
                    TRANSFER_TIME_SERIALIZE_KEY: _safe_float(serialize_timer.last),
                    TRANSFER_TIME_PUT_KEY: _safe_float(put_timer.last),
                    **adapter.metrics(),
                }
            )
            if isinstance(adapter, _MooncakeStoreTransportAdapter):
                bulk_chunk_count = len(payload.get("bulk_chunks") or [])
                _maybe_set_profiling_metric(
                    stats,
                    data,
                    _profile_key("buffer_count", "backend_payload_multi_buffer"),
                    float(bulk_chunk_count + 1),
                )
                _maybe_set_profiling_metric(
                    stats,
                    data,
                    _profile_key("scatter_max_bytes", "backend_payload_multi_buffer"),
                    float(_mooncake_scatter_max_bytes()),
                )
                _maybe_set_profiling_metric(
                    stats,
                    data,
                    _profile_key("entry_count", "backend_payload_multi_buffer"),
                    float(len((handle.transport_info or {}).get("actual_sizes", []))),
                )
            else:
                _maybe_set_profiling_metric(
                    stats,
                    data,
                    _profile_key("time_seconds", "backend_payload_pickle"),
                    _safe_float(backend_pickle_timer.last),
                )
            _add_transfer_throughput_metrics(
                stats,
                logical_bytes=logical_bytes,
                wire_bytes=wire_bytes,
                serialize_seconds=_safe_float(serialize_timer.last),
                put_seconds=_safe_float(put_timer.last),
                e2e_seconds=_safe_float(serialize_timer.last) + _safe_float(put_timer.last),
            )
        handle.stage = stage
        handle.stats = stats
        handle.transport_info = {"mode": adapter.mode(), **(handle.transport_info or {})}
        return handle

    def get(self, handle: RolloutTransferHandle) -> DataProto:
        adapter = self._get_adapter()
        rss_before = _cpu_rss_gb()
        backend_unpickle_seconds: Optional[float] = None
        with Timer(logger=None) as get_timer:
            if isinstance(adapter, _MooncakeStoreTransportAdapter) and "padded_sizes" in (handle.transport_info or {}):
                payload = adapter.get_payload(handle)
            else:
                encoded_payload = adapter.get_bytes(handle)
                with Timer(logger=None) as backend_unpickle_timer:
                    payload = pickle.loads(encoded_payload)
                backend_unpickle_seconds = _safe_float(backend_unpickle_timer.last)
        with Timer(logger=None) as deserialize_timer:
            if "buffer_owner" in payload and os.getenv("ROLL_MOONCAKE_DIRECT_VIEW", "0") == "1":
                data = DataProto.from_transfer_payload_view(payload, buffer_owner=payload["buffer_owner"])
            else:
                data = DataProto.from_transfer_payload(payload)
        if data.meta_info.get("rollout_transfer_metrics_enabled", False):
            stats = dict(handle.stats)
            logical_bytes = int(stats.get(TRANSFER_BYTES_TOTAL_KEY, 0))
            wire_bytes = int((handle.transport_info or {}).get("size", logical_bytes))
            stats[TRANSFER_TIME_GET_KEY] = _safe_float(get_timer.last)
            stats[TRANSFER_TIME_DESERIALIZE_KEY] = _safe_float(deserialize_timer.last)
            stats.update(adapter.metrics())
            if backend_unpickle_seconds is not None:
                _maybe_set_profiling_metric(
                    stats,
                    data,
                    _profile_key("time_seconds", "backend_payload_unpickle"),
                    backend_unpickle_seconds,
                )
            _maybe_set_profiling_metric(stats, data, _profile_key("peak_rss_gb", "backend_get"), max(rss_before, _cpu_rss_gb()))
            _maybe_set_profiling_metric(stats, data, _profile_key("time_seconds", "backend_get"), _safe_float(get_timer.last) + _safe_float(deserialize_timer.last))
            _add_transfer_throughput_metrics(
                stats,
                logical_bytes=logical_bytes,
                wire_bytes=wire_bytes,
                get_seconds=_safe_float(get_timer.last),
                deserialize_seconds=_safe_float(deserialize_timer.last),
                e2e_seconds=_safe_float(get_timer.last) + _safe_float(deserialize_timer.last),
            )
            self._maybe_record_transfer_metrics(data, stats)
            _record_transfer_profile_sample(data, stats)
        return data

    def cleanup(self, handle: RolloutTransferHandle) -> None:
        self._get_adapter().cleanup(handle)


class LegacyRolloutTransferBackend(RolloutTransferBackend):
    def put(self, data: DataProto, stage: str) -> RolloutTransferHandle:
        return RolloutTransferHandle(
            backend="legacy",
            protocol="legacy",
            stage=stage,
            payload=data.to_transfer_payload(stage=stage, protocol="legacy"),
            stats={
                TRANSFER_STAGE_KEY: stage,
                TRANSFER_BACKEND_KEY: "legacy",
                TRANSFER_PROTOCOL_KEY: "legacy",
            },
        )

    def get(self, handle: RolloutTransferHandle) -> DataProto:
        assert handle.payload is not None, "legacy transfer handle requires inline payload"
        data = DataProto.from_transfer_payload(handle.payload)
        self._maybe_record_transfer_metrics(data, handle.stats)
        return data


class RayOptimizedRolloutTransferBackend(RolloutTransferBackend):
    def __init__(self, protocol: str):
        if protocol != "v1":
            raise ValueError("ray_optimized transfer backend requires rollout_transfer_protocol='v1'")
        super().__init__(protocol=protocol)

    def put(self, data: DataProto, stage: str) -> RolloutTransferHandle:
        with Timer(logger=None) as serialize_timer:
            payload = data.to_transfer_payload(stage=stage, protocol=self.protocol)
        with Timer(logger=None) as put_timer:
            obj_ref = ray.put(payload)
        stats = dict(payload.get("transfer_stats", {}))
        if data.meta_info.get("rollout_transfer_metrics_enabled", False):
            wire_bytes = len(pickle.dumps(payload, protocol=5))
            logical_bytes = int(stats.get(TRANSFER_BYTES_TOTAL_KEY, 0))
            stats.update(
                {
                    TRANSFER_BACKEND_RAY_OPTIMIZED_KEY: 1.0,
                    TRANSFER_BACKEND_MOONCAKE_KEY: 0.0,
                    TRANSFER_TIME_SERIALIZE_KEY: _safe_float(serialize_timer.last),
                    TRANSFER_TIME_PUT_KEY: _safe_float(put_timer.last),
                }
            )
            _maybe_set_profiling_metric(stats, data, _profile_key("wire_bytes_estimated", "ray_put"), 1.0)
            _add_transfer_throughput_metrics(
                stats,
                logical_bytes=logical_bytes,
                wire_bytes=wire_bytes,
                serialize_seconds=_safe_float(serialize_timer.last),
                put_seconds=_safe_float(put_timer.last),
                e2e_seconds=_safe_float(serialize_timer.last) + _safe_float(put_timer.last),
            )
        return RolloutTransferHandle(
            backend="ray_optimized",
            protocol=self.protocol,
            stage=stage,
            obj_ref=obj_ref,
            stats=stats,
        )

    def get(self, handle: RolloutTransferHandle) -> DataProto:
        assert handle.obj_ref is not None, "ray_optimized transfer handle requires object ref"
        rss_before = _cpu_rss_gb()
        with Timer(logger=None) as get_timer:
            payload = ray.get(handle.obj_ref)
        with Timer(logger=None) as deserialize_timer:
            data = DataProto.from_transfer_payload(payload)
        if data.meta_info.get("rollout_transfer_metrics_enabled", False):
            stats = dict(handle.stats)
            logical_bytes = int(stats.get(TRANSFER_BYTES_TOTAL_KEY, 0))
            wire_bytes = int(stats.get(TRANSFER_BYTES_WIRE_KEY, logical_bytes))
            stats[TRANSFER_TIME_GET_KEY] = _safe_float(get_timer.last)
            stats[TRANSFER_TIME_DESERIALIZE_KEY] = _safe_float(deserialize_timer.last)
            _maybe_set_profiling_metric(stats, data, _profile_key("peak_rss_gb", "backend_get"), max(rss_before, _cpu_rss_gb()))
            _maybe_set_profiling_metric(stats, data, _profile_key("time_seconds", "backend_get"), _safe_float(get_timer.last) + _safe_float(deserialize_timer.last))
            _add_transfer_throughput_metrics(
                stats,
                logical_bytes=logical_bytes,
                wire_bytes=wire_bytes,
                get_seconds=_safe_float(get_timer.last),
                deserialize_seconds=_safe_float(deserialize_timer.last),
                e2e_seconds=_safe_float(get_timer.last) + _safe_float(deserialize_timer.last),
            )
            self._maybe_record_transfer_metrics(data, stats)
            _record_transfer_profile_sample(data, stats)
        return data


_ROLLOUT_TRANSFER_BACKEND_CACHE: dict[tuple[str, str], RolloutTransferBackend] = {}


def get_rollout_transfer_backend(backend_name: str, protocol: str) -> RolloutTransferBackend:
    cache_key = (backend_name, protocol)
    if cache_key in _ROLLOUT_TRANSFER_BACKEND_CACHE:
        return _ROLLOUT_TRANSFER_BACKEND_CACHE[cache_key]
    if backend_name == "legacy":
        backend = LegacyRolloutTransferBackend(protocol="legacy")
    elif backend_name == "ray_optimized":
        backend = RayOptimizedRolloutTransferBackend(protocol=protocol)
    elif backend_name == "mooncake":
        backend = MooncakeRolloutTransferBackend(protocol=protocol)
    else:
        raise ValueError(f"Unsupported rollout transfer backend: {backend_name}")
    _ROLLOUT_TRANSFER_BACKEND_CACHE[cache_key] = backend
    return backend


def uses_optimized_rollout_transfer(backend_name: str) -> bool:
    return backend_name != "legacy"


def materialize_rollout_transfer(handle: Union[DataProto, RolloutTransferHandle], backend_name: str, protocol: str) -> DataProto:
    if isinstance(handle, DataProto):
        return handle
    start_time = time.perf_counter()
    backend = get_rollout_transfer_backend(backend_name=backend_name, protocol=protocol)
    try:
        data = backend.get(handle)
        _record_profile_metrics(
            data,
            {
                _profile_key("time_seconds", "materialize_rollout_transfer"): time.perf_counter() - start_time,
                _profile_key("peak_rss_gb", "materialize_rollout_transfer"): _cpu_rss_gb(),
            },
        )
        return data
    finally:
        backend.cleanup(handle)


class ObjectRefWrap:
    def __init__(self, obj_ref: ray.ObjectRef, collected=False):
        self.obj_ref = obj_ref
        self.collected = collected
