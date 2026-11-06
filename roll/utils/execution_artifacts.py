import base64
import binascii
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


ATTENTION_SELECTION_KEY = "attention_selection"
ATTENTION_SELECTION_VALID_COUNT_KEY = "attention_selection_valid_count"
EXECUTION_ARTIFACT_PROFILE_ID_KEY = "execution_artifact_profile_id"
ATTENTION_SELECTION_PROFILE_KEY = "_attention_selection_profile"
ACTOR_REPLAY_ARTIFACT_KEYS = (
    "routed_experts",
    ATTENTION_SELECTION_KEY,
    ATTENTION_SELECTION_VALID_COUNT_KEY,
)


def get_expert_dtype(num_moe_experts: int) -> torch.dtype:
    """Choose expert-index storage from the model's static expert count."""
    if num_moe_experts <= 256:
        return torch.uint8
    if num_moe_experts <= 65536:
        return torch.uint16
    return torch.uint32


@dataclass(frozen=True)
class AttentionSelectionProfile:
    """Job-level description of one sparse-attention selection layout.

    ``row_alignment_bytes`` pads the stored row stride. Mooncake owns the
    independent base-address alignment of its registered buffers.
    """

    profile_id: str
    model_family: str
    backend_family: str
    num_layers: int
    topk: int
    protocol_version: int = 1
    coordinate_space: Literal["token", "block", "compressed_kv", "pool", "region"] = "token"
    stored_dtype: Literal["int32"] = "int32"
    padding_value: int = -1
    preserves_selection_order: bool = True
    row_alignment_bytes: int | None = None

    def __post_init__(self):
        if self.protocol_version != 1:
            raise ValueError(f"unsupported attention selection protocol_version: {self.protocol_version}")
        for name, value in (("profile_id", self.profile_id), ("model_family", self.model_family), ("backend_family", self.backend_family)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if self.stored_dtype != "int32":
            raise ValueError(f"unsupported stored_dtype: {self.stored_dtype}")
        if self.coordinate_space not in {"token", "block", "compressed_kv", "pool", "region"}:
            raise ValueError(f"unsupported coordinate_space: {self.coordinate_space}")
        if not isinstance(self.preserves_selection_order, bool):
            raise ValueError("preserves_selection_order must be a bool")
        if self.row_alignment_bytes is not None:
            alignment = self.row_alignment_bytes
            if (
                isinstance(alignment, bool)
                or not isinstance(alignment, int)
                or alignment < np.dtype(self.stored_dtype).itemsize
                or alignment & (alignment - 1)
            ):
                raise ValueError(
                    "row_alignment_bytes must be a power of two no smaller than the stored dtype"
                )
        if not np.iinfo(np.int32).min <= self.padding_value <= np.iinfo(np.int32).max:
            raise ValueError(f"padding_value does not fit int32: {self.padding_value}")

    @property
    def row_width(self) -> int:
        return self.num_layers * self.topk

    @property
    def row_stride_bytes(self) -> int:
        row_bytes = self.row_width * np.dtype(self.stored_dtype).itemsize
        if self.row_alignment_bytes is None:
            return row_bytes
        return (row_bytes + self.row_alignment_bytes - 1) // self.row_alignment_bytes * self.row_alignment_bytes

    @property
    def storage_row_width(self) -> int:
        return self.row_stride_bytes // np.dtype(self.stored_dtype).itemsize

    def estimate_payload_bytes(self, num_tokens: int, batch_size: int = 1) -> int:
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return batch_size * num_tokens * self.row_stride_bytes

    def estimate_packed_bytes(self, num_tokens: int, batch_size: int = 1) -> int:
        """Estimate the canonical DataProto tensors, including valid row counts."""
        selection_bytes = self.estimate_payload_bytes(num_tokens, batch_size)
        valid_count_bytes = batch_size * np.dtype(np.int32).itemsize
        return selection_bytes + valid_count_bytes

    def max_concurrent_requests(self, num_tokens: int, max_bytes: int) -> int:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        per_request = self.estimate_payload_bytes(num_tokens)
        if per_request == 0:
            return 0
        return max_bytes // per_request

    def validate_producer(self, backend_family: str) -> None:
        if self.backend_family != backend_family:
            raise ValueError(
                f"attention selection profile backend {self.backend_family} does not match producer {backend_family}"
            )

    def validate_vllm_profile(self, value: dict) -> None:
        """Validate the homogeneous raw DeepSeek profile returned by vLLM.

        QSA/Pool/Region codecs deliberately fail here; they need their own
        source-aware layout adapter instead of being mistaken for token IDs.
        """
        if not isinstance(value, dict):
            raise TypeError(f"vLLM indexer_profile must be a dict, got {type(value)}")
        if value.get("kind") != "indexer" or value.get("codec_version") != 1:
            raise ValueError("vLLM indexer_profile has an unsupported kind or codec version")
        if not isinstance(value.get("model_identity"), str) or not value["model_identity"]:
            raise ValueError("vLLM indexer_profile requires model_identity")
        if value.get("storage_dtype") != self.stored_dtype:
            raise ValueError(
                f"vLLM indexer_profile dtype {value.get('storage_dtype')} != {self.stored_dtype}"
            )
        sources = value.get("sources")
        if not isinstance(sources, list) or len(sources) != self.num_layers:
            raise ValueError(
                f"vLLM indexer_profile sources {0 if not isinstance(sources, list) else len(sources)} "
                f"!= profile num_layers {self.num_layers}"
            )
        source_layer_ids = []
        for source in sources:
            if not isinstance(source, dict):
                raise TypeError(f"vLLM indexer source must be a dict, got {type(source)}")
            layer_id = source.get("layer_id")
            if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
                raise ValueError(f"vLLM indexer source has invalid layer_id {layer_id}")
            source_layer_ids.append(layer_id)
            if self.coordinate_space != "token" or source.get("semantic") != "logical_token_id":
                raise ValueError(
                    "the raw DeepSeek adapter requires logical token coordinates"
                )
            if source.get("codec", "int32") != "int32":
                raise ValueError(
                    f"vLLM indexer codec {source.get('codec')} requires a source-aware adapter"
                )
            shape = source.get("shape")
            valid_shape = (
                isinstance(shape, (list, tuple))
                and bool(shape)
                and all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in shape)
            )
            if not valid_shape or math.prod(shape) != self.topk:
                raise ValueError(
                    f"vLLM indexer source shape {shape} does not match raw topk {self.topk}"
                )
        if len(set(source_layer_ids)) != len(source_layer_ids):
            raise ValueError("vLLM indexer source layer_ids must be unique")
        expected_layer_ids = list(range(self.num_layers))
        if source_layer_ids != expected_layer_ids:
            raise ValueError(
                "the raw DeepSeek adapter requires one source per model layer in layer order; "
                f"got {source_layer_ids}"
            )
        for source in sources:
            layer_id = source["layer_id"]
            consumers = source.get("consumer_layer_ids", [layer_id])
            if not isinstance(consumers, (list, tuple)) or list(consumers) != [layer_id]:
                raise ValueError(
                    "shared sparse-attention sources require a source-aware training adapter; "
                    f"layer {layer_id} has consumers {consumers}"
                )


def decode_attention_selection(
    value,
    profile: AttentionSelectionProfile,
    *,
    expected_rows: int | None = None,
) -> torch.Tensor:
    """Decode a producer payload into the canonical contiguous int32 layout [T, L, K]."""

    if isinstance(value, str):
        try:
            value = base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
            raise ValueError("attention selection is not valid base64") from exc
    if isinstance(value, (bytes, bytearray, memoryview)):
        if len(value) % np.dtype(np.int32).itemsize != 0:
            raise ValueError("attention selection byte length is not aligned to int32")
        # Buffer inputs may be immutable. Copy once so the returned Torch tensor
        # never aliases read-only memory.
        value = np.frombuffer(value, dtype=np.int32).copy()

    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"attention selection must contain integers, got {value.dtype}")
        array = value
        if array.dtype != np.int32:
            array = array.astype(np.int32, copy=False)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if not array.flags.writeable:
            array = array.copy()
        # vLLM's in-process API already returns a CPU int32 NumPy array. Keep a
        # view until pack_attention_selection copies it into the final batch;
        # an eager torch.tensor() copy doubles the transient top-2048 payload.
        selection = torch.from_numpy(array)
    elif isinstance(value, torch.Tensor):
        if value.dtype.is_floating_point or value.dtype.is_complex or value.dtype == torch.bool:
            raise TypeError(f"attention selection must contain integers, got {value.dtype}")
        selection = value.to(dtype=torch.int32).contiguous().clone()
    else:
        raise TypeError(f"unsupported attention selection payload: {type(value)}")

    row_width = profile.row_width
    if selection.numel() % row_width != 0:
        raise ValueError(
            f"attention selection has {selection.numel()} values, not divisible by row width {row_width}"
        )
    rows = selection.numel() // row_width
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"attention selection rows {rows} != expected {expected_rows}")

    return selection.reshape(rows, profile.num_layers, profile.topk).contiguous()


def pack_attention_selection(
    selections: list[torch.Tensor],
    valid_counts: list[int],
    sequence_length: int,
    profile: AttentionSelectionProfile,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-sample [T, L, K] selections to a DataProto batch tensor."""

    if len(selections) != len(valid_counts):
        raise ValueError(f"selection count {len(selections)} != valid-count size {len(valid_counts)}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    max_rows = sequence_length - 1
    if profile.storage_row_width == profile.row_width:
        packed = torch.full(
            (len(selections), max_rows, profile.num_layers, profile.topk),
            fill_value=profile.padding_value,
            dtype=torch.int32,
        )
        semantic_rows = packed
    else:
        packed = torch.full(
            (len(selections), max_rows, profile.storage_row_width),
            fill_value=profile.padding_value,
            dtype=torch.int32,
        )
        semantic_rows = packed[..., : profile.row_width].view(
            len(selections), max_rows, profile.num_layers, profile.topk
        )
    for index, (selection, valid_count) in enumerate(zip(selections, valid_counts)):
        if valid_count < 0:
            raise ValueError(f"attention selection valid_count must be non-negative, got {valid_count}")
        if valid_count > max_rows:
            raise ValueError(f"attention selection rows {valid_count} exceed packed capacity {max_rows}")
        expected_shape = (valid_count, profile.num_layers, profile.topk)
        if not (
            isinstance(selection, torch.Tensor)
            and selection.dtype == torch.int32
            and selection.is_contiguous()
            and tuple(selection.shape) == expected_shape
        ):
            selection = decode_attention_selection(selection, profile, expected_rows=valid_count)
        semantic_rows[index, :valid_count].copy_(selection)

    return packed.contiguous(), torch.tensor(valid_counts, dtype=torch.int32)


def unpack_attention_selection(
    value: torch.Tensor,
    profile: AttentionSelectionProfile,
) -> torch.Tensor:
    """Return the semantic ``[B, T, L, K]`` view of a stored selection tensor."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"attention selection must be a torch.Tensor, got {type(value)}")
    if value.dtype != torch.int32:
        raise TypeError(f"attention selection must use int32 storage, got {value.dtype}")
    if not value.is_contiguous():
        raise ValueError("attention selection storage must be contiguous")

    if profile.storage_row_width == profile.row_width:
        expected_tail = (profile.num_layers, profile.topk)
        if value.ndim != 4 or tuple(value.shape[-2:]) != expected_tail:
            raise ValueError(
                "attention selection must have shape [B, T, L, K] with "
                f"L/K={expected_tail}, got {tuple(value.shape)}"
            )
        return value

    if value.ndim != 3 or value.shape[-1] != profile.storage_row_width:
        raise ValueError(
            "aligned attention selection must have shape [B, T, row_stride_elements] with "
            f"row_stride_elements={profile.storage_row_width}, got {tuple(value.shape)}"
        )
    return value[..., : profile.row_width].view(
        value.shape[0], value.shape[1], profile.num_layers, profile.topk
    )
