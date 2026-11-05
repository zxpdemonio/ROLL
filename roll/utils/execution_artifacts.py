import base64
import binascii
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch


ATTENTION_SELECTION_KEY = "attention_selection"
ATTENTION_SELECTION_VALID_COUNT_KEY = "attention_selection_valid_count"
EXECUTION_ARTIFACT_PROFILE_ID_KEY = "execution_artifact_profile_id"
ATTENTION_SELECTION_PROFILE_KEY = "_attention_selection_profile"


@dataclass(frozen=True)
class AttentionSelectionProfile:
    """Job-level description of one sparse-attention selection layout."""

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
        if not np.iinfo(np.int32).min <= self.padding_value <= np.iinfo(np.int32).max:
            raise ValueError(f"padding_value does not fit int32: {self.padding_value}")

    @property
    def row_width(self) -> int:
        return self.num_layers * self.topk

    def estimate_payload_bytes(self, num_tokens: int, batch_size: int = 1) -> int:
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return batch_size * num_tokens * self.row_width * np.dtype(self.stored_dtype).itemsize

    def validate_producer(self, backend_family: str) -> None:
        if self.backend_family != backend_family:
            raise ValueError(
                f"attention selection profile backend {self.backend_family} does not match producer {backend_family}"
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
        value = np.frombuffer(value, dtype=np.int32)

    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"attention selection must contain integers, got {value.dtype}")
        selection = torch.tensor(value, dtype=torch.int32)
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
    packed = torch.full(
        (len(selections), max_rows, profile.num_layers, profile.topk),
        fill_value=profile.padding_value,
        dtype=torch.int32,
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
        packed[index, :valid_count].copy_(selection)

    return packed.contiguous(), torch.tensor(valid_counts, dtype=torch.int32)
