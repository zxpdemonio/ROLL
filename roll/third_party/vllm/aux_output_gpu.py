"""GPU array operations for the existing vLLM auxiliary block algorithm.

Only the worker/artifacts modules receive this dispatcher. Ordinary NumPy
arrays retain their existing behavior; no global NumPy or request logic changes.
"""

from contextlib import contextmanager
from threading import local

import numpy as np
import torch

_context = local()


@contextmanager
def gpu_allocation(writer):
    previous = getattr(_context, "writer", None)
    _context.writer = writer
    try:
        yield
    finally:
        _context.writer = previous


def current_writer():
    return getattr(_context, "writer", None)


class GPURows:
    """Small array adapter; converting it into host bytes is prohibited."""

    def __init__(self, tensor, ready=None):
        self.tensor = tensor
        self.ready = ready

    def __len__(self):
        return len(self.tensor)

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def dtype(self):
        return np.dtype(str(self.tensor.dtype).removeprefix("torch."))

    def __array__(self, *args, **kwargs):
        raise TypeError("GPU auxiliary rows cannot be converted to a host ndarray")

    def __getitem__(self, item):
        return type(self)(self.tensor[item], self.ready)

    def __setitem__(self, item, value):
        self.tensor[item].copy_(value.tensor)
        if self.tensor.is_cuda:
            # A sample=0 partial tail may never reach a PUT/publication fence.
            torch.cuda.current_stream(self.tensor.device).synchronize()

    def copy(self):
        return type(self)(self.tensor.clone(memory_format=torch.contiguous_format))

    def view(self, dtype):
        return type(self)(self.tensor.view(getattr(torch, np.dtype(dtype).name)), self.ready)

    def reshape(self, *shape):
        return type(self)(self.tensor.reshape(*shape), self.ready)

    def tobytes(self, order="C"):
        # publish_aux_output enqueues the result and immediately reuses pool
        # slots. Own a separate allocation, not merely a view of that slot.
        if order != "C":
            raise ValueError("GPU auxiliary blocks require C order")
        owned = self.copy()
        if owned.tensor.is_cuda:
            owned.ready = torch.cuda.Event()
            owned.ready.record(torch.cuda.current_stream(owned.tensor.device))
            # vLLM releases snapshot/pool slots immediately after publication.
            # Finish the read of those slots before allowing their reuse.
            owned.ready.synchronize()
        owned.tensor = owned.tensor.view(torch.uint8).reshape(-1)
        return owned


class GPUArrayOps:
    def __getattr__(self, name):
        return getattr(np, name)

    def empty(self, shape, dtype):
        if current_writer() is None:
            return np.empty(shape, dtype=dtype)
        import math

        current_writer().validate_workspace(
            math.prod(shape) * np.dtype(dtype).itemsize, math.prod(shape[2:]) * np.dtype(dtype).itemsize
        )
        return GPURows(torch.empty(shape, dtype=getattr(torch, np.dtype(dtype).name), device="cuda"))

    def asarray(self, value, *args, **kwargs):
        return value if isinstance(value, GPURows) else np.asarray(value, *args, **kwargs)

    def ascontiguousarray(self, value, *args, **kwargs):
        return (
            GPURows(value.tensor.contiguous())
            if isinstance(value, GPURows)
            else np.ascontiguousarray(value, *args, **kwargs)
        )

    def concatenate(self, values, axis=0):
        if any(isinstance(value, GPURows) for value in values):
            if not all(isinstance(value, GPURows) for value in values):
                raise TypeError("Cannot combine CPU and GPU auxiliary rows")
            result = torch.cat([value.tensor for value in values], dim=axis)
            if result.is_cuda:
                torch.cuda.current_stream(result.device).synchronize()
            return GPURows(result)
        return np.concatenate(values, axis=axis)

    def frombuffer(self, value, dtype):
        return value.view(dtype) if isinstance(value, GPURows) else np.frombuffer(value, dtype=dtype)
