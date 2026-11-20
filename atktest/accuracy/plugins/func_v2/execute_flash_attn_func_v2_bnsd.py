"""ATK CPU/GPU/NPU adapter for the BNSD ``flash_attn_func`` (v2) accuracy cases.

The NPU branch only changes how the operands are laid out in memory: each of q/k/v
is handed in as the BSND-shaped transposed view of a contiguous BNSD storage, which
is exactly what the backend's stride probe expects.  The CPU/GPU branches keep the
raw contiguous BSND operands, so the comparison verifies that the BNSD path returns
the same logical ``(B, S, H, D)`` result.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_func_v2 import flash_attn_func as cpu_flash_attn_func


def _bnsd_view(tensor):
    """Hand a contiguous BNSD ``(B, H, S, D)`` storage in as its ``(B, S, H, D)`` view."""
    view = tensor.transpose(1, 2).contiguous().transpose(1, 2)
    assert not view.is_contiguous(), "BNSD view degenerated to contiguous; the case missed the BNSD path"
    return view


@register("flash_attn_func_v2_bnsd_accuracy")
class FlashAttnFuncV2BnsdApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_func(**kwargs)
        if self.device == "gpu":
            # TriDao official FA2 benchmark; imported lazily because the
            # flash_attn CUDA extension is only importable on GPU hosts.
            from flash_attn import flash_attn_func as gpu_flash_attn_func

            kwargs = {
                name: value.to(f"cuda:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            return gpu_flash_attn_func(**kwargs)
        if self.device == "npu":
            import flash_attn_npu.flash_attn_npu_interface as interface
            from flash_attn_npu import flash_attn_func

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            for name in ("q", "k", "v"):
                kwargs[name] = _bnsd_view(kwargs[name])
            if os.environ.get("FA_ATK_USE_SCHEDULER_METADATA", "1") != "0":
                return flash_attn_func(**kwargs)
            original = interface.get_scheduler_metadata
            interface.get_scheduler_metadata = lambda *args, **params: None
            try:
                return flash_attn_func(**kwargs)
            finally:
                interface.get_scheduler_metadata = original
        raise RuntimeError(f"Unsupported backend: {self.device}")
