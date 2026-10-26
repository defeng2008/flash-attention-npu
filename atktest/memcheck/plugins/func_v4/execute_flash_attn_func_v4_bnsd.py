"""ATK CPU/NPU adapter for the 950 v4 BNSD ``flash_attn_func`` cases.

The NPU branch only changes how the operands are laid out in memory: each of q/k/v
is handed in as the BSND-shaped transposed view of a contiguous BNSD storage, which
is exactly what the backend's stride probe expects.  The CPU branch keeps the raw
contiguous BSND operands, so the ATK npu-vs-cpu comparison verifies that the BNSD
path returns the same logical ``(B, S, H, D)`` result.  The op builds its own
scheduler metadata from the pre-detection BSND shapes, so nothing else has to change.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_func_v4_bnsd import flash_attn_func as cpu_flash_attn_func


def _bnsd_view(tensor):
    """Hand a contiguous BNSD ``(B, H, S, D)`` storage in as its ``(B, S, H, D)`` view."""
    return tensor.transpose(1, 2).contiguous().transpose(1, 2)


@register("flash_attn_func_v4_bnsd_accuracy")
class FlashAttnFuncV4BnsdApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_func(**kwargs)
        if self.device == "npu":
            from flash_attn_npu_4 import flash_attn_func

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            for name in ("q", "k", "v"):
                kwargs[name] = _bnsd_view(kwargs[name])
            return flash_attn_func(**kwargs)
        raise RuntimeError(f"Unsupported backend: {self.device}")
