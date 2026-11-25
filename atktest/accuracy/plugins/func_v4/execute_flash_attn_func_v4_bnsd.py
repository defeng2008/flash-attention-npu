"""ATK CPU/GPU/NPU adapter for the BNSD ``flash_attn_func`` (v4) accuracy cases.

See ``execute_flash_attn_func_v2_bnsd.py`` for the BNSD rationale.  The op builds
its own scheduler metadata from the pre-detection BSND shapes, so nothing else has
to change.  ``return_lse`` is an NPU-only spelling and is dropped for the official
GPU reference.
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
    view = tensor.transpose(1, 2).contiguous().transpose(1, 2)
    assert not view.is_contiguous(), "BNSD view degenerated to contiguous; the case missed the BNSD path"
    return view


@register("flash_attn_func_v4_bnsd_accuracy")
class FlashAttnFuncV4BnsdApi(BaseApi):
    def init_by_input_data(self, input_data):
        """ATK 在 to_device 之后、内存采样之前回调：NPU 上只做一次 BNSD 布局转换。

        避免每轮 forward 都执行 ``.contiguous()``，额外多分配一份 BNSD 内存。
        """
        if self.device == "npu":
            for name in ("q", "k", "v"):
                input_data.kwargs[name] = _bnsd_view(input_data.kwargs[name])
        return input_data

    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_func(**kwargs)
        if self.device == "gpu":
            # TriDao official FA3 (hopper) reference; imported lazily because
            # flash_attn_interface is only importable on Hopper GPU hosts.
            try:
                import flash_attn_interface as gpu_fa3
            except ImportError:
                from flash_attn_3 import flash_attn_interface as gpu_fa3

            kwargs = {
                name: value.to(f"cuda:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            kwargs.pop("return_lse", None)
            return gpu_fa3.flash_attn_func(**kwargs)
        if self.device == "npu":
            from flash_attn_npu_4 import flash_attn_func

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            return flash_attn_func(**kwargs)
        raise RuntimeError(f"Unsupported backend: {self.device}")
