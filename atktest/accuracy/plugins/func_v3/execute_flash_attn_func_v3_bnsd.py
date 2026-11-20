"""ATK CPU/GPU/NPU adapter for the BNSD ``flash_attn_func`` (v3) accuracy cases.

See ``execute_flash_attn_func_v2_bnsd.py`` for the BNSD rationale.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_func_v3 import flash_attn_func as cpu_flash_attn_func


def _bnsd_view(tensor):
    """Hand a contiguous BNSD ``(B, H, S, D)`` storage in as its ``(B, S, H, D)`` view."""
    view = tensor.transpose(1, 2).contiguous().transpose(1, 2)
    assert not view.is_contiguous(), "BNSD view degenerated to contiguous; the case missed the BNSD path"
    return view


@register("flash_attn_func_v3_bnsd_accuracy")
class FlashAttnFuncV3BnsdApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_func(**kwargs)
        if self.device == "gpu":
            # TriDao official FA3 (hopper) benchmark; imported lazily because
            # flash_attn_interface is only importable on Hopper GPU hosts.
            try:
                import flash_attn_interface as gpu_fa3
            except ImportError:
                from flash_attn_3 import flash_attn_interface as gpu_fa3

            kwargs = {
                name: value.to(f"cuda:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            return gpu_fa3.flash_attn_func(**kwargs)
        if self.device == "npu":
            from flash_attn_npu_3 import flash_attn_func

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            for name in ("q", "k", "v"):
                kwargs[name] = _bnsd_view(kwargs[name])
            if os.environ.get("FA_ATK_USE_SCHEDULER_METADATA", "1") != "0":
                return flash_attn_func(**kwargs)
            # 910/950 的接口实现在不同模块里，按实际加载的模块取
            interface = sys.modules[flash_attn_func.__module__]
            original = interface.get_scheduler_metadata
            interface.get_scheduler_metadata = lambda *args, **params: None
            try:
                return flash_attn_func(**kwargs)
            finally:
                interface.get_scheduler_metadata = original
        raise RuntimeError(f"Unsupported backend: {self.device}")
