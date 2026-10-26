"""ATK CPU/GPU/NPU adapter for v2 flash_attn_func."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_func_v2 import flash_attn_func as cpu_flash_attn_func


@register("flash_attn_func_v2_accuracy")
class FlashAttnFuncV2Api(BaseApi):
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
            if os.environ.get("FA_ATK_USE_SCHEDULER_METADATA", "1") != "0":
                return flash_attn_func(**kwargs)
            original = interface.get_scheduler_metadata
            interface.get_scheduler_metadata = lambda *args, **params: None
            try:
                return flash_attn_func(**kwargs)
            finally:
                interface.get_scheduler_metadata = original
        raise RuntimeError(f"Unsupported backend: {self.device}")
