"""ATK cpu/gpu/npu adapter for fa4_varlen (double_benchmark)."""

import inspect
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _parent in _HERE.parents:
    if (_parent / "flash-attention-npu").is_dir():
        sys.path.insert(0, str(_parent / "flash-attention-npu"))
        break

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_fa4_varlen import flash_attn_varlen_func as cpu_impl


def _npu_fn():
    """惰性导入 NPU 候选实现：CPU/GPU 机器没有 torch_npu，
    模块级导入会让 @register 直接失败。"""
    from flash_attn_npu_4 import flash_attn_varlen_func as fn
    return fn


@register("fa4_varlen_mla_accuracy")
class Fa4VarlenApi(BaseApi):
    _NPU_ACCEPTED = None

    @classmethod
    def _npu_accepted(cls):
        """NPU 入口参数集合：首次调用才导入 NPU 实现，CPU/GPU 机器上模块导入不应失败。"""
        if cls._NPU_ACCEPTED is None:
            cls._NPU_ACCEPTED = set(inspect.signature(_npu_fn()).parameters)
        return cls._NPU_ACCEPTED

    @staticmethod
    def _move(kw, dev):
        return {n: (v.to(dev) if isinstance(v, torch.Tensor) else v) for n, v in kw.items()}

    def __call__(self, input_data, with_output=False):
        kw = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_impl(**kw)
        if self.device == "gpu":
            return self._gpu(kw)
        if self.device == "npu":
            npu_fn = _npu_fn()
            return npu_fn(**self._move({n: v for n, v in kw.items() if n in self._npu_accepted()},
                                       f"npu:{self.device_id}"))
        raise RuntimeError(f"Unsupported backend: {self.device}")

    def _gpu(self, kw):
        """官方 TriDao GPU 标杆（惰性导入，无 GPU 环境时不执行）。"""
        from flash_attn.cute import interface as gpu_mod
        gpu_fn = gpu_mod.flash_attn_func
        accepted = set(inspect.signature(gpu_fn).parameters)
        call = {n: v for n, v in kw.items() if n in accepted}
        return gpu_fn(**self._move(call, f"cuda:{self.device_id}"))
