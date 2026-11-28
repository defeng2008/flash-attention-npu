"""ATK cpu/npu adapter for fa3_kvcache (double_benchmark)."""

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

from benchmark_fa3_kvcache import flash_attn_with_kvcache as cpu_impl
from benchmark_fa3_kvcache import flash_attn_with_kvcache_fp32 as cpu_fp32_impl

# nodes_accuracy_cpu.yaml 里 fp32 节点的 name（同组最后一个节点 = ATK 的 cpu_benchmark）
FP32_NODE_NAME = "fp32"


def _npu_fn():
    """惰性导入 NPU 候选实现：CPU 标杆节点不应该依赖 torch_npu，
    模块级导入会让 @register 在纯 CPU 机器上直接失败。"""
    from flash_attn_npu_3 import flash_attn_with_kvcache as fn
    return fn


@register("fa3_kvcache_mla_accuracy")
class Fa3KvcacheApi(BaseApi):
    _NPU_ACCEPTED = None

    @classmethod
    def _npu_accepted(cls):
        """NPU 入口参数集合：首次调用才导入 NPU 实现，纯 CPU 机器上模块导入不应失败。"""
        if cls._NPU_ACCEPTED is None:
            cls._NPU_ACCEPTED = set(inspect.signature(_npu_fn()).parameters)
        return cls._NPU_ACCEPTED

    @staticmethod
    def _move(kw, dev):
        return {n: (v.to(dev) if isinstance(v, torch.Tensor) else v) for n, v in kw.items()}

    def _case_dtype(self):
        """用例配置里精度张量（q）的 dtype（bf16/fp16），默认 bf16。

        ATK 的 benchmark 任务会把 bf16/fp16 输入统一升到 fp32（InputDataset.get_benchmark_data），
        标杆节点的输出随之为 fp32；而候选节点仍按用例原精度输出。比较器要求两侧 dtype
        完全一致（base_compare.check_output_size），所以标杆结果必须按用例 dtype 回落。
        """
        inputs = getattr(getattr(self.task_result, "case_config", None), "inputs", None) or []
        for entry in inputs:
            if not isinstance(entry, list) and getattr(entry, "name", None) == "q":
                name = str(getattr(entry, "dtype", "")).lower()
                return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(name, torch.bfloat16)
        return torch.bfloat16

    def __call__(self, input_data, with_output=False):
        kw = dict(input_data.kwargs)
        if self.device == "cpu":
            # fp32 节点是纯 fp32 高精度标杆；其余 cpu 节点是低精度标杆（P 按用例 dtype 量化）
            fn = cpu_fp32_impl if getattr(self, "name", "") == FP32_NODE_NAME else cpu_impl
            return fn(**kw).to(self._case_dtype())
        if self.device == "npu":
            npu_fn = _npu_fn()
            return npu_fn(**self._move({n: v for n, v in kw.items() if n in self._npu_accepted()},
                                       f"npu:{self.device_id}"))
        raise RuntimeError(f"Unsupported backend: {self.device}")
