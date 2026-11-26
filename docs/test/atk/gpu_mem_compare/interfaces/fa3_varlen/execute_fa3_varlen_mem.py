"""ATK cpu/gpu/npu adapter for fa3_varlen (gpu_mem_compare)."""

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

from benchmark_fa3_varlen_mem import flash_attn_varlen_func as cpu_impl


def _npu_fn():
    """惰性导入 NPU 候选实现：CPU/GPU 机器没有 torch_npu，
    模块级导入会让 @register 直接失败。"""
    from flash_attn_npu_3 import flash_attn_varlen_func as fn
    return fn


FIA_NODE_NAME = "fia_mla"


@register("fa3_varlen_mla_mem")
class Fa3VarlenMem(BaseApi):
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
        if self.device == "npu":
            if getattr(self, "name", "") == FIA_NODE_NAME:
                return self._fia(kw)
            npu_fn = _npu_fn()
            return npu_fn(**self._move({n: v for n, v in kw.items() if n in self._npu_accepted()},
                                       f"npu:{self.device_id}"))
        raise RuntimeError(f"Unsupported backend: {self.device}")

    def _case_dtype(self):
        """用例配置里精度张量（q）的 dtype（bf16/fp16），默认 bf16。

        ATK 标杆路径会把 bf16/fp16 输入都升精度到 fp32（见 InputDataset.get_benchmark_data），
        单看张量 dtype 已无法区分用例原本是 bf16 还是 fp16，必须回到 case_config 读取。
        """
        inputs = getattr(getattr(self.task_result, "case_config", None), "inputs", None) or []
        for entry in inputs:
            if not isinstance(entry, list) and getattr(entry, "name", None) == "q":
                name = str(getattr(entry, "dtype", "")).lower()
                return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(name, torch.bfloat16)
        return torch.bfloat16

    def _fia(self, kw):
        """CANN npu_fused_infer_attention_score 标杆（映射对齐 docs/test/perf/bench_mla_vs_fia.py）。"""
        import torch_npu
        fia = torch_npu.npu_fused_infer_attention_score
        dev = f"npu:{self.device_id}"
        target_dtype = self._case_dtype()

        def _dt(t):
            # 标杆路径下 ATK 会把 bf16/fp16 输入都升精度到 fp32，而 aclnnFusedInferAttentionScore
            # 的 MLA 分支只支持 bf16/fp16，这里按用例配置的精度回落（bf16 用例->bf16，fp16 用例->fp16）。
            if isinstance(t, torch.Tensor) and t.dtype in (torch.float32, torch.float64):
                return t.to(target_dtype)
            return t

        q = _dt(kw["q"])
        k = _dt(kw.get("k", kw.get("k_cache")))
        v = _dt(kw.get("v", kw.get("v_cache")))
        qv = _dt(kw.get("qv"))
        scale = kw.get("softmax_scale")
        causal = bool(kw.get("causal", False))
        if kw.get("cu_seqlens_q") is not None:      # TND 布局
            cu_q, cu_k = kw["cu_seqlens_q"], kw["cu_seqlens_k"]
            if qv is None:
                half, rope = q.shape[-1] - 64, 64
                q_nope, q_rope = q[..., :half].contiguous(), q[..., half:].contiguous()
                k_nope, k_rope = k[..., :half].contiguous(), k[..., half:].contiguous()
                mask = None
                if causal:
                    mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool().to(dev)
                return fia(q_nope.to(dev), k_nope.to(dev), v.to(dev),
                           atten_mask=mask, query_rope=q_rope.to(dev), key_rope=k_rope.to(dev),
                           actual_seq_lengths=cu_q.tolist(), actual_seq_lengths_kv=cu_k.tolist(),
                           num_heads=q.shape[1], scale=scale, input_layout="TND",
                           num_key_value_heads=k.shape[1], sparse_mode=3 if causal else 0)[0]
            mask = None
            if causal:
                mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool().to(dev)
            return fia(qv.to(dev), v.to(dev), v.to(dev),        # TND decode (absorbed)
                       atten_mask=mask, query_rope=q.to(dev), key_rope=k.to(dev),
                       actual_seq_lengths=cu_q.tolist(), actual_seq_lengths_kv=cu_k.tolist(),
                       num_heads=qv.shape[1], scale=scale, input_layout="TND",
                       num_key_value_heads=v.shape[1], sparse_mode=3 if causal else 0)[0]
        if qv is None:                              # BSND prefill
            half, rope = q.shape[-1] - 64, 64
            q_nope, q_rope = q[..., :half].contiguous().to(dev), q[..., half:].contiguous().to(dev)
            k_nope, k_rope = k[..., :half].contiguous().to(dev), k[..., half:].contiguous().to(dev)
            mask = None
            if causal:
                mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool().to(dev)
            return fia(q_nope, k_nope, v.to(dev), atten_mask=mask,
                       query_rope=q_rope, key_rope=k_rope, num_heads=q.shape[2], scale=scale,
                       input_layout="BSND", num_key_value_heads=k.shape[2],
                       sparse_mode=3 if causal else 0)[0]
        mask = None
        if causal:
            mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool().to(dev)
        return fia(qv.to(dev), v.to(dev), v.to(dev),           # BSND decode (absorbed)
                   atten_mask=mask, query_rope=q.to(dev), key_rope=k.to(dev),
                   num_heads=qv.shape[2], scale=scale,
                   input_layout="BSND", num_key_value_heads=v.shape[2],
                   sparse_mode=3 if causal else 0)[0]
