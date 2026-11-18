"""ATK cpu/gpu/npu adapter for fa3_varlen (single_benchmark_cpu)."""

import inspect
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(PROJECT_ROOT / "flash-attention-npu"))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_fa3_varlen import flash_attn_varlen_func as cpu_impl

from flash_attn_npu_3 import flash_attn_varlen_func as npu_fn

FIA_NODE_NAME = ""


@register("fa3_varlen_mla_accuracy")
class Fa3VarlenApi(BaseApi):
    _ACCEPTED = set(inspect.signature(npu_fn).parameters)

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
            if FIA_NODE_NAME and getattr(self, "name", "") == FIA_NODE_NAME:
                return self._fia(kw)
            return npu_fn(**self._move({n: v for n, v in kw.items() if n in self._ACCEPTED},
                                       f"npu:{self.device_id}"))
        raise RuntimeError(f"Unsupported backend: {self.device}")

    def _gpu(self, kw):
        """官方 TriDao GPU 标杆（惰性导入，无 GPU 环境时不执行）。"""
        try:
            import flash_attn_interface as gpu_mod
        except ImportError:
            from flash_attn_3 import flash_attn_interface as gpu_mod
        gpu_fn = gpu_mod.flash_attn_with_kvcache
        accepted = set(inspect.signature(gpu_fn).parameters)
        call = {n: v for n, v in kw.items() if n in accepted}
        return gpu_fn(**self._move(call, f"cuda:{self.device_id}"))

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
        """CANN npu_fused_infer_attention_score 标杆（映射对齐 tests/perf/bench_mla_vs_fia.py）。"""
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
            return fia(qv.to(dev), v.to(dev), v.to(dev),        # TND decode (absorbed)
                       query_rope=q.to(dev), key_rope=k.to(dev),
                       actual_seq_lengths=cu_q.tolist(), actual_seq_lengths_kv=cu_k.tolist(),
                       num_heads=qv.shape[1], scale=scale, input_layout="TND",
                       num_key_value_heads=v.shape[1], sparse_mode=0)[0]
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
        return fia(qv.to(dev), v.to(dev), v.to(dev),           # BSND decode (absorbed)
                   query_rope=q.to(dev), key_rope=k.to(dev), num_heads=qv.shape[2], scale=scale,
                   input_layout="BSND", num_key_value_heads=v.shape[2],
                   sparse_mode=0)[0]
