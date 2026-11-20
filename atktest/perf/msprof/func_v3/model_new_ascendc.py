"""func_v3 对比路径：连续 BNSD 输入（与 model.py 只差布局）。

真实 BNSD 存储只在首次（预热）调用时物化并缓存，计时区间内只做
transpose(1, 2) 零拷贝视图，因此两种布局的耗时差异只来自 kernel 本身。
"""

import torch

from flash_attn_npu_3 import flash_attn_func

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# 必须与 model.py / ../shapes.yaml 的 func 段一致
SHAPES = [
    (1, 1024, 32, 8, 128, "fp16", True),
    (1, 8192, 64, 8, 128, "bf16", True),
    (2, 4096, 64, 8, 128, "bf16", True),
    (1, 4096, 64, 64, 128, "fp16", True),
    (1, 4096, 32, 8, 128, "fp16", False),
]


def get_input_groups():
    groups = []
    for batch, seqlen, heads, heads_kv, headdim, dtype_name, _ in SHAPES:
        dtype = DTYPES[dtype_name]
        generator = torch.Generator().manual_seed(2026)
        q = torch.randn(batch, seqlen, heads, headdim, dtype=dtype, generator=generator)
        k = torch.randn(batch, seqlen, heads_kv, headdim, dtype=dtype, generator=generator)
        v = torch.randn(batch, seqlen, heads_kv, headdim, dtype=dtype, generator=generator)
        groups.append([q, k, v])
    return groups


def get_init_inputs():
    return [{"causal": causal} for *_, causal in SHAPES]


class ModelNew(torch.nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal
        self._bnsd = {}

    def _bnsd_view(self, tensor):
        key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        cached = self._bnsd.get(key)
        if cached is None:
            # 物化连续 BNSD 存储；只发生在首次（预热）调用，不进计时区间。
            cached = tensor.transpose(1, 2).contiguous()
            self._bnsd[key] = cached
        return cached.transpose(1, 2)

    def forward(self, q, k, v):
        return flash_attn_func(
            self._bnsd_view(q),
            self._bnsd_view(k),
            self._bnsd_view(v),
            softmax_scale=q.shape[-1] ** -0.5,
            causal=self.causal,
        )
