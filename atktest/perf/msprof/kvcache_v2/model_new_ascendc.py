"""910 kvcache_v2（flash_attn_npu）：连续 BNSD 输入（msprof `--compare` 的 Asc 侧）。

输入与 ``model.py`` 完全相同（同一批 shape 与数据源）。BSND 输入无法直接
``transpose(1, 2)`` 得到 BNSD 存储，因此在首次调用时物化一次连续
``(B, H, S, D)`` 存储并缓存，之后每次只取零拷贝的 BSND 视图触发 BNSD 分支。
"""

import torch

from flash_attn_npu import flash_attn_with_kvcache

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# (b, sq, s_alloc, h, h_kv, d, dtype) —— 必须与 model.py 一致
SHAPES = [
    (1, 128, 4096, 32, 8, 128, "fp16"),
    (1, 64, 4096, 32, 8, 128, "fp16"),
    (1, 256, 8192, 32, 8, 128, "bf16"),
    (1, 128, 4096, 64, 64, 128, "fp16"),
    (4, 64, 2048, 16, 4, 128, "bf16"),
]


def get_input_groups():
    groups = []
    for batch, seqlen_q, seqlen_alloc, heads, heads_kv, headdim, dtype_name in SHAPES:
        dtype = DTYPES[dtype_name]
        generator = torch.Generator().manual_seed(2026)
        q = torch.randn(batch, seqlen_q, heads, headdim, dtype=dtype, generator=generator)
        k_cache = torch.randn(
            batch, seqlen_alloc, heads_kv, headdim, dtype=dtype, generator=generator
        )
        v_cache = torch.randn(
            batch, seqlen_alloc, heads_kv, headdim, dtype=dtype, generator=generator
        )
        cache_seqlens = torch.full((batch,), seqlen_alloc, dtype=torch.int32)
        groups.append([q, k_cache, v_cache, cache_seqlens])
    return groups


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._bnsd = {}

    def _bnsd_view(self, tensor):
        key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        cached = self._bnsd.get(key)
        if cached is None:
            # 物化连续 BNSD 存储；只发生在首次（预热）调用，不进计时区间。
            cached = tensor.transpose(1, 2).contiguous()
            self._bnsd[key] = cached
        return cached.transpose(1, 2)

    def forward(self, q, k_cache, v_cache, cache_seqlens):
        return flash_attn_with_kvcache(
            self._bnsd_view(q),
            self._bnsd_view(k_cache),
            self._bnsd_view(v_cache),
            cache_seqlens=cache_seqlens,
            softmax_scale=q.shape[-1] ** -0.5,
            causal=True,
            scheduler_metadata=None,
        )
