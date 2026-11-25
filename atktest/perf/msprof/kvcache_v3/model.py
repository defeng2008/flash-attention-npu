"""950 v3 kvcache：BSND 基准位（msprof `--compare` 的 Ref 侧）。

同一批 5 条 shape 由 ``get_input_groups()`` 生成，msprof 按 case 索引取第 i 组；
``kvcache_v3_perf_cases.jsonl`` 只负责枚举用例数量与报告里的 shape/dtype。
张量按 ``(B, S, H, D)`` 连续生成后直接传入，走 BSND 分支。
"""

import torch

from flash_attn_npu_3 import flash_attn_with_kvcache

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# (b, sq, s_alloc, h, h_kv, d, dtype)
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


class Model(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, cache_seqlens):
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=q.shape[-1] ** -0.5,
            causal=True,
            scheduler_metadata=None,
        )
