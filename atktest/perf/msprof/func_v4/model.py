"""func_v4 标杆路径：BSND 连续输入。

msprof wrapper 按 case 序号调用 get_input_groups() 取真实输入、调用
get_init_inputs() 取该 case 的 causal 开关；SHAPES 必须与 ../shapes.yaml 的
func 段以及 func_v4_perf_cases.jsonl 完全一致。
"""

import torch

from flash_attn_npu_4 import flash_attn_func

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# (b, s, h, h_kv, d, dtype, causal)
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


class Model(torch.nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return flash_attn_func(
            q, k, v, softmax_scale=q.shape[-1] ** -0.5, causal=self.causal
        )
