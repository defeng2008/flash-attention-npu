"""MLA user-scenario configs (from ``usershape.txt``), shared by baseline scripts.

Task 0.3: torch-only description of the two real MLA user shapes, importable by
both the ops-transformer FIA baseline (``bench_mla_fia.py``) and later
flash-attn-npu side scripts:

  * ``prefill`` -> FIA ``ROPE_COMBINE_D128`` (qkHeadDim=192 > vHeadDim=128)
  * ``decode``  -> FIA ``ROPE_SPLIT_D512``   (qkHeadDim=512 == vHeadDim=512, rope=64)

``make_inputs`` performs the 512/64 split, so the merged ``(..., 576)`` tensors
are never fed to FIA directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# softmax_scale from usershape.txt. Owned by the framework: it is close to but
# not equal to the DeepSeek MLA scale 192 ** -0.5 (= 0.0721688); it is
# 2.0047x that value (== 1/sqrt(47.773)). Both sides of the benchmark pass this
# same explicit value, so the comparison is like-for-like.
SOFTMAX_SCALE = 0.14467962580268923

# Fixed causal mask side used by the prefill user scenario (sparse_mode=3).
CAUSAL_MASK_SIZE = 2048


@dataclass(frozen=True)
class MLAScenario:
    """One MLA user scenario, expressed in the FIA MLA calling convention."""

    name: str
    form: str  # "prefill" or "decode"
    dtype: torch.dtype
    total_q: int
    total_kv: int
    num_heads_q: int
    num_heads_kv: int
    headdim_q_combined: int  # 192 (prefill) / 576 (decode)
    headdim_k_combined: int
    headdim_v: int
    rope_dim: int  # 0 (prefill) / 64 (decode)
    cu_seqlens_q: Tuple[int, ...]
    cu_seqlens_k: Tuple[int, ...]
    causal: bool
    softmax_scale: float = SOFTMAX_SCALE
    window_size: Tuple[int, int] = (-1, -1)
    softcap: float = 0.0
    input_layout: str = "TND"

    @property
    def qk_nope_dim(self) -> int:
        """Head dim after removing the rope part (== FIA qkHeadDim)."""
        return self.headdim_q_combined - self.rope_dim

    @property
    def out_shape(self) -> Tuple[int, int, int]:
        return (self.total_q, self.num_heads_q, self.headdim_v)

    @property
    def fia_form(self) -> str:
        return "ROPE_COMBINE_D128" if self.form == "prefill" else "ROPE_SPLIT_D512"


# --- usershape.txt prefill: q/k=192, v=128, heads=32, causal, total=198 --------
PREFILL_SCENARIO = MLAScenario(
    name="prefill_mla_combine_d128",
    form="prefill",
    dtype=torch.bfloat16,
    total_q=198,
    total_kv=198,
    num_heads_q=32,
    num_heads_kv=32,
    headdim_q_combined=192,
    headdim_k_combined=192,
    headdim_v=128,
    rope_dim=0,
    cu_seqlens_q=(0, 198),
    cu_seqlens_k=(0, 198),
    causal=True,
)

# --- usershape.txt decode: q/k=576 (split 512+64), v=512, KV heads=1 ----------
_DECODE_FIELDS = dict(
    form="decode",
    dtype=torch.bfloat16,
    total_q=1,
    num_heads_q=32,
    num_heads_kv=1,
    headdim_q_combined=576,
    headdim_k_combined=576,
    headdim_v=512,
    rope_dim=64,
    cu_seqlens_q=(0, 1),
    causal=False,
)

DECODE_SCENARIO = MLAScenario(
    name="decode_mla_split_d512",
    total_kv=1025,
    cu_seqlens_k=(0, 1025),
    **_DECODE_FIELDS,
)

# Variant kept for later cross-check against the "996" value seen in the log.
DECODE_SCENARIO_S2_996 = MLAScenario(
    name="decode_mla_split_d512_s2_996",
    total_kv=996,
    cu_seqlens_k=(0, 996),
    **_DECODE_FIELDS,
)

SCENARIOS: Dict[str, MLAScenario] = {
    PREFILL_SCENARIO.name: PREFILL_SCENARIO,
    DECODE_SCENARIO.name: DECODE_SCENARIO,
    DECODE_SCENARIO_S2_996.name: DECODE_SCENARIO_S2_996,
}


def make_causal_mask(size: int = CAUSAL_MASK_SIZE, device: Optional[str] = "npu") -> torch.Tensor:
    """Upper-triangular causal mask (True == masked), as required by sparse_mode=3."""
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask if device is None else mask.to(device)


def make_inputs(scenario: MLAScenario, seed: int = 0, device: Optional[str] = "npu") -> dict:
    """Build FIA-ready inputs for ``scenario``.

    Returns a dict of tensors already split into FIA's ``query/key/value`` and
    (decode) ``query_rope/key_rope`` form, the int32 ``cu_seqlens_*`` tensors,
    plus ``softmax_scale``. Small-magnitude normals keep bf16 scores stable.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=gen, dtype=torch.float32).mul_(0.1).to(scenario.dtype)

    cu_q = torch.tensor(scenario.cu_seqlens_q, dtype=torch.int32)
    cu_k = torch.tensor(scenario.cu_seqlens_k, dtype=torch.int32)

    if scenario.form == "prefill":
        inputs = dict(
            q=rand((scenario.total_q, scenario.num_heads_q, scenario.headdim_q_combined)),
            k=rand((scenario.total_kv, scenario.num_heads_kv, scenario.headdim_k_combined)),
            v=rand((scenario.total_kv, scenario.num_heads_kv, scenario.headdim_v)),
            atten_mask=make_causal_mask(device=None),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
        )
    else:
        r = scenario.rope_dim
        q_combined = rand((scenario.total_q, scenario.num_heads_q, scenario.headdim_q_combined))
        k_combined = rand((scenario.total_kv, scenario.num_heads_kv, scenario.headdim_k_combined))
        # Merged layout keeps rope in the last ``r`` dims: [..., :D-r] + [..., D-r:].
        inputs = dict(
            q=q_combined[..., :-r].contiguous(),
            q_rope=q_combined[..., -r:].contiguous(),
            k=k_combined[..., :-r].contiguous(),
            k_rope=k_combined[..., -r:].contiguous(),
            v=rand((scenario.total_kv, scenario.num_heads_kv, scenario.headdim_v)),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
        )

    if device is not None:
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    inputs["softmax_scale"] = scenario.softmax_scale
    return inputs
