# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""User-scenario definitions for MLA on Ascend 910 (source: usershape.txt).

These two shapes are the acceptance targets for the FA3/FA4 MLA work:
  * prefill  (training / unabsorbed): q/k head_dim 192, v head_dim 128
  * decode   (inference / absorbed):  q/k head_dim 576 (rope 64 + nope 512),
                                       v head_dim 512, KV heads == 1

The 576 "merged rope" layout is split into an independent q(64)/qv(512) and
k(64)/v(512) before it enters the MLA path.

softmax_scale is passed explicitly with the DeepSeek convention 192 ** -0.5 so
that the training (prefill) and inference (decode) forms share one scale; this
is what makes the train/infer consistency check meaningful.
"""

from dataclasses import dataclass

# 192 ** -0.5, identical to usershape.txt.
DEEPSEEK_SOFTMAX_SCALE = 0.14467962580268923


@dataclass(frozen=True)
class MlaScenario:
    name: str
    form: str  # "prefill" | "decode"
    total_q: int
    total_kv: int
    heads_q: int
    heads_kv: int
    head_dim_q: int
    head_dim_k: int
    head_dim_v: int
    rope_dim: int
    causal: bool
    softmax_scale: float


def _q_dim(s: MlaScenario) -> int:
    # Across the public interfaces the MLA "q" carries only the rope part for
    # the absorbed form; the nope part travels in qv.
    return s.rope_dim if s.form == "decode" else s.head_dim_q


def _qv_dim(s: MlaScenario) -> int:
    return s.head_dim_q - s.rope_dim if s.form == "decode" else 0


def _k_dim(s: MlaScenario) -> int:
    return s.rope_dim if s.form == "decode" else s.head_dim_k


PREFILL = MlaScenario(
    name="prefill",
    form="prefill",
    total_q=198,
    total_kv=198,
    heads_q=32,
    heads_kv=32,
    head_dim_q=192,
    head_dim_k=192,
    head_dim_v=128,
    rope_dim=0,
    causal=True,
    softmax_scale=DEEPSEEK_SOFTMAX_SCALE,
)

DECODE = MlaScenario(
    name="decode",
    form="decode",
    total_q=1,
    total_kv=1025,
    heads_q=32,
    heads_kv=1,
    head_dim_q=576,
    head_dim_k=576,
    head_dim_v=512,
    rope_dim=64,
    causal=False,
    softmax_scale=DEEPSEEK_SOFTMAX_SCALE,
)

USER_SCENARIOS = [PREFILL, DECODE]


def scenario_summary(s: MlaScenario) -> str:
    if s.form == "decode":
        return (
            f"{s.name}: q=({s.total_q},{s.heads_q},{_q_dim(s)}) "
            f"qv=({s.total_q},{s.heads_q},{_qv_dim(s)}) "
            f"k=({s.total_kv},{s.heads_kv},{_k_dim(s)}) "
            f"v=({s.total_kv},{s.heads_kv},{s.head_dim_v}) "
            f"out=({s.total_q},{s.heads_q},{s.head_dim_v}) scale={s.softmax_scale}"
        )
    return (
        f"{s.name}: q=({s.total_q},{s.heads_q},{s.head_dim_q}) "
        f"k=({s.total_kv},{s.heads_kv},{s.head_dim_k}) "
        f"v=({s.total_kv},{s.heads_kv},{s.head_dim_v}) "
        f"out=({s.total_q},{s.heads_q},{s.head_dim_v}) scale={s.softmax_scale}"
    )


if __name__ == "__main__":
    for scenario in USER_SCENARIOS:
        print(scenario_summary(scenario))
