"""Cross-input constraints for the BNSD ``flash_attn_func`` (v2) accuracy cases.

The public API keeps its ``(B, S, H, D)`` signature, so the generated operands are
described in BSND terms; the ATK adapter re-views them as contiguous BNSD storages
right before calling the NPU op.

Two extra constraints make the layout probe fire at all: the adapter's BNSD view
short-circuits on ``is_contiguous()``, and the ``contiguous BNSD.transpose(1, 2)``
view is only non-contiguous when the sequence and head dims are both >= 2.  So the
generator clamps ``seqlen`` and ``heads`` (q heads *and* kv heads) to at least 2.
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


# BNSD needs a per-head plane, so a single head would collapse the stride pattern.
HEADS = (2, 4, 8, 16)
MIN_SEQLEN = 2


def _nearest(value, choices):
    return min(choices, key=lambda candidate: abs(candidate - max(int(value), 1)))


def _inputs(case_config):
    return {item.name: item for item in case_config.inputs if not isinstance(item, list)}


@GENERATOR_REGISTRY.register("flash_attn_func_v2_bnsd_constraints")
class FlashAttnFuncV2BnsdConstraints(CaseGenerator):
    def after_case_config(self, case_config):
        inputs = _inputs(case_config)
        batch, seqlen_q, heads_q, head_dim = inputs["q"].shape
        seqlen_k = inputs["k"].shape[1]

        heads_q = max(int(heads_q), 2)
        valid_heads = tuple(head for head in HEADS if head <= heads_q and heads_q % head == 0)
        heads_q = _nearest(heads_q, valid_heads)
        heads_kv = _nearest(inputs["k"].shape[2], valid_heads)
        dtype = inputs["q"].dtype

        seqlen_q = max(seqlen_q, MIN_SEQLEN)
        seqlen_k = max(seqlen_k, MIN_SEQLEN)
        # causal + Sq > Sk is a separately tracked known NPU issue.
        if inputs["causal"].range_values is True:
            seqlen_q = min(seqlen_q, seqlen_k)

        inputs["q"].shape = [batch, seqlen_q, heads_q, head_dim]
        for name, shape in (
            ("k", [batch, seqlen_k, heads_kv, head_dim]),
            ("v", [batch, seqlen_k, heads_kv, head_dim]),
        ):
            inputs[name].dtype = dtype
            inputs[name].shape = shape

        inputs["alibi_slopes"].range_values = "default"
        return case_config
