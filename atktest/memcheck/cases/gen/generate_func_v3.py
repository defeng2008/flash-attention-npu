"""Cross-input constraints for v3 ``flash_attn_func``."""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


HEADS = (1, 2, 4, 8, 16)


def _nearest(value, choices):
    return min(choices, key=lambda candidate: abs(candidate - max(int(value), 1)))


def _inputs(case_config):
    return {item.name: item for item in case_config.inputs if not isinstance(item, list)}


@GENERATOR_REGISTRY.register("flash_attn_func_v3_constraints")
class FlashAttnFuncV3Constraints(CaseGenerator):
    def after_case_config(self, case_config):
        inputs = _inputs(case_config)
        batch, seqlen_q, heads_q, head_dim = inputs["q"].shape
        seqlen_k = inputs["k"].shape[1]
        valid_kv_heads = tuple(head for head in HEADS if head <= heads_q and heads_q % head == 0)
        heads_kv = _nearest(inputs["k"].shape[2], valid_kv_heads)
        dtype = inputs["q"].dtype

        if inputs["causal"].range_values is True:
            seqlen_q = min(seqlen_q, seqlen_k)

        inputs["q"].shape = [batch, seqlen_q, heads_q, head_dim]
        for name, shape in (
            ("k", [batch, seqlen_k, heads_kv, head_dim]),
            ("v", [batch, seqlen_k, heads_kv, head_dim]),
        ):
            inputs[name].dtype = dtype
            inputs[name].shape = shape

        # qv is exposed but not consumed by the current 910B/C kernel.
        inputs["qv"].dtype = dtype
        inputs["qv"].shape = [batch, seqlen_q, heads_q, head_dim]
        inputs["qv"].range_values = "default"
        for name in ("q_descale", "k_descale", "v_descale"):
            inputs[name].range_values = "default"
        return case_config
