"""Cross-input constraints for v2 ``flash_attn_with_kvcache``."""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


CACHE_CAPACITIES = (16, 32, 64, 96, 128, 256)
HEADS = (1, 2, 4, 8, 16)


def _nearest(value, choices):
    return min(choices, key=lambda candidate: abs(candidate - max(int(value), 1)))


def _inputs(case_config):
    return {item.name: item for item in case_config.inputs if not isinstance(item, list)}


@GENERATOR_REGISTRY.register("flash_attn_with_kvcache_v2_constraints")
class FlashAttnKvCacheV2Constraints(CaseGenerator):
    def after_case_config(self, case_config):
        inputs = _inputs(case_config)
        batch, seqlen_q, heads_q, head_dim = inputs["q"].shape
        capacity = inputs["k_cache"].shape[1]
        valid_kv_heads = tuple(head for head in HEADS if head <= heads_q and heads_q % head == 0)
        heads_kv = _nearest(inputs["k_cache"].shape[2], valid_kv_heads)
        dtype = inputs["q"].dtype

        if inputs["causal"].range_values is True and capacity < seqlen_q:
            capacity = min(value for value in CACHE_CAPACITIES if value >= seqlen_q)

        for name in ("k_cache", "v_cache"):
            inputs[name].dtype = dtype
            inputs[name].shape = [batch, capacity, heads_kv, head_dim]
        inputs["cache_seqlens"].dtype = "int32"
        inputs["cache_seqlens"].shape = [batch]
        lower = seqlen_q if inputs["causal"].range_values else 1
        # Non-paged B>1 currently requires every effective length to equal capacity.
        inputs["cache_seqlens"].range_values = (
            [capacity, capacity] if batch > 1 else [lower, capacity]
        )

        new_length = inputs["k"].shape[1]
        for name in ("k", "v"):
            inputs[name].dtype = dtype
            inputs[name].shape = [batch, new_length, heads_kv, head_dim]
            inputs[name].range_values = "default"
        for name in (
            "rotary_cos", "rotary_sin", "cache_batch_idx", "cache_leftpad",
            "block_table", "alibi_slopes", "scheduler_metadata",
        ):
            inputs[name].range_values = "default"
        return case_config
