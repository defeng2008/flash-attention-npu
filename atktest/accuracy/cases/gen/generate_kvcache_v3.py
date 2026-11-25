"""Cross-input constraints for the BNSD ``flash_attn_with_kvcache`` (v3) accuracy cases.

See ``generate_kvcache_v2.py`` for the BNSD rationale.  v3 uses ``page_table``
instead of ``block_table``; it is forced to ``default`` here so the cache is a
plain non-paged ``(B, capacity, H_kv, D)`` tensor.
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


CACHE_CAPACITIES = (16, 32, 64, 128)
HEADS = (2, 4, 8, 16)
MIN_SEQLEN = 2


def _nearest(value, choices):
    return min(choices, key=lambda candidate: abs(candidate - max(int(value), 1)))


def _inputs(case_config):
    return {item.name: item for item in case_config.inputs if not isinstance(item, list)}


@GENERATOR_REGISTRY.register("flash_attn_with_kvcache_v3_bnsd_constraints")
class FlashAttnKvCacheV3BnsdConstraints(CaseGenerator):
    def after_case_config(self, case_config):
        inputs = _inputs(case_config)
        batch, seqlen_q, heads_q, head_dim = inputs["q"].shape
        capacity = inputs["k_cache"].shape[1]

        heads_q = max(int(heads_q), 2)
        valid_kv_heads = tuple(head for head in HEADS if head <= heads_q and heads_q % head == 0)
        heads_q = _nearest(heads_q, valid_kv_heads)
        heads_kv = _nearest(inputs["k_cache"].shape[2], valid_kv_heads)
        dtype = inputs["q"].dtype

        seqlen_q = max(seqlen_q, MIN_SEQLEN)
        capacity = max(capacity, MIN_SEQLEN)
        if inputs["causal"].range_values is True and capacity < seqlen_q:
            capacity = min(value for value in CACHE_CAPACITIES if value >= seqlen_q)
        capacity = max(capacity, seqlen_q)

        inputs["q"].shape = [batch, seqlen_q, heads_q, head_dim]
        for name in ("k_cache", "v_cache"):
            inputs[name].dtype = dtype
            inputs[name].shape = [batch, capacity, heads_kv, head_dim]
        inputs["cache_seqlens"].dtype = "int32"
        inputs["cache_seqlens"].shape = [batch]
        lower = seqlen_q if inputs["causal"].range_values else 1
        if batch > 1:
            # Non-paged B>1 currently requires every effective length to equal capacity.
            length_range = [capacity, capacity]
        else:
            length_range = [lower, capacity]
        inputs["cache_seqlens"].range_values = length_range

        # page_table empty -> non-paged; k/v/qv empty -> no append-KV.
        inputs["page_table"].dtype = "int32"
        inputs["page_table"].shape = [batch, 1]
        inputs["page_table"].range_values = "default"
        for name in ("k", "v"):
            inputs[name].dtype = dtype
            inputs[name].shape = [batch, MIN_SEQLEN, heads_kv, head_dim]
            inputs[name].range_values = "default"
        inputs["qv"].dtype = dtype
        inputs["qv"].shape = [batch, seqlen_q, heads_q, head_dim]
        inputs["qv"].range_values = "default"
        for name in (
            "rotary_cos", "rotary_sin", "cache_batch_idx", "cache_leftpad",
            "cu_seqlens_q", "cu_seqlens_k_new", "rotary_seqlens",
            "q_descale", "k_descale", "v_descale", "scheduler_metadata",
        ):
            inputs[name].range_values = "default"
        inputs["max_seqlen_q"].range_values = "default"
        return case_config
