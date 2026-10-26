"""Cross-input constraints for v3 ``flash_attn_with_kvcache``."""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


CACHE_CAPACITIES = (16, 32, 64, 96, 128, 256)
PAGES_PER_BATCH = (1, 2, 4)
HEADS = (1, 2, 4, 8, 16)
MAX_CACHE_ELEMENTS = 8_388_608


def _nearest(value, choices):
    return min(choices, key=lambda candidate: abs(candidate - max(int(value), 1)))


def _inputs(case_config):
    return {item.name: item for item in case_config.inputs if not isinstance(item, list)}


@GENERATOR_REGISTRY.register("flash_attn_with_kvcache_v3_constraints")
class FlashAttnKvCacheV3Constraints(CaseGenerator):
    def after_case_config(self, case_config):
        inputs = _inputs(case_config)
        batch, seqlen_q, heads_q, head_dim = inputs["q"].shape
        cache_shape = inputs["k_cache"].shape
        valid_kv_heads = tuple(head for head in HEADS if head <= heads_q and heads_q % head == 0)
        heads_kv = _nearest(cache_shape[2], valid_kv_heads)
        dtype = inputs["q"].dtype
        paged = inputs["page_table"].range_values != "default"

        if paged:
            page_size = cache_shape[1]
            max_pages = max(
                MAX_CACHE_ELEMENTS // (batch * page_size * heads_kv * head_dim),
                1,
            )
            valid_pages = tuple(pages for pages in PAGES_PER_BATCH if pages <= max_pages)
            pages = _nearest(inputs["page_table"].shape[1], valid_pages)
            if inputs["causal"].range_values is True and page_size * pages < seqlen_q:
                pages = min(value for value in valid_pages if page_size * value >= seqlen_q)
            blocks = batch * pages
            capacity = page_size * pages
            final_cache_shape = [blocks, page_size, heads_kv, head_dim]
            inputs["page_table"].dtype = "int32"
            inputs["page_table"].shape = [batch, pages]
            inputs["page_table"].range_values = [0, blocks - 1]
        else:
            capacity = cache_shape[1]
            if inputs["causal"].range_values is True and capacity < seqlen_q:
                capacity = min(value for value in CACHE_CAPACITIES if value >= seqlen_q)
            final_cache_shape = [batch, capacity, heads_kv, head_dim]
            inputs["page_table"].dtype = "int32"
            inputs["page_table"].shape = [batch, 1]
            inputs["page_table"].range_values = "default"

        for name in ("k_cache", "v_cache"):
            inputs[name].dtype = dtype
            inputs[name].shape = final_cache_shape
        inputs["cache_seqlens"].dtype = "int32"
        inputs["cache_seqlens"].shape = [batch]
        lower = seqlen_q if inputs["causal"].range_values else 1
        if batch > 1 and not paged:
            length_range = [capacity, capacity]
        elif batch > 1:
            representative = max(lower, capacity // 2)
            length_range = [representative, representative]
        else:
            length_range = [lower, capacity]
        inputs["cache_seqlens"].range_values = length_range

        new_length = inputs["k"].shape[1]
        for name in ("k", "v"):
            inputs[name].dtype = dtype
            inputs[name].shape = [batch, new_length, heads_kv, head_dim]
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
