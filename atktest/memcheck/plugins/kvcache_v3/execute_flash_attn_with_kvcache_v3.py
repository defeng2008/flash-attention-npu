"""ATK CPU/GPU/NPU adapter for v3 flash_attn_with_kvcache."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_with_kvcache_v3 import flash_attn_with_kvcache as cpu_flash_attn_with_kvcache


@register("flash_attn_with_kvcache_v3_accuracy")
class FlashAttnWithKvCacheV3Api(BaseApi):
    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_with_kvcache(**kwargs)
        if self.device == "gpu":
            # TriDao official FA3 (hopper) benchmark; imported lazily because
            # flash_attn_interface is only importable on Hopper GPU hosts.
            try:
                import flash_attn_interface as gpu_fa3
            except ImportError:
                from flash_attn_3 import flash_attn_interface as gpu_fa3

            kwargs = {
                name: value.to(f"cuda:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            return gpu_fa3.flash_attn_with_kvcache(**kwargs)
        if self.device == "npu":
            import torch

            from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            q = kwargs["q"]
            k_cache = kwargs["k_cache"]
            v_cache = kwargs["v_cache"]
            page_table = kwargs.get("page_table")
            cache_seqlens = kwargs.get("cache_seqlens")
            if page_table is None:
                page_size = None
                max_seqlen_k = k_cache.shape[1]
            else:
                page_size = k_cache.shape[1]
                max_seqlen_k = page_size * page_table.shape[1]
            if cache_seqlens is None:
                cache_seqlens = torch.full(
                    (q.shape[0],), max_seqlen_k, dtype=torch.int32, device=q.device
                )
                kwargs["cache_seqlens"] = cache_seqlens
            cu_seqlens_q = kwargs.get("cu_seqlens_q")
            max_seqlen_q = (
                kwargs.get("max_seqlen_q") if cu_seqlens_q is not None else q.shape[1]
            )
            if os.environ.get("FA_ATK_USE_SCHEDULER_METADATA", "1") != "0":
                kwargs["scheduler_metadata"] = get_scheduler_metadata(
                    q.shape[0],
                    max_seqlen_q,
                    max_seqlen_k,
                    q.shape[2],
                    k_cache.shape[2],
                    q.shape[3],
                    cache_seqlens,
                    qkv_dtype=q.dtype,
                    headdim_v=v_cache.shape[3],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=kwargs.get("cu_seqlens_k_new"),
                    cache_leftpad=kwargs.get("cache_leftpad"),
                    page_size=page_size,
                    max_seqlen_k_new=0 if kwargs.get("k") is None else kwargs["k"].shape[1],
                    causal=kwargs.get("causal", False),
                    window_size=kwargs.get("window_size", (-1, -1)),
                    attention_chunk=kwargs.get("attention_chunk", 0),
                    softcap=kwargs.get("softcap", 0.0),
                    num_splits=kwargs.get("num_splits", 0),
                    pack_gqa=kwargs.get("pack_gqa"),
                    sm_margin=kwargs.get("sm_margin", 0),
                    softmax_scale=kwargs.get("softmax_scale"),
                )
            else:
                kwargs["scheduler_metadata"] = None
            return flash_attn_with_kvcache(**kwargs)
        raise RuntimeError(f"Unsupported backend: {self.device}")
