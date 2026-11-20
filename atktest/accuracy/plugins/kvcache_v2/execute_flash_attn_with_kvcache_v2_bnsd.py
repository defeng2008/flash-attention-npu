"""ATK CPU/GPU/NPU adapter for the BNSD ``flash_attn_with_kvcache`` (v2) accuracy cases.

The NPU branch only changes the memory layout of the BNSD operands: q, k_cache and
v_cache are handed in as the BSND-shaped transposed view of a contiguous BNSD
storage.  The cache is always non-paged (``block_table`` empty) and no new KV is
appended (``k``/``v`` empty), which is required for BNSD to apply.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_with_kvcache_v2 import (
    flash_attn_with_kvcache as cpu_flash_attn_with_kvcache,
)


def _bnsd_view(tensor):
    """Hand a contiguous BNSD ``(B, H, S, D)`` storage in as its ``(B, S, H, D)`` view."""
    view = tensor.transpose(1, 2).contiguous().transpose(1, 2)
    assert not view.is_contiguous(), "BNSD view degenerated to contiguous; the case missed the BNSD path"
    return view


@register("flash_attn_with_kvcache_v2_bnsd_accuracy")
class FlashAttnWithKvCacheV2BnsdApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_with_kvcache(**kwargs)
        if self.device == "gpu":
            # TriDao official FA2 benchmark; imported lazily because the
            # flash_attn CUDA extension is only importable on GPU hosts.
            from flash_attn import flash_attn_with_kvcache as gpu_flash_attn_with_kvcache

            kwargs = {
                name: value.to(f"cuda:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            # scheduler_metadata is an NPU-only extension unknown to official FA2.
            kwargs.pop("scheduler_metadata", None)
            return gpu_flash_attn_with_kvcache(**kwargs)
        if self.device == "npu":
            from flash_attn_npu import flash_attn_with_kvcache, get_scheduler_metadata

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            q = kwargs["q"]
            k_cache = kwargs["k_cache"]
            v_cache = kwargs["v_cache"]
            cache_seqlens = kwargs.get("cache_seqlens")
            if cache_seqlens is None:
                cache_seqlens = torch.full(
                    (q.shape[0],), k_cache.shape[1], dtype=torch.int32, device=q.device
                )
                kwargs["cache_seqlens"] = cache_seqlens
            if os.environ.get("FA_ATK_USE_SCHEDULER_METADATA", "1") != "0":
                kwargs["scheduler_metadata"] = get_scheduler_metadata(
                    q.shape[0],
                    q.shape[1],
                    k_cache.shape[1],
                    q.shape[2],
                    k_cache.shape[2],
                    q.shape[3],
                    cache_seqlens,
                    qkv_dtype=q.dtype,
                    headdim_v=v_cache.shape[3],
                    causal=kwargs.get("causal", False),
                    window_size=kwargs.get("window_size", (-1, -1)),
                    softcap=kwargs.get("softcap", 0.0),
                    softmax_scale=kwargs.get("softmax_scale"),
                )
            else:
                kwargs["scheduler_metadata"] = None
            for name in ("q", "k_cache", "v_cache"):
                kwargs[name] = _bnsd_view(kwargs[name])
            return flash_attn_with_kvcache(**kwargs)
        raise RuntimeError(f"Unsupported backend: {self.device}")
