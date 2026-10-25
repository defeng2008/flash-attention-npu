"""``kv_seqlen_allocated`` regression cases for the Ascend 950 v3 backend.

``kv_seqlen_allocated`` carries the **allocated** KV plane length ``S_alloc`` of a
BNSD KV cache (``k.size(2)``).  On the BSND path the batch-to-batch region stride
is taken from the *used* length (``gActualKvseqlen``), so a pre-allocated BSND
cache can only be expressed with ``cache_seqlens`` equal to the cache's own
sequence dimension.  BNSD planes are ``S_alloc`` sized instead, which is exactly
why the allocated length has to be passed to the kernel as a separate scalar.

Two cases, one per layout:

* ``test_bsnd_used_length_equals_cache_length`` - BSND boundary case.  With
  ``cache_seqlens == k_cache.size(1)`` the BSND result must match the float32
  reference, and reading the very same storage as BNSD (where ``S_alloc``
  degenerates to the used length) must reproduce it.
* ``test_bnsd_allocated_cache_longer_than_used`` - BNSD case.  With
  ``S_alloc > used``, the head-plane pitch must come from the allocated length:
  a pitch derived from the used length would read the pre-allocated tail, so the
  tail is filled with a large sentinel and with zeros in a second run - both must
  give identical, reference-matching results.  A ``cache_seqlens`` beyond
  ``S_alloc`` must be rejected before launch.

Run:
    PYTHONPATH=$(pwd) python -m pytest docs/enable-qkv-bnsd/test_kv_seqlen_allocated.py -q
or standalone:
    PYTHONPATH=$(pwd) python docs/enable-qkv-bnsd/test_kv_seqlen_allocated.py
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401  registers the npu backend used by torch.npu below

from tests.common.attention_ref import ref_flash_attention
from tests.common.test_utils import make_golden_attention_mask

from flash_attn_npu_3 import flash_attn_with_kvcache

torch.npu.set_device(0)

B, H, H_KV, D = 2, 8, 2, 128
SQ = 32
S_KV = 512  # case 1: BSND cache, sized exactly to the used length
S_ALLOC = 1024  # case 2: BNSD pre-allocated plane length
USED = [300, 512]
SCALE = 1.0 / math.sqrt(D)
JUNK = 1e4
DTYPE = torch.bfloat16
TOL = {"atol": 2e-2, "rtol": 2e-2}


def _reference(q_bsnd, k_balloc, v_balloc, used):
    """Per-batch float32 reference over each batch's valid KV prefix only."""
    outs = []
    for b, n in enumerate(used):
        mask, _, _ = make_golden_attention_mask(SQ, n, True, -1, -1)
        out_b, _ = ref_flash_attention(
            q_bsnd[b : b + 1],
            k_balloc[b : b + 1, :n],
            v_balloc[b : b + 1, :n],
            SCALE,
            mask,
            DTYPE,
            0.0,
            upcast=True,
            reorder_ops=False,
        )
        outs.append(out_b)
    return torch.cat(outs, 0)


def _run_kvcache(q, k_cache, v_cache, used):
    out, lse, *_ = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=torch.tensor(used, dtype=torch.int32).npu(),
        softmax_scale=SCALE,
        causal=True,
        return_softmax_lse=True,
    )
    return out, lse


def _run_bnsd(q_storage, k_storage, v_storage, used):
    """Hand contiguous BNSD storage to the API as its BSND-shaped view."""
    return _run_kvcache(
        q_storage.transpose(1, 2),
        k_storage.transpose(1, 2),
        v_storage.transpose(1, 2),
        used,
    )


def _fill_plane_tail(storage, used, junk):
    """Fill the region beyond ``used`` of every (batch, head) BNSD plane."""
    for b, n in enumerate(used):
        storage[b, :, n:, :] = junk
    return storage


def test_bsnd_used_length_equals_cache_length():
    """BSND: the region stride is the used length, so ``S_alloc`` == used."""
    gen = torch.Generator().manual_seed(2026)
    used = [S_KV, S_KV]
    q = torch.randn(B, SQ, H, D, dtype=DTYPE, generator=gen).npu()
    k_alloc = torch.randn(B, S_KV, H_KV, D, dtype=DTYPE, generator=gen).npu()
    v_alloc = torch.randn(B, S_KV, H_KV, D, dtype=DTYPE, generator=gen).npu()

    out_ref = _reference(q.cpu(), k_alloc.cpu(), v_alloc.cpu(), used)

    out, lse = _run_kvcache(q, k_alloc, v_alloc, used)
    assert tuple(out.shape) == (B, SQ, H, D)
    assert tuple(lse.shape) == (B, H, SQ)
    torch.testing.assert_close(out.float().cpu(), out_ref.float().cpu(), **TOL)

    # The same storage read as BNSD sits on the S_alloc == used boundary, so the
    # BNSD path must reproduce the BSND result element for element.
    q_bnsd = q.transpose(1, 2).contiguous()
    k_bnsd = k_alloc.transpose(1, 2).contiguous()
    v_bnsd = v_alloc.transpose(1, 2).contiguous()

    out_bnsd, lse_bnsd = _run_bnsd(q_bnsd, k_bnsd, v_bnsd, used)
    assert tuple(out_bnsd.shape) == (B, SQ, H, D)
    torch.testing.assert_close(out_bnsd.float().cpu(), out.float().cpu(), rtol=0, atol=1e-2)
    torch.testing.assert_close(lse_bnsd.float().cpu(), lse.float().cpu(), rtol=0, atol=1e-2)


def test_bnsd_allocated_cache_longer_than_used():
    """BNSD: the head-plane pitch comes from ``S_alloc``, not from ``used``."""
    assert S_ALLOC > max(USED)
    gen = torch.Generator().manual_seed(2026)
    q_alloc = torch.randn(B, H, SQ, D, dtype=DTYPE, generator=gen).npu()
    k_alloc = torch.randn(B, H_KV, S_ALLOC, D, dtype=DTYPE, generator=gen)
    v_alloc = torch.randn(B, H_KV, S_ALLOC, D, dtype=DTYPE, generator=gen)

    out_ref = _reference(
        q_alloc.transpose(1, 2).contiguous().cpu(),
        k_alloc.transpose(1, 2).contiguous(),
        v_alloc.transpose(1, 2).contiguous(),
        USED,
    )

    k_junk = _fill_plane_tail(k_alloc.clone(), USED, JUNK).npu()
    v_junk = _fill_plane_tail(v_alloc.clone(), USED, JUNK).npu()

    out, lse = _run_bnsd(q_alloc, k_junk, v_junk, USED)
    assert tuple(out.shape) == (B, SQ, H, D)  # out follows the caller's q shape
    assert tuple(lse.shape) == (B, H, SQ)
    # A pitch taken from the used length would read the sentinel tail here.
    torch.testing.assert_close(out.float().cpu(), out_ref.float().cpu(), **TOL)

    # A different pre-allocated tail must not change a single element, in the
    # output or in the LSE: the tail lies beyond the allocated plane pitch.
    k_zero = _fill_plane_tail(k_alloc.clone(), USED, 0.0).npu()
    v_zero = _fill_plane_tail(v_alloc.clone(), USED, 0.0).npu()
    out_zero, lse_zero = _run_bnsd(q_alloc, k_zero, v_zero, USED)
    torch.testing.assert_close(out_zero.float().cpu(), out.float().cpu(), rtol=0, atol=1e-3)
    torch.testing.assert_close(lse_zero.float().cpu(), lse.float().cpu(), rtol=0, atol=1e-3)

    # cache_seqlens beyond the allocated plane length must be rejected.
    with pytest.raises(ValueError):
        _run_bnsd(q_alloc, k_junk, v_junk, [S_ALLOC + 1024, USED[1]])


if __name__ == "__main__":
    test_bsnd_used_length_equals_cache_length()
    print("[PASS] test_bsnd_used_length_equals_cache_length")
    test_bnsd_allocated_cache_longer_than_used()
    print("[PASS] test_bnsd_allocated_cache_longer_than_used")
    print("ALL CHECKS PASSED")
