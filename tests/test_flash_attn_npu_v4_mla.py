# Copyright (c) 2026, Minghua Shen.
# MLA forward port of tests/test_flash_attn_npu_v3_mla.py to the FA4 public API.

"""Public MLA forward tests for flash_attn_npu_4 (FA4). Run with pytest.

MLA shares the FA4 entry points: ``flash_attn_func`` (dense BSND) and
``flash_attn_varlen_func`` (packed TND / paged KV). Compressed MLA passes ``qv``
(Dq=64, Dv=512) while prefill MLA has Dq=192 and Dv=128. Both return exactly two
tensors ``(out, softmax_lse)``.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401

import flash_attn_npu_4 as fa4
from tests.common.compare import assert_fa_close
from tests.common.golden_cache import get_or_compute_golden
from tests.common.test_utils import make_cu_seqlens


DTYPES = [torch.float16, torch.bfloat16]
REPRESENTATIONS = ["compressed", "prefill"]
LAYOUTS = ["bsnd", "tnd", "mixed", "paged_bsnd", "paged_tnd"]


@pytest.fixture(scope="module", autouse=True)
def _reference_threads():
    if not torch.npu.is_available():
        pytest.skip("MLA requires Ascend 910B/C")
    if not torch.npu.get_device_name(0).startswith(("Ascend910B", "Ascend910C")):
        pytest.skip("MLA requires Ascend 910B/C")
    # These small CPU references cost much more with a machine-wide thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _view(tensor, kind):
    """Allocate poisoned backing storage, preserving legal outer strides."""
    if kind == "contiguous" or tensor.numel() == 0:
        return tensor.clone()
    shape = list(tensor.shape)
    if kind == "broadcast":
        return tensor.narrow(0, 0, 1).expand_as(tensor)
    axis = -1 if kind == "scale-head" else -2 if kind == "head-gap" and tensor.ndim >= 3 else 0
    shape[axis] = 2 * shape[axis] + 2
    fill = float("nan") if tensor.is_floating_point() else 117
    storage = torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)
    selection = [slice(None)] * tensor.ndim
    selection[axis] = slice(1, -1, 2)
    result = storage[tuple(selection)]
    result.copy_(tensor)
    return result


def _make_case(
    layout="bsnd",
    representation="compressed",
    dtype=torch.bfloat16,
    *,
    q=(7, 3),
    k=(65, 33),
    used_q=None,
    used_k=None,
    heads=8,
    kv_heads=2,
    splits=1,
    causal=False,
    window=(-1, -1),
    cap=0.0,
    scale=None,
    page=16,
    view="contiguous",
    repeated_pages=False,
):
    packed = layout in ("tnd", "mixed", "paged_tnd")
    paged = layout.startswith("paged")
    batch, sq, sk = len(q), max(q, default=0), max(k, default=0)
    qa = list(q) if packed else [sq] * batch
    ka = list(k) if layout == "tnd" else [sk] * batch
    if paged:
        sk = math.ceil(sk / page) * page
        ka = [sk] * batch
    qused = list(used_q if used_q is not None else qa)
    kused = list(used_k if used_k is not None else k)
    hk = 1 if representation == "compressed" else kv_heads
    dq, dv = (64, 512) if representation == "compressed" else (192, 128)
    raw = {
        name: torch.randn(batch, seq, nh, dim).mul_(0.1).to(dtype)
        for name, seq, nh, dim in (("q", sq, heads, dq), ("k", sk, hk, dq), ("v", sk, hk, dv))
    }
    if representation == "compressed":
        raw["qv"] = torch.randn(batch, sq, heads, dv).mul_(0.1).to(dtype)
    physical = {}
    table = None
    if paged:
        npages = batch * (sk // page)
        table = torch.arange(npages - 1, -1, -1, dtype=torch.int32).reshape(batch, sk // page)
        for name in ("k", "v"):
            pool = torch.empty(npages, page, hk, raw[name].shape[-1], dtype=dtype)
            for b in range(batch):
                pool[table[b].long()] = raw[name][b].reshape(-1, page, hk, raw[name].shape[-1])
            physical[name] = pool
        if repeated_pages and table.shape[1] > 1:
            table[:, -1] = table[:, 0]
    for name, value in raw.items():
        if name in ("q", "qv") and packed:
            value = (
                torch.cat([value[b, :n] for b, n in enumerate(qa)])
                if batch
                else value.reshape(0, heads, value.shape[-1])
            )
        elif name in ("k", "v") and layout == "tnd":
            value = (
                torch.cat([value[b, :n] for b, n in enumerate(ka)])
                if batch
                else value.reshape(0, hk, value.shape[-1])
            )
        physical.setdefault(name, value)
    tensors = {name: _view(value.to("npu"), view) for name, value in physical.items()}
    if view == "broadcast":
        physical = {name: _view(value, view) for name, value in physical.items()}
    if paged:
        for name in ("k", "v"):
            for b in range(batch):
                raw[name][b] = physical[name][table[b].long()].reshape(sk, hk, raw[name].shape[-1])
    if view == "broadcast":
        for name, value in physical.items():
            if name in ("k", "v") and paged:
                continue
            if value.ndim == 4:
                raw[name] = value.clone()
            else:
                offset = 0
                for b, n in enumerate(qa if name in ("q", "qv") else ka):
                    raw[name][b, :n] = value[offset : offset + n]
                    offset += n
    options = dict(softmax_scale=scale, causal=causal, window_size=window, softcap=cap, num_splits=splits)
    lengths = torch.tensor(kused, dtype=torch.int32, device="npu")
    prefix = make_cu_seqlens(qa).to("npu") if packed else None
    return dict(
        layout=layout,
        representation=representation,
        raw=raw,
        t=tensors,
        q=qa,
        k=ka,
        qused=qused,
        kused=kused,
        sq=sq,
        sk=sk,
        page=page,
        lengths=lengths,
        prefix=prefix,
        options=options,
        used_q=torch.tensor(qused, dtype=torch.int32, device="npu") if used_q is not None else None,
        prefix_k=make_cu_seqlens(ka).to("npu") if layout == "tnd" else None,
        table=_view(table.to("npu"), "offset") if table is not None else None,
    )


def _call(case, *, lse=True, **extra):
    """Dispatch through the FA4 public API (2 return values, no with_kvcache)."""
    t = case["t"]
    args = dict(t, **case["options"])
    args.update(extra)
    layout = case["layout"]
    if layout == "tnd":
        return fa4.flash_attn_varlen_func(
            **args,
            cu_seqlens_q=case["prefix"],
            cu_seqlens_k=case["prefix_k"],
            max_seqlen_q=case["sq"],
            max_seqlen_k=case["sk"],
            seqused_q=case["used_q"],
            seqused_k=case["lengths"],
            return_lse=lse,
        )
    if layout == "mixed" or layout == "paged_tnd":
        # Packed Q with BSND/paged KV; per-batch KV lengths come from seqused_k,
        # so max_seqlen_k must stay unset (only cu-packed KV may carry it).
        return fa4.flash_attn_varlen_func(
            **args,
            cu_seqlens_q=case["prefix"],
            max_seqlen_q=case["sq"],
            seqused_q=case["used_q"],
            seqused_k=case["lengths"],
            page_table=case["table"],
            return_lse=lse,
        )
    # Dense BSND (optionally paged) KV: FA4 has no with_kvcache, so seqused_k on
    # the varlen entry point carries the per-batch KV lengths.
    return fa4.flash_attn_varlen_func(
        **args,
        cu_seqlens_q=None,
        seqused_q=case["used_q"],
        seqused_k=case["lengths"],
        page_table=case["table"],
        return_lse=lse,
    )


def _attention_reference(q, k, v, qv, options, upcast):
    # Preserve the independent FP64/input-dtype equations used by the MLA E2Es.
    sq, sk = q.shape[1], k.shape[1]
    dtype = torch.float64 if upcast else q.dtype
    scale = options["softmax_scale"]
    if scale is None:
        scale = (q.shape[-1] + (0 if qv is None else qv.shape[-1])) ** -0.5
    left, right = options["window_size"]
    if left >= sk - 1:
        left = -1
    if right >= sq - 1:
        right = -1
    causal = options["causal"]
    if sq == 1 and left == right == -1:
        causal = False
    if causal:
        right = 0
    center = torch.arange(sq)[:, None] + sk - sq
    position = torch.arange(sk)[None, :]
    mask = torch.zeros(sq, sk, dtype=torch.bool)
    if left >= 0:
        mask |= position < center - left
    if right >= 0:
        mask |= position > center + right
    groups = q.shape[-2] // k.shape[-2]
    query = q.to(dtype).transpose(1, 2)
    key = k.to(dtype).repeat_interleave(groups, dim=2).transpose(1, 2)
    value = v.to(dtype).repeat_interleave(groups, dim=2).transpose(1, 2)
    scores = (query if upcast else query * scale) @ key.transpose(-1, -2)
    if qv is not None:
        content = qv.to(dtype).transpose(1, 2)
        scores = scores + (content if upcast else content * scale) @ value.transpose(-1, -2)
    scores = scores * scale if upcast else scores.float()
    cap = options["softcap"]
    if cap:
        scores = cap * torch.tanh(scores / cap)
    scores.masked_fill_(mask, -torch.inf)
    lse = scores.logsumexp(-1).float()
    empty = mask.all(-1)
    safe = scores.masked_fill(empty[:, None], 0)
    if upcast:
        weights = (safe - safe.amax(-1, keepdim=True)).exp().masked_fill(empty[:, None], 0)
        out = (weights @ value) / weights.sum(-1, keepdim=True).clamp_min(1)
    else:
        weights = safe.softmax(-1).masked_fill(empty[:, None], 0)
        out = weights.to(dtype) @ value
    return out.transpose(1, 2).to(q.dtype), lse


def _check(case, outputs):
    out, lse = (value.cpu() for value in outputs[:2])
    t = case["t"]
    assert out.shape == (*t["q"].shape[:-1], t["v"].shape[-1])
    assert out.dtype == t["q"].dtype and lse.dtype == torch.float32
    offset = 0
    for b, (allocated, nq, nk) in enumerate(zip(case["q"], case["qused"], case["kused"])):
        packed = case["prefix"] is not None
        actual = out[offset : offset + nq].unsqueeze(0) if packed else out[b : b + 1, :nq]
        logs = lse[:, offset : offset + nq].unsqueeze(0) if packed else lse[b : b + 1, :, :nq]
        offset += allocated
        if nq == 0:
            continue
        if nk == 0:
            assert actual.count_nonzero() == 0
            expected = math.inf if case["table"] is None and case["sk"] == 0 else -math.inf
            assert (logs == expected).all()
            continue
        raw = {
            name: value[b : b + 1, : nq if name in ("q", "qv") else nk]
            for name, value in case["raw"].items()
        }

        def compute():
            values = []
            for upcast in (True, False):
                values.extend(
                    _attention_reference(
                        raw["q"], raw["k"], raw["v"], raw.get("qv"), case["options"], upcast
                    )
                )
            return dict(zip(("out", "lse", "out_pt", "lse_pt"), values))

        refs = get_or_compute_golden(
            nodeid="mla-public-attention-v4",
            metadata=case["options"],
            inputs=raw,
            compute_fn=compute,
            expected_keys=("out", "lse", "out_pt", "lse_pt"),
            source_files=[__file__],
        )
        assert_fa_close(
            actual, refs["out"], refs["out_pt"], softcap=case["options"]["softcap"], name="MLA out"
        )
        assert_fa_close(
            logs, refs["lse"], refs["lse_pt"], softcap=case["options"]["softcap"], name="MLA LSE"
        )


def _equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        actual, expected = (actual,), (expected,)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert (a.shape, a.stride(), a.dtype, a.device) == (b.shape, b.stride(), b.dtype, b.device)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "causal,window,cap,splits",
    [
        (False, (-1, -1), 0, 1),
        (True, (-1, -1), 0, 0),
        (False, (2, 1), 15, 3),
    ],
    ids=["unmasked", "causal", "window-softcap-split"],
)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_mla_forward(dtype, representation, layout, causal, window, cap, splits):
    case = _make_case(
        layout,
        representation,
        dtype,
        q=(3, 1),
        k=(129, 65),
        splits=splits,
        cap=cap,
        causal=causal,
        window=window,
        scale=0.23 if cap else None,
    )
    full = _call(case)
    assert len(full) == 2
    _check(case, full)
    _equal(_call(case, lse=False), full[0])
    explicit = case["options"]["softmax_scale"] or (
        576 if representation == "compressed" else 192
    ) ** -0.5
    _equal(_call(case, softmax_scale=explicit), full)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_mla_dense_bsnd(dtype, representation):
    # Dense BSND path through flash_attn_func.
    case = _make_case("bsnd", representation, dtype, q=(7, 3), k=(65, 33), splits=1)
    outputs = _call(case)
    assert len(outputs) == 2
    _check(case, outputs)
    _equal(_call(case, lse=False), outputs[0])


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("sq,sk", [(0, 3), (3, 0), (0, 0)])
def test_mla_empty_q_or_kv(dtype, representation, sq, sk):
    case = _make_case("bsnd", representation, dtype, q=(sq,), k=(sk,), splits=1)
    _check(case, _call(case))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mla_page_table_and_views(dtype, representation, layout):
    # Page-table KV plus non-contiguous inputs (offset / head-gap / broadcast).
    for view in ("offset", "head-gap", "broadcast"):
        case = _make_case(
            layout, representation, dtype, view=view, repeated_pages=True, splits=3, causal=True
        )
        _check(case, _call(case))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("layout", ["bsnd", "paged_bsnd"])
def test_mla_prefill_gqa(representation, layout):
    case = _make_case(layout, representation, torch.bfloat16, heads=8, kv_heads=2, splits=1)
    _check(case, _call(case))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_mla_bnsd_storage(dtype, representation):
    # Keep BSND logical axes while matching ordinary FA3's BNSD storage pattern.
    # MLA accepts the strides and reports ``out`` in the caller's BSND shape.
    case = _make_case("bsnd", representation, dtype, q=(7, 3), k=(65, 33))
    dense = dict(case["t"])
    expected = _call(case)
    _check(case, expected)

    for name, tensor in case["t"].items():
        tensor = tensor.transpose(1, 2).contiguous().transpose(1, 2)
        if tensor.shape[-2] > 1:  # a single KV head leaves the transpose contiguous
            assert not tensor.is_contiguous()
        case["t"][name] = tensor

    # The strides carry the layout: the same operands must give bit-identical
    # results whether the caller hands them over densely or as a BNSD view.
    out, lse = _call(case)
    assert out.shape == (*case["t"]["q"].shape[:-1], case["t"]["v"].shape[-1])
    torch.testing.assert_close(out, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(lse, expected[1], rtol=0, atol=0)

    # ``flash_attn_func`` shares the MLA entry point and must accept the view too.
    outputs = fa4.flash_attn_func(**dict(case["t"], **case["options"]), return_lse=True)
    reference = fa4.flash_attn_func(**dict(dense, **case["options"]), return_lse=True)
    assert outputs[0].shape == (*case["t"]["q"].shape[:-1], case["t"]["v"].shape[-1])
    torch.testing.assert_close(outputs[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], reference[1], rtol=0, atol=0)


def test_mla_rejects_gather_kv_indices():
    case = _make_case("bsnd", "compressed", torch.bfloat16, q=(3,), k=(65,))
    indices = torch.arange(64, dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match="gather_kv_indices"):
        fa4.flash_attn_func(**case["t"], gather_kv_indices=indices, **case["options"])
