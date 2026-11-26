"""MLA fp32 CPU golden for fa3_kvcache (gpu_mem_compare)."""

import torch

BLOCK = 128


def _mask(sq, sk, causal, window_size):
    left, right = window_size
    if sk > 0 and left >= sk - 1:
        left = -1
    if sq > 0 and right >= sq - 1:
        right = -1
    if causal:
        right = 0
    if left < 0 and right == 0:
        return torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=sk - sq + 1)
    if left < 0 and right <= 0:
        return None
    q_pos = torch.arange(sq)[:, None]
    k_pos = torch.arange(sk)[None, :]
    center = q_pos + sk - sq
    return ((left >= 0) & (k_pos < center - left)) | ((right >= 0) & (k_pos > center + right))


def _paged(cache, table):
    blocks = [cache[int(p)] for p in table if int(p) >= 0]
    if not blocks:
        return cache[:0]
    return torch.cat(blocks, dim=0)


def _core(q, k, v, qv, scale, causal, window_size, length):
    """q/k/v: [S, H, D] single sequence; length: valid KV rows (None = all)."""
    qh = q.float().transpose(0, 1)
    if qv is not None:
        qnh = qv.float().transpose(0, 1)
        kh = k.float().transpose(0, 1)
        vh = v.float().transpose(0, 1)
        scores = (qnh @ vh.transpose(-1, -2) + qh @ kh.transpose(-1, -2)) * scale
        value = vh
    else:
        kh = k.float().transpose(0, 1)
        vh = v.float().transpose(0, 1)
        scores = (qh @ kh.transpose(-1, -2)) * scale
        value = vh
    if length is not None:
        scores = scores[..., :length]
        value = value[..., :length, :]
    mask = _mask(scores.shape[-2], scores.shape[-1], causal, window_size)
    if mask is not None:
        scores = scores + mask.to(scores.dtype) * -1e4
    # 内核 flash-attention 口径：P = exp(s - max) 先量化到输入精度（内核 Cast 到 PType，
    # 设备默认 RNE，已实测），再与 V 做 fp32 累加；行和 l 取 fp32 的 exp 之和
    # （内核 CalcLocalRowSum 用的是 fp32 ls），最后除以 l。标杆必须复现整条链路：
    # 量化「未归一化的 exp」而不是「softmax 归一化后的概率」，否则近零元素的相对误差会被放大。
    e = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
    length_sum = e.sum(dim=-1, keepdim=True)
    probs = e.to(q.dtype).float()
    if value.shape[0] != probs.shape[0] and probs.shape[0] % value.shape[0] == 0:
        value = value.repeat_interleave(probs.shape[0] // value.shape[0], dim=0)
    return (probs @ value / length_sum).transpose(0, 1)


def _bsnd(q, k, v, qv, scale, causal, window_size, lengths, page_table):
    outs = []
    for b in range(q.shape[0]):
        kb, vb = k[b], v[b]
        length = int(lengths[b]) if lengths is not None else None
        if page_table is not None:
            kb, vb = _paged(k, page_table[b]), _paged(v, page_table[b])
            kb, vb = kb[:length], vb[:length]
            length = None
        outs.append(_core(q[b], kb, vb, None if qv is None else qv[b], scale, causal, window_size, length))
    return torch.stack(outs).to(q.dtype)


def _tnd(q, k, v, qv, cu_q, cu_k, scale, causal, window_size, page_table=None, seqlens=None):
    outs = []
    for b in range(len(cu_q) - 1):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        kb, vb = k, v
        if page_table is not None:
            # paged KV：内核只接受 page_table + seqused_k，每 batch 的 KV 长度取自 seqused_k。
            kb, vb = _paged(k, page_table[b]), _paged(v, page_table[b])
            seql = int(seqlens[b])
            kb, vb = kb[:seql], vb[:seql]
            ks, ke = 0, seql
        else:
            ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        outs.append(_core(q[qs:qe], kb[ks:ke], vb[ks:ke],
                          None if qv is None else qv[qs:qe], scale, causal, window_size, None))
    return torch.cat(outs, dim=0).to(q.dtype)


def flash_attn_with_kvcache(**kw):
    q = kw["q"]
    k = kw.get("k", kw.get("k_cache"))
    v = kw.get("v", kw.get("v_cache"))
    qv = kw.get("qv")
    scale = kw.get("softmax_scale")
    if scale is None:
        scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** -0.5
    causal = bool(kw.get("causal", False))
    window = tuple(kw.get("window_size") or (-1, -1))
    if kw.get("cu_seqlens_q") is not None:
        return _tnd(q, k, v, qv, kw["cu_seqlens_q"], kw.get("cu_seqlens_k"), scale, causal, window,
                    kw.get("page_table"), kw.get("seqused_k"))
    lengths = None
    if kw.get("cache_seqlens") is not None:
        cs = kw["cache_seqlens"]
        lengths = [cs] * q.shape[0] if isinstance(cs, int) else list(cs)
    if kw.get("seqused_k") is not None:
        lengths = list(kw["seqused_k"])
    return _bsnd(q, k, v, qv, scale, causal, window, lengths, kw.get("page_table"))
