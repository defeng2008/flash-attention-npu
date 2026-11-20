"""Standalone v2 KV-cache benchmark adapted from the repository test."""

import torch


def group_matmul(head, kv_head, left, right, high_prec=1):
    group_num = head // kv_head
    score = None
    for index in range(kv_head):
        group_score = torch.matmul(
            left[index * group_num:(index + 1) * group_num].to(torch.float32),
            right[index:index + 1].to(torch.float32),
        ).to(torch.float32)
        score = group_score if score is None else torch.cat((score, group_score), 0)
    return score


def softmax1(qk_result, is_first, gm, interm_dtype=torch.float16):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm, dm = lm, 0
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
    sim_sub = torch.exp((sim - hm).to(interm_dtype))
    return sim_sub, torch.sum(sim_sub, dim=-1, keepdims=True), dm, hm


def qkMM1(query, key):
    result = None
    for start in range(0, key.shape[1], 128):
        split = group_matmul(query.shape[0], key.shape[0], query[:, :, start:start + 128], key[:, start:start + 128], 0)
        result = split if result is None else result + split
    return result


def pvMM2(p, value):
    result = None
    for start in range(0, value.shape[1], 128):
        split = group_matmul(p.shape[0], value.shape[0], p[:, :, start:start + 128], value[:, start:start + 128], 0)
        result = split if result is None else result + split
    return result


def ref_flash_attention(query, key, value, scale, mask, data_type, softcap=0.0):
    interm_dtype = torch.float32
    query = query.cpu().permute(1, 0, 2)
    key = key.cpu().permute(1, 2, 0)
    value = value.cpu().permute(1, 0, 2)
    scale = torch.tensor(scale, dtype=interm_dtype)
    gl = go = gm = None
    for kv_start in range(0, key.shape[2], 512):
        sub_key = key[:, :, kv_start:kv_start + 512]
        sub_value = value[:, kv_start:kv_start + 512]
        qk_result = qkMM1(query, sub_key).to(interm_dtype) * scale
        if softcap > 0.0:
            qk_result = softcap * torch.tanh(qk_result / softcap)
        if mask is not None:
            qk_result += mask[:query.shape[1], kv_start:kv_start + sub_key.shape[2]].to(interm_dtype) * -1e4
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        lo = pvMM2(p_result.to(data_type), sub_value).to(interm_dtype)
        if kv_start == 0:
            gl, go = row_sum, lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm + row_sum
            go = go * dm + lo
    return (go / gl).permute(1, 0, 2).to(data_type), torch.squeeze(torch.log(gl) + gm, -1).float()


def _attention_mask(sq, sk, causal, window_size):
    left, right = window_size
    if sk > 0 and left >= sk - 1:
        left = -1
    if sq > 0 and right >= sq - 1:
        right = -1
    if causal:
        right = 0
    if left < 0 and right == 0:
        return torch.triu(torch.ones(sq, sk), diagonal=sk - sq + 1).bool()
    if left < 0 and right <= 0:
        return None
    q_pos = torch.arange(sq)[:, None]
    k_pos = torch.arange(sk)[None, :]
    center = q_pos + sk - sq
    return ((left >= 0) & (k_pos < center - left)) | ((right >= 0) & (k_pos > center + right))


def flash_attn_with_kvcache(
    q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None,
    cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None, block_table=None,
    softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0,
    rotary_interleaved=True, alibi_slopes=None, num_splits=0, scheduler_metadata=None,
    return_softmax_lse=False,
):
    if any(item is not None for item in (rotary_cos, rotary_sin, cache_leftpad, alibi_slopes, block_table)):
        raise NotImplementedError("The generated repository-style benchmark subset excludes these options")
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    output = []
    for index in range(q.shape[0]):
        cache_index = index if cache_batch_idx is None else int(cache_batch_idx[index])
        length = k_cache.shape[1] if cache_seqlens is None else int(cache_seqlens[index])
        if k is not None:
            new_length = k.shape[1]
            k_cache[cache_index, length:length + new_length] = k[index]
            v_cache[cache_index, length:length + new_length] = v[index]
            length += new_length
        key = k_cache[cache_index, :length]
        value = v_cache[cache_index, :length]
        output.append(ref_flash_attention(
            q[index], key, value, scale,
            _attention_mask(q.shape[1], length, causal, window_size), q.dtype, softcap,
        )[0])
    return torch.stack(output)
