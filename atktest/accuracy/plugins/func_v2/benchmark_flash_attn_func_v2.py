"""Standalone v2 flash_attn_func benchmark adapted from the repository test."""

import torch


def group_matmul(head, kv_head, left, right, dtype=torch.float32):
    group_num = head // kv_head
    score = None
    for index in range(kv_head):
        group_score = torch.matmul(
            left[index * group_num:(index + 1) * group_num].to(dtype),
            right[index:index + 1].to(dtype),
        ).to(dtype)
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


def qkMM1(query, key, dtype=torch.float32):
    result = None
    for start in range(0, key.shape[1], 128):
        result_split = group_matmul(
            query.shape[0], key.shape[0], query[:, :, start:start + 128], key[:, start:start + 128], dtype
        )
        result = result_split if result is None else result + result_split
    return result


def pvMM2(p, value, dtype=torch.float32):
    result = None
    for start in range(0, value.shape[1], 128):
        result_split = group_matmul(
            p.shape[0], value.shape[0], p[:, :, start:start + 128], value[:, start:start + 128], dtype
        )
        result = result_split if result is None else result + result_split
    return result


def ref_flash_attention(query, key, value, scale, mask, data_type, softcap=0.0):
    # ATK upcasts fp16/bf16 cases to fp32 before invoking the double-benchmark
    # golden; compute that baseline in fp64 so it is a true higher-precision ref.
    interm_dtype = torch.float64 if data_type in (torch.float32, torch.float64) else torch.float32
    query = query.cpu().permute(1, 0, 2)
    key = key.cpu().permute(1, 2, 0)
    value = value.cpu().permute(1, 0, 2)
    scale = torch.tensor(scale, dtype=interm_dtype)
    gl = go = gm = None
    for kv_start in range(0, key.shape[2], 512):
        sub_key = key[:, :, kv_start:kv_start + 512]
        sub_value = value[:, kv_start:kv_start + 512]
        qk_result = qkMM1(query, sub_key, interm_dtype).to(interm_dtype) * scale
        if softcap > 0.0:
            qk_result = softcap * torch.tanh(qk_result / softcap)
        if mask is not None:
            # Mask with -inf instead of a fixed -1e4: large-value ranges and
            # outliers push qk logits far past 1e4, so the constant no longer
            # suppresses the masked entries.
            qk_result = qk_result.masked_fill(
                mask[:query.shape[1], kv_start:kv_start + sub_key.shape[2]], float("-inf")
            )
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        lo = pvMM2(p_result.to(data_type), sub_value, interm_dtype).to(interm_dtype)
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


def flash_attn_func(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
    softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False,
):
    if dropout_p != 0.0 or alibi_slopes is not None or return_attn_probs:
        raise NotImplementedError("The generated repository-style benchmark subset excludes these options")
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return torch.stack([
        ref_flash_attention(
            q[index], k[index], v[index], scale,
            _attention_mask(q.shape[1], k.shape[1], causal, window_size), q.dtype, softcap,
        )[0]
        for index in range(q.shape[0])
    ])
