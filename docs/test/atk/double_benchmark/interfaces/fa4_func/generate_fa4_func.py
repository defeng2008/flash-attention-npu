"""fa4_func (MLA) 的 ATK 跨输入约束钩子。

ATK 每条用例都会逐维独立随机各个输入，本钩子把它们改写成 MLA 的合法组合：

* prefill（q 末维 192）：Dq=Dk=192(nope128+rope64)，Dv=128，Hk=Hq=32，**摘掉 qv**
* decode （q 末维 64 ）：q 只留 rope64，qv=512，k 只留 rope64，v=512，Hk=1（吸收态）
* paged  ：KV 存成 (num_blocks, block_size, H, D)；page_table 固定成
           "第 b 个 batch 用第 b*nblk..b*nblk+nblk-1 页"；seqused_k / cache_seqlens
           取实际 KV 长度（= num_blocks * block_size）
* 索引张量（cu_seqlens_* / page_table / seqused_k）改成 int32_fixed 写死取值，
  否则 ATK 会随机出一组非法的（非单调的）累计长度。

注意：range_values 会被 ATK 拿去 hash 去重，嵌套结构必须用 tuple 而不是 list。
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY

BLOCK = 128   # paged KV 的 block_size，与交付用例保持一致
ROPE = 64     # MLA 的 rope 维度
NOPE = 128    # MLA 的 nope 维度
LATENT = 512  # 吸收态 v / qv 的维度
HEADS = 32    # q 的头数


def _entries(case_config):
    """按名字取输入；attr_tuple 在输入列表里是嵌套 list，这里按类型过滤掉。"""
    return {i.name: i for i in case_config.inputs if not isinstance(i, list)}


def _drop(case_config, *names):
    case_config.inputs = [i for i in case_config.inputs
                          if getattr(i, "name", None) not in names]


def _fixed(entry, values, shape=None):
    """把索引张量的取值写死：换成 ATK 的 *_fixed 数据类型。"""
    entry.dtype = "int32_fixed"
    if values and isinstance(values[0], (list, tuple)):
        entry.range_values = tuple(tuple(v) for v in values)
    else:
        entry.range_values = tuple(values)
    entry.shape = list(shape) if shape else [len(values)]


def _blocks(batch, nblk):
    """page_table：第 b 个 batch 占用全局页号 b*nblk .. b*nblk+nblk-1。"""
    return [tuple(b * nblk + i for i in range(nblk)) for b in range(batch)]


@GENERATOR_REGISTRY.register("fa4_func_mla_constraints")
class Fa4FuncConstraints(CaseGenerator):
    """把 ATK 随机出来的用例改写成合法的 MLA 组合。"""

    def after_case_config(self, case_config):
        ins = _entries(case_config)
        q = ins["q"]
        decode = q.shape[-1] == ROPE        # 形态判据：q 末维
        bsnd = len(q.shape) == 4
        qv_entry = ins.get("qv") if decode else None
        if not decode:
            _drop(case_config, "qv")        # qv 只在吸收态出现

        # ---- q / qv ------------------------------------------------------------
        if decode:
            dq, hk, dk, dv = ROPE, 1, ROPE, LATENT
        else:
            dq, hk, dk, dv = NOPE + ROPE, HEADS, NOPE + ROPE, NOPE
        if bsnd:
            batch, seq_q = q.shape[0], q.shape[1]
            q.shape = [batch, seq_q, HEADS, dq]
        else:
            total_q = q.shape[0]
            q.shape = [total_q, HEADS, dq]
        if qv_entry is not None:
            qv_entry.shape = ([batch, seq_q, HEADS, LATENT] if bsnd
                              else [total_q, HEADS, LATENT])

        # ---- KV（普通 / paged）--------------------------------------------------
        # 非 paged：k 的 seq 维就是 KV 长度；paged：页数取自 page_table 的第 1 维。
        # 硬约束：KV 长度必须 >= Q 长度。causal / 滑窗掩码在 KV<Q（或窗口过窄）时会把
        # 尾部整行 mask 掉，而这类行在真实 MLA 场景并不存在；softmax 对全 -inf 行无
        # 定义，候选与参考必然对不上（实测 fa3_func 21/100 失败全部由此导致）。
        q_len = seq_q if bsnd else total_q
        k_entry = ins["k_cache"] if "k_cache" in ins else ins["k"]
        v_entry = ins["v_cache"] if "v_cache" in ins else ins["v"]
        paged = "page_table" in ins
        if paged:
            pt = ins["page_table"]
            nblk = max(pt.shape[1], -(-q_len // BLOCK))   # 向上取整，保证 nblk*BLOCK >= q_len
            kv_len = nblk * BLOCK
            batch_pt = batch if bsnd else 1        # TND 按单序列打包
            _fixed(pt, _blocks(batch_pt, nblk), shape=[batch_pt, nblk])
        elif bsnd:
            batch_pt = batch
            kv_len = max(k_entry.shape[1], q_len)
        else:
            batch_pt = 1
            kv_len = max(k_entry.shape[0], q_len)
        if paged:
            k_entry.shape = [batch_pt * nblk, BLOCK, hk, dk]
            v_entry.shape = [batch_pt * nblk, BLOCK, hk, dv]
        elif bsnd:
            k_entry.shape = [batch, kv_len, hk, dk]
            v_entry.shape = [batch, kv_len, hk, dv]
        else:
            k_entry.shape = [kv_len, hk, dk]
            v_entry.shape = [kv_len, hk, dv]

        # ---- 索引张量 / 长度 attr ----------------------------------------------
        if not bsnd:
            _fixed(ins["cu_seqlens_q"], [0, total_q], shape=[2])
            if "cu_seqlens_k" in ins:
                _fixed(ins["cu_seqlens_k"], [0, kv_len], shape=[2])
            if "max_seqlen_q" in ins:
                ins["max_seqlen_q"].range_values = total_q
            if "max_seqlen_k" in ins:
                ins["max_seqlen_k"].range_values = kv_len
        if "seqused_k" in ins:
            _fixed(ins["seqused_k"], [kv_len] * batch_pt, shape=[batch_pt])
        if "cache_seqlens" in ins:
            ins["cache_seqlens"].range_values = kv_len
        return case_config
