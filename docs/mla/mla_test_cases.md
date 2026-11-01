# MLA 前向测试用例文档（flash_attn_npu_3 / flash_attn_npu_4）

> 任务 5.2。本文档以表格形式整理 MLA 前向的功能、一致性/精度、`softmax_scale` 口径、
> 拒绝路径与回归用例。用例名与仓库真实测试函数名一一对应。
>
> 配套设计文档：`docs/mla/mla_design.md`。

## 测试文件与规模

| 文件 | 测试函数数 | 运行结果（规格给定） | 状态 |
| --- | --- | --- | --- |
| `tests/test_flash_attn_npu_v3_mla.py` | 31 | 1253 passed | 本次**新增**（工作区未跟踪） |
| `tests/test_flash_attn_npu_v4_mla.py` | 7 | 105 passed | 本次**新增**（工作区未跟踪） |
| `tests/test_flash_attn_npu_mla_consistency.py` | 7 | 18 passed | 本次**新增**（工作区未跟踪） |
| `tests/test_flash_attn_npu_v3_metadata.py`（新增函数） | — | — | **既有文件扩展**（回归） |
| `tests/test_torch_compile_fa3.py`（新增类/函数） | — | — | **既有文件扩展**（回归） |

> 说明：单个 `test_*` 函数通常带大量 `@pytest.mark.parametrize`，上表「运行结果」是可执行
> 用例（参数组合）总数，而非函数数。

---

## 1. 功能用例

### 1.1 v3 入口（`tests/test_flash_attn_npu_v3_mla.py`）

| 用例名 / 文件 | 目的 | 输入要点 | 期望 | 状态 |
| --- | --- | --- | --- | --- |
| `test_mla_forward` | MLA 前向核心正确性（含 prefill/decode、TND/BSND、分页、因果/窗口） | 参数化 `dtype × representation × layout × splits × cap × causal × window`；`_call` 按 layout 路由到 `flash_attn_func`/`flash_attn_varlen_func`/`flash_attn_with_kvcache`，与 CPU golden 比对 | 与参考 `assert_fa_close` 一致 | 通过 |
| `test_mla_func` | 公共 `flash_attn_func` MLA 路径 | `dtype × representation × splits × causal × window × cap` | 输出/lse 正确 | 通过 |
| `test_mla_func_empty` | 空 query / 空 KV 边界（`flash_attn_func`） | `sq`/`sk` 取 0 等 | 形状正确、无越界 | 通过 |
| `test_mla_varlen_allocation_used` | varlen 分配/used 长度语义 | `flash_attn_varlen_func`，`cu_seqlens_q/k`、`seqused_q/k`、`max_seqlen_q/k` | 分配形状保持，used 前缀限长 | 通过 |
| `test_mla_page_boundaries` | 分页 KV（`flash_attn_with_kvcache`）边界 | page_size 取边界值，`representation × layout × page` | 分页读正确 | 通过 |
| `test_mla_prefill_gqa` | prefill 形态 GQA（KV 头 >1） | `dtype × heads × kv_heads × layout` | GQA 结果正确 | 通过 |
| `test_mla_boundaries` | 形状边界 | `dtype × representation × layout × spec` | 边界接受/正确处理 | 通过 |
| `test_mla_block_cancellation` | 分块/因果抵消逻辑 | `dtype × layout × splits × cap` | 结果正确 | 通过 |
| `test_mla_softcap_normalization` | softcap 归一化 | `dtype × layout × splits`，`softcap>0` | 与参考一致 | 通过 |
| `test_mla_default_options` | 默认参数路径 | `representation × layout` | 默认值行为正确 | 通过 |
| `test_mla_views_and_reuse` | 非连续视图 / metadata 复用 | `dtype × representation × layout × view` | 视图与记录复用正确 | 通过 |
| `test_mla_accumulator_buffer_reuse` | accumulator 复用（splits） | `dtype × representation × layout × pack` | 累加器正确重算 | 通过 |
| `test_mla_auto_partial_outputs` | 自动拆分 partial 输出 | `dtype × representation × layout × query × keys × split` | partial O/LSE 语义正确 | 通过 |
| `test_mla_cache_length_and_record_forms` | cache 长度与记录形态 | `representation × (layout, form)`（含 cloned/offset 视图；int 形态只覆盖 BSND 类 layout，打包 TND 需传张量） | 记录形式接受规则正确 | 通过 |
| `test_mla_empty_accumulators_default_dtype` | 空 accumulator 的默认 dtype | `dtype × representation × layout` | dtype/shape 正确 | 通过 |
| `test_mla_metadata_scale_and_softcap_reuse` | metadata 复用时 scale/softcap 可变 | `dtype × representation × layout` | scale/positive softcap 可变，激活状态变化需重生成 | 通过 |
| `test_mla_stream_lifetime` | 流生命周期/记录 stream | `dtype × representation × layout × splits` | 无 use-after-free | 通过 |
| `test_flash_attn_metadata_signature` | `get_scheduler_metadata` 公开签名 | `inspect.signature` | 参数名/顺序固定 | 通过 |
| `test_flash_attn_forward_signature` | 三个公共入口签名 | `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` | 参数名/顺序固定 | 通过 |
| `test_flash_attn_metadata_real_fake_contract` | metadata real/fake 契约 | `dtype × has_softcap` | 形状/dtype 一致，非法 kwarg 报错 | 通过 |
| `test_mla_prefill_bnsd_storage` | BSND 逻辑轴 + BNSD 存储模式（非连续 view） | `dtype × layout`，`transpose(1,2).contiguous().transpose(1,2)` | 接受并正确计算 | 通过 |
| `test_mla_fake_workspace_overflow` / `test_mla_fake_wide_cache` | Fake 下的溢出/大 cache 边界 | `layout`/`dtype` | 正确报错/通过 | 通过 |

### 1.2 v4 入口（`tests/test_flash_attn_npu_v4_mla.py`）

| 用例名 / 文件 | 目的 | 输入要点 | 期望 | 状态 |
| --- | --- | --- | --- | --- |
| `test_mla_forward` | v4 MLA 前向核心正确性 | `dtype × representation × layout × causal × window × cap × splits` | 与 golden 一致 | 通过 |
| `test_mla_dense_bsnd` | v4 dense BSND MLA | `dtype × representation` | 输出正确 | 通过 |
| `test_mla_empty_q_or_kv` | 空 query / KV | `dtype × representation × sq × sk` | 形状正确 | 通过 |
| `test_mla_page_table_and_views` | v4 分页（经 `flash_attn_varlen_func` 的 `page_table`/`seqused_k`）与视图 | `dtype × representation × layout` | 分页读正确 | 通过 |
| `test_mla_prefill_gqa` | v4 prefill GQA | `representation × layout` | GQA 正确 | 通过 |

> v4 MLA 无 `flash_attn_with_kvcache`；decode/分页统一走 `flash_attn_varlen_func` + `page_table`/`seqused_k`。

---

## 2. 一致性 / 精度用例

（`tests/test_flash_attn_npu_mla_consistency.py`，同一 latent `c_KV` 上 prefill=未吸收、decode=吸收）

| 用例名 / 文件 | 目的 | 输入要点 | 期望 | 状态 |
| --- | --- | --- | --- | --- |
| `test_prefill_decode_consistency` | v3 prefill vs decode 相对 L2 | `dtype × (seq,heads) × causal`；显式统一 `softmax_scale=192**-0.5` | decode 前 128 通道 ≈ prefill；bf16 rel-L2 < 5e-3、fp16 < 2e-3；并交叉核对 FP64/输入dtype CPU golden | 通过 |
| `test_prefill_decode_consistency_v4` | v4 同一致性 | `dtype × (seq,heads)`，`flash_attn_varlen_func` + `seqused_k` | `out_d[..., :128] ≈ out_p`，rel-L2 < 阈值 | 通过 |
| `_assert_close_to_golden`（辅助，被上述调用） | NPU vs FP64/输入dtype golden | — | 两者 rel-L2 均 < 5e-3 | 通过 |

> 说明：一致性用例**必须显式传入统一 scale**，因为两形态缺省 scale 不同（见 §3）。

---

## 3. `softmax_scale` 口径用例

| 用例名 / 文件 | 目的 | 输入要点 | 期望 | 状态 |
| --- | --- | --- | --- | --- |
| `test_mla_scale_default_derivation` | 缺省 scale 推导契约 | 不传 `softmax_scale` | compressed 得 `576^-0.5`、prefill 得 `192^-0.5`，比值 `sqrt(3)` | 通过 |
| `test_mla_default_scale` | 缺省 vs 显式 scale 行为 | `dtype`；对照 `PREFILL_SCALE=192^-0.5` 与 `COMPRESSED_DEFAULT_SCALE=576^-0.5` | 显式传入时两形态口径统一；缺省时口径不同 | 通过 |

---

## 4. 拒绝路径用例

| 用例名 / 文件 | 目的 | 输入要点 | 期望 | 状态 |
| --- | --- | --- | --- | --- |
| `test_mla_describe_rejected`（consistency） | 量化 descale 拒绝 | 传 `q_descale` | `RuntimeError` 匹配 `descale` | 通过 |
| `test_mla_invalid_layout_rejected`（consistency） | 非法 layout | 直接调 native op，`layout ∈ {2,3}` | `layout must be TND or BSND` | 通过 |
| `test_mla_v4_bnsd_rejected`（consistency） | v4 BNSD 拒绝 | BNSD storage view（KV 头>1 以便可检测） | 匹配 `BNSD` 报错 | 通过 |
| `test_mla_func_rejects_bnsd_layout`（v4） | v4 `flash_attn_func` BNSD 拒绝 | BNSD 输入 | 报错 | 通过 |
| `test_mla_rejects_gather_kv_indices`（v4） | `gather_kv_indices` 拒绝 | 传该参数 | 报错 | 通过 |
| `test_mla_nonstandard_parameters_rejected`（v3） | 非标准参数拒绝 | `layout × name` | 报错 | 通过 |
| `test_mla_invalid_inputs`（v3） | 非法输入 | `dtype × fault` | 报错 | 通过 |
| `test_mla_unsupported_options`（v3） | 不支持选项 | `representation × layout × name × value` | 报错 | 通过 |
| `test_mla_cache_update_rejected`（v3） | append-KV/rotary 等 cache 更新拒绝 | `layout × name` | 报错 | 通过 |
| `test_mla_metadata_and_length_rejections`（v3） | metadata/长度拒绝 | `layout × fault` | 报错 | 通过 |
| `test_mla_prefix_rejections`（v3） | 分配前缀校验拒绝 | `representation × layout × fault` | 报错 | 通过 |
| `test_mla_metadata_input_rejections`（v3） | metadata 输入拒绝 | `representation × fault` | 报错 | 通过 |
| `test_mla_metadata_device_only`（v3） | metadata 仅 device 生成 | `layout × monkeypatch` | 正确约束 | 通过 |

---

## 5. 回归用例（非 MLA 输入不被接管）

| 用例名 / 文件 | 目的 | 输入要点 | 期望 | 状态 |
| --- | --- | --- | --- | --- |
| `test_metadata_input_rejections`（`tests/test_flash_attn_npu_v3_metadata.py`，新增） | 普通 FA3 metadata 仍按原逻辑拒绝非法输入 | `fault` | 报错不变 | 通过 |
| `test_metadata_fake_matches_real` / `test_metadata_fake_without_device_runtime` / `test_metadata_fresh_storage_and_compiled_fingerprint`（同上，新增） | 普通 FA3 metadata real/fake、编译指纹 | `varlen/compiled/inference/symbolic` | 行为不变 | 通过 |
| `TestForwardRegistration::test_kvcache_backward_rejected` / `::test_kvcache_fresh_outputs` / `::test_forward_preallocated_output_rejected`（`tests/test_torch_compile_fa3.py`，新增） | 普通 FA3 kvcache/前向注册行为不变 | `execution/dtype/invalid_dtype` | 行为不变 | 通过 |
| `test_fa3_forward_schema_and_fake` / `test_fa3_bnsd_public_forward` / `test_fa3_bnsd_schema_and_fake`（同上，新增） | 普通 FA3 schema、BNSD 前向 | `dtype/strided/entry/causal` | 行为不变（BNSD 非 MLA 仍可用） | 通过 |

> `_is_mla` 分流保证：`qv is None` 且 `q.size(-1) == v.size(-1)` 的普通调用
> 直接 `return std::nullopt`，公共入口执行原有 FA3/FA4 逻辑（设计文档 §3.2）。

---

## 6. 运行方式

```bash
source /home/w00811046/miniconda3/etc/profile.d/conda.sh && \
conda activate cann && \
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh && \
export ASCEND_RT_VISIBLE_DEVICES=0

# 全部 MLA 用例
python -m pytest tests/test_flash_attn_npu_v3_mla.py -q
python -m pytest tests/test_flash_attn_npu_v4_mla.py -q
python -m pytest tests/test_flash_attn_npu_mla_consistency.py -q

# 回归
python -m pytest tests/test_flash_attn_npu_v3_metadata.py -q
python -m pytest tests/test_torch_compile_fa3.py -q
```

> 需要 Ascend 910B/910C 设备；三个 MLA 测试文件均带 `module` 级 fixture，
> 非目标设备时自动 `skip`（`tests/test_flash_attn_npu_mla_consistency.py:41-52`）。
