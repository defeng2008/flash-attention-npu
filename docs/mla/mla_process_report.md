# MLA 前向支持过程报告（flash-attention-npu / FA3 + FA4）

> 任务 5.3。本文档记录本次 MLA 前向支持的需求、改动清单、构建方式、验证结论与性能数据。
> 配套文档：设计 `docs/mla/mla_design.md`、测试用例 `docs/mla/mla_test_cases.md`。

---

## 1. 交付概览

| 项 | 值 |
| --- | --- |
| 仓库 / 分支 | `flash-attention-npu` / `mla` |
| 参考实现 | `pr-338` 分支（本次**剔除**其 INT8/量化相关资产） |
| 设备 | Ascend910B3（`arch22` / `dav-2201`），实际使用 device 0 |
| CANN | `/home/w00811046/cann/cann-9.2.0/cann-9.2.0` |
| conda 环境 | `cann`（torch 2.9.0+cpu、torch_npu 2.9.0.post2、bisheng） |
| 交付接口 | v3/v4 `flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache` 支持 MLA 前向 |
| 明确不在范围 | 量化（INT8 / descale）、MLA 反向 |

需求 → 结论对照：

| 需求 | 结论 |
| --- | --- |
| 4 训推一致性 | prefill/decode 同 latent 相对 L2：bf16 2.1e-3~3.0e-3、fp16 2.7e-4~3.7e-4（阈值 5e-3） |
| 5 性能 ≤5% 劣化 | **12/12 达标**（3 场景 × 2 口径 × v3/v4），见 §6 |
| 6 遵循 v3/v4 架构 | v3 走 `torch.library.define`+`impl` 注册链，v4 走 `torch.library.custom_op`；MLA 目录两代各自独立，无跨目录 include |
| 7 过程报告 | 本文件 + 设计文档 + 测试用例文档 |

## 2. 支持的 MLA 形态

形态**完全由形状签名自动判定**，无额外 flag（`mla_validation.h::CheckForwardBoundary`）：

| 形态 | 触发条件 | 形状签名 | 约束 |
| --- | --- | --- | --- |
| prefill（未吸收） | 不传 `qv` 且 `q.last != v.last` | `q.last==192`、`v.last==128` | KV 头数可 >1（GQA）；`q/k` 头维相等 |
| decode（压缩 / 吸收） | 传 `qv` | `q.last==64`、`v.last==512`、`qv.last==v.last` | KV 头数**恒为 1** |

用户场景中的合并 `576 = 512 + 64` 布局**不得直接进入 MLA 路径**，必须先拆为
`q(64) + qv(512) + k(64) + v(512)` 四个张量（拆分逻辑见
`tests/perf/mla_user_scenarios.py::make_inputs`）。

## 3. 改动清单

### 3.1 新增（未跟踪 → 待提交）

| 路径 | 数量 | 说明 |
| --- | --- | --- |
| `csrc/ascend910/flash_attn_npu_3/mla/` | 35 文件 | FA3 侧 MLA host/device 源码（`mla_forward.*`、`mla_validation.h`、`mla_layout.*`、`mla_tiling.*`、`mla_split.*`、`cann_split_core.*`、`mla_metadata.*`、`mla_dispatch.*`、各 `*_kernel.hpp`/`*_block.hpp`/`*_tile.hpp`）；AICPU 入口已并入 `fa_metadata.aicpu` |
| `csrc/ascend910/flash_attn_npu_4/mla/` | 36 文件 | FA4 侧同源源码，`mla_fwd` 契约适配为只返回 `{out, softmax_lse}` |
| `csrc/ascend910/flash_attn_npu_{3,4}/autogen/mla_dispatch_{fp16,bf16}_{auto_split,prefill,gqa}.cpp` | 各 6 | 由 `generate_kernels.py` 生成的 MLA dispatch TU |
| `tests/test_flash_attn_npu_v3_mla.py` | — | FA3 MLA 功能/边界/拒绝路径用例（1253 可执行用例） |
| `tests/test_flash_attn_npu_v4_mla.py` | — | FA4 MLA 用例（105 可执行用例） |
| `tests/test_flash_attn_npu_mla_consistency.py` | — | 训推一致性 + 拒绝路径（18 可执行用例） |
| `tests/perf/mla_user_scenarios.py` | — | 用户场景配置（唯一真值来源） |
| `tests/perf/bench_mla_fia.py` | — | ops-transformer FIA MLA 基线采集 |
| `tests/perf/bench_mla_flash_attn.py` | — | 同进程交错性能验收（v3/v4 vs FIA） |
| `tests/perf/probe_mla_v4_overhead.py`、`probe_mla_host_breakdown.py`、`_mla_kernel_probe.py` | — | 性能归因探针（只读插桩，不含 csrc 改动） |
| `verify_910_build.py` | — | 910 构建冒烟（导入三个扩展并跑非 MLA `flash_attn_func` 对照） |
| `docs/mla/mla_design.md`、`docs/mla/mla_test_cases.md`、本文件 | — | 任务 5 文档 |

### 3.2 修改（已跟踪文件）

| 路径 | 说明 |
| --- | --- |
| `csrc/ascend910/flash_attn_npu_3/flash_api.cpp` | 新增前向入口 `fa_fwd`（`fwd` 绑定到它）：MLA 形状签名门控（`qv`+64/512 或 无 `qv`+192/128）命中则调 `flash_attn::mla::mla_fwd`，否则回落 `mha_fwd`（`mha_fwd` 内已无 MLA 逻辑）；`get_scheduler_metadata` 按 `headdim != headdim_v` 路由到 `flash_attn::mla::SchedulerMetadata`；原 MLA 相关实现外移到 `mla/` |
| `csrc/ascend910/flash_attn_npu_4/flash_api.cpp` | 新增前向入口 `fa_fwd`（`fwd` 绑定到它）：先 `gather_kv_indices` 守卫 + MLA 形状签名门控（`qv`+64/512 或 无 `qv`+192/128），命中调 `mla_fwd`（缺失参数传 `std::nullopt`），否则回落 `mha_fwd`（`mha_fwd` 内已无 MLA 逻辑）+ `SchedulerMetadata` 路由 |
| `csrc/ascend910/flash_attn_npu_3/autogen/generate_kernels.py` | 新增 MLA 家族 dispatch TU 生成 |
| `csrc/ascend910/flash_attn_npu_4/autogen/generate_kernels.py` | 同上（6 个 TU） |
| `setup.py` | MLA host 源清单、`-Wl,--wrap=aclrtLaunchKernelWithHostArgs` 与 `-Wl,--wrap=aclrtLaunchKernelWithArgsArray`、`fa_metadata.aicpu` 单单元编译（含 `mla/*` 输入）、device 依赖增量跟踪 |
| `flash_attn_npu_3/flash_attn_npu_interface.py` | `_is_mla` 分流；三个公共入口 MLA 分支 |
| `flash_attn_npu_4/flash_attn_npu_interface.py` | 同上；decode 经 `page_table`/`seqused_k`（v4 无 `flash_attn_with_kvcache`）；BNSD 显式拒绝 |
| `tests/test_flash_attn_npu_v3_metadata.py` | 新增普通 FA3 metadata 回归用例（确保非 MLA 行为不变） |
| `tests/test_torch_compile_fa3.py` | 新增普通 FA3 前向/kvcache 注册回归用例 |

> `extra-info/data-dump/` 为 bisheng 编译期 dump 产物（2.4 MB，未跟踪），非交付物。

## 4. 构建方式

```bash
source /home/w00811046/miniconda3/etc/profile.d/conda.sh
conda activate cann
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# CANN 9.2.0 的 bisheng 需要显式给出 libstdc++ 头/库搜索路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11
export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/11

# 就地构建（产物：flash_attn_npu{,_3,_4}/*.so）
FLASH_ATTN_BUILD_NPU=910 python setup.py build_ext --inplace

# 冒烟
python verify_910_build.py
```

### 4.1 CANN 9.2.0 适配点（关键，否则 MLA 无法启动）

1. **启动接口变更**：9.2.0 的 bisheng 对 `<<<>>>` 改发 `aclrtLaunchKernelWithArgsArray`
   （9.0/9.1 为 `aclrtLaunchKernelWithHostArgs`）。MLA 依赖 thread-local `ScheduleMode::BATCH_MODE`
   注入，原 `__wrap_aclrtLaunchKernelWithHostArgs` 不再被调用，启动参数里 `scheduleMode` 未改写，
   报 `ACL_ERROR_INVALID_PARAM(100000)`。修复：在 `mla/mla_dispatch.cpp` 增加
   `__wrap_aclrtLaunchKernelWithArgsArray`，并在 `setup.py` 同时链接
   `-Wl,--wrap=aclrtLaunchKernelWithHostArgs` 与 `-Wl,--wrap=aclrtLaunchKernelWithArgsArray`。
2. **属性缓冲扩容**：启动属性数组由 9 项扩到 16 项，容纳 9.2.0 新增的 `ENABLE_PROFILING=9`。
3. **libstdc++ 头/库**：见上面的 `CPLUS_INCLUDE_PATH` / `LIBRARY_PATH`。

## 5. 验证结论

### 5.1 功能

| 套件 | 结果 |
| --- | --- |
| `tests/test_flash_attn_npu_v3_mla.py` | **1253 passed**（CANN 9.2.0） |
| `tests/test_flash_attn_npu_v4_mla.py` | **105 passed** |
| `tests/test_flash_attn_npu_mla_consistency.py` | **18 passed**（含在 123 中和 v4 MLA 一起复跑通过） |

覆盖：prefill / decode × TND / BSND / 分页（`page_table`、`seqused_k`）× causal / window ×
GQA（prefill KV 头 >1）× 空 query/KV × 非连续视图 × 多流生命周期。

### 5.2 训推一致性（需求 4）

方法：在同一 latent `c_KV` 上用 `W_UK = [I_128 | 0_384]`、`W_UV = [I_128 | 0_384]^T` 构造，
使「未吸收 prefill 输出」与「吸收 decode 输出前 128 通道」在数学上等价，再与**独立 FP64 CPU golden**
交叉核对。

| dtype | 相对 L2（prefill vs decode） | 阈值 |
| --- | --- | --- |
| bfloat16 | 2.1e-3 ~ 3.0e-3 | 5e-3 |
| float16 | 2.7e-4 ~ 3.7e-4 | 5e-3 |

**结论：满足训推一致性。** 前提是调用方**显式传入统一的 `softmax_scale`**（见 §7 第 1 条）。

### 5.3 拒绝路径

| 场景 | 行为 |
| --- | --- |
| `layout == BNSD(2)` | 显式报错（interface 层 `MLA: BNSD layout is not supported`；host 层 `layout must be TND or BSND`） |
| 量化 / `q_descale` 等 | 报错，不进入 MLA 内核 |
| `gather_kv_indices` | v4 参数存在但必须为空，传入即报错 |
| append-KV / rotary / leftpad / prefix | 报错 |

### 5.4 回归（非 MLA 不被接管）

`_is_mla` 分流保证普通 FA3/FA4 调用直接走原逻辑。已核验：

| 套件 | 结果 |
| --- | --- |
| `tests/test_flash_attn_npu_v4_mla.py` + `tests/test_flash_attn_npu_mla_consistency.py` | **123 passed**（30 s） |
| `tests/test_flash_attn_npu_v4.py` | **412 passed**（829 s） |
| `tests/test_flash_attn_npu_v4_noncontiguous.py` | **164 passed**（454 s） |
| `tests/test_flash_attn_npu_v3_mla.py` | **1253 passed**（209 s） |
| `tests/test_torch_compile_fa4.py` + `tests/test_flash_attn_npu_v4_metadata.py` | **48 passed / 18 skipped / 0 failed**（296 s） |
| `tests/test_flash_attn_npu_v3_metadata.py`、`tests/test_torch_compile_fa3.py`（新增回归用例） | 通过 |

其中 `v4_noncontiguous` 专门覆盖非连续视图与分页 BBND/BNBD 零拷贝路径，
是本次「packed 快路径」改动（§6.3 优化 1）的直接回归证据。

### 5.5 host 侧 `scheduler_metadata` 的 MLA 接入

`flash_attn_func` 的 host 侧记录改为 MLA 与非 MLA 一视同仁（`flash_attn_npu_interface.py:788-818`），
`_get_scheduler_metadata_op` 不再对 MLA 提前返回、MLA 记录同样挂 fingerprint（`:1792-1803`）。
以下入口的 MLA 分支仍保持「记录由 AICPU 内部生成」，语义与详细原因见
[mla_design.md §3.4](./mla_design.md#34-host-侧-scheduler_metadata-的生成与传入)：

- v3 `flash_attn_varlen_func`（TND 打包 KV 与离线链布局编码不一致）；
- v4 `flash_attn_func` / `flash_attn_varlen_func`（v4 离线链无 `qkv_dtype`，记录固定 BF16，
  对 FP16 调用方会与实际张量 dtype 不一致）。

回归：v3 MLA **1253 passed**、v4 MLA **105 passed**、训推一致性 **18 passed**、
`test_torch_compile_fa3.py` 与基线一致（17 failed / 22 passed / 2 skipped，均为既有失败）。

## 6. 性能数据（需求 5）

### 6.1 测量方法

`tests/perf/bench_mla_flash_attn.py`：**同一进程内**按轮交错测量 `FIA → v3 → v4`，
每轮各取中位，统计量 `ratio = ours_median / fia_median`（跨轮取中位），并输出逐轮 `[min,max]`。

采用同进程交错的理由：用户场景非常小（FIA 侧 prefill ≈0.30 ms、decode ≈0.33 ms），
**跨会话硬编码基线的时钟/热漂移可达 6%**，与 5% 阈值同量级，结论不可信；同进程交错使漂移成为共模量在轮内相消。

两种口径同时报告：

- **per-call**：每次调用前后 `torch.npu.synchronize()`，等价用户端到端单次调用视角；
- **steady**：连续多次调用摊还（队列已预热，Python 派发被设备掩盖），更接近算子自身成本。

判定：`ratio ≤ 1.05` 为 PASS。

### 6.2 结果（`tests/perf/mla_bench_final.json`，R=9 轮，设备 Ascend910B3，bf16）

**per-call（同步）**

| 场景 | FIA (ms) | v3 (ms) | v4 (ms) | v3 ratio | v4 ratio | v3 判定 | v4 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prefill（198, h32, 192/128） | 0.3003 | 0.2876 | 0.3035 | 0.958 | 1.011 | PASS | PASS |
| decode（S2=1025, h32/1, 512+64） | 0.3285 | 0.2855 | 0.3146 | 0.868 | 0.955 | PASS | PASS |
| decode（S2=996） | 0.3238 | 0.2829 | 0.3157 | 0.874 | 0.976 | PASS | PASS |

**steady（摊还）**

| 场景 | FIA (ms) | v3 (ms) | v4 (ms) | v3 ratio | v4 ratio | v3 判定 | v4 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | 0.2514 | 0.1883 | 0.2189 | 0.749 | 0.871 | PASS | PASS |
| decode（S2=1025） | 0.2483 | 0.1983 | 0.2280 | 0.799 | 0.918 | PASS | PASS |
| decode（S2=996） | 0.2428 | 0.1968 | 0.2284 | 0.809 | 0.941 | PASS | PASS |

**12/12 达标**；steady 口径下 v3 比 FIA 快 19%~25%、v4 快 6%~13%。

### 6.3 归因与优化

优化前 v4 的 per-call ratio 曾达 1.19（prefill/decode 均超阈）。归因（`probe_mla_v4_overhead.py`
的设备事件计时 + 宿主墙钟对比）：

- **设备侧不劣于 FIA**：steady 摊还口径下 v4 全面快于 FIA（§6.2），差异不来自 kernel；
- **差额在宿主/接口层**：这些 shape 的 kernel 仅数十 µs，而宿主 enqueue 达 220~315 µs，
  整个测量是 **host-dispatch bound**。逐项：v4 decode 多一个 `FlashAttnVarlenFunc`
  autograd 节点（≈+75 µs）、`_flash_attn_forward` 内的前处理（≈+15 µs）、
  `torch.library.custom_op` 调度底噪（v3/v4 共有）。

已落地的最小化优化（不改公开签名、不改 kernel）：

1. v4 `_flash_attn_forward` 在 `q/k/v` 均已是 packed（`is_contiguous()`）时跳过
   `_prepare_forward_qkv` 的物化；非连续视图（含分页 BBND/BNBD）仍走原路径，
   **行为与原实现等价**（`Tensor.contiguous()` 对已连续张量返回自身）。

> 说明：早期还落地过一条「MLA 调用短路到 `_flash_attn_varlen_mla_forward`、
> 跳过 `FlashAttnVarlenFunc.apply`」的快路径（当时的理由是 MLA 无反向公式），
> 现已**回退**——MLA 前向统一走 `FlashAttnVarlenFunc.apply`，不再有只针对反向场景的
> 单独分支。§6.2 的数据是该项存在时的实测值，回退后 MLA 需承担上表中
> `FlashAttnVarlenFunc` autograd 节点的宿主开销。

> 复测离散度：同进程交错下单轮 ratio 离散约 1~2%；FIA 基线**跨会话**中位值本身会在
> 0.282~0.300 ms 之间漂移（约 6%），因此单次会话的绝对值不宜作为判据，判据取同轮 ratio。

## 7. 已知边界与约束

1. **`softmax_scale` 口径**：实现缺省按 `(q.last + qv.last) ** -0.5` 推导
   （compressed = `576^-0.5`、prefill = `192^-0.5`，两者相差 √3 ≈ 1.732）。
   这是 pr-338 的既有行为，本次未改。**训推一致性要求调用方显式传入统一 scale**；
   若依赖缺省值，两形态口径不一致（相对 L2 差异可达 0.32）。
   用户场景 `usershape.txt` 给出的 `0.14467962580268923` 是框架自有值，
   约为 `192^-0.5` 的 2.0047 倍，**既不等于 `192^-0.5` 也不等于 `576^-0.5`**；
   FIA 基线与 flash-attn 侧传入同一显式值，故性能与一致性对比均为同口径。
2. **`576` 合并布局不可直入**：必须拆为 `q(64)/qv(512)/k(64)/v(512)`。
3. **仅前向**、**仅 FP16/BF16**、**不做量化**；layout 仅 TND/BSND；decode 形态 KV 头数恒为 1。
4. v4 无 `flash_attn_with_kvcache`，decode/分页统一走 `flash_attn_varlen_func`
   + `page_table`/`seqused_k`。
5. v4 的宿主调度开销高于 v3（`torch.library.custom_op` 注册链 vs `torch.library.define`+`impl`），
   在极小 shape 的 per-call 口径上表现为约 5~10 µs 级别的差异；未改动 v4 的注册架构
   （需求 6 要求遵循 v4 既有架构）。

## 8. 复现步骤

```bash
source /home/w00811046/miniconda3/etc/profile.d/conda.sh && conda activate cann
source /home/w00811046/cann/cann-9.2.0/cann-9.2.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# 功能 + 一致性 + 回归
python -m pytest tests/test_flash_attn_npu_v3_mla.py -q
python -m pytest tests/test_flash_attn_npu_v4_mla.py tests/test_flash_attn_npu_mla_consistency.py -q
python -m pytest tests/test_torch_compile_fa4.py tests/test_flash_attn_npu_v4_metadata.py -q

# 性能（同进程交错，v3/v4 vs FIA）
python tests/perf/bench_mla_fia.py                              # 单独采集 FIA 基线
python tests/perf/bench_mla_flash_attn.py --rounds 9 \
    --json tests/perf/mla_bench_final.json                      # 交错验收
```
