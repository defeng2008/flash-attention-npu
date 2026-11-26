# flash-attention-npu msSanitizer 内存检查报告

- 检查对象：`flash_attn_npu_3`（Ascend 910 / FA3）
- 参考流程：`flash-attention-npu/README.md`
- 日期：2026-11-26

## 1. 运行方式（严格按 README）

| 项目 | 内容 |
|---|---|
| conda 环境 | `cann` |
| CANN 环境 | `/home/w00811046/cann/cann-9.2.0-clean`（`source ascend-toolkit/set_env.sh`） |
| 设备 | 910B3（`dav-2201`） |
| 构建命令 | `FLASH_ATTN_ENABLE_MSSANITIZER=TRUE FLASH_ATTN_BUILD_VERSION=v3 python setup.py build_ext --inplace` |
| 插桩方式 | 910 → 静态插桩（`-g --cce-enable-sanitizer`，编译与链接均启用） |
| 插桩校验 | `.so` 由 57 MB 增至 2.2 GB，且含 sanitizer 符号，确认生效 |
| 运行命令 | `mssanitizer --tool=memcheck -- python -m pytest -q -s tests/test_flash_attn_npu_v3.py` |
| 用例范围 | 全量 595 条在 memcheck 下不可行，采用 CI 快速模式 `--random-sample=1`（每个测试函数取 1 条） |
| 实际规模 | 29 条用例、603 个 kernel，耗时 15 分钟 |
| 用例结果 | **29/29 pytest 通过** |

> 注：README 中可选的 `FLASH_ATTN_MEMCHECK=TRUE`（跳过 golden 比对）在当前测试代码中**并未实现**（仅出现在 README），本次运行包含完整精度比对与反向，故耗时较长。

## 2. 检查结论

| 检查项 | 结果 |
|---|---|
| 非法访问 / 段错误 | 无 |
| 越界**读** | 无（178786 条告警全部为 GM 越界**写**） |
| 未初始化读 / 数据竞争 | 无 |
| GM 越界写告警 | **178786 条** |
| 寄存器告警 | `FFTS_BASE_ADDR` 未复位（信息级） |
| 缺失 `debug_line` | 均为 torch_npu 内置算子（Cast / Sub / ReduceMax 等），非本仓 kernel |

## 3. 越界写按 kernel 分布

| kernel | 告警数 | 归属 |
|---|---|---|
| `FAGGeneral<...>` | ≈ 178253 | 本仓，**后向** kernel（绝对多数） |
| `SplitFuse::FAInfer<...>` | 504 | 本仓，前向 kernel |
| `SplitFuse::MlaDecodeGemv` | 22 | 本仓，MLA decode |
| combine | 6 | 本仓 |
| torch_npu `Transpose` | 少量 | 框架内置 |

## 4. 越界写 Top 代码位置（按 frame #0 聚合）

| 次数 | 位置 | 访问 |
|---|---|---|
| 129552 | `csrc/catlass/include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp:172` | `AscendC::Fixpipe` L0C→GM 整 tile 落盘 |
| 18625 | `csrc/ascend910/flash_attn_npu/fag_epilogue_op.hpp:825` | `DataCopyPad` 写 drop workspace |
| 17787 | `csrc/ascend910/flash_attn_npu/fag_epilogue_op.hpp:1009` | 后向 epilogue 写 GM |
| 5085 / 3652 / 3279 | `csrc/ascend910/flash_attn_npu/fag_epilogue_pre.hpp:95 / 99 / 100` | 后向 pre-epilogue 写 GM（含 `InitOutput` 初始化 dq workspace） |
| 312 | `csrc/ascend910/flash_attn_npu_3/rescale_o.hpp:294` | GM 写 |
| 273 | `csrc/ascend910/flash_attn_npu/fag_epilogue_sfmg.hpp:294` | GM 写 |
| 208 | `csrc/ascend910/flash_attn_npu_3/online_softmax.hpp:526` | `CopyPUbToGm` 用 `DataCopy`（非 `DataCopyPad`）写 P |
| 4 / 2 | `csrc/ascend910/flash_attn_npu_3/combine_scale.hpp:309 / 216` | `DataCopyPad` 写 O / LSE |

## 5. 判读

全部告警为同一形态：**按对齐块 / 整 tile 写 GM，末块越过逻辑 tensor 尾部，落进对齐 padding**。

- 典型例子：[online_softmax.hpp:526](file:///home/w00811046/code/fan/mlatest/flash-attention-npu/csrc/ascend910/flash_attn_npu_3/online_softmax.hpp#L526) 的 `CopyPUbToGm` 使用 `DataCopy` 按 32B 块写，靠 `columnNumPad - columnNumRound` 的 stride 间距填充，末尾 tile 会写超分配边界；Catlass Fixpipe 同理整 tile 落盘。
- 29 条用例结果全部正确、无崩溃，说明实际落在被过度分配的 padding 内，属"越界但无害"级别。
- 但从 memcheck 语义看确为真实越界写，建议重点复核**后向 `FAGGeneral` 的 epilogue** 与 **Catlass Fixpipe 落盘**是否有尾部 clamp。

## 6. 产物与还原

- 原始日志：`/home/w00811046/tmp/msan_sample1.msan.log`（mssanitizer 明细）、`/home/w00811046/tmp/msan_sample1.log`（终端输出）
- 检查完成后已将 `flash_attn_npu_3` 还原为正常构建（56.9 MB、无 sanitizer 符号），避免插桩版 `.so` 无法脱离 mssanitizer 运行。
- 复现：重新执行第 1 节构建命令 + 运行命令即可。
