# Host / AICPU / AI Core 打日志方式总结

本文档针对 `flash-attn-npu` 在 Ascend 910（CANN 9.2.0）上的三类执行体，给出**经过实测验证**的日志方案、落盘位置、编译前提和已知死路。

---

## 0. 一张表看懂

| 执行体 | 推荐接口 | 输出位置 | 编译/链接前提 | 终端可见 |
| --- | --- | --- | --- | --- |
| Host（宿主进程） | `TORCH_WARN` / `TORCH_CHECK` | stdout / stderr | 无 | 是 |
| Host（宿主进程） | `AlogRecord(...)` | `~/ascend/log/run/plog/plog-*.log` | 链接 `libascendalog` | 否 |
| AICPU（`-x aicpu` 编译的 `.so`） | `ASC_CPU_LOG_INFO / ERROR / WARNING / DEBUG` | `~/ascend/log/debug/device-<n>/device-*.log` | `-I $CANN/<arch>/asc/include` + `-I $CANN/<arch>/include` | 否（设备侧 plog） |
| AI Core（`-x asc` 编译的 kernel） | `AscendC::printf` / `AscendC::PRINTF` | 宿主 stdout（dump 通道） | 需 runtime 提供系统打印 FIFO | 是（有条件） |
| AI Core（`-x asc` 编译的 kernel） | `AscendC::DumpTensor` / `DumpAccChkPoint` | dump 文件 / profiler | 需开启 dump 配置 | 否 |

`$CANN` 指 `ASCEND_TOOLKIT_HOME`（本机为 `/home/w00811046/cann/cann-9.2.0/cann-9.2.0`），`<arch>` 为 `aarch64-linux`。

---

## 1. Host 侧

### 1.1 `TORCH_WARN` / `TORCH_CHECK`（最省事）

宿主进程里直接写，不需要任何 CANN 依赖，输出走 stdout/stderr，`grep` 得到。

```cpp
TORCH_WARN("[FA3 MLA mla_fwd] wdflayout=", layout, " causal=", is_causal);
TORCH_CHECK(layout == 0 || layout == 1, "MLA: layout must be TND or BSND");
```

- 每条 `TORCH_WARN` 只打印一次同类消息，重复调用会被去重。
- 若日志在静态库/插件里，注意 `TORCH_WARN` 触发时会带 `(Triggered internally at <file>:<line>)`，可据此确认代码版本是否为最新编译产物。

### 1.2 `AlogRecord`（宿主 plog）

头文件：`$CANN/<arch>/include/base/alog_pub.h`

```cpp
#include "base/alog_pub.h"   // 需 -I $CANN/<arch>/include

AlogRecord(SLOG, DLOG_TYPE_DEBUG, DLOG_INFO, "value=%d\n", v);
```

- 落盘到 `~/ascend/log/run/plog/plog-<pid>_<timestamp>.log`。
- 链接 `-L$CANN/<arch>/lib64 -lascendalog`（该库导出 `AlogRecord`、`AlogCheckDebugLevel` 为 weak 符号）。
- 与 AICPU 的区别见 2.3：**同名接口在设备侧不可用**。

### 1.3 环境变量

| 变量 | 作用 |
| --- | --- |
| `ASCEND_GLOBAL_LOG_LEVEL=1` | 全局日志级别调到 INFO（0=DEBUG,1=INFO,2=WARN,3=ERROR） |
| `ASCEND_SLOG_PRINT_TO_STDOUT=1` | 宿主 plog 额外打到 stdout。**只对宿主进程生效** |
| `ASCEND_PROCESS_LOG_PATH=<dir>` | 自定义 plog 落盘目录 |

---

## 2. AICPU 侧（`bisheng -x aicpu`）

AICPU kernel 被编译成一个独立 `.so`，由 `aicpu_custom_scheduler` 子进程在设备侧加载执行。它既不是宿主进程、也不是 AI Core，日志设施自成一套。

### 2.1 正确接口：`ASC_CPU_LOG_*`

```cpp
#include "utils/log/asc_cpu_log.h"

ASC_CPU_LOG_INFO("[MLA-BUILD] b=%u sq=%u words=%u", a.context.batch, a.context.sq, a.words);
ASC_CPU_LOG_WARNING(...);
ASC_CPU_LOG_ERROR(...);
ASC_CPU_LOG_DEBUG(...);
```

展开后是 `dlog_info(ASCENDCKERNEL, "[%s:%d][%s] " fmt "\n", ...)`，由 `CheckLogLevel` 门控，实际写入设备 plog。

模块名固定为 `ASCENDCKERNEL`，因此日志行形如：

```
[INFO] ASCENDCKERNEL(16200,aicpu_custom_scheduler):...[mla_metadata.cpp:19][mla_metadata.cpp:19][BuildNonquantMetadata] [MLA-BUILD] b=2 sq=64 words=404
```

### 2.2 编译前提（关键）

该头文件内部有两级相对 include：

```cpp
#include "../pkg_inc/base/dlog_pub.h"   // 必须以 asc/include 为根
#include "aicpu/cust_cpu_utils.h"       // 必须以 <arch>/include 为根
```

所以 AICPU 编译命令必须补两条 `-I`（本项目已固化在 `setup.py` 的 `aicpu_cmd` 中）：

```python
f"-I{aicpu_inc}",                                                # asc/include/aicpu_api
f"-I{os.path.join(ascend_home, cann_arch_dir, 'asc/include')}",  # utils/log/asc_cpu_log.h
f"-I{os.path.join(ascend_home, cann_arch_dir, 'include')}",      # aicpu/cust_cpu_utils.h
```

漏掉第一条的典型报错：

```
mla_metadata.cpp:14:10: fatal error: 'utils/log/asc_cpu_log.h' file not found
```

**只能加在 AICPU 编译命令上**。改完后 `setup.py` 的 mtime 会进入 AICPU 增量签名，强制重编该 `.o`，无需手工删旧产物。

### 2.3 落盘与查看

输出**不在终端**，在设备 debug 日志里：

```bash
grep -rh "MLA-BUILD" ~/ascend/log/debug/device-*/
# 或者按模块过滤
grep -rh "ASCENDCKERNEL.*mla_metadata" ~/ascend/log/debug/device-*/
```

实测输出：

```
[MLA-BUILD] b=2 sq=64 words=404
[MLA-BUILD] b=2 sq=64 words=400
[MLA-BUILD] b=2 sq=64 words=400
```

- `device-<n>` 中 `n` 是 device id + 1。
- 一次进程运行可能产生多个 `device-*.log`（每次 so 重载一个新文件），所以用 `-r` 递归。
- 若看不到，先确认 `ASCEND_GLOBAL_LOG_LEVEL` 不大于 1（INFO）。宿主侧的 `ASCEND_SLOG_PRINT_TO_STDOUT=1` **对它无效**。

### 2.4 三条死路（均已踩过）

| 方式 | 现象 | 根因 |
| --- | --- | --- |
| `printf(...)` | 编译通过、运行无输出，`~/ascend/log` 里也搜不到 | AICPU 的 stdout 不接宿主终端管道 |
| `AlogRecord(...)`（`base/alog_pub.h`） | `aicpu exception, errorCode=0x2a, funcName=_Z18ComputeMlaMetadataPv` | 该符号在宿主 `libascendalog.so`，设备侧不可解析；声明为 weak，编译链接都过，运行跳空地址 |
| `#include "kernel_operator.h"` 或 `AscendC::*` | `'kernel_operator.h' file not found` | AICPU 编译单元没有 Ascend C 头文件搜索路径，也不该有 |

### 2.5 另一条通道：`CustCpuKernelUtils`

`aicpu/cust_cpu_utils.h` 提供 `CustLogInfo/Warning/Error(aicpu::CpuKernelContext&, fmt, ...)`，但需要拿到 `CpuKernelContext`。本项目的 `ComputeMlaMetadata(aicpuContext*, size)` 签名里只有裸指针，不持有 context，因此不适用；普通自定义 AICPU 算子可以直接用。

### 2.6 零依赖兜底：写回 metadata 缓冲区

如果不信任任何日志通道，最稳的做法是把值写进**已经分配好、宿主能读回**的缓冲区。

统一结构下 MLA 记录把 header 织进 `FAInferTilingData`，逻辑字 29/30/31（原 `reserved[3]`）已无落点。
可用于调试的是 MLA 路径下**不被写入的 FA 区间**——`FAInferTilingData` 物理字 `44 ~ 593`
（原 `coreInfo[25]` / `splitInfo[25]`）在 MLA 路径下恒为 0（`memset` 后无人再写）：

```cpp
// AICPU 侧：写入物理字 44 起的空闲区
auto* out = reinterpret_cast<uint32_t*>(a.metaOutAddr);
out[44] = a.batch;   // 调试值
```

```python
# 宿主侧按 dword 读回（缓冲是 uint8）
print(meta.view(torch.int32)[44].item())
```

---

## 3. AI Core 侧（`bisheng -x asc`）

### 3.1 `AscendC::printf`

```cpp
#include "kernel_operator.h"

AscendC::printf("block=%d value=%u\n", AscendC::GetBlockIdx(), v);
AscendC::PRINTF("...");   // 同义
```

- 910B 对应 `__NPU_ARCH__ == 2201`，该架构在支持列表内（2002/2201/3002/3510）。
- 实现方式是写一段 TLV 到 GM 里的系统打印 FIFO（`g_sysPrintFifoSpace`），kernel 结束后由 runtime 回读到宿主。
- **前提**：`g_sysPrintFifoSpace` 必须非空，否则 `get_block_info()` 直接返回 `nullptr`，`printf` 静默丢弃，不报错。直调（`aclrtLaunchKernelWithArgsArray`）场景下如果没配 dump/print 空间，就会完全无输出。
- 不要在循环里密集打印，TLV 环形缓冲区溢出后同样静默丢数据。

### 3.2 `AscendC::DumpTensor`

```cpp
#include "kernel_operator.h"

AscendC::DumpTensor(tensor, 100 /*desc, 唯一标识*/, 256 /*dumpSize 元素个数*/);
AscendC::DumpAccChkPoint(tensor, index, countOff, dumpSize);
```

- 走 ACL dump 通道，结果落到 dump 文件 / profiler，不开 dump 配置同样没有输出。
- 适合看"某一块模板计算完是什么值"，不适合当作流程日志。

### 3.3 断言与时间戳

- `ASCENDC_ASSERT(cond, { ... })`：条件不满足时触发断言日志并终止 kernel。
- `asc_time_stamp(desc_id)`（`utils/debug/asc_time.h`）：打时间戳点，配合 profiler 看流水。

### 3.4 编译单元隔离（重要）

| 文件类型 | 编译方式 | 可用 |
| --- | --- | --- |
| `mla_prefill_dispatch_impl.hpp` 等 ASC device 目标 | `bisheng -x asc` | `AscendC::*`、`DumpTensor`、`printf` |
| `fa_metadata.aicpu`（内含 `mla_metadata.cpp` / `mla_metadata_consumer.h` 等） | `bisheng -x aicpu` | 只有 AICPU 接口，**不能** include `kernel_operator.h` |

判断方法：看 `setup.py` 里该源文件走的是 `compile_common`（`-x asc`）还是 `aicpu_cmd`（`-x aicpu`）。

---

## 4. 选型决策

1. **先问这条路径是哪一类执行体**——不看代码位置看编译方式。
2. Host 边界（tiling、形状校验、张量分配、参数拼装）→ `TORCH_WARN`/`TORCH_CHECK`，成本最低。
3. AICPU 元数据生成（`BuildNonquantMetadata`、`PrepareMlaGeometry`、分核）→ `ASC_CPU_LOG_INFO` + `grep ~/ascend/log/debug/device-*/`。
4. AI Core kernel 内部 → 优先 `DumpTensor` 看数据，流程性信息改用 GM 回读，`printf` 只在确认 print FIFO 可用时用。
5. **任何方案都不要留在提交里**；调试完成后清理，只保留 `setup.py` 的两条 `-I`（零成本，便于复现）。

---

## 5. 复现步骤（本项目实测）

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11
export LIBRARY_PATH=/usr/lib/gcc/aarch64-linux-gnu/11
cd /home/w00811046/code/mla/flash-attention-npu
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python setup.py build_ext --inplace

# 确认日志串已进 .so
grep -c "MLA-BUILD" flash_attn_npu_3/flash_attn_npu_3.cpython-310-aarch64-linux-gnu.so

python tests/manual_mla_check.py
grep -rh "MLA-BUILD" ~/ascend/log/debug/device-*/
```

`CPLUS_INCLUDE_PATH` / `LIBRARY_PATH` 不可省：bisheng 用 hcc 自带的 GCC 14.3.0 交叉工具链，找不到宿主 libstdc++ 头时会报 `'cstddef' file not found`。
