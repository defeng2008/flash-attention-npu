# atktest：flash-attn 内存检查（ATK memory_check）

用 ATK 的 `memory_check` 任务（底层 msSanitizer）检查 flash-attn 算子的内存越界 / 未初始化访问 / 泄漏等问题。
只覆盖 `flash_attn_func` 与 `flash_attn_with_kvcache` 两类接口，910 与 950 两代设备。

## 1. 测试矩阵

| 接口 key | 版本 | NPU 模块 | 910 | 950 |
|---|---|---|---|---|
| `func_v2` | v2 | `flash_attn_npu` | ✅ | ❌ |
| `kvcache_v2` | v2 | `flash_attn_npu` | ✅ | ❌ |
| `func_v3` | v3 | `flash_attn_npu_3` | ✅ | ✅ |
| `kvcache_v3` | v3 | `flash_attn_npu_3` | ✅ | ✅ |
| `func_v4` | v4 | `flash_attn_npu_4` | ✅ | ✅ |
| ~~`kvcache_v4`~~ | v4 | `flash_attn_npu_4` | — | — |

两条硬约束（来自代码，不是配置选择）：

- **v2 仅支持 910**：`flash_attn_npu/__init__.py` 在非 910 设备 import 时直接 `raise RuntimeError`。
- **v4 没有 `flash_attn_with_kvcache`**：`flash_attn_npu_4/__init__.py` 只导出 `flash_attn_func` / `flash_attn_varlen_func` / `get_scheduler_metadata`。

因此 910 跑 5 个接口，950 跑 3 个。每个接口固定 200 条用例：

| 设备 | 接口 | 用例数 |
|---|---|---|
| 910 | `func_v2` / `kvcache_v2` / `func_v3` / `kvcache_v3` / `func_v4` | 5 × 200 = **1000** |
| 950 | `func_v3` / `kvcache_v3` / `func_v4` | 3 × 200 = **600** |

910 与 950 共用同一份 `cases/source/`，不再按设备裁剪。

## 2. 目录内容

```
atktest/memcheck/
├── atk_memory_check_matrix.yaml   # 接口 → 用例源 / plugin 目录映射
├── nodes_memory_check.yaml        # 单 npu 节点，task: [memory_check]
├── cases/
│   ├── gen/                       # 生成配置 <key>.yaml + 约束脚本 generate_<key>.py
│   ├── source/<key>.json          # 全量用例（自包含，不再引用交付目录）
│   ├── filter_cases.py            # 按设备挑选接口并规范化用例
│   └── <device>/<key>.json        # 生成物（可随时重建）
├── plugins/<key>/                 # 执行插件：execute_*.py + benchmark_*.py
├── gen_cases.sh                   # 重新生成 cases/source 的入口
├── run_memory_check.sh            # 统一运行入口
└── reports/<device>/<key>/        # 归档：atk_output 报告 + mss_logs
```

用例的生成配置、生成脚本与产物均已复制进本目录，运行时不依赖
`flash_attention_v2_v3_atk_delivery/` 与 `flash_attention_bnsd_atk_delivery/`。

## 3. 环境准备

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ascend
source ~/miniconda3/Ascend/cann/set_env.sh
npu-smi info      # 确认目标卡空闲且健康
python3 -c "import torch_npu; print(torch_npu.npu.get_device_name())"
```

按设备准备算子包（关键差异）：

```bash
cd flash-attention-npu

# 910：必须带 msSanitizer 静态插桩编译
FLASH_ATTN_ENABLE_MSSANITIZER=TRUE python setup.py install

# 950：普通编译，由 ATK 运行时注入（-ms 负责包装）
FLASH_ATTN_BUILD_NPU=950 python setup.py install
```

## 4. 操作步骤

### 步骤 0：重新生成用例（可选，仅在需要换参数网格时执行）

```bash
cd flash-attention-npu/atktest/memcheck
./gen_cases.sh --dry-run                # 只生成到 cases/gen/result 并打印条数，不覆盖 source
./gen_cases.sh                          # 重建 cases/source/ 下全部 5 份用例
./gen_cases.sh --interface func_v3 --seed 2026   # 只重建一个接口
```

生成命令固定带 `-en 0`：关闭边界用例，否则 atk 在 shape 尚未填充时会调用约束脚本而报
`ValueError: not enough values to unpack`。

### 步骤 1：按设备派生用例

```bash
cd flash-attention-npu/atktest/memcheck
python3 cases/filter_cases.py --device 950 --dry-run    # 只打印各接口条数
python3 cases/filter_cases.py --device 950              # 写入 cases/950/
```

910 与 950 共用同一份 `cases/source/`，不做裁剪；脚本只按设备挑选接口
（v2 仅 910，其余 910/950 均支持）、重排 `id`、替换私有精度比较器。
`run_memory_check.sh` 每次启动会自动执行这一步。

### 步骤 2：跑内存检查

全量测试命令（会先自动派生用例，再逐接口串行跑完所有用例）：

```bash
# 950 全量：600 条（func_v3 200 + kvcache_v3 200 + func_v4 200）
./run_memory_check.sh --device 950 --card 4

# 910 全量：1000 条（5 个接口各 200）
./run_memory_check.sh --device 910 --card 0
```

常用变体：

```bash
# 先冒烟 5 条
./run_memory_check.sh --device 950 --card 4 --interface func_v3 --limit 5

# 只跑指定多个接口（950 仅有 func_v3 / kvcache_v3 / func_v4）
./run_memory_check.sh --device 950 --card 4 --interface func_v3,kvcache_v3

# 910 上只跑 v2 的两个接口（400 条）
./run_memory_check.sh --device 910 --card 0 --interface func_v2,kvcache_v2
```

脚本做的事：刷新用例 → 逐接口 `cd` 到 `plugins/<key>/`，用
`mssanitizer --log-file <log> -- atk task ... -ms -msl <log> --single_process` 执行
→ 把本次 `atk_output` 归档到 `reports/<device>/<key>/`。

两点容易踩坑：

- **必须由外层 `mssanitizer` 包住 atk**。ATK 的 `-msl` 只是事后读日志，日志本身由
  msSanitizer 运行时写出；不包起来则日志不存在，所有用例都会以
  `No such file or directory` 失败。
- **atk 的参数前要加 `--`**。mssanitizer 会继续解析后面的选项，否则报
  `invalid option -- 'c'`。

接口是**串行**执行的：msSanitizer 日志由所有用例共享，靠偏移量切片归属，并发会互相污染。
`--max-parallel` 只设置 ATK 的 `-mt`（默认 100 并发），不影响这里的串行约束。

### 步骤 3：看结果

- 汇总报告：`reports/<device>/<key>/<run>/report/*`
- 失败明细：`reports/<device>/<key>/**/mssanitizer.log`（**只有失败用例才会留档**）
- 原始 sanitizer 日志：`reports/<device>/<key>/mss_logs/<key>.log`

判定规则：用例日志片段中出现 `ERROR` 即该用例失败；`mem_check_pass` 默认要求 100% 通过。

### 步骤 4：失败排查

```bash
# 看失败用例的报错行
grep -n "ERROR" reports/950/func_v3/**/mssanitizer.log

# 缩到最小复现
./run_memory_check.sh --device 950 --card 4 --interface func_v3 --limit 1
```

910 上若日志全空且"全部通过"，先确认算子是否真的用 `FLASH_ATTN_ENABLE_MSSANITIZER=TRUE` 编译过。

## 5. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `atk: command not found` | 未 `conda activate ascend`，或用 `--atk` / `ATK_BIN` 指定绝对路径 |
| `找不到 mssanitizer` | 未 `source .../cann/set_env.sh`，或用 `MSAN_BIN` 指定路径 |
| `mssanitizer: invalid option -- 'c'` | 手工调用时漏了 `--` 分隔符，应为 `mssanitizer <opts> -- atk task ...` |
| 所有用例都报 `mss_logs/<key>.log` 不存在 | atk 没有跑在 mssanitizer 里，日志没生成；用 `run_memory_check.sh` 或补上包装 |
| `KeyError: 'cv_fused_double_benchmark'` | 私有精度比较器未安装；`filter_cases.py` 已把该键换成内置的 `mixed_tolerance_bm_v2`，重跑生成用例即可 |
| 进程卡在 D 状态、`wchan=devdrv_sync_non_trans_msg_send` | 驱动管理通道被占满/卡死，`kill -9` 无效，等驱动恢复后再跑 |
| `ImportError: libhccl.so` | 漏了 `source .../cann/set_env.sh`，或先 activate 后 source 的顺序反了 |
| 950 上出现 page_size 报错 | 用例不按页大小裁剪，该 page_size 属覆盖范围；若确认算子不支持，需从 `cases/source/` 移除对应用例 |
| 跑得很慢 | msSanitizer 有数倍到数十倍开销，属正常；先用 `--limit` 冒烟 |

## 6. 注意事项

- 一次只在一张卡上跑，且该卡不要同时跑性能测试（sanitizer 有额外 shadow memory 开销，结果也不可信）。
- `cases/<device>/` 与 `reports/` 都是生成物，可随时删除重建。
- 本目录自包含：用例源在 `cases/source/`、执行插件在 `plugins/`，不读取任何 delivery 目录。
- 每轮开始会重置 `mss_logs/<key>.log` 与 `plugins/<key>/atk_output/mss/offset.txt`；
  两者必须成对清理，否则旧偏移会让日志切片为空而误判通过。
