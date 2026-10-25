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

因此 910 跑 5 个接口，950 跑 3 个。

## 2. 目录内容

```
atktest/
├── atk_memory_check_matrix.yaml   # 接口 → 用例源 / plugin 目录映射
├── nodes_memory_check.yaml        # 单 npu 节点，task: [memory_check]
├── cases/
│   ├── filter_cases.py            # 按设备裁剪用例
│   └── <device>/<key>.json        # 生成物（可随时重建）
├── run_memory_check.sh            # 统一运行入口
└── reports/<device>/<key>/        # 归档：atk_output 报告 + mss_logs
```

用例与 adapter **不复制**，直接引用既有两个交付目录：

- `flash_attention_v2_v3_atk_delivery/`（v2/v3 的 func 与 kvcache，各 200 条用例）
- `flash_attention_bnsd_atk_delivery/`（v4 目前只有 BNSD 版 func 用例与 adapter）

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

### 步骤 1：生成用例

```bash
cd flash-attention-npu/atktest
python3 cases/filter_cases.py --device 950 --dry-run    # 先看会剔除多少
python3 cases/filter_cases.py --device 950              # 写入 cases/950/
```

950 会剔除两类用例：分页 KV cache 中 `page_size` 不是 128/256/512/1024 的，以及 `softcap > 0` 的。
910 原样透传。

### 步骤 2：跑内存检查

```bash
# 950（3 个接口）：先冒烟 5 条
./run_memory_check.sh --device 950 --card 4 --interface func_v3 --max-task 5

# 950 全量
./run_memory_check.sh --device 950 --card 4

# 910（5 个接口）
./run_memory_check.sh --device 910 --card 0

# 只跑指定多个接口
./run_memory_check.sh --device 910 --interface func_v2,kvcache_v3
```

脚本做的事：刷新用例 → 逐接口 `cd` 到对应 adapter 目录执行
`atk task --task memory_check -ms -msl <log> --single_process` → 把本次 `atk_output` 归档到 `reports/<device>/<key>/`。

接口是**串行**执行的：msSanitizer 日志由所有用例共享，靠偏移量切片归属，并发会互相污染。

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
./run_memory_check.sh --device 950 --card 4 --interface func_v3 --max-task 1
```

910 上若日志全空且"全部通过"，先确认算子是否真的用 `FLASH_ATTN_ENABLE_MSSANITIZER=TRUE` 编译过。

## 5. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `atk: command not found` | 未 `conda activate ascend`，或用 `--atk` / `ATK_BIN` 指定绝对路径 |
| 进程卡在 D 状态、`wchan=devdrv_sync_non_trans_msg_send` | 驱动管理通道被占满/卡死，`kill -9` 无效，等驱动恢复后再跑 |
| `ImportError: libhccl.so` | 漏了 `source .../cann/set_env.sh`，或先 activate 后 source 的顺序反了 |
| 950 上出现 page_size 报错 | 用例未经过滤，先跑 `filter_cases.py --device 950` 重建 |
| 跑得很慢 | msSanitizer 有数倍到数十倍开销，属正常；先用 `--max-task` 冒烟 |

## 6. 注意事项

- 一次只在一张卡上跑，且该卡不要同时跑性能测试（sanitizer 有额外 shadow memory 开销，结果也不可信）。
- `cases/<device>/` 与 `reports/` 都是生成物，可随时删除重建。
- 本目录不改动既有两个 delivery 目录，只读取其中的用例与 adapter。
