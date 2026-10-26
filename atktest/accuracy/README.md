# BNSD 精度测试 (atktest/accuracy)

针对 **BNSD (`(B, H, S, D)`) QKV 布局特性** 的 ATK 精度测试，支持**单标杆 (npu vs cpu)** 与
**双标杆 (npu vs cpu, gpu)** 两种跑法。用例、插件、脚本、报告全部落在本目录内，
不依赖 `flash_attention_v2_v3_atk_delivery/`、不修改开源 ATK、不修改算子源码。

- 接口范围：`flash_attn_func` (v2 / v3 / v4) + `flash_attn_with_kvcache` (v2 / v3)
- 设备范围：910 / 950；v2 系仅 910，v4 无 kvcache 入口
- GPU 标杆：Dao-AILab 官方实现（v2 用 `flash_attn`，v3/v4 用 `flash_attn_interface` / `flash_attn_3`）
- 精度标准：`cv_fused_double_benchmark`，阈值 `max_re_ratio=2.0` / `avg_re_ratio=1.2` /
  `root_mean_squared_ratio=1.2`（与
  [flash_attn_func_v3.yaml](../../flash_attention_v2_v3_atk_delivery/interfaces/flash_attn_func_v3/flash_attn_func_v3.yaml#L8-L13) 一致）；
  比较器由私有包 **`atk_private_precision`** 提供，无需自带实现，前置安装见 §4

## 1. 目录结构

```
atktest/accuracy/
├── atk_accuracy_matrix.yaml            # key -> 模块 / api_type / 设备 / 用例源 / 插件目录
├── nodes_accuracy_single_bm.yaml       # 单标杆: npu + cpu
├── nodes_accuracy_double_bm.yaml       # 双标杆: npu + cpu + gpu(远端，host 待替换)
├── cases/
│   ├── gen/<key>.yaml                  # 生成配置（含精度标准与 shape 网格）
│   ├── gen/generate_<key>.py           # BNSD 交叉约束脚本
│   ├── filter_cases.py                 # 按设备派生 cases/<device>/<key>.json
│   ├── source/<key>.json               # 生成后入库的自包含用例
│   └── <device>/<key>.json             # 可直接执行的用例（可重建）
├── plugins/<key>/
│   ├── execute_<...>.py                # cpu/gpu/npu 三分支适配器
│   └── benchmark_<...>.py              # CPU golden（纯 torch）
│                                       # 精度比较器 cv_fused_double_benchmark 来自
│                                       # 私有包 atk_private_precision，不在本目录
├── gen_cases.sh                        # 重建 cases/source
├── run_accuracy.sh                     # 统一运行入口
└── reports/<device>/<key>/<mode>/      # 归档的 atk_output
```

`key` ∈ `func_v2` / `func_v3` / `func_v4` / `kvcache_v2` / `kvcache_v3`。

## 2. BNSD 是怎么被测出来的

对外接口签名仍是 BSND `(B, S, H, D)`，BNSD 只在 **execute 插件的 npu 分支**注入：

```python
def _bnsd_view(tensor):
    return tensor.transpose(1, 2).contiguous().transpose(1, 2)
```

即把 BSND 张量先重排为**连续的 BNSD 存储**，再以形状仍为 `(B, S, H, D)` 的转置视图喂入算子，
让算子内部的 stride 判据命中 BNSD 分支。cpu / gpu 分支保持原始 BSND，数值等价，
所以比对验证的是「BNSD 路径返回了与 BSND 语义一致的 `(B, S, H, D)` 结果」。

命中 BNSD 判据的前提（生成器已强制，见 [generate_func_v2.py](cases/gen/generate_func_v2.py)）：

- `H ≥ 2` **且** `H_kv ≥ 2`（`H_kv == 1` 时转置视图仍被判连续）
- `S ≥ 2`（q 与 kv 的序列长度都要）
- 非 paged（kvcache 的 `page_table` / `block_table` 置空）
- 无 append-KV（kvcache 的 `k` / `v` / `qv` 置空）
- kvcache 容量 `capacity ≥ seqlen_q`

npu 分支还带一行防御性断言 `assert not view.is_contiguous()`：若某条用例因 shape 退化
没走到 BNSD 路径，会立刻暴露而不是「静默通过」。

## 3. 单标杆 vs 双标杆

两者用**同一份用例、同一个比较器**，差别只有节点表：

| 模式 | 节点表 | atk 参数 | 比对次数 |
|---|---|---|---|
| `single` | `nodes_accuracy_single_bm.yaml` (npu + cpu) | `--bm_device cpu --single_process` | 1（npu vs cpu） |
| `double` | `nodes_accuracy_double_bm.yaml` (npu + cpu + gpu) | `--bm_device cpu` | 2（npu vs cpu，npu vs gpu） |

依据：ATK 的 `accuracy_calc` 会遍历**所有非主节点**逐一比对，节点表里有几个标杆就比几次。
远端 GPU 节点必须多进程模式，因此 `double` 不能加 `--single_process`。

> **运行双标杆前必须先把 [nodes_accuracy_double_bm.yaml](nodes_accuracy_double_bm.yaml) 里的
> `host: 192.168.0.100` 换成真实 GPU 主机 IP**，并保证该主机的 `atk` 服务可达、装了对应版本的
> Dao-AILab flash-attention。GPU 与 NPU 结果本来就可能不一致（不同 kernel / 不同累加顺序），
> 双标杆的价值是把两边差异都量化出来。

## 4. 快速开始

前置：`cv_fused_double_benchmark` 比较器由私有包提供，**必须先装**，否则 post_process 阶段
会 `KeyError: 'cv_fused_double_benchmark'`。

```bash
conda activate ascend
source ~/miniconda3/Ascend/cann/set_env.sh

# 0) 安装私有精度包（一次性）
pip install /path/to/atk_private_precision-0.1.0-cp39-abi3-linux_aarch64.whl
python -c "from atk.tasks.post_process import ACCURACY_REGISTRY as R; print('cv_fused_double_benchmark' in R)"  # True

cd atktest/accuracy

# 1) 重建用例（可选，cases/source 已入库）
./gen_cases.sh --dry-run          # 先看每个接口生成多少条
./gen_cases.sh                    # 落盘 cases/source/<key>.json

# 2) 冒烟：单标杆，只跑 3 条
./run_accuracy.sh --device 910 --interface func_v3 --mode single --limit 3

# 3) 全量单标杆（910 上 5 个接口）
./run_accuracy.sh --device 910 --mode single

# 4) 双标杆（先改好 GPU host）
./run_accuracy.sh --device 910 --interface func_v3 --mode double --limit 3

# 5) 后续 950（v2 自动跳过）
./run_accuracy.sh --device 950 --mode single
```

指定卡号：`--card 4`；指定 python 环境里的 atk：`--atk /path/to/atk`。

## 5. 精度比较器口径

比较器 `cv_fused_double_benchmark` 由私有包 `atk_private_precision`（0.1.0）提供，
类为 `atk_private_precision.post_process.cv_fused_double_benchmark_compare.CvFusedDoubleBenchmarkAccuracyCompare`，
算法为编译后的二进制扩展（`abi3.so`，Cython 编译），**源码不可得**。本目录**不再自带实现**。

ATK 侧的加载链路（[private_loader.py](file:///home/w00811046/miniconda3/envs/ascend/lib/python3.10/site-packages/atk/tasks/post_process/private_loader.py)）：
`import atk.tasks.post_process` 时 import 私有包并调用其 `register_accuracy_plugins(registry, api_version=1)`，
把名字注册进 `ACCURACY_REGISTRY`（provider 标记为 `private`）。私有包同时注册了
`cv_fused_double_benchmark` 与 `cv_fused_double_benchmark_ct` 两个名字。

由报告可见该标准实际输出的 5 个指标（`精度详情` 里的 `new_benchmark_indicate`）：

| 字段 | 含义 |
|---|---|
| `Actual_最大相对误差` | 最大相对误差 |
| `Actual_平均相对误差` | 平均相对误差 |
| `Actual_均方根误差` | RMSE |
| `Actual_误差均衡性` | 误差均衡性（对应符号 `_stat_eb_info`） |
| `Actual_小值域数错数量` | **小值域**错误数量（对应 `partition_value_range` / `get_result_small_domain`）|

即该标准是**值域分段 + 小值域专用路径**的口径，不是简单的"全局 max/avg/rms 分别除以
dtype 基础容差"。yaml 里的三个 ratio（`max_re_ratio` / `avg_re_ratio` /
`root_mean_squared_ratio`）是这套分段判据的阈值倍数，由私有实现内部解释。

> 本目录早期曾自带一份 `compare_cv_fused_double_benchmark.py` 作为替代实现（口径为
> `max_re` 按整张量量程归一）。装上私有包后该文件已删除，避免与私有实现抢同名注册
> （`atk task -p .` 注册在私有包之后，会**覆盖**掉私有实现）。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `KeyError: 'cv_fused_double_benchmark'` | 私有包没装：`pip install atk_private_precision-*.whl`，用 §4 的注册检查确认 |
| `ct-tool not installed` | `cv_fused_double_benchmark_ct` 才能用的可选工具缺失，跑 `cv_fused_double_benchmark` 时只是提示，可忽略 |
| 想知道实际用的是私有实现还是自带插件 | `python -c "from atk.tasks.post_process import AccuracyFactory as A; print(A.get_accuracy('cv_fused_double_benchmark').__module__)"`，应是 `atk_private_precision.*` |
| `atk: command not found` | 先 `conda activate ascend`，或用 `--atk` / `ATK_BIN` 指定 |
| `ImportError: libhccl.so` 之类 | 先 `activate` 再 `source set_env.sh`，顺序不能反 |
| 生成用例报 `not enough values to unpack` | 漏了 `-en 0`（边界用例不含完整 shape） |
| 双标杆卡住 | GPU 主机不可达：核对 `nodes_accuracy_double_bm.yaml` 的 host/port，或退回 `--mode single` |
| npu 分支断言失败（BNSD 未命中） | 该条用例 shape 退化（H/H_kv/S < 2）；检查对应生成器里 `HEADS` / `MIN_SEQLEN` 约束 |

## 7. 报告

`reports/<device>/<接口>/<mode>/<run>/` 下：

- `report/*.xlsx` — ATK 精度报告（通过率与逐用例误差）
- `log/atk.log` — 运行日志，含比较器 `error_info`（`max_re=... (..x <= 2.0)` 等）
- `input/`、`output/` — 保存的输入与 npu 输出张量（`--save_data`）
