# T2I-imgbench

T2I（文生图）模型评测编排工具：用生图模型（API 或本地）在 **GenEval / DPG-Bench / LongText-Bench / Qwen-Image-Bench** 四个 benchmark 上生图，再调用各 benchmark 官方脚本评分。支持任意中转站平台（OpenAI 兼容，如 yunwu）/ 千问（DashScope）API 生图与本地 Z-Image 生图。

## 目录结构

```
T2I-imgbench/
├── README.md                # 本文档（唯一文档）
├── configs/
│   ├── base.yaml            # 公共配置：paths + judges + benchmarks 注册表
│   └── <model>.yaml         # 模型配置：model_name + api:/model: + generation: 采样参数
├── Benchmarks/              # 各 benchmark 官方 repo / 数据（见「数据准备」）
├── Models/                  # judge 模型与本地生图权重（见「模型准备」）
├── scripts/                 # 编排脚本（生图 / 整理评测输入 / 评分 / 失败汇总）
├── jobs/                    # slurm 作业脚本（含 tests/ 全链路测试：API / 本地生图 + 评分）
│   └── tests/
└── outputs/                 # 生图与评分输出（gitignore）：logs/、cache/ 在其中
```

所有路径统一由 `configs/base.yaml` 以**相对项目根**的方式声明，`scripts/common.py` 在运行时解析为绝对路径；需要迁移机器时改 `base.yaml` 即可。如需把整个项目放在任意位置，也可设置环境变量 `T2I_IMGBENCH_ROOT` 覆盖项目根。

## 快速开始

```bash
bash setup.sh   # 一键完成数据/模型/环境准备并提交 smoke 测试；分阶段幂等可重跑，选项见脚本头部注释
```

## 运行流程

1. **配置 API key 或者配置本地生图模型**：复制 `.env.example` 为 `.env`，填入 `QIANWEN_API_KEY` / `OPENAI_COMPAT_API_KEY`。
2. **生图**
  - API，以 qwen_image_2 为例：
    ```bash
     python scripts/gen_images.py --config qwen_image_2 --benchmark geneval --tag run01
     # --benchmark: geneval | dpg | longtext_en | longtext_zh | qwenimagebench_cn | qwenimagebench_en
     # --tag 缺省为时间戳 YYYYMMDD-HHMMSS；--limit N / --prompt-ids a,b,c 用于 smoke
    ```
  - 本地 Z-Image 生图（GPU 24G+）：
    ```bash
     python scripts/gen_images_zimage.py --config z_image --benchmark geneval --tag run01
    ```
3. **单 model 单 benchmark 评分示例**
  ```bash
   # geneval, dpg, longtext
   python scripts/run_geneval_eval.py    --config qwen_image_2 --tag run01
   python scripts/run_dpg_eval.py        --config qwen_image_2 --tag run01
   python scripts/run_longtext_eval.py   --config qwen_image_2 --tag run01 --mode both
   # Qwen-Image-Bench（本地生图判分，官方口径每 prompt 1 张，分 0/1/2/3 四片 shard 同时运行）：
   python scripts/run_qwenimagebench_eval.py --config qwen_image_2 \
       --source run --benchmark qwenimagebench_cn --tag run01 --shard 0/4
   python scripts/run_qwenimagebench_eval.py --config qwen_image_2 \
       --source run --benchmark qwenimagebench_cn --tag run01 --merge
  ```
4. **outputs 目录结构**（`outputs/<model>/<benchmark>/<tag>/`）：
  ```
   outputs/qwen_image_2/geneval/run01/
   ├── images/<prompt_id>/          # seed{k}_raw.png / seed{k}_processed.png / seed{k}_meta.json
   ├── failures.jsonl               # 生图失败样本（口径 A：单列失败率）
   ├── _eval/<benchmark>/           # 官方评测输入 + 逐图完整评分（results.jsonl / eval_results/ 等）
   └── scores/                      # 汇总 JSON（<model>_<benchmark>_summary.json）
  ```

## jobs/ 用法（slurm）

在 **T2I-imgbench 项目根目录**下 `sbatch`（脚本内 `cd "$SLURM_SUBMIT_DIR"`）；日志写入 `outputs/logs/%x.%j.out`。
每个脚本头部的「使用前请按本机集群修改」注释标出了两处必改项：

- `#SBATCH -p YOUR_PARTITION` 的分区名
- 「按本机集群修改」变量区（`CONDA_ENV` / `CONDA_SH` 等）；
- (按需) `module load anaconda3` 行取消注释。

每个 slurm 脚本内注释都写有具体用法与示例命令，提交前参考对应脚本头部注释即可。
