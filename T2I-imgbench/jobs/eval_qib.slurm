#!/bin/bash
# ===== 使用前请按本机集群修改 =====
# 1) 下方 #SBATCH -p 后的分区名（#SBATCH 指令不支持 shell 变量，只能直接改文本）
# 2) 「按本机集群修改」变量区（CONDA_ENV / CONDA_SH 等）
#SBATCH -J ev_qib
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o outputs/logs/%x.%j.out

# ===== 按本机集群修改以下变量 =====
CONDA_ENV="qwenimagebench"         # Qwen-Image-Bench judge conda 环境
CONDA_SH="/hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh"   # conda 初始化脚本
# module load anaconda3            # 如集群需要请先取消注释
#SBATCH 的 -p 分区名请与下方用途保持一致（#SBATCH 指令不支持 shell 变量）

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$SLURM_SUBMIT_DIR"             # 在 T2I-imgbench 项目根目录下 sbatch
mkdir -p outputs/logs
set -euo pipefail

set -x
# 用法: sbatch jobs/eval_qib.slurm <run_qwenimagebench_eval.py 参数...>
# 例（released 参考图，复现榜单；--released-model 为 Benchmarks/Qwen-Image-Bench/images/ 下的目录名，config 任选其一）:
#   sbatch jobs/eval_qib.slurm --config qwen_image_2 --source released --released-model GPT-Image-1.5 --shard 0/4
# 例（本地生图判分，tag 与生图一致）:
#   sbatch jobs/eval_qib.slurm --config qwen_image_2 --source run --benchmark qwenimagebench_cn --tag 20261004-120000 --shard 0/4
# 合并判分:
#   python scripts/run_qwenimagebench_eval.py --config qwen_image_2 --source run --benchmark qwenimagebench_cn --tag 20261004-120000 --merge

python scripts/run_qwenimagebench_eval.py "$@"
set +x
echo "=== qib eval done ==="
