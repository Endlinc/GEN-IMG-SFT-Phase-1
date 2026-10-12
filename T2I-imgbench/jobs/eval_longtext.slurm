#!/bin/bash
# ===== 使用前请按本机集群修改 =====
# 1) 下方 #SBATCH -p 后的分区名（#SBATCH 指令不支持 shell 变量，只能直接改文本）
# 2) 「按本机集群修改」变量区（CONDA_ENV / CONDA_SH 等）
#SBATCH -J ev_longtext
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o outputs/logs/%x.%j.out

# ===== 按本机集群修改以下变量 =====
CONDA_ENV="longtext"               # LongText-Bench 评分 conda 环境（Qwen2.5-VL）
CONDA_SH="/hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh"   # conda 初始化脚本
# module load anaconda3            # 如集群需要请先取消注释
#SBATCH 的 -p 分区名请与下方用途保持一致（#SBATCH 指令不支持 shell 变量）

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$SLURM_SUBMIT_DIR"             # 在 T2I-imgbench 项目根目录下 sbatch
mkdir -p outputs/logs
set -euo pipefail

# 用法: sbatch jobs/eval_longtext.slurm <model yaml> [tag] [en|zh|both]
# 例: sbatch jobs/eval_longtext.slurm qwen_image_2 20261004-120000 both
CONFIG=${1:-qwen_image_2}
TAG=${2:-}
MODE=${3:-en}
if [[ -n "$TAG" ]]; then
  python scripts/run_longtext_eval.py --config "$CONFIG" --tag "$TAG" --mode "$MODE"
else
  python scripts/run_longtext_eval.py --config "$CONFIG" --mode "$MODE"
fi
echo "=== longtext eval ($MODE) done ==="
