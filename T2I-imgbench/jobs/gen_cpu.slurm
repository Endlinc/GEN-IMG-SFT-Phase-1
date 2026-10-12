#!/bin/bash
# ===== 使用前请按本机集群修改 =====
# 1) 下方 #SBATCH -p 后的分区名（#SBATCH 指令不支持 shell 变量，只能直接改文本）
# 2) 「按本机集群修改」变量区（CONDA_ENV / CONDA_SH 等）
#SBATCH -J t2i_gen
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o outputs/logs/%x.%j.out

# ===== 按本机集群修改以下变量 =====
CONDA_ENV="geneval"                # 生图/编排 conda 环境
CONDA_SH="/hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh"   # conda 初始化脚本
# module load anaconda3            # 如集群需要请先取消注释
#SBATCH 的 -p 分区名请与下方用途保持一致（#SBATCH 指令不支持 shell 变量）

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$SLURM_SUBMIT_DIR"             # 在 T2I-imgbench 项目根目录下 sbatch
mkdir -p outputs/logs

set -x

# 用法（在脚本下方注释修改要跑的命令后 sbatch）：
# python scripts/gen_images.py --config qwen_image_2 --benchmark geneval --tag run01
# python scripts/gen_images.py --config qwen_image_2 --benchmark longtext_en --tag run01
# python scripts/gen_images.py --config qwen_image_2 --benchmark longtext_zh --tag run01
python scripts/gen_images.py --config qwen_image_2 --benchmark qwenimagebench_cn --tag run01

set +x
echo "=== gen done ==="
MODEL=qwen_image_2
BENCH=qwenimagebench_cn
TAG=run01
d="outputs/$MODEL/$BENCH/$TAG/images"
echo "$BENCH: $(ls "$d" 2>/dev/null | wc -l) prompts, $(find "$d" -name '*_processed.png' 2>/dev/null | wc -l) processed pngs"
