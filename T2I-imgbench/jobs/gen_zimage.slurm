#!/bin/bash
# ===== 使用前请按本机集群修改 =====
# 1) 下方 #SBATCH -p 后的分区名（#SBATCH 指令不支持 shell 变量，只能直接改文本）
# 2) 「按本机集群修改」变量区（CONDA_ENV / CONDA_SH 等）
#SBATCH -J gen_zimage
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o outputs/logs/%x.%j.out

# ===== 按本机集群修改以下变量 =====
CONDA_ENV="zimage"                 # 本地 Z-Image 生图 conda 环境（需 24G+ 显存）
CONDA_SH="/hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh"   # conda 初始化脚本
# module load anaconda3            # 如集群需要请先取消注释
#SBATCH 的 -p 分区名请与下方用途保持一致（#SBATCH 指令不支持 shell 变量）

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$SLURM_SUBMIT_DIR"             # 在 T2I-imgbench 项目根目录下 sbatch
mkdir -p outputs/logs
set -euo pipefail

CFG=${CFG:-z_image}
TAG=${TAG:-run01}
set -x
# 用法: sbatch jobs/gen_zimage.slurm [benchmark] [额外参数...]
# 默认跑全部四个 benchmark 全量; 例: sbatch jobs/gen_zimage.slurm geneval --limit 20
if [ $# -ge 1 ]; then
  python scripts/gen_images_zimage.py --config $CFG --tag $TAG --benchmark "$1" "${@:2}"
else
  for b in geneval dpg longtext_en longtext_zh; do
    python scripts/gen_images_zimage.py --config $CFG --tag $TAG --benchmark $b
  done
fi
set +x
echo "=== gen zimage done ==="
