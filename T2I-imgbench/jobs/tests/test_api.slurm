#!/bin/bash
# ===== 使用前请按本机集群修改 =====
# 1) 下方 #SBATCH -p 后的分区名（#SBATCH 指令不支持 shell 变量，只能直接改文本）
# 2) 「按本机集群修改」变量区（各 conda 环境名 / CONDA_SH）
#SBATCH -J test_api
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o outputs/logs/%x.%j.out

# 全链路测试：API 生图 + 六个 benchmark 评分（geneval / dpg / longtext_en / longtext_zh / qwenimagebench_cn / qwenimagebench_en，每 benchmark 30 条 prompt）。
# 用法: sbatch jobs/tests/test_api.slurm
#       CFG=gpt_image_2 sbatch jobs/tests/test_api.slurm   # 换 API 生图模型（环境变量覆盖）
#       TAG=run02 sbatch jobs/tests/test_api.slurm         # 换输出 tag（缺省 test）

# ===== 按本机集群修改以下变量 =====
GEN_ENV="geneval"                  # 生图/编排 + GenEval 评分 conda 环境
LONGTEXT_ENV="longtext"            # LongText-Bench 评分 conda 环境
DPG_ENV="dpgbench"                 # DPG-Bench 评分 conda 环境
QIB_ENV="qwenimagebench"           # Qwen-Image-Bench judge 评分 conda 环境
CONDA_SH="/hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh"   # conda 初始化脚本
# module load anaconda3            # 如集群需要请先取消注释

source "$CONDA_SH"
cd "$SLURM_SUBMIT_DIR"             # 在 T2I-imgbench 项目根目录下 sbatch
mkdir -p outputs/logs
set -euo pipefail

CFG=${CFG:-qwen_image_2}
TAG=${TAG:-test}
LIM=30
BENCHES="geneval dpg longtext_en longtext_zh qwenimagebench_cn qwenimagebench_en"

# 1) API 生图（qib 官方口径每 prompt 1 张，由 base.yaml 注册表自动决定）
conda activate "$GEN_ENV"
set -x
for b in $BENCHES; do
  python scripts/gen_images.py --config $CFG --benchmark $b --tag $TAG --limit $LIM
done
set +x

# 2) 评分（各评分脚本自动调用 prepare_eval_inputs 整理评测输入）
set -x
conda activate "$GEN_ENV"
python scripts/run_geneval_eval.py  --config $CFG --tag $TAG
conda activate "$LONGTEXT_ENV"
python scripts/run_longtext_eval.py --config $CFG --tag $TAG --mode both
conda activate "$DPG_ENV"
python scripts/run_dpg_eval.py      --config $CFG --tag $TAG
conda activate "$QIB_ENV"
for lang in cn en; do
  python scripts/run_qwenimagebench_eval.py --config $CFG \
      --source run --benchmark qwenimagebench_$lang --tag $TAG --shard 0/1
  python scripts/run_qwenimagebench_eval.py --config $CFG \
      --source run --benchmark qwenimagebench_$lang --tag $TAG --merge
done
set +x

# 3) 失败汇总 + 产物清点
conda activate "$GEN_ENV"
for b in $BENCHES; do
  python scripts/summarize_failures.py --config $CFG --benchmark $b --tag $TAG || true
done
for b in $BENCHES; do
  d="outputs/$CFG/$b/$TAG/images"
  s=$(find "outputs/$CFG/$b/$TAG" -maxdepth 2 \( -name '*summary*.json' -o -name '*_bench_scores.json' \) 2>/dev/null | tr '\n' ' ')
  echo "$b: $(ls "$d" 2>/dev/null | wc -l) prompts | scores: $s"
done
echo "=== test_api done ==="
