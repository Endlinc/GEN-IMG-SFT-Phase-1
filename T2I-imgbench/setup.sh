#!/usr/bin/env bash
# T2I-imgbench 一键配置脚本
# 按顺序执行 5 个阶段：repos -> data -> models -> envs -> smoke
#   repos   克隆 4 个 benchmark 官方 repo 到 Benchmarks/
#   data    下载 Qwen-Image-Bench HF 数据集 + 生成 prompts_manifest.json 校验清单
#   models  下载评价（judge）模型与可选 Z-Image 本地生图权重到 Models/
#   envs    创建 6 个 conda 环境并安装依赖（已存在的环境跳过，除非 --force）
#   smoke   校验依赖后 sbatch 提交两条全链路测试（jobs/tests/test_api.slurm 需 API key；test_local.slurm 需 Z-Image 权重）
#
# 用法：
#   bash setup.sh                  # 全流程
#   bash setup.sh --only repos,data    # 只跑指定阶段（逗号分隔）
#   bash setup.sh --skip models,envs   # 跳过指定阶段
#   bash setup.sh --force          # 已存在的 conda 环境也强制重装依赖
#   SKIP_ZIMAGE=1 bash setup.sh    # 跳过 Z-Image 权重下载（约 20.5G，仅本地生图需要）
#   HF_ENDPOINT=https://hf-mirror.com bash setup.sh   # 无法直连 HuggingFace 时用镜像
set -euo pipefail
cd "$(dirname "$0")"

ONLY=""
SKIP=""
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="${2:?--only 需要阶段列表，如 --only repos,data}"; shift 2;;
    --skip) SKIP="${2:?--skip 需要阶段列表，如 --skip models,envs}"; shift 2;;
    --force) FORCE=1; shift;;
    *) echo "未知参数: $1（支持 --only/--skip/--force）"; exit 1;;
  esac
done

log()  { echo; echo "==> [$(date +%H:%M:%S)] $*"; }
warn() { echo "WARNING: $*"; }
stage_enabled() {
  local s=$1
  [ -n "$ONLY" ] && [[ ",$ONLY," != *",$s,"* ]] && return 1
  [ -n "$SKIP" ] && [[ ",$SKIP," == *",$s,"* ]] && return 1
  return 0
}

# ---------------------------------------------------------------- 前置检查
log "阶段 0/5：前置检查（git / conda / python）"
for tool in git python3; do
  command -v "$tool" >/dev/null || { echo "缺少 $tool，请先安装"; exit 1; }
done
CONDA_SH=${CONDA_SH:-/hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh}
[ -f "$CONDA_SH" ] || { echo "找不到 conda 初始化脚本: $CONDA_SH（可用 CONDA_SH=... 指定）"; exit 1; }
# 下载工具（huggingface-cli / modelscope / pyyaml）装到 user site，避免依赖 conda 环境
if ! python3 -c "import huggingface_hub, modelscope, yaml" 2>/dev/null; then
  log "安装下载工具到 user site（huggingface_hub / modelscope / pyyaml）"
  python3 -m pip install --user -q huggingface_hub modelscope pyyaml
fi

# ---------------------------------------------------------------- repos
if stage_enabled repos; then
  log "阶段 1/5：克隆 benchmark 官方 repo -> Benchmarks/"
  mkdir -p Benchmarks/DPG-Bench Benchmarks/LongText-Bench
  clone_repo() {  # $1=url $2=dest
    if [ -d "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
      echo "  已存在，跳过: $2"
    else
      git clone --depth 1 "$1" "$2"
    fi
  }
  clone_repo https://github.com/djghosh13/geneval.git                 Benchmarks/geneval
  clone_repo https://github.com/TencentQQGYLab/ELLA.git               Benchmarks/DPG-Bench/ELLA
  clone_repo https://github.com/X-Omni-Team/X-Omni.git                Benchmarks/LongText-Bench/X-Omni
  clone_repo https://github.com/QwenLM/Qwen-Image-Bench.git           Benchmarks/Qwen-Image-Bench-gh
else
  log "阶段 1/5：repos（跳过）"
fi

# ---------------------------------------------------------------- data
if stage_enabled data; then
  log "阶段 2/5：下载 Qwen-Image-Bench HF 数据集（prompts + 官方参考图）"
  huggingface-cli download Qwen/Qwen-Image-Bench --repo-type dataset \
      --local-dir Benchmarks/Qwen-Image-Bench
  log "生成 prompts 校验清单 Benchmarks/prompts_manifest.json"
  python3 scripts/make_prompts_manifest.py
else
  log "阶段 2/5：data（跳过）"
fi

# ---------------------------------------------------------------- models
if stage_enabled models; then
  log "阶段 3/5：下载 judge 模型与生图权重 -> Models/"
  mkdir -p Models/geneval Models/openclip

  if [ ! -f Models/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth ]; then
    log "GenEval 检测器 Mask2Former（OpenMMLab 官方权重）"
    wget -c -O Models/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
      https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224732-8b0fa8de.pth
  else
    echo "  已存在，跳过: Models/geneval/mask2former...pth"
  fi

  if [ ! -f Models/openclip/open_clip_pytorch_model.bin ]; then
    log "GenEval CLIP 权重（laion2B bigG）"
    huggingface-cli download laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
        open_clip_pytorch_model.bin --local-dir Models/openclip
  fi

  if [ ! -f Models/Qwen2.5-VL-7B-Instruct/config.json ]; then
    log "LongText-Bench OCR judge（Qwen2.5-VL-7B-Instruct）"
    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Models/Qwen2.5-VL-7B-Instruct
  fi

  if [ ! -f Models/mplug/config.yaml ]; then
    log "DPG-Bench VQA judge（mPLUG，经 modelscope）"
    modelscope download --model damo/mplug_visual-question-answering_coco_large_en \
        --local_dir Models/mplug
  fi

  if [ ! -f Models/Q-Judger/config.json ]; then
    log "Qwen-Image-Bench judge（Q-Judger）"
    huggingface-cli download Qwen/Qwen-Image-Bench --local-dir Models/Q-Judger
  fi

  if [ "${SKIP_ZIMAGE:-0}" = "1" ]; then
    echo "  SKIP_ZIMAGE=1，跳过 Z-Image 代码与权重"
  else
    if [ ! -d Models/Z-Image/src ]; then
      log "Z-Image 代码仓库"
      git clone --depth 1 https://github.com/Tongyi-MAI/Z-Image.git Models/Z-Image
    fi
    if [ ! -f Models/Z-Image/ckpts/Z-Image/config.json ]; then
      log "Z-Image 底座权重（bf16 约 20.5G，可用 SKIP_ZIMAGE=1 跳过）"
      huggingface-cli download Tongyi-MAI/Z-Image --local-dir Models/Z-Image/ckpts/Z-Image
    fi
  fi
else
  log "阶段 3/5：models（跳过）"
fi

# ---------------------------------------------------------------- envs
if stage_enabled envs; then
  log "阶段 4/5：创建 conda 环境并安装依赖"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }
  # 返回 0 表示需要装依赖：环境不存在则创建；已存在时仅 --force 才重装
  ensure_env() {
    if env_exists "$1"; then
      if [ "$FORCE" = 1 ]; then echo "  --force：重装依赖 $1"; return 0; fi
      echo "  环境已存在，跳过（--force 可重装）: $1"; return 1
    fi
    conda create -n "$1" python=3.10 -y
  }
  in_env() { local env=$1; shift; conda run -n "$env" "$@"; }

  # 编排层
  if ensure_env imgbench; then
    in_env imgbench pip install -q requests pyyaml python-dotenv pillow tqdm pandas numpy
  fi

  # GenEval 评分（torch2.1.2 生态；本集群 mmdet 也可用预编译 wheel，见 README）
  if ensure_env geneval; then
    in_env geneval pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    in_env geneval pip install -q mmdet==3.3.0 mmcv==2.1.0 numpy==1.26.4
    in_env geneval pip install -q open-clip-torch clip-benchmark pytorch-lightning einops
  fi

  # LongText-Bench OCR
  if ensure_env longtext; then
    in_env longtext pip install -q torch==2.6 torchvision --index-url https://download.pytorch.org/whl/cu121
    in_env longtext pip install -q "transformers>=4.51.3" qwen-vl-utils accelerate sentencepiece "protobuf==3.20.3"
  fi

  # DPG-Bench（mPLUG via modelscope；fairseq-fixed 编译需 setuptools<81）
  if ensure_env dpgbench; then
    in_env dpgbench pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    in_env dpgbench pip install -q modelscope "transformers==4.46.3" opencv-python timm addict simplejson sortedcontainers datasets "ftfy>=6.0.3" rapidfuzz "rouge_score<=0.0.4"
    in_env dpgbench pip install -q "setuptools<81"
    in_env dpgbench pip install -q fairseq-fixed torchaudio librosa unicodedata2
  fi

  # Qwen-Image-Bench judge
  if ensure_env qwenimagebench; then
    in_env qwenimagebench pip install -q -r Benchmarks/Qwen-Image-Bench-gh/requirements.txt
  fi

  # 本地生图 Z-Image
  if [ "${SKIP_ZIMAGE:-0}" != "1" ] && [ -d Models/Z-Image ]; then
    if ensure_env zimage; then
      in_env zimage pip install -q -e Models/Z-Image
    fi
  fi
  echo "conda 环境就绪: $(conda env list | awk 'NR>2{print $1}' | tr '\n' ' ')"
else
  log "阶段 4/5：envs（跳过）"
fi

# ---------------------------------------------------------------- smoke
if stage_enabled smoke; then
  log "阶段 5/5：全链路 smoke 测试（30 条 prompt/benchmark，sbatch 提交）"
  if [ ! -f .env ]; then
    cp .env.example .env
    warn "已从 .env.example 生成 .env，请填入 QIANWEN_API_KEY 或 OPENAI_COMPAT_API_KEY 后重跑 smoke"
  fi
  if ! command -v sbatch >/dev/null; then
    warn "当前机器没有 sbatch，跳过提交（到集群登录节点执行 sbatch jobs/tests/test_api.slurm / test_local.slurm）"
  else
    # API 生图链路（需要 API key）
    if grep -qsE '^(QIANWEN_API_KEY|OPENAI_COMPAT_API_KEY)=.+' .env; then
      sbatch jobs/tests/test_api.slurm
      echo "已提交 test_api，跟踪日志: tail -f outputs/logs/test_api.*.out"
    else
      warn ".env 中没有可用的 API key，跳过 test_api（填好 key 后: sbatch jobs/tests/test_api.slurm）"
    fi
    # 本地生图链路（需要 Z-Image 权重，无需 API key）
    if [ "${SKIP_ZIMAGE:-0}" != "1" ] && [ -f Models/Z-Image/ckpts/Z-Image/config.json ]; then
      sbatch jobs/tests/test_local.slurm
      echo "已提交 test_local，跟踪日志: tail -f outputs/logs/test_local.*.out"
    else
      echo "跳过 test_local（无 Z-Image 权重；下载后执行: sbatch jobs/tests/test_local.slurm）"
    fi
  fi
else
  log "阶段 5/5：smoke（跳过）"
fi

log "setup.sh 完成"
