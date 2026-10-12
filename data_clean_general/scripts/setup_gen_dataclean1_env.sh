#!/usr/bin/env bash
set -euo pipefail

# Override via environment variables, e.g.
#   CONDA_SH=$HOME/anaconda3/etc/profile.d/conda.sh bash scripts/setup_gen_dataclean1_env.sh
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
ENV="${CONDA_ENV:-gen-dataclean1}"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  conda create -n "$ENV" python=3.10 pip -y \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    --override-channels
fi

conda activate "$ENV"
PY="$(command -v python)"

"$PY" -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url "$TORCH_INDEX"
"$PY" -m pip install pillow imagehash langdetect pyyaml numpy transformers pyarrow \
  -i "$PIP_MIRROR" --trusted-host pypi.tuna.tsinghua.edu.cn
"$PY" -c "import torch, torchvision, transformers, imagehash, pyarrow; print('ok', torch.__version__, torch.cuda.is_available())"
