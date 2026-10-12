# 通用图文数据清洗工具（data_clean_general）

本工具用于文生图（T2I）训练数据的**通用质量清洗**：对「图片 + 文本」数据集逐条运行一组可配置的过滤器，剔除损坏、低分辨率、重复、非目标语言、NSFW、带水印、模糊或压缩严重的样本，并把保留 / 剔除结果连同原始文本一起导出，便于人工抽检。

目前支持以下 6 个数据集：

| 数据集 | `--format` | 配置文件 |
|---|---|---|
| Alchemist | `alchemist` | `config/config_alchemist.yaml` |
| Conceptual Captions | `conceptual-captions` | `config/config_conceptual_captions.yaml` |
| DenseFusion-4V-100K | `densefusion-4v-100k` | `config/config_densefusion_4v_100k.yaml` |
| DiffusionDB-2M-Random-50K | `diffusiondb_2m_random_50k` | `config/config_diffusiondb_2m_random_50k.yaml` |
| Fine-T2I | `fine-t2i` | `config/config_fine_t2i.yaml`（curated）/ `config/config_fine_t2i_synthetic.yaml`（synthetic 子集） |
| WikiArt | `wikiart` | `config/config_wikiart.yaml` |

主要特点：

- **可配置**：用 YAML 文件选择启用哪些过滤器、各自阈值；每个数据集一份配置。
- **流式读取**：tar、zip、parquet 等格式**无需先解压**即可逐条读取。
- **断点续跑**：全量任务支持中断后从断点继续；分片型数据集可按分片分批处理。
- **结果可追溯**：报告 JSON 记录每张被剔除图片的原因和具体数值；导出时保留数据集原始的 caption / 元数据。
- **抽检工具**：附带一个本地网页看图器，用于浏览保留 / 剔除结果并做人工复核。

### 使用前需要自行准备 / 修改的内容

| 内容 | 怎么做 | 见 |
|---|---|---|
| Python 环境 | 安装 PyTorch 及依赖 | [第 2 节](#2-环境安装) |
| 两个模型（NSFW、水印检测） | 从 HuggingFace 下载，并把 `config/*.yaml` 中的占位路径 `/path/to/models/...` 改成你的模型目录 | [第 3 节](#3-模型准备) |
| 数据集 | 从 HuggingFace 下载；Conceptual Captions、Alchemist 需按 URL 下载图片，DiffusionDB 需先导出图片 | [第 4 节](#4-数据集准备) |
| 命令中的路径 | 文档中的 `/path/to/...` 均为占位符，运行时替换为你自己的数据集目录和输出目录 | [第 5 节](#5-快速开始) |

---

## 目录

1. [目录结构](#1-目录结构)
2. [环境安装](#2-环境安装)
3. [模型准备](#3-模型准备)
4. [数据集准备](#4-数据集准备)
5. [快速开始](#5-快速开始)
6. [命令行参数](#6-命令行参数)
7. [过滤器说明](#7-过滤器说明)
8. [配置文件](#8-配置文件)
9. [输出结果](#9-输出结果)
10. [断点续跑与分批处理](#10-断点续跑与分批处理)
11. [性能说明](#11-性能说明)
12. [辅助工具](#12-辅助工具)
13. [扩展新的数据集格式](#13-扩展新的数据集格式)
14. [常见问题](#14-常见问题)

---

## 1. 目录结构

```
data_clean_general/
├── README.md                      # 本说明文件
├── run_filters.py                 # 命令行入口
├── config/                        # 各数据集的过滤器配置
│   ├── config_alchemist.yaml
│   ├── config_conceptual_captions.yaml
│   ├── config_densefusion_4v_100k.yaml
│   ├── config_diffusiondb_2m_random_50k.yaml
│   ├── config_fine_t2i.yaml
│   ├── config_fine_t2i_synthetic.yaml
│   └── config_wikiart.yaml
├── src/
│   ├── cli.py                     # 命令行参数解析与运行调度
│   ├── core.py                    # Sample / Filter / Pipeline 等核心数据结构
│   ├── filters.py                 # 全部过滤器实现
│   ├── loaders.py                 # 各数据集格式的读取器
│   ├── checkpoint.py              # 断点续跑、流水线并行、增量导出
│   ├── export.py                  # 报告与保留/剔除图片导出
│   └── config.py                  # 由配置构建过滤器
├── scripts/
│   ├── setup_gen_dataclean1_env.sh        # 一键创建 conda 环境（见第 2 节）
│   └── make_formal_samplecheck.py         # 从正式结果中随机抽 10% 用于人工抽检
└── webview/
    ├── app.py                     # 本地看图 / 人工复核服务
    └── static/index.html
```

---

## 2. 环境安装

推荐环境：Linux、Python 3.10、带 NVIDIA GPU（CUDA 12.x）。

已验证的版本：

| 依赖 | 版本 |
|---|---|
| Python | 3.10 |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| transformers | 5.x |
| pillow | 12.x |
| imagehash | 4.3 |
| numpy | 2.2 |
| pyyaml | 6.0 |
| langdetect | 1.0.9 |
| pyarrow | 用于 parquet 格式（wikiart / alchemist / diffusiondb_2m_random_50k） |

安装示例：

```bash
conda create -n dataclean python=3.10 -y
conda activate dataclean

# GPU 版 PyTorch（按本机 CUDA 版本选择对应的 index-url）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 其余依赖
pip install pillow imagehash langdetect pyyaml numpy transformers pyarrow
```

检查是否能用 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> 也可以使用一键安装脚本 `scripts/setup_gen_dataclean1_env.sh`（创建名为 `gen-dataclean1` 的环境，安装 CUDA 12.4 版 PyTorch，使用清华 pip 镜像）。脚本通过环境变量 `CONDA_SH` 指定 conda 初始化脚本的位置（默认 `$HOME/miniconda3/etc/profile.d/conda.sh`），例如：
>
> ```bash
> CONDA_SH=$HOME/anaconda3/etc/profile.d/conda.sh bash scripts/setup_gen_dataclean1_env.sh
> ```

---

## 3. 模型准备

只有两个过滤器需要深度学习模型，其余过滤器只依赖 CPU 计算：

| 过滤器 | HuggingFace 模型 | 大小 |
|---|---|---|
| `NSFWFilter` | [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection)（ViT 二分类：normal / nsfw） | 约 1.1 GB |
| `WatermarkFilter` | [`prithivMLmods/Watermark-Detection-SigLIP2`](https://huggingface.co/prithivMLmods/Watermark-Detection-SigLIP2)（SigLIP2 二分类：No Watermark / Watermark） | 约 355 MB |

**第一步：从 HuggingFace 下载模型到本地**（目录可自选，下面以 `/path/to/models` 为例）：

```bash
pip install -U huggingface_hub
# 国内网络可先设置镜像：export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Falconsai/nsfw_image_detection --local-dir /path/to/models/nsfw_image_detection
huggingface-cli download prithivMLmods/Watermark-Detection-SigLIP2 --local-dir /path/to/models/Watermark-Detection-SigLIP2
```

**第二步：修改配置文件中的模型路径。** 仓库中 `config/*.yaml` 的 `model_name` 是占位路径 `/path/to/models/...`，请把它替换成你实际下载的目录（每个配置文件各有两处）：

```yaml
  - name: NSFWFilter
    model_name: "/path/to/models/nsfw_image_detection"         # 改成你的路径
  - name: WatermarkFilter
    model_name: "/path/to/models/Watermark-Detection-SigLIP2"  # 改成你的路径
```

如果所有配置都使用同一个模型目录，可以一次性批量替换：

```bash
sed -i 's#/path/to/models#/your/real/models/dir#g' config/*.yaml
```

也可以直接把 `model_name` 写成 HuggingFace 模型名（如 `Falconsai/nsfw_image_detection`），首次运行时会自动下载。

> **注意**：如果不修改占位路径 `/path/to/models/...`，运行时会因找不到模型而报错。若在配置中删掉 `model_name` 这一行，则使用 `src/filters.py` 中的默认值（HuggingFace 模型名，需能联网下载）。不需要这两个过滤器时，也可以把对应的 `enabled` 设为 `false`。

---

## 4. 数据集准备

### 4.1 下载来源与目录要求

| 数据集 | 下载地址 | `--dataset` 应指向 | 目录结构要求 |
|---|---|---|---|
| Alchemist | [yandex/alchemist](https://huggingface.co/datasets/yandex/alchemist) | 数据集根目录 | `data/*.parquet`（或 csv，含 `img_key` / `url` / `prompt`）+ `images/{img_key}.{jpg,png,webp,...}`。**图片需按 `url` 自行下载**，见 4.3 |
| Conceptual Captions | [google-research-datasets/conceptual_captions](https://huggingface.co/datasets/google-research-datasets/conceptual_captions) | 数据集根目录 | `cc_clean_index.jsonl` + `cc_images/`。**图片需按 URL 自行下载并生成索引**，见 4.3 |
| DenseFusion-4V-100K | [BAAI/DenseFusion-1M](https://huggingface.co/datasets/BAAI/DenseFusion-1M)（仓库内的 4V-100K 部分） | DenseFusion 仓库根目录 | `DenseFusion-4V-100k/DenseFusion-4V-100k.jsonl` + `images/DenseFusion-4V-100K/*.zip`（zip 无需解压） |
| DiffusionDB-2M-Random-50K | [svjack/diffusiondb_2m_random_50k](https://huggingface.co/datasets/svjack/diffusiondb_2m_random_50k) | 数据集根目录 | `data/*.parquet` + `images/{parquet 文件名}_idx{6 位行号}.png`。**图片需先从 parquet 导出**，见 4.3 |
| Fine-T2I | [ma-xu/fine-t2i](https://huggingface.co/datasets/ma-xu/fine-t2i) | **子集目录**（如 `fine-t2i/curated`），不能是 `fine-t2i` 根目录 | 目录下直接有 `*.tar`（或在 `data/` 下），每个 tar 内为 `{id}.jpg` + `{id}.txt` + `{id}.json` |
| WikiArt | [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) | 数据集根目录 | `data/*.parquet`（图片字节内嵌）+ `dataset_infos.json`（用于把标签编号转为名称） |

### 4.2 下载命令

```bash
pip install -U huggingface_hub
# 国内网络可先设置镜像：export HF_ENDPOINT=https://hf-mirror.com
DATA=/path/to/datasets    # 替换为你的数据集存放目录

# Alchemist（只有 prompt + 图片 URL，图片需另行下载，见 4.3）
huggingface-cli download yandex/alchemist --repo-type dataset --local-dir $DATA/alchemist

# Conceptual Captions（只有 caption + 图片 URL，图片需另行下载，见 4.3）
huggingface-cli download google-research-datasets/conceptual_captions --repo-type dataset --local-dir $DATA/conceptual_captions_hf

# DenseFusion-4V-100K（只下载 4V-100K 部分，不需要 DenseFusion-1M）
huggingface-cli download BAAI/DenseFusion-1M --repo-type dataset \
  --include "DenseFusion-4V-100k/*" "images/DenseFusion-4V-100K/*" \
  --local-dir $DATA/DenseFusion

# DiffusionDB-2M-Random-50K（下载后需导出图片，见 4.3）
huggingface-cli download svjack/diffusiondb_2m_random_50k --repo-type dataset --local-dir $DATA/diffusiondb_2m_random_50k

# Fine-T2I（全量约 2 TB，可用 --include 只下载需要的子集）
huggingface-cli download ma-xu/fine-t2i --repo-type dataset --include "curated/*" --local-dir $DATA/fine-t2i
huggingface-cli download ma-xu/fine-t2i --repo-type dataset --include "synthetic_*/*" --local-dir $DATA/fine-t2i

# WikiArt
huggingface-cli download huggan/wikiart --repo-type dataset --local-dir $DATA/wikiart
```

### 4.3 需要额外处理的数据集

**DiffusionDB-2M-Random-50K：从 parquet 导出图片。** 读取器直接从 `images/` 读 PNG，不读 parquet 内嵌的图片字节，因此需要先导出一次（文件名规则必须与下面一致）：

```bash
cd $DATA/diffusiondb_2m_random_50k
python - <<'EOF'
import glob, os
import pyarrow.parquet as pq

os.makedirs("images", exist_ok=True)
for path in sorted(glob.glob("data/*.parquet")):
    stem = os.path.splitext(os.path.basename(path))[0]
    row = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["image"]):
        for img in batch.column("image").to_pylist():
            out = f"images/{stem}_idx{row:06d}.png"
            if img and img.get("bytes") and not os.path.exists(out):
                with open(out, "wb") as f:
                    f.write(img["bytes"])
            row += 1
    print("done", path, row)
EOF
```

**Alchemist：按 URL 下载图片。** 读取 `data/train-00000-of-00001.parquet` 中的 `img_key` 与 `url`，把每张图片保存为 `images/{img_key}.{扩展名}`（扩展名可以是 jpg / jpeg / png / webp / bmp，读取器按文件名主干 `img_key` 匹配）。下载失败的样本会被 `BrokenFilesFilter` 以 `missing_image` 剔除。

**Conceptual Captions：按 URL 下载图片并生成索引。** HuggingFace 上的数据只有 caption 和图片 URL。需要自行下载图片到 `cc_images/`，并为**下载成功**的样本生成 `cc_clean_index.jsonl`，每行一条 JSON：

```json
{"original_id": 288, "file_name": "00000288.jpg", "url": "http://...", "caption": "new headboard for the guest bedroom"}
```

其中 `file_name`（相对 `cc_images/` 的文件名）和 `caption` 为必填；`original_id` 用作样本 ID（缺省时用文件名主干），`url` 等其余字段会原样保留到导出的 `.json` 中。最终目录：

```
conceptual_captions/
├── cc_clean_index.jsonl
└── cc_images/
    ├── 00000000.jpg
    └── ...
```

---

## 5. 快速开始

所有命令都在 `data_clean_general/` 目录下执行。以下命令中的 `/path/to/datasets/...` 请替换为第 4 节中你的数据集目录；`--output` 可以是任意目录，这里使用相对路径，结果会写在 `data_clean_general/` 下（看图器默认读取这些目录，见第 12 节）。

### 5.1 小样本试跑（`--limit`）

先用少量样本确认数据路径、格式、模型路径和阈值是否合适：

```bash
cd data_clean_general

python run_filters.py \
  --config config/config_conceptual_captions.yaml \
  --dataset /path/to/datasets/conceptual_captions \
  --format conceptual-captions \
  --output ./output_conceptual_captions \
  --save-mode split \
  --run-name cc_smoke \
  --limit 100
```

运行结束后，终端会打印一个汇总 JSON（每个过滤器的输入 / 保留 / 剔除数量），结果写在 `./output_conceptual_captions/cc_smoke/` 下：

- `images/`：保留的图片 + 原始 caption / 元数据
- `deleted/`：被剔除的图片 + 原始 caption / 元数据
- `deleted_summary.txt`：每张被剔除图片的文件名、所属分片、过滤器、原因
- `filter_report.json`：完整报告

> 分片型数据集（Fine-T2I、DenseFusion-4V-100K、WikiArt、DiffusionDB）试跑时建议加 `--tar-limit 1`，只读第一个分片，启动更快。

### 5.2 全量运行（`--resume`）

确认效果后，把 `--limit` 换成 `--resume` 即可全量处理。**第一次运行和中断后重新运行都用同一条命令**：程序会检查 `<output>/<run-name>/checkpoint/`，有断点就从断点继续，没有就从头开始。

```bash
python run_filters.py \
  --config config/config_conceptual_captions.yaml \
  --dataset /path/to/datasets/conceptual_captions \
  --format conceptual-captions \
  --output ./output_conceptual_captions \
  --save-mode split \
  --run-name conceptual_captions_formal \
  --resume
```

建议在 `tmux` / `screen` 中运行全量任务，避免终端断开导致任务中断。

### 5.3 六个数据集的完整命令

```bash
# Alchemist
python run_filters.py --config config/config_alchemist.yaml \
  --dataset /path/to/datasets/alchemist --format alchemist \
  --output ./output_alchemist --save-mode split \
  --run-name alchemist_formal --resume

# Conceptual Captions
python run_filters.py --config config/config_conceptual_captions.yaml \
  --dataset /path/to/datasets/conceptual_captions --format conceptual-captions \
  --output ./output_conceptual_captions --save-mode split \
  --run-name conceptual_captions_formal --resume

# DenseFusion-4V-100K（--dataset 指向 DenseFusion 仓库根目录）
python run_filters.py --config config/config_densefusion_4v_100k.yaml \
  --dataset /path/to/datasets/DenseFusion --format densefusion-4v-100k \
  --output ./output_densefusion_4v_100k --save-mode split \
  --run-name densefusion_4v_100k_formal --resume --tar-batch-size 4

# DiffusionDB-2M-Random-50K（需先按 4.3 导出图片）
python run_filters.py --config config/config_diffusiondb_2m_random_50k.yaml \
  --dataset /path/to/datasets/diffusiondb_2m_random_50k --format diffusiondb_2m_random_50k \
  --output ./output_diffusiondb_2m_random_50k --save-mode split \
  --run-name diffusiondb_2m_random_50k_formal --resume

# Fine-T2I curated（--dataset 指向子集目录；每个子集单独一个 --run-name）
python run_filters.py --config config/config_fine_t2i.yaml \
  --dataset /path/to/datasets/fine-t2i/curated --format fine-t2i \
  --output ./output_finet2i/finet2i_formal --save-mode split \
  --run-name curated --resume --tar-batch-size 5

# Fine-T2I synthetic 子集（以 synthetic_enhanced_prompt_random_resolution 为例，
# 其余三个 synthetic_* 子集只需替换 --dataset 与 --run-name）
python run_filters.py --config config/config_fine_t2i_synthetic.yaml \
  --dataset /path/to/datasets/fine-t2i/synthetic_enhanced_prompt_random_resolution --format fine-t2i \
  --output ./output_finet2i/finet2i_formal --save-mode split \
  --run-name synthetic_enhanced_prompt_random_resolution --resume --tar-batch-size 5

# WikiArt
python run_filters.py --config config/config_wikiart.yaml \
  --dataset /path/to/datasets/wikiart --format wikiart \
  --output ./output_wikiart --save-mode split \
  --run-name wikiart_formal --resume
```

带 `--tar-batch-size N` 时，每次运行只处理 N 个未完成的分片后退出，重复执行同一条命令直到汇总中的 `pending_tar_shards` 为 0（见第 10 节）；不加则一次处理全部分片。

---

## 6. 命令行参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--config` | 是 | 过滤器配置文件。只写文件名时会自动到 `config/` 下查找，例如 `--config config_wikiart.yaml` |
| `--dataset` | 是 | 数据集目录（各数据集要求见第 4.1 节） |
| `--format` | 是 | 数据集格式：`alchemist`、`conceptual-captions`、`densefusion-4v-100k`、`diffusiondb_2m_random_50k`、`fine-t2i`、`wikiart` |
| `--output` | 是 | 输出根目录，结果写到 `<output>/<run-name>/` |
| `--limit N` | 二选一 | 小样本试跑：只处理前 N 条，不写断点。与 `--resume` 互斥 |
| `--resume` | 二选一 | 全量运行并支持断点续跑。必须同时给 `--run-name` |
| `--run-name` | `--resume` 时必填 | 本次结果的子目录名 |
| `--save-mode` | 否 | `report`：只写报告 JSON；`kept`：再导出保留的图片；`split`：再导出保留和剔除的图片（推荐，便于抽检）。默认取配置文件中的 `save_mode` |
| `--report-name` | 否 | 报告文件名，默认 `filter_report.json` |
| `--tar-batch-size N` | 否 | 分片型数据集：每次只处理接下来 N 个未完成分片，需配合 `--resume`（见第 10 节） |
| `--tar-offset` / `--tar-limit` | 否 | 分片型数据集：手动指定跳过前多少个分片、最多处理多少个分片。不能与 `--tar-batch-size` 同时用 |
| `--tar-glob` | 否 | `fine-t2i` 匹配分片文件的通配符，默认 `*.tar` |
| `--workers` | 否 | GPU 之后 CPU 过滤器（Compression / Blur / Entropy）的并行线程数，0 为自动 |
| `--pre-workers` | 否 | Duplicate 之前 CPU 过滤器（Broken / Resolution / AspectRatio）的并行线程数，0 为自动 |
| `--dup-hash-workers` | 否 | 覆盖配置中 DuplicateFilter 的 `hash_workers` |
| `--prefetch` | 否 | 样本预读队列长度，0 为自动 |
| `--write-workers` | 否 | 导出图片的并行写线程数，默认 8 |
| `--checkpoint-every` | 否 | 每处理多少条样本保存一次去重状态，默认 100 |

分片型数据集及其分片单位：Fine-T2I 为 tar，DenseFusion-4V-100K 为 zip，WikiArt 与 DiffusionDB 为 parquet；Alchemist 和 Conceptual Captions 不分片。

---

## 7. 过滤器说明

### 7.1 执行顺序

过滤器按配置文件中的书写顺序执行，样本一旦被某个过滤器剔除，就不再进入后续过滤器。在 `--resume` 模式下，程序把过滤器分成几段做流水线并行：

```
BrokenFiles → Resolution → AspectRatio    （CPU 多线程）
      → Duplicate                          （哈希多线程计算，去重索引串行更新，"先出现者保留"）
      → Language                           （CPU）
      → NSFW → Watermark                   （GPU 批量推理）
      → CompressionQuality → Blur → Entropy（CPU 多线程）
```

建议把便宜的过滤器（文件检查、分辨率）放在前面，先剔除大量样本，再跑昂贵的模型推理。

每个过滤器都可以用 `enabled: true/false` 开关。

### 7.2 BrokenFilesFilter

剔除图片缺失、无法解码，以及文本缺失 / 为空 / 过短的样本。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `check_missing_image` | `true` | 检查图片是否存在（本地文件、tar/zip 成员或已读取的字节） |
| `check_image_loadable` | `true` | 用 PIL 解码图片，剔除损坏图 |
| `check_missing_text` | `true` | 检查文本是否存在 |
| `min_text_length` | `0` | 文本最短字符数 |

剔除原因：`missing_image`、`broken_image`、`missing_or_empty_text`、`text_too_short`。

### 7.3 ResolutionFilter

| 参数 | 默认值 | 说明 |
|---|---|---|
| `min_width` / `min_height` | `0` | 最小宽 / 高 |
| `max_width` / `max_height` | `0` | 最大宽 / 高，0 表示不限制 |
| `min_pixels` / `max_pixels` | `0` | 最小 / 最大总像素数，0 表示不限制 |

剔除原因：`width_too_small` / `width_too_large`、`height_too_small` / `height_too_large`、`pixels_too_small` / `pixels_too_large`。

### 7.4 AspectRatioFilter

| 参数 | 默认值 | 说明 |
|---|---|---|
| `min_ratio` / `max_ratio` | `0` / `inf` | 宽高比（宽 / 高）范围 |
| `allowed_ratios` | `[]` | 只允许的固定比例，如 `["1:1", "16:9"]`；为空表示不限制 |
| `tolerance` | `1e-3` | 与 `allowed_ratios` 比较时的容差 |

### 7.5 DuplicateFilter

基于感知哈希去除完全重复和近似重复的图片。默认的 `hybrid` 模式要求 pHash **和** dHash 的汉明距离都不超过阈值才判为重复，偏向“宁可漏判也不误删”。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `method` | `hybrid` | `hybrid`（pHash + dHash）、`perception`/`phash`、`difference`、`average` |
| `hash_size` | `16` | 哈希边长，至少 8 |
| `threshold` | `6` | 主哈希（pHash）汉明距离阈值，0 表示只去完全相同 |
| `secondary_threshold` | 同 `threshold` | `hybrid` 模式下 dHash 的阈值 |
| `hash_workers` | `1` | 并行计算哈希的线程数 |
| `hash_batch_size` | `32` | 每批计算哈希的样本数 |

去重是**同一次运行（同一个 `--run-name`）内**的全局去重；断点续跑时去重索引会从 `checkpoint/duplicate_state.json` 恢复。

### 7.6 LanguageFilter

用 `langdetect` 检测文本语言，剔除不在允许列表中的样本。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `allowed_languages` | `["en", "zh", "zh-cn", "zh-tw"]` | 允许的语言代码（`zh-*` 统一归为 `zh`） |
| `fallback_keep` | `true` | 文本太短或检测失败时是否保留 |
| `min_text_length` | `5` | 参与检测的最短文本长度 |

### 7.7 NSFWFilter（GPU）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_name` | `Falconsai/nsfw_image_detection` | 模型目录或 HuggingFace 模型名（见第 3 节） |
| `threshold` | `0.5` | NSFW 概率**高于**该值则剔除 |
| `device` | `auto` | `auto` / `cuda` / `cpu` |
| `batch_size` | `8` | 推理批大小 |

### 7.8 WatermarkFilter（GPU）

当以下两个条件**同时**满足时判为有水印并剔除：

- `watermark_prob >= threshold`
- `watermark_prob - no_watermark_prob >= min_margin`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_name` | `prithivMLmods/Watermark-Detection-SigLIP2` | 模型目录或 HuggingFace 模型名（见第 3 节） |
| `threshold` | `0.98` | 水印概率阈值（各数据集配置中实际使用 0.70 ~ 0.75） |
| `min_margin` | `0.2` | 两类概率的最小差值，用于排除模型拿不准的样本 |
| `device` | `auto` | `auto` / `cuda` / `cpu` |
| `batch_size` | `8` | 推理批大小 |

调参方向：**调小** `threshold` / `min_margin` → 删得更多（更严格去水印，误删也会增多）；**调大** → 删得更少（更少误删，漏检增多）。

### 7.9 CompressionQualityFilter

基于两个指标剔除压缩伪影严重的图片：

- `blockiness`：8×8 块边界梯度与块内梯度之比，衡量 JPEG 块效应；
- `high_freq_ratio`：FFT 高频能量与低频能量之比，衡量振铃、噪声等。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_blockiness` | `0` | 超过则剔除，0 表示不检查 |
| `max_high_freq_ratio` | `0` | 超过则剔除，0 表示不检查 |

### 7.10 BlurFilter

把灰度图缩放到固定边长后计算 Laplacian 方差，方差越低越模糊。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `min_laplacian_var` | `200` | 低于该值则剔除 |
| `resize_to` | `512` | 计算前缩放到的边长，0 表示不缩放 |

### 7.11 EntropyFilter

灰度直方图的香农熵（0 ~ 8）。熵过低通常是纯色、空白图；熵过高通常是噪声图。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `min_entropy` | `0` | 低于则剔除 |
| `max_entropy` | `8` | 高于则剔除 |

---

## 8. 配置文件

每个数据集一份配置，放在 `config/` 下。以 `config/config_conceptual_captions.yaml` 为例：

```yaml
save_mode: split

filters:
  - name: BrokenFilesFilter
    enabled: true
    check_missing_image: true
    check_missing_text: true
    check_image_loadable: true
    min_text_length: 1

  - name: ResolutionFilter
    enabled: true
    min_width: 512
    min_height: 512
    min_pixels: 100000

  - name: AspectRatioFilter
    enabled: true
    min_ratio: 0.25
    max_ratio: 4.0

  - name: DuplicateFilter
    enabled: true
    hash_size: 16
    threshold: 6
    secondary_threshold: 6
    method: hybrid
    hash_workers: 16
    hash_batch_size: 64

  - name: LanguageFilter
    enabled: true
    allowed_languages: [en, zh, zh-cn, zh-tw]
    fallback_keep: true
    min_text_length: 5

  - name: NSFWFilter
    enabled: true
    model_name: "/path/to/models/nsfw_image_detection"
    threshold: 0.5
    device: auto
    batch_size: 32

  - name: WatermarkFilter
    enabled: true
    model_name: "/path/to/models/Watermark-Detection-SigLIP2"
    threshold: 0.70
    min_margin: 0.20
    device: auto
    batch_size: 32

  - name: CompressionQualityFilter
    enabled: true
    max_blockiness: 2.5
    max_high_freq_ratio: 5.0

  - name: BlurFilter
    enabled: true
    min_laplacian_var: 100
    resize_to: 512

  - name: EntropyFilter
    enabled: true
    min_entropy: 2
    max_entropy: 8.5
```

各数据集配置与上面的主要差异：

| 配置文件 | 与 Conceptual Captions 配置的差异 |
|---|---|
| `config_alchemist.yaml` | 水印阈值 `0.73`；关闭 CompressionQuality 与 Entropy |
| `config_densefusion_4v_100k.yaml` | 水印阈值 `0.75` / `min_margin 0.25`；哈希和推理批次更小（节省内存） |
| `config_diffusiondb_2m_random_50k.yaml` | 关闭 CompressionQuality 与 Entropy |
| `config_fine_t2i.yaml` / `config_fine_t2i_synthetic.yaml` | 关闭 CompressionQuality 与 Entropy |
| `config_wikiart.yaml` | 关闭 CompressionQuality 与 Entropy |

> **断点续跑期间不要修改过滤器阈值**：程序会对过滤器配置计算指纹并保存在 `checkpoint/meta.json` 中，`--resume` 时如果发现过滤规则变了会直接报错，避免同一批结果用了两套标准。只改性能参数（`batch_size`、`device`、`hash_workers`、`hash_batch_size`）不受影响。若要换阈值，请换一个新的 `--run-name`。

---

## 9. 输出结果

`--save-mode split` 时，一次运行的输出目录结构如下：

```
<output>/<run-name>/
├── filter_report.json      # 完整报告
├── deleted_summary.txt     # 被剔除图片清单：文件名 \t 分片 \t 过滤器 \t 原因
├── images/                 # 保留的图片 + 同名 .json / .txt（原始文本与元数据）
├── deleted/                # 被剔除的图片 + 同名 .json / .txt
└── checkpoint/             # 仅 --resume 模式
    ├── meta.json           # 配置指纹、已完成的分片列表
    ├── progress.jsonl      # 每条样本的处理结果（kept / dropped）
    ├── dropped.jsonl       # 每条被剔除样本的详细原因
    └── duplicate_state.json# 去重索引
```

导出时**保持数据集原始的文本形态**，不改写、不合成 caption：

- `fine-t2i`：与原 tar 一致，写出 `.txt`（原始 caption）+ `.json`（原始元数据）。其中 `synthetic_enhanced_prompt_*` 子集的 `.txt` 本身就是 enhanced prompt。
- `conceptual-captions` / `densefusion-4v-100k`：写出 `.json`，内容为原始 jsonl 中的那一行记录（caption、url、image_path 等）。
- `wikiart`：数据集没有 caption，`.json` 中保留原始 `artist` / `genre` / `style` 标签。
- `alchemist`：`.json` 中保留原始 `prompt`、`url`、`img_key`。
- `diffusiondb_2m_random_50k`：保留原始 prompt 和生成参数（seed / cfg / sampler 等），写出 `.json` + `.txt`。

`filter_report.json` 示例片段：

```json
{
  "initial_count": 150,
  "final_count": 144,
  "total_dropped": 6,
  "filters": [
    {
      "name": "ResolutionFilter",
      "input_count": 150,
      "kept_count": 145,
      "dropped_count": 5,
      "dropped": [
        {
          "sample_id": "12_2047c8d3-8ee4-4c53-81bf-559ae49b3929",
          "filter_name": "ResolutionFilter",
          "reason": "width_too_small",
          "details": {"width": 512, "height": 512, "pixels": 262144, "min_width": 768},
          "source": {"...": "..."}
        }
      ]
    }
  ]
}
```

`reason` 是剔除原因，`details` 是对应的具体数值（分辨率、模型分数、检测到的语言等），可用于追溯和调参。

---

## 10. 断点续跑与分批处理

- `--resume` 模式下，每条样本处理完会立即写入 `checkpoint/progress.jsonl` 并导出图片；中断后用**完全相同的命令**重新运行，已处理的样本会被跳过。
- 对于分片型数据集（Fine-T2I、DenseFusion-4V-100K、WikiArt、DiffusionDB），可加 `--tar-batch-size N`：每次运行只处理接下来 N 个未完成的分片，跑完后退出；再执行同一条命令即处理下一批。分片完成情况记录在 `checkpoint/meta.json` 的 `completed_tar_files` 中，去重状态跨批次保留。
- 运行结束时的汇总会给出 `completed_tar_shards` / `pending_tar_shards` / `total_tar_shards`，可据此判断是否全部完成。
- 需要无人值守跑完全部分片时，重复执行同一条命令，直到 `pending_tar_shards` 为 0 即可。

---

## 11. 性能说明

- **一定要在有 GPU 的机器上运行全量任务**。NSFW 与 Watermark 是整个流程中最耗时的部分，`device: auto` 在没有 GPU 时会退回 CPU，速度会慢一个数量级左右。日志中的 `GPU micro-batching enabled` 只表示启用了批量推理，并不代表真的在用 GPU，请用 `python -c "import torch; print(torch.cuda.is_available())"` 确认。
- 参考数据：在没有 GPU 的节点上处理 Conceptual Captions（约 115 万条），速度约 2.7 条/秒。
- `--save-mode split` 会把每张图片都写一份到输出目录，输出体积与原数据集同量级；只需要统计结果时可用 `--save-mode report`。
- 分片型数据集建议配合 `--tar-batch-size`，单次运行时间可控，也便于中途检查结果。
- 并行度可通过 `--workers`、`--pre-workers`、`--dup-hash-workers`、`--write-workers`、`--prefetch` 调整；内存紧张时调小这些参数和配置中的 `batch_size`。

---

## 12. 辅助工具

### 12.1 结果看图器（webview）

清洗结果目录通常有几十万张图片，直接用文件浏览器打开很容易卡死。`webview/app.py` 是一个零依赖（只用 Python 标准库）的本地网页服务，一次只加载一张图：

```bash
cd data_clean_general/webview
python app.py --host 0.0.0.0 --port 8765
# 浏览器打开 http://<服务器IP>:8765
```

功能：

- 按数据集 / 运行版本浏览 `images/`（保留）和 `deleted/`（剔除）中的图片，同时显示原始 caption 与剔除原因；
- 人工复核：把误判的图片标记到 `images_after_check/`（复核后保留）或 `deleted_after_check/`（复核后删除），并可填写原因；
- 图片索引缓存在 `webview/.cache/` 中（已加入 `.gitignore`）。

看图器默认读取 `data_clean_general/` 下的 `output_alchemist`、`output_conceptual_captions`、`output_densefusion_4v_100k`、`output_diffusiondb_2m_random_50k`、`output_finet2i`、`output_wikiart` 这几个输出目录（与第 5.3 节命令中的 `--output` 一致），每个 `--run-name` 显示为一个版本；Fine-T2I 的各子集显示为 `finet2i_formal` 版本下的子集。如使用其他输出目录，请修改 `app.py` 中的 `DATASETS` 列表。

### 12.2 抽检样本生成

```bash
python scripts/make_formal_samplecheck.py
```

从各数据集正式结果的 `images/` 中随机抽取 10%（固定随机种子 42），复制（优先硬链接）到 `<run-name>_samplecheck` 版本目录，用于在看图器中人工抽检。默认处理第 5.3 节命令产生的 `*_formal` 结果；抽取的数据集与目录在脚本开头的 `JOBS` 列表中配置，已抽过的版本会跳过。

---

## 13. 扩展新的数据集格式

1. 在 `src/loaders.py` 中新增一个 Loader 类，实现 `load()` 方法，逐条 `yield` `Sample` 对象：
   - `id`：样本唯一 ID（用于去重索引和断点记录，同一数据集内不能重复）；
   - `image_path`（本地文件）或 `image_bytes`（压缩包 / parquet 内的图片字节）二选一；
   - `text`：用于语言检测等文本过滤器的文本；
   - `metadata`：原始记录，导出时会原样写入 sidecar `.json`；
   - `source`：来源信息（如所属分片、文件名），用于报告和导出命名。
2. 若数据集是分片存储的，实现 `list_tar_shards()`，并支持 `offset` / `limit` / `batch_size` / `completed_*_names` / `on_shard_complete`，即可复用 `--tar-batch-size` 断点分批逻辑（可参考 `DenseFusion4V100kLoader`、`WikiArtLoader`）。
3. 在 `src/cli.py` 的 `--format` 选项和 `build_loader()` 中注册新格式；分片型格式还需加入 `_SHARDED_FORMATS`。
4. 在 `config/` 下新增该数据集的配置文件，先用 `--limit` 小样本试跑。

---

## 14. 常见问题

**Q：全部样本都被 BrokenFilesFilter 剔除了。**
A：通常是 `--dataset` 指错了层级，或者 `--format` 与目录结构不匹配（对照第 4.1 节）。例如 `fine-t2i` 必须指向含 `*.tar` 的子集目录，而不是 `fine-t2i` 根目录；DiffusionDB 需要先导出图片。可查看 `deleted_summary.txt` 中的剔除原因（`missing_image` / `broken_image` / `missing_or_empty_text`）。

**Q：报错找不到模型 / `OSError: ... is not a local folder`。**
A：配置文件中的 `model_name` 仍是占位路径 `/path/to/models/...`，请按第 3 节从 HuggingFace 下载模型，并替换为你自己的路径。

**Q：`--resume` 时报 `Config filters changed since this run was started`。**
A：断点续跑期间修改了过滤器阈值。改回原配置，或换一个新的 `--run-name` 重新跑。

**Q：`--tar-batch-size requires --resume`。**
A：分批处理依赖断点记录，只能在 `--resume` 模式下使用；`--limit` 试跑时请用 `--tar-limit` 控制分片数。

**Q：运行很慢。**
A：先确认 GPU 是否可用（第 11 节）；其次可以把配置中耗时的过滤器暂时关闭，或用 `--save-mode report` 减少磁盘写入。
