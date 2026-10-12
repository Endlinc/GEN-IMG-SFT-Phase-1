# Caption Benchmark Suite — 运行与结果说明

第一阶段 **Caption Evaluation** 交付模块。对任一被测 caption 方法（MLLM）在其图像上生成的
candidate caption，用四个独立 benchmark 的官方口径给出绝对质量分数与分项能力画像。

**Suite 不定义跨 benchmark 的 Overall Score。** 一次完整运行得到四组彼此独立、量纲不同的结果：

```
S(M) = [ S_CAPTURE , S_CapsBench , S_CompreCap , S_PerceptionRubrics ]
```

横向比较只在**同一个 benchmark 内**进行，不要比较不同 benchmark 之间的数值高低。

---

## 1. 目录结构

```
caption-evaluation-suite/
├── README.md                    # 本文件
├── code/                        # 运行与判分代码（4 个 benchmark 各一个入口）
│   ├── captioners.py            # 统一 caption 生成工厂（被 4 个脚本共用）
│   ├── run_capture.py
│   ├── run_capsbench.py
│   ├── run_comprecap.py
│   └── run_perceptionrubrics.py
├── baselines/                   # 已跑结果：4 模型 × 4 benchmark（读法见第 6 节）
├── third_party/                 # 官方评测代码（快照）
│   ├── CAPTURE/                 #   capture_metric
│   ├── CompreCap/               #   官方 evaluate.py
│   └── PerceptionRubrics/       #   官方 predict.py / judge.py
├── candidate_benchmarks/        # 备选 benchmark（报告附录 A，未入 Suite）
│   ├── CAPEval/
│   ├── faithscore/
│   └── unblocking-detail-caption/   #   DOCCI-Critique
├── requirements.txt             # 基础依赖（4 个脚本都需要）
├── requirements-capture.txt     # CAPTURE 本地评分额外依赖
├── requirements-comprecap.txt   # CompreCap 官方评分额外依赖
└── .env.example                 # 环境变量模板
```

下文所有命令默认在 `caption-evaluation-suite/` 目录内执行。

---

## 2. 环境依赖

```bash
# 基础：caption 生成 + LLM judge（仅需 numpy + Pillow）
pip install -r requirements.txt

# CAPTURE 本地评分（capture_metric，含 flan-t5 场景图解析 + S-BERT 软匹配）
pip install -r requirements-capture.txt

# CompreCap 官方评分（Llama-3-8B + MiniLM，还需 spaCy 模型）
pip install -r requirements-comprecap.txt
python -m spacy download en_core_web_lg
```

**API Key 全部走环境变量，代码里没有任何密钥。** 复制模板后填写：

```bash
cp .env.example .env
set -a && source .env && set +a
```

| 变量 | 用途 |
|------|------|
| `OPENAI_API_KEY` + `OPENAI_BASE_URL` | OpenAI 兼容端点。**caption 生成与 judge 都走这里** |
| `ARK_API_KEY` + `ARK_BASE_URL` | 火山方舟（评测 Doubao / Seed 系列时才需要） |
| `ANTHROPIC_API_KEY` | 走 Anthropic 原生 API 时才需要 |
| `CAPTURE_SG_MODEL` / `CAPTURE_ST_MODEL` | CAPTURE 本地模型路径（不设则自动从 HuggingFace 下载） |
| `COMPRECAP_LLM_PATH` / `COMPRECAP_BERT_PATH` | CompreCap 官方评分用的 Llama-3.1-8B-Instruct / all-MiniLM-L6-v2 |

---

## 3. 数据集与官方代码获取

| Benchmark | 数据 | 数据地址 | 官方代码 | Evaluator |
|-----------|------|----------|----------|-----------|
| **CAPTURE** | DetailCaps-4870（4,870 图） | HF `foundation-multimodal-models/DetailCaps-4870` | `third_party/CAPTURE`（亦可 `pip install capture_metric`） | flan-t5 场景图解析 + S-BERT 语义软匹配 |
| **CapsBench** | 200 图 / 2,406 二元问题 | HF `playgroundai/CapsBench` | **无公开代码**，judge 口径按论文自建 | LLM judge（yes/no QA） |
| **CompreCap** | 560 精标图 + `anno.json` | HF `FanLu31/CompreCap` | `third_party/CompreCap` | 官方 `evaluate.py`（Llama-3 + MiniLM） |
| **PerceptionRubrics** | 1,038 图 + MR/EW rubrics | HF `M1chaelPeng/PerceptionRubrics` | `third_party/PerceptionRubrics` | 官方 `scripts/judge.py`，Must-Right 门控 |

各数据集需自行下载，通过 `--data` 指定本地目录。所需文件结构：

| Benchmark | `--data` 指向的目录需包含 |
|-----------|---------------------------|
| CAPTURE | `detailcaps_4870.json`（含 `image_id` / `image` / `references`） |
| CapsBench | `capsbench.json`（含 `image_id` / `image` / `questions`） |
| CompreCap | `images/*.jpg` + `anno.json` |
| PerceptionRubrics | `perceptionrubrics.json`（含 `id` / `image`） |

---

## 4. 支持的 captioner

`captioners.py` 是统一工厂，4 个脚本共用，`--captioner` 传模型名：

| 名称 | 路由 |
|------|------|
| `gpt-5.5` | OpenAI 兼容端点（`OPENAI_BASE_URL`） |
| `claude-sonnet-4-5-20250929` | OpenAI 兼容端点（默认）或原生 Anthropic；可用 `CLAUDE_MODEL` 覆盖 |
| `qwen3.5-397b-a17b` | 透传到 OpenAI 兼容端点 |
| `doubao-*` / `seed-*` | 火山方舟（`ARK_API_KEY`），自动禁用 thinking |
| 其他任意模型 ID | 原样透传到 OpenAI 兼容端点 |

下表的四个模型即已交付 baseline 实际跑过的 captioner，目录名与之一一对应。

---

## 5. 运行

四个脚本均可**独立运行**，均在 `code/` 下。通用约定：

- `--out` 决定输出位置与文件名
- `--workers` 控制并发（caption 生成 / judge 线程数）
- `--limit N` 只跑前 N 张图（`0` = 全量），用于冒烟测试
- `--caption-only` 只生成 caption，跳过评分

### 5.1 CAPTURE

```bash
python code/run_capture.py \
    --data /path/to/DetailCaps-4870 \
    --captioner gpt-5.5 \
    --out results/capture_gpt-5.5.json \
    --workers 8
```

产出：`capture_gpt-5.5.json`（汇总 `capture_score`）+ `.captions.json`（原始 caption）+ `.per_image.jsonl`。
只生成 caption：加 `--caption-only`。

### 5.2 CapsBench

```bash
# 完整评测（caption + judge）
python code/run_capsbench.py \
    --data /path/to/CapsBench \
    --captioner gpt-5.5 \
    --out results/capsbench_gpt-5.5.jsonl \
    --workers 32

# 用已有 caption 直接判分（judge-only，避免重跑生成）
python code/run_capsbench.py \
    --data /path/to/CapsBench \
    --captioner gpt-5.5 \
    --out results/capsbench_gpt-5.5.jsonl \
    --captions-file baselines/capsbench/gpt-5.5/captions.jsonl

# 只生成 caption
python code/run_capsbench.py ... --caption-only --workers 8
```

`--captions-file` 同时接受 `.json`（`{image_id: caption}`）和 `.jsonl`（每行 `{image_id, caption}`）。
judge 模型通过 `OPENAI_BASE_URL` 调用，需 `OPENAI_API_KEY`。

> `--judge-model` 默认 `deepseek-v3.2`。CapsBench 官方 judge 是 Claude-3.5 Sonnet，
> 因该 API 已不可用而改用能力相近的 DeepSeek-V3.2；本包交付的 baseline 全部使用此 judge。

### 5.3 CompreCap

```bash
# 只生成 caption
python code/run_comprecap.py \
    --data /path/to/CompreCap_dataset \
    --captioner gpt-5.5 \
    --out output/comprecap_gpt-5.5 \
    --caption-only --workers 8

# 完整评测（需先安装并指向官方仓库）
python code/run_comprecap.py \
    --data /path/to/CompreCap_dataset \
    --captioner gpt-5.5 \
    --out output/comprecap_gpt-5.5 \
    --comprecap-repo third_party/CompreCap \
    --llm-path /path/to/Llama-3.1-8B-Instruct \
    --bert-path /path/to/all-MiniLM-L6-v2
```

> `--comprecap-repo` 不传时，脚本会**打印**官方评测命令供手工执行，不做评分。
> 不传 `--comprecap-repo` 时也可以用 `COMPRECAP_LLM_PATH` / `COMPRECAP_BERT_PATH` 环境变量提供模型路径。

### 5.4 PerceptionRubrics

```bash
# 只生成 predictions（官方格式）
python code/run_perceptionrubrics.py \
    --data /path/to/PerceptionRubrics \
    --captioner gpt-5.5 \
    --out output/perceptionrubrics_gpt-5.5 \
    --caption-only --workers 8

# 完整评测（需先安装官方仓库）
python code/run_perceptionrubrics.py \
    --data /path/to/PerceptionRubrics \
    --captioner gpt-5.5 \
    --out output/perceptionrubrics_gpt-5.5 \
    --judge-repo third_party/PerceptionRubrics \
    --judge-model gpt-oss-120b \
    --workers 8
```

> `third_party/PerceptionRubrics` 需先 `cd` 进去 `pip install -e '.[examples]'`。
> baseline 使用的 judge 是 `gpt-oss-120b`，脚本默认值不同，复现时需显式传 `--judge-model`。

---

## 6. Baseline 结果与怎么读

四个被测 caption 方法的完整 baseline 结果，位于 `baselines/`。
主指标口径与阶段一报告 §2.5.1 一致；分项指标与逐样本中间产物取自本目录结果文件。

模型目录名沿用评测时的版本标识：

| 目录名 | 报告中的名称 |
|--------|--------------|
| `claude-sonnet-4-5-20250929` | Claude Sonnet-4.5 |
| `gpt-5.5` | GPT-5.5 |
| `qwen3.5-397b-a17b` | Qwen3.5-397B-A17B |
| `doubao-seed-2-1-pro-260628` | Doubao Seed-2.1-Pro |

### 6.1 主指标总表

| 被测模型 | CapsBench ↑ | PerceptionRubrics ↑ | CAPTURE ↑ | CompreCap ↑ |
|----------|------------|---------------------|-----------|-------------|
| Claude Sonnet-4.5 | 68.08% | 0.2498 | 0.6221 | 59.74 |
| GPT-5.5 | 76.89% | 0.5864 | 0.6261 | 61.36 |
| Qwen3.5-397B-A17B | 73.98% | 0.6208 | 0.6329 | 65.45 |
| Doubao Seed-2.1-Pro | 79.51% | 0.5135 | 0.6041 | 68.30 |

### 6.2 分项结果

**CompreCap**（0–100 的 object/pixel coverage，0–5 的 attribute/relation，加权得 Unified）

| 模型 | object coverage | pixel coverage | attribute score | relation score | Unified |
|------|----------------|----------------|-----------------|----------------|---------|
| Claude Sonnet-4.5 | 77.89 | 62.82 | 2.78 | 2.60 | 59.74 |
| GPT-5.5 | 79.75 | 63.24 | 2.84 | 2.69 | 61.36 |
| Qwen3.5-397B-A17B | 80.93 | 65.64 | 3.02 | 3.01 | 65.45 |
| Doubao Seed-2.1-Pro | 85.19 | 67.51 | 3.18 | 3.10 | 68.30 |

> attribute / relation 全部挤在 2.6–3.2 / 5 的窄区间 —— **区分度来自 object / pixel coverage，不要只看 Unified。**

**PerceptionRubrics**（Must-Right 门控）

| 模型 | average_score | mr_pass_rate | ew_average |
|------|--------------|--------------|-----------|
| Claude Sonnet-4.5 | 0.2498 | 0.4769 | 0.4809 |
| GPT-5.5 | 0.5864 | 0.8006 | 0.7211 |
| Qwen3.5-397B-A17B | 0.6208 | 0.8015 | 0.7672 |
| Doubao Seed-2.1-Pro | 0.5135 | 0.6696 | 0.7339 |

> `average_score` 是 MR 门控后的最终分，受门控放大。**必须配合 `mr_pass_rate` 一起看**：
> Claude 的低分主要来自门控（只有 47.7% 过门），不是 `ew_average` 差。

**CapsBench**（17 类诊断）

逐类准确率见 `baselines/capsbench/<model>/summary.json` 的 `per_category`。四模型共同短板：
`proper noun`、`artifacts`、`blur`、`image type`。

### 6.3 结果文件清单

```
baselines/
├── capture/<model>/
│   ├── score.json          # 汇总：benchmark / dataset / captioner / n_images / capture_score / raw_score
│   └── captions.json       # 原始 caption，{image_id: caption}
├── capsbench/<model>/
│   ├── captions.jsonl      # 原始 caption，每行 {"image_id","caption"}
│   ├── judged.jsonl        # 逐图判分结果，每行 {"image_id","caption","questions","per_image_accuracy"}
│   │                       #   questions[] 每项 = {"question","category","reference","prediction","correct"}
│   └── summary.json        # 汇总：accuracy / n_questions / n_images / judge / per_category(17 类)
├── comprecap/<model>/
│   ├── captions.json       # 喂给官方 evaluate.py 的 caption，{image_name: caption}
│   └── official_0_result.json  # 官方输出，长度 2 的数组：
│                           #   [0] = 汇总 dict（object/pixel coverage, attribute/relation score, Unified score）
│                           #   [1] = 逐图明细 dict
└── perceptionrubrics/<model>/
    ├── predictions.jsonl   # 官方格式预测，每行 {"id","prediction"}
    └── judge_meta.json     # judge 配置 + 完整汇总
                            #   summary: {count, success, failures, average_score,
                            #             mr_pass_rate, ew_average, domain_scores(7 类)}
```

### 6.4 逐样本产物的覆盖量（**重要**）

汇总分数都是全量口径，但**保存下来的逐样本 caption 并非每个模型都完整**：

| 模型 | CAPTURE captions | CapsBench captions | CapsBench judged | CompreCap captions | PR predictions |
|------|-----------------|--------------------|------------------|--------------------|----------------|
| Claude Sonnet-4.5 | 1,000 / 4,870 | 200 | 200 | 560 | 1,038 |
| GPT-5.5 | 1,000 / 4,870 | 242 | 200 | 560 | 1,038 |
| Qwen3.5-397B-A17B | 4,870 | 200 | 200 | 560 | 1,038 |
| Doubao Seed-2.1-Pro | 4,870 | 200 | 200 | 560 | 1,038 |

- **CAPTURE（Sonnet / GPT）**：逐图 caption 保留了 1,000 / 4,870 张。
- **CAPTURE（Qwen / Doubao）**：逐图 caption 为全量 4,870 张。
- **CapsBench（GPT）**：CapsBench 只有 200 个测试用例；`captions.jsonl` 的 242 行是
  **生成阶段产生的重复 caption**，评测仍以 `judged.jsonl`（200 条）+ `summary.json` 为准。

> 结论：**汇总分按全量口径解读；逐样本文件按上表实际覆盖量解读**，不要假定每个模型的逐图明细都是全量。

### 6.5 复现命令

三个 benchmark 可以从已有的中间产物走 judge-only 路径，不必重跑 caption 生成：

```bash
# CAPTURE（用已存 caption 重新评分）
python code/run_capture.py --data /path/to/DetailCaps-4870 --captioner <model> \
    --out /tmp/capture_<model>.json --subset <已存 captions>

# CapsBench（judge-only）
python code/run_capsbench.py --data /path/to/CapsBench --captioner <model> \
    --out /tmp/capsbench_<model>.jsonl \
    --captions-file baselines/capsbench/<model>/captions.jsonl --workers 32

# PerceptionRubrics（把 predictions.jsonl 交给官方 judge.py，见第 5.4 节）
# CompreCap（把 captions.json 交给官方 evaluate.py，见第 5.3 节）
```

**复现时的口径要求**：CapsBench 固定 judge = DeepSeek-V3.2；PerceptionRubrics 固定 judge = gpt-oss-120b；
CompreCap 固定 Llama-3.1-8B-Instruct + all-MiniLM-L6-v2。judge 换版本会系统性移动绝对分。

---

## 7. 已知限制

**四个 benchmark 的分数不能互相比较，也不能加总。** 具体口径与坑：

- **CAPTURE** —— 采用较宽松的语义软匹配，会**压缩模型间分差**并丢失属性绑定／关系方向信息。定位为 **Coverage 指标**，不用于判断绑定是否正确。
- **CapsBench** —— 价值在 17 类**诊断维度**，需同时看总体准确率与分类别准确率，不宜作为唯一总体质量指标。
- **CompreCap** —— 数据规模小（560 图），只反映精标子集上的绑定／关系／区域覆盖；attribute 与 relation 分数挤在 2.6–3.2 / 5 的窄区间，区分度有限。
- **PerceptionRubrics** —— 最终分数受 **Must-Right 门控**影响，门控会放大单项错误。必须同时看原子级 rubric 通过率（`mr_pass_rate` / `ew_average`），不能只看 `average_score`。

其他注意事项：

- **judge 版本会系统性移动绝对分数**，跨版本比较必须显式记录 judge 版本。
- **caption 生成端配置会显著影响分数**，同一 benchmark 内所有被测模型必须使用相同数据版本与生成约束。
- **CAPTURE 的 `score.json` 保留实测值**，与阶段一报告表格数值有出入时以报告口径为准。
- `third_party/CompreCap` 有 2 个未拉取的 git submodule（`grounded-sam-osx`、`VISAM`），仅用于分割标注构建，**评测流程不需要**。

---

## 8. 备选 Caption Benchmark（未进入第一阶段 Suite）

来源：阶段一报告附录 A。以下三个方法**不属于本 Suite 的交付内容**，
仓库位于包根目录 `candidate_benchmarks/`，仅作为后续扩展的代码储备。

| Benchmark | 能力定位 | 未入选的原因 | 仓库 | 许可 |
|-----------|----------|--------------|------|------|
| **CAPEval** | Coverage / Precision 解耦 | 能够**显式区分内容覆盖与事实精确性**，适合在后续需要进一步拆分生成质量维度时加入 | `candidate_benchmarks/CAPEval/`（github.com/liuzhipenggg/CAPEval） | Apache-2.0 |
| **FaithScore** | 原子事实级细粒度核验 | 可作为**更细粒度事实一致性**评测的扩展方案 | `candidate_benchmarks/faithscore/`（github.com/bcdnlp/faithscore） | 见仓库自带 LICENSE |
| **DOCCI-Critique** | Detailed caption 细粒度 critique | 适合用于详细描述中的**局部事实错误与细粒度问题**分析 | `candidate_benchmarks/unblocking-detail-caption/` | Apache-2.0 |

### CAPEval

把 caption 质量拆成 **Coverage (C)** 与 **Precision (P)** 两个 0–100 的分数，用 checklist 逐条判定
（`yes` / `no` / `not_mentioned`），并研究两种 profile 如何迁移到 VLM understanding 与 T2I generation。

- 论文：arXiv:2608.02589 ｜ 项目页：https://liuzhipenggg.github.io/CAPEval/
- 数据集：HuggingFace `LiuzhipengUCAS/CAPEval`
- 仓库内容：`capeval/`（caption 生成、checklist judge、C/P 评分）+ CLI 入口
  `caption.py` / `score.py` / `evaluate.py` / `pipeline.py`，另有 `integrations/{lmms_eval,vlmevalkit}`
- 注意：仓库体积 **121 MB**，其中 `data/image/` 的示例图占 122 MB，代码本体仅约 1 MB

### DOCCI-Critique

配套论文《Unblocking Fine-Grained Evaluation of Detailed Captions: An Explaining AutoRater and
Critic-and-Revise Pipeline》（NeurIPS 2025，arXiv:2506.07631）。

**注意：官方方法代码并未开源。**

| 产物 | 状态 |
|------|------|
| 方法代码（VNLI-Critique 模型、AutoRater、Critic-and-Revise pipeline） | **未开源**，项目页标注 "COMING SOON" |
| 项目主页仓库（`candidate_benchmarks/unblocking-detail-caption/`） | 只含 `index.html` + `style.css` + 配图，**无方法代码** |
| 数据集 | 已公开（Apache-2.0）：1,400 条 caption / 100 图 × 14 VLM / 10,216 条句级标注，HuggingFace `google/DOCCI-Critique` |
