"""LongText-Bench 评分（EN/ZH）：导出 {pid}_{rep}.png -> 官方 OCR(Qwen2.5-VL) -> Text Accuracy。

单卡运行：torchrun --nproc_per_node=1。官方 evaluate_text_reward.py 会读取
textbench/ 下的 prompt 文件，因此在 textbench 目录内执行（路径见 base.yaml
benchmarks 注册表的 workdir）。

用法:
  python scripts/run_longtext_eval.py --config qwen_image_2 --tag 20261004-120000 [--mode en]
"""
import argparse
import json
import os
import socket
import subprocess
import sys

from common import load_config, out_dir


def find_free_port() -> int:
    """挑一个当前空闲的 TCP 端口（torchrun rendezvous 用，避免同节点 29500 冲突）。"""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_one(cfg, mode, tag):
    model = cfg["model_name"]
    bench = f"longtext_{mode}"
    repo = cfg["benchmarks"][bench]["workdir"]           # 官方 textbench 目录（注册表）

    tag_dir = out_dir(cfg, bench, tag)
    work = os.path.join(tag_dir, "_eval", bench)
    img_dir = os.path.join(work, "images")
    eval_out = os.path.join(work, "eval_results")
    score_dir = os.path.join(tag_dir, "scores")
    os.makedirs(score_dir, exist_ok=True)

    # 1. 导出 {prompt_id}_{repeat_id}.png
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "prepare_eval_inputs.py"),
                    "--config", cfg["_config_path"], "--benchmark", bench,
                    "--tag", os.path.basename(tag_dir), "--eval-dir", img_dir],
                   check=True)

    # 2. 官方 OCR 评测（单卡）
    env = os.environ.copy()
    env.setdefault("HF_HOME", cfg["paths"]["cache_root"])
    env["LTB_MODEL_PATH"] = cfg["judges"]["longtext"]["ocr_model"]
    subprocess.run(["torchrun", "--nnodes=1", "--node-rank=0", "--nproc_per_node=1",
                    "--master_port", str(find_free_port()),
                    "evaluate_text_reward.py",
                    "--sample_dir", img_dir, "--output_dir", eval_out, "--mode", mode],
                   cwd=repo, check=True, env=env)

    # 3. 汇总
    with open(os.path.join(eval_out, "results.jsonl"), "w") as fout:
        for fn in sorted(os.listdir(eval_out)):
            if fn.startswith("results_chunk"):
                with open(os.path.join(eval_out, fn)) as fin:
                    fout.write(fin.read())
    subprocess.run([sys.executable, "summary_scores.py",
                    os.path.join(eval_out, "results.jsonl"), "--mode", mode],
                   cwd=repo, check=True, env=env)

    # 4. 归档
    with open(os.path.join(eval_out, "scores.txt")) as f:
        score_str = f.read().strip()
    summary = {
        "benchmark": bench, "model": model, "tag": os.path.basename(tag_dir),
        "text_accuracy": float(score_str.split(":")[1].strip()),
        "raw_result_dir": eval_out,
    }
    with open(os.path.join(score_dir, f"{model}_{bench}_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="configs/ 下模型 yaml：可写文件名（如 z_image）或路径")
    ap.add_argument("--tag", default=None,
                    help="生图时的 tag（缺省为时间戳）；用于定位 outputs/<model>/longtext_*/<tag>")
    ap.add_argument("--mode", choices=["en", "zh", "both"], default="both")
    args = ap.parse_args()

    cfg = load_config(args.config)
    modes = ["en", "zh"] if args.mode == "both" else [args.mode]
    for m in modes:
        run_one(cfg, m, args.tag)


if __name__ == "__main__":
    main()
