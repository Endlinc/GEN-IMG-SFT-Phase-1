"""DPG-Bench 评分：导出 2x2 拼图 -> 官方 compute_dpg_bench.py (mPLUG VQA) -> 汇总 JSON。

单卡环境下用 PROCESSES=1 调 dist_eval.sh；脚本会对每张拼图按 --resolution 裁回 4 格。

用法:
  python scripts/run_dpg_eval.py --config qwen_image_2 --tag 20261004-120000
"""
import argparse
import json
import os
import re
import subprocess
import sys

from common import load_config, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="configs/ 下模型 yaml：可写文件名（如 z_image）或路径")
    ap.add_argument("--tag", default=None,
                    help="生图时的 tag（缺省为时间戳）；用于定位 outputs/<model>/dpg/<tag>")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="覆盖 base.yaml 注册表中 dpg 的官方 n_samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = cfg["model_name"]
    repo = cfg["benchmarks"]["dpg"]["repo"]              # 官方评测 repo（base.yaml 注册表）
    n_samples = args.n_samples or cfg["benchmarks"]["dpg"]["n_samples"]

    tag_dir = out_dir(cfg, "dpg", args.tag)
    work = os.path.join(tag_dir, "_eval", "dpg")
    img_dir = os.path.join(work, "images")
    score_dir = os.path.join(tag_dir, "scores")
    os.makedirs(score_dir, exist_ok=True)

    # 1. 导出 2x2 拼图（单格 1024）
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "prepare_eval_inputs.py"),
           "--config", cfg["_config_path"], "--benchmark", "dpg",
           "--tag", os.path.basename(tag_dir), "--eval-dir", img_dir,
           "--n-samples", str(n_samples)]
    subprocess.run(cmd, check=True)

    # 2. 官方评测（单卡：PROCESSES=1）
    env = os.environ.copy()
    env.update({
        "PROCESSES": "1", "PIC_NUM": str(n_samples),
        "MODELSCOPE_CACHE": cfg["paths"]["cache_root"],
        "HF_HOME": cfg["paths"]["cache_root"],
        "MPLUG_PATH": cfg["judges"]["dpg"]["vqa_model"],
    })
    subprocess.run(["bash", os.path.join(repo, "dpg_bench", "dist_eval.sh"), img_dir, "1024"],
                   cwd=repo, check=True, env=env)

    # dist_eval.sh 默认把结果写在 image_root_path 下，找到最新结果文件
    cands = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if f.startswith("dpg-bench_") and f.endswith("_results.txt")]
    if cands:
        latest = max(cands, key=os.path.getmtime)
        with open(latest) as f:
            content = f.read()
        m = re.search(r"DPG-Bench score: ([\d.]+)", content)
        summary = {
            "benchmark": "dpg_bench", "model": model, "tag": os.path.basename(tag_dir),
            "dpg_score": float(m.group(1)) if m else None,
            "raw_result_file": latest,
        }
        # L1/L2 分项
        l1 = dict(re.findall(r"\t([\w\- ]+): ([\d.]+)\n", content))
        summary["category_scores"] = {k.strip(): float(v) for k, v in l1.items()}
        with open(os.path.join(score_dir, f"{model}_dpg_summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, indent=2))
    else:
        print("未找到 DPG 结果文件，请检查上方官方脚本输出。")


if __name__ == "__main__":
    main()
