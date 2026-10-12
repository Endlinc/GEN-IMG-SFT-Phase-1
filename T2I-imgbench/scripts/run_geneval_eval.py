"""GenEval 评分：导出官方格式 -> 官方 evaluate_images.py -> summary_scores.py -> 汇总 JSON。

用法:
  python scripts/run_geneval_eval.py --config qwen_image_2 --tag 20261004-120000 [--limit 5]
依赖: geneval 环境（torch2.1 + mmdet3.3 + mmcv2.1 + open_clip + clip_benchmark）；
     检测器/CLIP 权重见 configs/base.yaml 的 judges.geneval 配置。
"""
import argparse
import json
import os
import subprocess
import sys

from common import load_config, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="configs/ 下模型 yaml：可写文件名（如 z_image）或路径")
    ap.add_argument("--tag", default=None,
                    help="生图时的 tag（缺省为时间戳）；用于定位 outputs/<model>/geneval/<tag>")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="覆盖 base.yaml 注册表中 geneval 的官方 n_samples")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = cfg["model_name"]
    repo = cfg["benchmarks"]["geneval"]["repo"]          # 官方评测 repo（base.yaml 注册表）
    ckpt_dir = cfg["judges"]["geneval"]["detector_checkpoint"]

    tag_dir = out_dir(cfg, "geneval", args.tag)
    work = os.path.join(tag_dir, "_eval", "geneval")
    img_dir = os.path.join(work, "images")
    results_jsonl = os.path.join(work, "results.jsonl")
    score_dir = os.path.join(tag_dir, "scores")
    os.makedirs(score_dir, exist_ok=True)

    # 1. 导出官方格式
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "prepare_eval_inputs.py"),
           "--config", cfg["_config_path"], "--benchmark", "geneval",
           "--tag", os.path.basename(tag_dir), "--eval-dir", img_dir]
    if args.n_samples:
        cmd += ["--n-samples", str(args.n_samples)]
    subprocess.run(cmd, check=True)

    # 2. 官方评测（imagedir 需以数字子目录组织）
    env = os.environ.copy()
    env.setdefault("HF_HOME", cfg["paths"]["cache_root"])
    clip_ckpt = cfg["judges"]["geneval"].get("clip_checkpoint")
    eval_cmd = [sys.executable, os.path.join(repo, "evaluation", "evaluate_images.py"), img_dir,
                "--outfile", results_jsonl, "--model-path", ckpt_dir]
    if clip_ckpt:
        eval_cmd += ["--options", f"clip_path={clip_ckpt}"]
    subprocess.run(eval_cmd, cwd=repo, check=True, env=env)

    # 3. 官方汇总（打印任务分项）
    subprocess.run([sys.executable, os.path.join(repo, "evaluation", "summary_scores.py"),
                    results_jsonl], cwd=repo, check=True, env=env)

    # 4. 汇总 JSON 归档
    import pandas as pd
    df = pd.read_json(results_jsonl, orient="records", lines=True)
    summary = {
        "benchmark": "geneval", "model": model, "tag": os.path.basename(tag_dir),
        "n_images": int(len(df)),
        "n_prompts": int(df.groupby("metadata").ngroups),
        "overall_correct_rate_images": float(df["correct"].mean()),
        "per_task": {},
    }
    for tag, g in df.groupby("tag"):
        summary["per_task"][tag] = float(g["correct"].mean())
    if summary["per_task"]:
        summary["overall_geneval_official"] = float(
            sum(summary["per_task"].values()) / len(summary["per_task"]))
    with open(os.path.join(score_dir, f"{model}_geneval_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
