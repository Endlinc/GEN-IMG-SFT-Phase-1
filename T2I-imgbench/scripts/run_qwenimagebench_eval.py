"""Qwen-Image-Bench 评测编排：准备 judge 输入 -> 调用官方 judge.py -> 合并分片成绩。

两种图片来源：
- released：Benchmarks/Qwen-Image-Bench/images/<模型>/ 官方发布的参考图（复现榜单用），
  prompt 取 jsonl 的 prompt_cn（与官方判分协议一致，官方响应中引用的即中文 prompt）。
- run：outputs/<model>/<benchmark>/<tag>/images/ 下本地生成的 seed0_processed.png。

判分以子进程调用 Benchmarks/Qwen-Image-Bench-gh/judge.py（推理参数由官方 backend 固化），
输入 jsonl 放在 <tag>/_eval/qib/ 下，judge.py 的 _judged.jsonl / _bench_scores.* 随之落在同目录。
数据集 jsonl / 参考图目录 / judge 模型路径均来自 configs/base.yaml 注册表。

用法示例：
  # smoke（released，前 5 条）
  python scripts/run_qwenimagebench_eval.py --config qwen_image_2 \
      --source released --released-model GPT-Image-1.5 --limit 5
  # 全量分片（本地生图判分，tag 与生图一致）
  python scripts/run_qwenimagebench_eval.py --config z_image \
      --source run --benchmark qwenimagebench_cn --tag 20261004-120000 --shard 0/4
  # 合并分片成绩
  python scripts/run_qwenimagebench_eval.py --config z_image \
      --source run --benchmark qwenimagebench_cn --tag 20261004-120000 --merge
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

from common import load_config, out_dir


def load_bench_records(cfg):
    with open(cfg["benchmarks"]["qwenimagebench_cn"]["prompts"]) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_rows(args, cfg):
    """返回 (rows, bench_key, skipped)。rows: [{ID, prompt, image_path}]，按 ID 排序。"""
    bench_cfg = cfg["benchmarks"]["qwenimagebench_cn"]
    records = load_bench_records(cfg)
    rows, skipped = [], []
    if args.source == "released":
        bench_key = "qwenimagebench"
        img_root = bench_cfg["released_images"]
        for r in records:
            rel = r.get(args.released_model)
            img = os.path.join(img_root, rel) if rel else None
            if not rel or not os.path.exists(img):
                skipped.append(r["ID"])
                continue
            rows.append({"ID": r["ID"], "prompt": r["prompt_cn"], "image_path": img})
    else:
        bench_key = args.benchmark
        lang = args.benchmark.rsplit("_", 1)[1]  # cn | en
        src_dir = os.path.join(out_dir(cfg, args.benchmark, args.tag), "images")
        for r in records:
            img = os.path.join(src_dir, f"{r['ID']:06d}", "seed0_processed.png")
            if not os.path.exists(img):
                skipped.append(r["ID"])
                continue
            rows.append({"ID": r["ID"], "prompt": r[f"prompt_{lang}"], "image_path": img})
    rows.sort(key=lambda x: x["ID"])
    if args.prompt_ids:
        keep = {int(x) for x in args.prompt_ids.split(",") if x.strip()}
        rows = [row for row in rows if row["ID"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        if not (0 <= shard_i < shard_n):
            raise SystemExit(f"非法 --shard {args.shard}，要求 0 <= i < n")
        rows = [row for idx, row in enumerate(rows) if idx % shard_n == shard_i]
    return rows, bench_key, skipped


def name_suffix(args):
    s = ""
    if args.prompt_ids:
        s += "_patch"
    if args.limit:
        s += f"_limit{args.limit}"
    if args.shard:
        i, n = args.shard.split("/")
        s += f"_shard{i}of{n}"
    return s


def judge_dir_of(cfg, bench_key, tag):
    """<tag>/_eval/qib：judge 输入 / 逐图完整评分 / bench scores 都落在此目录。"""
    return os.path.join(out_dir(cfg, bench_key, tag), "_eval", "qib")


def run_judge(args, cfg):
    rows, bench_key, skipped = build_rows(args, cfg)
    bench_repo = cfg["benchmarks"]["qwenimagebench_cn"]["repo"]  # 官方 judge.py 所在 repo
    judge_dir = judge_dir_of(cfg, bench_key, args.tag)
    os.makedirs(judge_dir, exist_ok=True)
    input_path = os.path.join(judge_dir, f"judge_input{name_suffix(args)}.jsonl")
    with open(input_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"judge input: {input_path} ({len(rows)} rows, skipped {len(skipped)}: {skipped[:10]}"
          f"{'...' if len(skipped) > 10 else ''})")
    if args.prepare_only:
        return
    if not rows:
        print("no rows to judge, exit.")
        return

    judge_model = args.model or cfg["judges"]["qib"]["judge_model"]
    cmd = [args.judge_python, os.path.join(bench_repo, "judge.py"),
           "--input", input_path, "--model", judge_model,
           "--max-batch-size", str(args.max_batch_size),
           "--max-new-tokens", str(args.max_new_tokens)]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=bench_repo, check=True)
    dt = time.time() - t0
    # 每行约 5 个 L1 维度 judge 任务，供估算全量时长
    n_tasks = len(rows) * 5
    print(f"judge finished in {dt/60:.1f} min; ~{n_tasks} tasks, "
          f"~{dt/max(n_tasks,1):.1f} s/task")


def merge_shards(args, cfg):
    """合并目录下所有 judge_input*_judged.jsonl，重算 bench scores。"""
    bench_key = "qwenimagebench" if args.source == "released" else args.benchmark
    bench_repo = cfg["benchmarks"]["qwenimagebench_cn"]["repo"]
    judge_dir = judge_dir_of(cfg, bench_key, args.tag)
    sys.path.insert(0, bench_repo)
    from judge import compute_bench_scores, print_bench_scores, save_bench_scores

    files = sorted(f for f in glob.glob(os.path.join(judge_dir, "judge_input*_judged.jsonl"))
                   if "_limit" not in os.path.basename(f))
    if not files:
        raise SystemExit(f"no judged files in {judge_dir}")
    seen, rows = set(), []
    for fp in files:
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["ID"] in seen:
                    continue
                seen.add(r["ID"])
                rows.append(r)
    rows.sort(key=lambda x: x["ID"])
    merged_path = os.path.join(judge_dir, "merged_judged.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged {len(files)} files -> {merged_path} ({len(rows)} rows)")

    all_dim_raw_scores = []
    for r in rows:
        raw = r.get("judge_model_output")
        all_dim_raw_scores.append(json.loads(raw) if raw else {})
    bench = compute_bench_scores(all_dim_raw_scores)
    json_path, xlsx_path = save_bench_scores(bench, merged_path)
    print(f"bench scores: {json_path} / {xlsx_path}")
    print_bench_scores(bench)

    # 汇总指针归档到 <tag>/scores/
    tag_dir = out_dir(cfg, bench_key, args.tag)
    score_dir = os.path.join(tag_dir, "scores")
    os.makedirs(score_dir, exist_ok=True)
    summary = {
        "benchmark": bench_key, "model": cfg["model_name"],
        "tag": os.path.basename(tag_dir),
        "n_prompts": len(rows),
        "merged_judged": merged_path,
        "bench_scores_json": json_path,
        "bench_scores_xlsx": xlsx_path,
    }
    with open(os.path.join(score_dir, f"{cfg['model_name']}_{bench_key}_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="configs/ 下模型 yaml：可写文件名（如 qwen_image_2）或路径")
    ap.add_argument("--source", required=True, choices=["released", "run"])
    ap.add_argument("--released-model", default=None,
                    help="source=released 时 jsonl 中的模型字段名，如 GPT-Image-1.5")
    ap.add_argument("--benchmark", default=None,
                    choices=["qwenimagebench_cn", "qwenimagebench_en"],
                    help="source=run 时 outputs 树中的 benchmark 目录名")
    ap.add_argument("--tag", default=None,
                    help="生图时的 tag（缺省为时间戳）；source=run 时用于定位产物，"
                         "判分输入/结果落在 <tag>/_eval/qib/")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 条（smoke 用）")
    ap.add_argument("--prompt-ids", default=None,
                    help="逗号分隔的 ID 白名单（补跑缺失样本用，文件名带 _patch 后缀）")
    ap.add_argument("--shard", default=None, metavar="i/n", help="按 ID 顺序取模分片（0 起）")
    ap.add_argument("--merge", action="store_true", help="合并分片 judged 结果并重算成绩")
    ap.add_argument("--prepare-only", action="store_true", help="只生成 judge 输入，不调 judge.py")
    ap.add_argument("--model", default=None,
                    help="judge 模型路径（缺省取 base.yaml judges.qib.judge_model）")
    ap.add_argument("--judge-python", default=sys.executable,
                    help="运行 judge.py 的 python（默认当前解释器，slurm 中已 activate qwenimagebench）")
    ap.add_argument("--max-batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source == "released" and not args.released_model:
        ap.error("--source released 需要 --released-model")
    if args.source == "run" and not args.benchmark:
        ap.error("--source run 需要 --benchmark")

    if args.merge:
        merge_shards(args, cfg)
    else:
        run_judge(args, cfg)


if __name__ == "__main__":
    main()
