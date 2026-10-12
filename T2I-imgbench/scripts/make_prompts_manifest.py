"""生成 Benchmarks/prompts_manifest.json：对每个 prompt 文件计算 sha256（协议 §1）。

覆盖:
- geneval:   prompts/evaluation_metadata.jsonl（553 条）
- dpg_bench: dpg_bench/prompts/*.txt（1065 个单文件，逐个 sha256）+ dpg_bench.csv（命题依赖）
- longtext:  textbench/text_prompts.jsonl / text_prompts_zh.jsonl（各 160 条）
- qwenimagebench: Qwen-Image-Bench/qwen_image_bench_hf_v0518.jsonl（1000 条双语 prompt）

路径均取自 configs/base.yaml 的 benchmarks 注册表；manifest 中的根路径写相对
（"Benchmarks"），作为数据准备完成后的校验/溯源产物。

用法: python scripts/make_prompts_manifest.py
产物: Benchmarks/prompts_manifest.json（随 manifest 重新生成而刷新）
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

from common import PROJECT_ROOT, load_config


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def file_entry(path, n_records=None):
    e = {"sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    if n_records is not None:
        e["n_records"] = n_records
    return e


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    cfg = load_config(None)   # 只加载 base.yaml（注册表 + 路径）
    root = cfg["paths"]["benchmarks_root"]
    out = os.path.join(root, "prompts_manifest.json")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "benchmarks_root": os.path.relpath(root, PROJECT_ROOT),   # 相对项目根写 "Benchmarks"
        "benchmarks": {},
    }

    # --- GenEval ---
    repo = cfg["benchmarks"]["geneval"]["repo"]
    meta = cfg["benchmarks"]["geneval"]["prompts"]
    manifest["benchmarks"]["geneval"] = {
        "repo_path": os.path.relpath(repo, root),
        "commit": git_commit(repo),
        "files": {"prompts/evaluation_metadata.jsonl": file_entry(meta, count_lines(meta))},
    }

    # --- DPG-Bench ---
    repo = cfg["benchmarks"]["dpg"]["repo"]
    pdir = cfg["benchmarks"]["dpg"]["prompts"]
    per_file = {}
    for fn in sorted(os.listdir(pdir)):
        if fn.endswith(".txt"):
            per_file[f"dpg_bench/prompts/{fn}"] = sha256_file(os.path.join(pdir, fn))
    # 目录聚合 hash:对 "filename:sha256" 排序拼接后再哈希,便于快速整体校验
    agg = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in per_file.items()).encode()).hexdigest()
    csv_path = os.path.join(repo, "dpg_bench", "dpg_bench.csv")
    manifest["benchmarks"]["dpg_bench"] = {
        "repo_path": os.path.relpath(repo, root),
        "commit": git_commit(repo),
        "files": {
            "dpg_bench/dpg_bench.csv": file_entry(csv_path, count_lines(csv_path) - 1),
            "dpg_bench/prompts": {
                "n_files": len(per_file),
                "aggregate_sha256": agg,
                "per_file_sha256": per_file,
            },
        },
    }

    # --- LongText-Bench ---
    repo = cfg["benchmarks"]["longtext_en"]["repo"]
    files = {}
    for key in ("longtext_en", "longtext_zh"):
        p = cfg["benchmarks"][key]["prompts"]
        files[os.path.relpath(p, repo)] = file_entry(p, count_lines(p))
    manifest["benchmarks"]["longtext_bench"] = {
        "repo_path": os.path.relpath(repo, root),
        "commit": git_commit(repo),
        "files": files,
    }

    # --- Qwen-Image-Bench ---
    jsonl = cfg["benchmarks"]["qwenimagebench_cn"]["prompts"]
    repo = cfg["benchmarks"]["qwenimagebench_cn"]["repo"]
    manifest["benchmarks"]["qwenimagebench"] = {
        "repo_path": os.path.relpath(repo, root),
        "commit": git_commit(repo),
        "files": {"qwen_image_bench_hf_v0518.jsonl": file_entry(jsonl, count_lines(jsonl))},
    }

    with open(out, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"written: {out}")
    for bench, b in manifest["benchmarks"].items():
        print(f"  {bench}: commit={str(b['commit'])[:7]}")


if __name__ == "__main__":
    main()
