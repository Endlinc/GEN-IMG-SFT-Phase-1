#!/usr/bin/env python3
"""使用 Qwen-Image 官方 Prompt Enhancer 改写 GenEval / DPG-Bench prompts。

依赖:
    pip install dashscope python-dotenv

环境变量优先级:
    1. 已导出到 shell 的 DASHSCOPE_API_KEY / DASH_API_KEY
    2. 项目根 .env 中的 QIANWEN_API_KEY（会被自动映射到对应环境变量）

示例:
    # dry-run: 不调用 API，仅展示前 3 条会如何处理
    python scripts/rewrite_prompts_qwenimage.py --dry-run --version 2512

    # 正式改写全部 benchmark（从项目根 .env 读取 QIANWEN_API_KEY）
    python scripts/rewrite_prompts_qwenimage.py --version 2512 --resume

成本估算 (qwen-plus, 华北 2):
    - original 版本: ~3.7 元 (553 + 1065 = 1618 条)
    - 2512 版本: ~7.6 元
    保守预算 5-15 元，具体取决于输出长度。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from common import load_config


# DashScope qwen-plus 价格（元 / 百万 tokens）
INPUT_PRICE_PER_1M = 0.8
OUTPUT_PRICE_PER_1M = 2.0


def approx_tokens(text):
    """粗略估算 token 数：按字符数 / 4，对中英混合较保守。"""
    return max(1, int(len(text) / 4))


def import_rewrite(qwen_image_root, version):
    """从 Qwen-Image 仓库导入 rewrite 函数。"""
    tools_dir = Path(qwen_image_root) / "src" / "examples" / "tools"
    if not tools_dir.exists():
        raise FileNotFoundError(f"Qwen-Image tools 目录不存在: {tools_dir}")

    sys.path.insert(0, str(tools_dir))
    try:
        if version == "2512":
            import prompt_utils_2512 as prompt_utils
        else:
            import prompt_utils
        return prompt_utils.rewrite
    finally:
        sys.path.pop(0)


def load_geneval_prompts(geneval_prompts_file):
    """读取 GenEval 官方 prompt 文件。

    格式: 每行一条 prompt，id 为 0-based 行号（与 evaluation_metadata.jsonl 顺序一致）。
    """
    prompts = []
    with open(geneval_prompts_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            prompts.append({"id": idx, "original": line})
    return prompts


def load_dpg_prompts(dpg_prompts_dir):
    """读取 DPG-Bench prompts 目录下的所有 txt 文件。

    每个文件名为 item_id，文件内容为一条 prompt。
    """
    prompts = []
    dpg_prompts_dir = Path(dpg_prompts_dir)
    for txt_file in sorted(dpg_prompts_dir.glob("*.txt")):
        item_id = txt_file.stem
        with open(txt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        prompts.append({"item_id": item_id, "original": prompt})
    return prompts


def load_done_ids(jsonl_path):
    """读取已有的 jsonl，返回已完成 id 集合。"""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("id") if "id" in rec else rec.get("item_id")
            if key is not None:
                done.add(key)
    return done


def estimate_cost(total_input_chars, total_output_chars, n_prompts, version):
    """基于字符数估算 token 与费用（包含每次请求的 system prompt）。"""
    # system prompt 是输入的主要成本来源
    system_prompt_tokens = 5000 if version == "2512" else 2000
    user_input_tokens = int(total_input_chars / 4)
    output_tokens = int(total_output_chars / 4)

    input_tokens = n_prompts * system_prompt_tokens + user_input_tokens
    input_cost = input_tokens * INPUT_PRICE_PER_1M / 1e6
    output_cost = output_tokens * OUTPUT_PRICE_PER_1M / 1e6
    return {
        "n_prompts": n_prompts,
        "system_prompt_tokens": system_prompt_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_yuan": input_cost,
        "output_cost_yuan": output_cost,
        "total_cost_yuan": input_cost + output_cost,
    }


def rewrite_prompts(prompts, rewrite_fn, out_jsonl, out_txt, benchmark, version="2512", resume=False, max_retries=5, sleep=0.5, dry_run=False):
    """逐条改写 prompts 并输出。"""
    out_jsonl = Path(out_jsonl)
    out_txt = Path(out_txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(out_jsonl) if resume else set()

    # 用于估算
    total_input_chars = 0
    total_output_chars = 0
    success = 0
    failed = 0

    mode = "a" if resume and done else "w"
    with open(out_jsonl, mode, encoding="utf-8") as fj:
        for i, rec in enumerate(prompts, 1):
            key = rec.get("id") if "id" in rec else rec.get("item_id")
            original = rec["original"]

            if key in done:
                print(f"[{benchmark}] {i}/{len(prompts)} 已存在，跳过 id={key}")
                continue

            print(f"[{benchmark}] {i}/{len(prompts)} 改写 id={key}: {original[:80]}...")

            if dry_run:
                enhanced = "[dry-run] " + original
                rec_out = {**rec, "enhanced": enhanced, "language": "en"}
                fj.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                success += 1
                continue

            # 实际调用 API
            enhanced = None
            for attempt in range(1, max_retries + 1):
                try:
                    enhanced = rewrite_fn(original)
                    break
                except Exception as e:
                    print(f"  重试 {attempt}/{max_retries}: {e}")
                    time.sleep(sleep * attempt)

            if enhanced is None:
                print(f"  失败 id={key}")
                failed += 1
                continue

            enhanced = enhanced.strip().replace("\n", " ")

            # 语言判断复用 Qwen-Image 逻辑：简单检测中文字符
            lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in original) else "en"

            rec_out = {**rec, "enhanced": enhanced, "language": lang}
            fj.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            fj.flush()

            total_input_chars += len(original)
            total_output_chars += len(enhanced)
            success += 1

            if sleep > 0:
                time.sleep(sleep)

    # 生成 .txt 纯文本文件
    if not dry_run and out_jsonl.exists():
        write_txt_from_jsonl(out_jsonl, out_txt, benchmark)

    cost = estimate_cost(total_input_chars, total_output_chars, success, version=version)
    print(f"\n[{benchmark}] 完成: 成功 {success}, 失败 {failed}, 跳过 {len(done)}")
    print(f"  单次 system prompt 约 {cost['system_prompt_tokens']} tokens")
    print(f"  估算输入 tokens: {cost['input_tokens']}, 输出 tokens: {cost['output_tokens']}")
    print(f"  估算费用: {cost['total_cost_yuan']:.2f} 元")
    return success, failed


def write_txt_from_jsonl(jsonl_path, txt_path, benchmark):
    """从 jsonl 生成与官方格式对齐的 txt 文件。"""
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    with open(txt_path, "w", encoding="utf-8") as f:
        for rec in records:
            if benchmark == "geneval":
                f.write(f"{rec['id']}\t{rec['enhanced']}\n")
            else:
                # DPG-Bench: item_id\nprompt\n 的交替格式
                f.write(f"{rec['item_id']}\n{rec['enhanced']}\n")
    print(f"  已生成 {txt_path}")


def main():
    # 以项目根为基准定位 Qwen-Image repo / benchmark prompts / 输出目录
    cfg = load_config(None)   # 只加载 base.yaml（paths + benchmarks 注册表）
    models_root = cfg["paths"]["models_root"]
    benchmarks_root = cfg["paths"]["benchmarks_root"]
    default_qwen_root = os.path.join(models_root, "Qwen-Image")
    default_out_dir = os.path.join(cfg["paths"]["output_root"], "qwen_prompt_enhance")
    default_env_file = os.path.join(os.path.dirname(models_root), ".env")  # 项目根/.env

    ap = argparse.ArgumentParser(description="使用 Qwen-Image Prompt Enhancer 改写 benchmark prompts")
    ap.add_argument("--qwen-image-root", default=str(default_qwen_root),
                    help="Qwen-Image 仓库路径")
    ap.add_argument("--benchmark", choices=["geneval", "dpg", "all"], default="all",
                    help="要改写的 benchmark")
    ap.add_argument("--out-dir", default=str(default_out_dir),
                    help="输出根目录")
    ap.add_argument("--version", choices=["original", "2512"], default="2512",
                    help="使用 Qwen-Image 的 prompt_utils 版本")
    ap.add_argument("--env-file", default=str(default_env_file),
                    help=".env 文件路径，用于读取 QIANWEN_API_KEY")
    ap.add_argument("--resume", action="store_true",
                    help="断点续传，跳过已存在 id")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="单条失败重试次数")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="请求间隔秒数")
    ap.add_argument("--limit", type=int, default=None,
                    help="仅处理前 N 条 prompt（用于小批量测试）")
    ap.add_argument("--dry-run", action="store_true",
                    help="不调用 API，仅写入前 3 条示例用于检查")
    args = ap.parse_args()

    # 统一使用绝对路径，避免 cwd 影响
    qwen_root = Path(args.qwen_image_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    env_file = Path(args.env_file).resolve()

    # 加载 .env：项目根 .env 中的 QIANWEN_API_KEY，需要映射到 prompt_utils 所需变量
    if load_dotenv and env_file.exists():
        load_dotenv(env_file, override=False)

    required_env = "DASH_API_KEY" if args.version == "2512" else "DASHSCOPE_API_KEY"
    if not os.environ.get(required_env):
        qianwen_key = os.environ.get("QIANWEN_API_KEY")
        if qianwen_key:
            os.environ[required_env] = qianwen_key
            print(f"已从 {env_file} 读取 QIANWEN_API_KEY 并映射到 {required_env}")

    # 导入 rewrite 函数
    rewrite_fn = import_rewrite(qwen_root, args.version)

    # 检查环境变量
    if not os.environ.get(required_env) and not args.dry_run:
        print(f"错误: 未找到可用的 API Key。请在 {env_file} 中设置 QIANWEN_API_KEY，")
        print(f"或手动导出环境变量 {required_env}")
        sys.exit(1)

    geneval_prompts_file = Path(benchmarks_root) / "geneval" / "prompts" / "generation_prompts.txt"
    dpg_prompts_dir = cfg["benchmarks"]["dpg"]["prompts"]

    benchmarks = []
    if args.benchmark in ("geneval", "all"):
        benchmarks.append(("geneval", geneval_prompts_file))
    if args.benchmark in ("dpg", "all"):
        benchmarks.append(("dpg", dpg_prompts_dir))

    total_success = 0
    total_failed = 0
    summary = {}

    for benchmark, src in benchmarks:
        print(f"\n========== 处理 {benchmark} ==========")
        if benchmark == "geneval":
            prompts = load_geneval_prompts(src)
        else:
            prompts = load_dpg_prompts(src)

        expected = len(prompts)
        if args.limit is not None:
            print(f"共读取 {expected} 条 prompts，本次处理前 {args.limit} 条")
            prompts = prompts[:args.limit]
            expected = min(expected, args.limit)
        else:
            print(f"共读取 {expected} 条 prompts")
        if args.dry_run:
            print("DRY-RUN 模式: 不调用 API")
            prompts = prompts[:3] if args.limit is None else prompts
            expected = len(prompts)

        out_jsonl = out_dir / benchmark / "enhanced_prompts.jsonl"
        out_txt = out_dir / benchmark / "enhanced_prompts.txt"

        s, f = rewrite_prompts(
            prompts, rewrite_fn, out_jsonl, out_txt, benchmark,
            version=args.version, resume=args.resume, max_retries=args.max_retries,
            sleep=args.sleep, dry_run=args.dry_run
        )
        total_success += s
        total_failed += f

        # 统计 jsonl 实际行数
        actual = 0
        if out_jsonl.exists():
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                actual = sum(1 for line in fh if line.strip())

        summary[benchmark] = {"expected": expected, "actual": actual, "success": s, "failed": f}

    # 最终汇总报告
    print("\n" + "=" * 50)
    print("最终统计")
    print("=" * 50)
    all_ok = True
    for benchmark, stats in summary.items():
        expected, actual, failed = stats["expected"], stats["actual"], stats["failed"]
        complete = (actual == expected and failed == 0)
        status = "✓ 完成" if complete else "✗ 不完整/有失败"
        print(f"  [{benchmark}] 预期 {expected} 条, 实际 {actual} 条, 失败 {failed} 条 -> {status}")
        if not complete:
            all_ok = False

    print(f"\n全部完成: 成功 {total_success}, 失败 {total_failed}")
    print(f"输出目录: {out_dir}")

    if not all_ok and not args.dry_run:
        print("\n警告: 部分 prompt 未成功改写，请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
