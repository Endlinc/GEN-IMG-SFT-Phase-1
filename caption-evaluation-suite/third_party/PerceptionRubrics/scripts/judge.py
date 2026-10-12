"""Judge Perception Rubric predictions with an OpenAI-compatible chat model.

Reads rubrics from a Hugging Face dataset (or a local JSONL) and predictions
from a JSONL, calls the judge on each pair, and writes per-sample results plus
a summary.

Environment variables ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` are honored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from perception_rubric import OpenAICompatibleJudge, evaluate_caption
from perception_rubric.io import load_jsonl, load_predictions, write_jsonl


def iter_rubric_rows(args) -> Iterable[Dict[str, Any]]:
    if args.input:
        for row in load_jsonl(args.input):
            yield row
        return
    try:
        from perception_rubric.data import load_eval_dataset
    except ImportError:
        print("Install extras with `pip install \".[examples]\"` to load from a dataset.", file=sys.stderr)
        sys.exit(2)
    dataset = load_eval_dataset(args.dataset, args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    for row in dataset:
        yield {
            "id": row[args.id_column],
            "category": row.get(args.category_column, "unknown") if hasattr(row, "get") else row[args.category_column] if args.category_column in row else "unknown",
            "rubrics": {
                "must_right": row["must_right"],
                "easy_wrong": row["easy_wrong"],
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local JSONL file with id + rubrics.")
    source.add_argument("--dataset", help="Hugging Face dataset repo id, e.g. M1chaelPeng/PerceptionRubrics.")
    parser.add_argument("--split", default="test", help="Dataset split when --dataset is used (default: test).")
    parser.add_argument("--id-column", default="id", help="Dataset id column (default: id).")
    parser.add_argument("--category-column", default="category", help="Dataset category/domain column (default: category).")
    parser.add_argument("--predictions", required=True, help="JSONL of {id, prediction}.")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory (default: outputs). Model subdirectory is created automatically.")
    parser.add_argument("--output", default=None, help="(Legacy) explicit output JSONL path. Overrides --output-dir structure.")
    parser.add_argument("--predict-model", default=None, help="Name of the predict model (used for directory naming when --output is not set).")
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL"), help="Judge model name. Defaults to $JUDGE_MODEL.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (default: None, i.e. API default).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling via extra_body (default: None).")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"], help="Reasoning effort for the judge model.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1).")
    parser.add_argument("--limit", type=int, help="Judge only the first N rows.")
    return parser


def write_judge_meta(meta_path: Path, args, summary: Dict[str, Any], elapsed: float):
    """Write judge meta file with hyperparameters and run info."""
    meta = {
        "task": "judge",
        "judge_model": args.model,
        "predict_model": args.predict_model,
        "dataset": args.dataset,
        "split": args.split,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "reasoning_effort": args.reasoning_effort,
        "retries": args.retries,
        "workers": args.workers,
        "limit": args.limit,
        "predictions_file": str(args.predictions),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.model:
        print("--model or JUDGE_MODEL is required", file=sys.stderr)
        return 2

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kwargs):  # type: ignore[no-redef]
            return x

    predictions = load_predictions(args.predictions)
    judge = OpenAICompatibleJudge(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Use predict model name for directory (fall back to judge model)
        model_dir_name = (args.predict_model or args.model).replace("/", "_")
        output_path = Path(args.output_dir) / model_dir_name / "results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume support: load existing results ---
    existing_records: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        for record in load_jsonl(output_path):
            existing_records[str(record["id"])] = record
        if existing_records:
            print(f"Resuming: loaded {len(existing_records)} existing results from {output_path}")

    rows = list(iter_rubric_rows(args))
    if args.input and args.limit is not None:
        rows = rows[: args.limit]

    # Filter out already-judged rows
    pending_rows = [r for r in rows if str(r["id"]) not in existing_records]
    print(f"Total: {len(rows)}, Already done: {len(existing_records)}, Pending: {len(pending_rows)}")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    def judge_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(sample["id"])
        category = sample.get("category", "unknown")
        caption = predictions.get(sample_id)
        rubrics = sample.get("rubrics") or {
            "must_right": sample.get("must_right", []),
            "easy_wrong": sample.get("easy_wrong", []),
        }
        if not caption:
            return {"id": sample_id, "category": category, "score": 0.0, "correct": False,
                    "mr_pass": False, "ew_score": 0.0, "error": "missing prediction"}

        result = evaluate_caption(caption, rubrics, judge, retry_times=args.retries)
        mr_pass = bool(result.judge_result["must_right"] and all(result.judge_result["must_right"]))
        ew_score = (sum(result.judge_result["easy_wrong"]) / len(result.judge_result["easy_wrong"])) if result.judge_result["easy_wrong"] else 0.0

        record: Dict[str, Any] = {
            "id": sample_id,
            "category": category,
            "score": result.score,
            "correct": result.correct,
            "mr_pass": mr_pass,
            "ew_score": round(ew_score, 4),
            "judge_result": result.judge_result,
        }
        if result.error:
            record["error"] = result.error
        return record

    if pending_rows:
        if args.workers > 1:
            new_records = [None] * len(pending_rows)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_idx = {executor.submit(judge_one, row): idx for idx, row in enumerate(pending_rows)}
                pbar = tqdm(as_completed(future_to_idx), total=len(pending_rows), desc="judge")
                for future in pbar:
                    idx = future_to_idx[future]
                    record = future.result()
                    new_records[idx] = record
                    if record.get("error"):
                        fail_count += 1
                    else:
                        success_count += 1
                    pbar.set_postfix(ok=success_count, fail=fail_count)
        else:
            new_records = []
            pbar = tqdm(pending_rows, desc="judge")
            for sample in pbar:
                record = judge_one(sample)
                new_records.append(record)
                if record.get("error"):
                    fail_count += 1
                else:
                    success_count += 1
                pbar.set_postfix(ok=success_count, fail=fail_count)
    else:
        new_records = []
        print("Nothing to do — all samples already judged.")

    elapsed = time.time() - start_time
    print(f"\nNew results: {len(new_records)} (success={success_count}, fail={fail_count}, elapsed={elapsed:.1f}s)")

    # Merge existing + new records, maintaining row order
    existing_records.update({str(r["id"]): r for r in new_records if r is not None})
    all_records = [existing_records.get(str(r["id"]), {"id": str(r["id"]), "category": r.get("category", "unknown"), "score": 0.0, "error": "missing"}) for r in rows]

    # --- Compute summary (failures count as 0 score, included in total) ---
    total_score = 0.0
    counted = 0
    failures = 0
    mr_pass_count = 0
    ew_total = 0.0
    domain_stats: Dict[str, Dict[str, Any]] = {}

    for record in all_records:
        category = record.get("category", "unknown")
        counted += 1
        if record.get("error"):
            failures += 1
            # Failures count as 0 score but are included in the denominator
            score = 0.0
            mr_pass = False
            ew_score = 0.0
        else:
            score = record.get("score", 0.0)
            mr_pass = record.get("mr_pass", False)
            ew_score = record.get("ew_score", 0.0)

        total_score += score
        mr_pass_count += int(mr_pass)
        ew_total += ew_score

        if category not in domain_stats:
            domain_stats[category] = {"total_score": 0.0, "count": 0, "mr_pass": 0, "ew_sum": 0.0, "failures": 0}
        domain_stats[category]["total_score"] += score
        domain_stats[category]["count"] += 1
        domain_stats[category]["mr_pass"] += int(mr_pass)
        domain_stats[category]["ew_sum"] += ew_score
        if record.get("error"):
            domain_stats[category]["failures"] += 1

    write_jsonl(output_path, all_records)
    domain_scores = {}
    for cat, stats in sorted(domain_stats.items()):
        domain_scores[cat] = {
            "count": stats["count"],
            "failures": stats["failures"],
            "average_score": round(stats["total_score"] / stats["count"], 4) if stats["count"] else 0.0,
            "mr_pass_rate": round(stats["mr_pass"] / stats["count"], 4) if stats["count"] else 0.0,
            "ew_average": round(stats["ew_sum"] / stats["count"], 4) if stats["count"] else 0.0,
        }
    summary = {
        "count": counted,
        "success": counted - failures,
        "failures": failures,
        "average_score": round(total_score / counted, 4) if counted else 0.0,
        "mr_pass_rate": round(mr_pass_count / counted, 4) if counted else 0.0,
        "ew_average": round(ew_total / counted, 4) if counted else 0.0,
        "domain_scores": domain_scores,
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Write judge meta file
    meta_path = output_path.parent / "judge_meta.json"
    write_judge_meta(meta_path, args, summary, elapsed)
    print(f"Meta: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
