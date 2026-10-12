"""Generate captions for the Perception Rubric benchmark with an OpenAI-compatible vision model.

Reads images from a Hugging Face dataset (column ``image``, expected to be a
PIL Image) and writes one prediction per line as JSONL:

    {"id": "...", "prediction": "...", "reasoning_content": "..."}

Environment variables ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` are honored.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from perception_rubric.io import write_jsonl


CAPTION_PROMPT = "Describe the image in detail."


def encode_image(image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def caption_one(client, model: str, image_data_uri: str, max_tokens: int,
                 temperature: float = 0.0, top_p: float = None, top_k: int = None,
                 reasoning_effort: str = None):
    """Call the model and return (content, reasoning_content, usage_dict)."""
    kwargs = dict(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p
    extra_body = {}
    if top_k is not None:
        extra_body["top_k"] = top_k
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = reasoning_effort
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    # Extract reasoning_content if present (reasoning/thinking models)
    reasoning_content = getattr(choice.message, "reasoning_content", None) or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    } if response.usage else {}
    return content, reasoning_content, usage


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo id, e.g. M1chaelPeng/PerceptionRubrics.")
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument("--model", default=os.getenv("PREDICT_MODEL"), help="Vision model name. Defaults to $PREDICT_MODEL.")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory (default: outputs). A model subdirectory is created automatically.")
    parser.add_argument("--output", default=None, help="(Legacy) explicit output JSONL path. Overrides --output-dir structure.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (default: None, i.e. API default).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling via extra_body (default: None).")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"], help="Reasoning effort for the model.")
    parser.add_argument("--limit", type=int, help="Caption only the first N rows.")
    parser.add_argument("--retries", type=int, default=5, help="Max retries per API call (default: 5).")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1).")
    parser.add_argument("--image-column", default="image", help="Dataset column holding the PIL Image (default: image).")
    parser.add_argument("--id-column", default="id", help="Dataset column holding the sample id (default: id).")
    return parser


def write_meta(meta_path: Path, args, num_samples: int, num_failures: int, elapsed: float):
    """Write evaluation meta file with hyperparameters and run info."""
    meta = {
        "task": "predict",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "reasoning_effort": args.reasoning_effort,
        "workers": args.workers,
        "limit": args.limit,
        "image_column": args.image_column,
        "id_column": args.id_column,
        "num_samples": num_samples,
        "num_failures": num_failures,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": CAPTION_PROMPT,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.model:
        print("--model or PREDICT_MODEL is required", file=sys.stderr)
        return 2

    try:
        from perception_rubric.data import load_eval_dataset
    except ImportError:
        print("Install extras with `pip install \".[examples]\"` to use this script.", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI
    except ImportError:
        print("Install OpenAI SDK with `pip install openai`.", file=sys.stderr)
        return 2
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kwargs):  # type: ignore[no-redef]
            return x

    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    dataset = load_eval_dataset(args.dataset, args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Determine output path
    model_dir_name = args.model.replace("/", "_")
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.output_dir) / model_dir_name / "predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume support: load existing results ---
    existing_ids: set = set()
    if output_path.exists():
        from perception_rubric.io import load_jsonl
        for record in load_jsonl(output_path):
            existing_ids.add(str(record["id"]))
        if existing_ids:
            print(f"Resuming: {len(existing_ids)} predictions already exist in {output_path}")

    all_rows = list(dataset)
    pending_rows = [r for r in all_rows if str(r[args.id_column]) not in existing_ids]
    print(f"Total: {len(all_rows)}, Already done: {len(existing_ids)}, Pending: {len(pending_rows)}")

    if not pending_rows:
        print("Nothing to do — all predictions already exist.")
        meta_path = output_path.parent / "predict_meta.json"
        write_meta(meta_path, args, num_samples=len(all_rows), num_failures=0, elapsed=0.0)
        print(f"Meta: {meta_path}")
        return 0

    success_count = 0
    fail_count = 0
    start_time = time.time()
    import threading
    write_lock = threading.Lock()

    def predict_one(row):
        sample_id = str(row[args.id_column])
        data_uri = encode_image(row[args.image_column])
        last_error = None
        for attempt in range(args.retries):
            try:
                content, reasoning_content, usage = caption_one(
                    client, args.model, data_uri, args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, reasoning_effort=args.reasoning_effort,
                )
                record = {"id": sample_id, "prediction": content}
                if reasoning_content:
                    record["reasoning_content"] = reasoning_content
                if usage:
                    record["usage"] = usage
                return record
            except Exception as exc:
                last_error = str(exc)
                if attempt < args.retries - 1:
                    time.sleep(min(30.0, 2.0 * (2 ** attempt)))
        return {"id": sample_id, "prediction": "", "error": last_error}

    # Stream results to file as they complete (append mode for resume)
    out_file = open(output_path, "a", encoding="utf-8")

    try:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_idx = {executor.submit(predict_one, row): idx for idx, row in enumerate(pending_rows)}
                pbar = tqdm(as_completed(future_to_idx), total=len(pending_rows), desc="predict")
                for future in pbar:
                    record = future.result()
                    with write_lock:
                        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_file.flush()
                    if record.get("error"):
                        fail_count += 1
                    else:
                        success_count += 1
                    pbar.set_postfix(ok=success_count, fail=fail_count)
        else:
            pbar = tqdm(pending_rows, desc="predict")
            for row in pbar:
                record = predict_one(row)
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                if record.get("error"):
                    fail_count += 1
                else:
                    success_count += 1
                pbar.set_postfix(ok=success_count, fail=fail_count)
    finally:
        out_file.close()

    elapsed = time.time() - start_time
    total_samples = len(existing_ids) + success_count + fail_count
    total_failures = fail_count
    print(f"\nWrote {success_count + fail_count} new rows to {output_path} (success={success_count}, fail={fail_count})")
    print(f"Total predictions: {total_samples} ({total_failures} failures, elapsed={elapsed:.1f}s)")

    # Write meta file alongside predictions
    meta_path = output_path.parent / "predict_meta.json"
    write_meta(meta_path, args, num_samples=total_samples, num_failures=total_failures, elapsed=elapsed)
    print(f"Meta: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
