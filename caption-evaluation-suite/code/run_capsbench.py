"""Reproduce CapsBench evaluation on arbitrary MLLM-generated captions.

Protocol (arXiv:2409.10695, Sec 7):
  - Captioners: PG Captioner, GPT-4o, Claude-3.5 Sonnet.
    Caption generation uses detailed instructions, output schema, few-shot
    prompting (3 high-quality examples), and chain-of-thought.
  - For each image, generate a candidate detailed caption.
  - For each pre-generated yes/no question, an LLM reads the caption ONLY and
    answers yes / no / n/a.
  - Judge: Claude-3.5 Sonnet, run 3 times per question, majority vote.
  - Accuracy = fraction matching the reference answer.

Two caption sources:
  default                    call the API and caption the images
  --captions-file PATH       judge captions generated earlier (canonical JSONL; see
                             caption_io.py), skipping the API entirely
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from caption_io import (add_caption_args, count_errors, load_injected,
                        resolve_prompt, save_captions)
from captioners import build_captioner
from datasets import load_samples, sample_key


def _image_to_b64(image: Any) -> str:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _llm_judge(base_url: str, api_key: str, model: str, caption: str, question: str,
               n_votes: int = 3) -> str:
    """Call model n_votes times and return majority answer (paper: 3x consensus)."""
    prompt = (
        "You are given a detailed image caption and a yes/no question about the image. "
        "Answer the question based ONLY on the caption.\n\n"
        f"Caption: {caption}\n\n"
        f"Question: {question}\n\n"
        "Reply with exactly one word: yes, no, or n/a (if the caption does not contain enough information)."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You answer yes/no/n/a questions from captions."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        # NOT 8. A thinking judge (deepseek-v4-flash, and the doubao/seed family)
        # spends its whole budget on `reasoning_content` and returns an EMPTY
        # `content` with finish_reason=length -- which would silently score every
        # question "n/a" and collapse accuracy to ~0. 512 leaves room either way.
        "max_tokens": 512,
    }
    # DeepSeek's reasoning trace is pure overhead for a one-word answer.
    if model.lower().startswith("deepseek"):
        payload["thinking"] = {"type": "disabled"}
    data = json.dumps(payload).encode("utf-8")

    import time
    votes: list[str] = []
    for _ in range(n_votes):
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    base_url.rstrip("/") + "/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {api_key}"},
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                text = resp["choices"][0]["message"].get("content", "").strip().lower()
                if "yes" in text:
                    votes.append("yes")
                elif "no" in text:
                    votes.append("no")
                else:
                    votes.append("n/a")
                break
            # A 200 carrying an empty or non-JSON body is an upstream hiccup, not
            # a verdict. It has to be caught here: the caller reads every verdict
            # through `future.result()`, so one unparseable response among the
            # ~7k judge calls of a run raises out of the pool and discards the
            # whole run.
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError,
                    TimeoutError, json.JSONDecodeError, KeyError, IndexError) as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    votes.append("n/a")  # default to n/a after all retries

    # majority vote
    from collections import Counter
    return Counter(votes).most_common(1)[0][0]


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples("capsbench", data_dir)
    if args.limit:
        samples = samples[: args.limit]
    ids = [sample_key("capsbench", s, i) for i, s in enumerate(samples)]

    captions_path = Path(args.out).with_suffix(".captions.jsonl")

    # --- caption source: injected file, or the API (lazily, only if needed) ---
    injected = load_injected(args)
    captions: Dict[str, str] = {}
    if injected is not None:
        missing = [k for k in ids if k not in injected]
        if missing:
            raise SystemExit(
                f"[CapsBench] --captions-file covers {len(ids) - len(missing)}/{len(ids)} "
                f"images; {len(missing)} missing (first: {missing[:3]})")
        captions = {k: injected[k] for k in ids}
        save_captions(captions_path, captions,
                      meta={"benchmark": "CapsBench", "source": args.captions_file,
                            "n_images": len(captions),
                            "n_errors": count_errors(captions)})
        print(f"[CapsBench] using {len(captions)} pre-generated captions from "
              f"{args.captions_file}", flush=True)
    else:
        captioner = build_captioner(args.captioner, device=args.device,
                                    prompt_override=resolve_prompt(args),
                                    max_tokens=args.max_tokens)

    # caption-only mode: generate + save captions concurrently, skip judging
    if args.caption_only:
        if injected is None:
            from captioners import caption_many
            from datasets import iter_items
            items = list(iter_items("capsbench", data_dir, limit=args.limit))
            cap_map = caption_many(captioner, items, max_workers=args.workers, label="CapsBench")
            captions = {k: cap_map.get(k, "") for k, _ in items}
            save_captions(captions_path, captions,
                          meta={"benchmark": "CapsBench", "captioner": args.captioner,
                                "n_images": len(captions),
                                "n_errors": count_errors(captions)})
        n_err = count_errors(captions)
        print(f"\n[CapsBench] captions saved: {len(captions)} ({n_err} errors) -> "
              f"{captions_path}", flush=True)
        return 0

    judge_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # The judge is a different model family from the captioner, so it has its own
    # key. Fall back to OPENAI_API_KEY for single-key endpoints.
    judge_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    judge_model = args.judge_model

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def judge_question(q):
        question_text = q.get("question", "")
        ref_answer = q.get("answer", "yes").lower().strip()
        pred = _llm_judge(judge_url, judge_key, judge_model, caption_ref[0], question_text)
        return {
            "question": question_text,
            "category": q.get("category", ""),
            "reference": ref_answer,
            "prediction": pred,
            "correct": (pred == ref_answer),
        }

    results = []
    total_correct = 0
    total_questions = 0

    for i, sample in enumerate(samples):
        image_id = sample_key("capsbench", sample, i)

        # Use pre-generated caption if available, otherwise generate
        if injected is not None:
            caption = captions[image_id]
            print(f"[{i+1}/{len(samples)}] judging {image_id} (pre-captioned) ...", flush=True)
        else:
            from datasets import to_array
            print(f"[{i+1}/{len(samples)}] captioning {image_id} ...", flush=True)
            caption = captioner.caption(to_array(sample["image"]))
            captions[image_id] = caption
        caption_ref = [caption]  # mutable ref for closure

        questions = sample.get("questions", [])
        judged = [None] * len(questions)

        # parallel judge: all questions for this image concurrently
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_idx = {ex.submit(judge_question, q): idx
                             for idx, q in enumerate(questions)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                judged[idx] = future.result()

        row = {"image_id": image_id, "caption": caption, "questions": judged}
        n_correct = sum(1 for q in judged if q["correct"])
        total_correct += n_correct
        total_questions += len(judged)
        row["per_image_accuracy"] = n_correct / max(1, len(judged))
        results.append(row)

        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # keep the canonical caption file for every path that generated in-line
    if injected is None:
        save_captions(captions_path, captions,
                      meta={"benchmark": "CapsBench", "captioner": args.captioner,
                            "n_images": len(captions),
                            "n_errors": count_errors(captions)})
        print(f"[CapsBench] captions saved -> {captions_path}", flush=True)

    accuracy = total_correct / max(1, total_questions)
    summary = {
        "benchmark": "CapsBench",
        "captioner": args.captioner,
        "judge": judge_model,
        "n_images": len(results),
        "n_questions": total_questions,
        "accuracy": accuracy,
        "per_category": {},
    }
    # per-category accuracy
    cat_correct: Dict[str, int] = {}
    cat_total: Dict[str, int] = {}
    for r in results:
        for q in r["questions"]:
            c = q["category"] or "unknown"
            cat_correct[c] = cat_correct.get(c, 0) + int(q["correct"])
            cat_total[c] = cat_total.get(c, 0) + 1
    summary["per_category"] = {c: cat_correct[c] / cat_total[c] for c in cat_total}

    summary_path = Path(args.out).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nCapsBench accuracy: {accuracy:.4f} ({total_correct}/{total_questions})")
    print(f"results -> {args.out}")
    print(f"summary -> {summary_path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reproduce CapsBench with an MLLM captioner")
    p.add_argument("--data", required=True, help="CapsBench data directory (contains capsbench.json)")
    p.add_argument("--captioner", required=True, help="captioner name (see captioners.py)")
    p.add_argument("--out", required=True, help="output .jsonl path")
    # Official CapsBench judge is Claude-3.5 Sonnet. It has been replaced by DeepSeek
    # (comparable capability) because the original judge endpoint is no longer available.
    # deepseek-v3.2 has since been retired upstream; v4-flash is its successor there.
    p.add_argument("--judge-model", default="deepseek-v4-flash",
                   help="OpenAI-compatible judge for yes/no QA (official paper judge is "
                        "Claude-3.5 Sonnet, replaced by DeepSeek due to API availability)")
    p.add_argument("--workers", type=int, default=32,
                   help="parallel judge threads per image (default 32)")
    p.add_argument("--device", default="cuda", help="torch device for local models")
    p.add_argument("--limit", type=int, default=0, help="evaluate first N images (0=all)")
    p.add_argument("--caption-only", action="store_true",
                   help="only generate + save captions, skip judging")
    add_caption_args(p)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
