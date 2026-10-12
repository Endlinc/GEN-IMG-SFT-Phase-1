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
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from captioners import build_captioner


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
        "max_tokens": 8,
    }
    data = json.dumps(payload).encode("utf-8")

    import time
    votes: list[str] = []
    for _ in range(n_votes):
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    base_url.rstrip("/") + "/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
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
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    votes.append("n/a")  # default to n/a after all retries

    # majority vote
    from collections import Counter
    return Counter(votes).most_common(1)[0][0]


def load_capsbench(data_dir: Path) -> List[Dict[str, Any]]:
    path = data_dir / "capsbench.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_datasets.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_capsbench(data_dir)
    captioner = build_captioner(args.captioner, device=args.device)

    # Load captions from pre-generated file (list or dict) when in judge-only mode.
    pre_captions: Dict[str, str] | None = None
    if getattr(args, "captions_file", ""):
        cap_path = Path(args.captions_file)
        if cap_path.suffix == ".jsonl":
            pre_captions = {}
            with open(cap_path) as f:
                for line in f:
                    row = json.loads(line)
                    pre_captions[row["image_id"]] = row["caption"]
        elif cap_path.suffix == ".json":
            pre_captions = json.load(open(cap_path))
        else:
            raise ValueError(f"Unsupported captions format: {cap_path.suffix}")

    # caption-only mode: generate + save captions concurrently, skip judging
    if getattr(args, "caption_only", False):
        from captioners import caption_many
        items = []
        for i, sample in enumerate(samples):
            if args.limit and i >= args.limit:
                break
            image_id = sample["image_id"]
            image = sample["image"]
            if isinstance(image, str) and os.path.exists(image):
                pil = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                pil = image
            items.append((image_id, np.array(pil)))
        cap_map = caption_many(captioner, items, max_workers=args.workers, label="CapsBench")
        cap_path = Path(args.out).with_suffix(".captions.json")
        with open(cap_path, "w", encoding="utf-8") as f:
            json.dump({k: cap_map[k] for k, _ in items}, f, indent=2, ensure_ascii=False)
        n_err = sum(1 for v in cap_map.values() if v.startswith("__ERROR__"))
        print(f"\n[CapsBench] captions saved: {len(cap_map)} ({n_err} errors) -> {cap_path}", flush=True)
        return 0

    judge_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    judge_key = os.environ.get("OPENAI_API_KEY", "")
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
        if args.limit and i >= args.limit:
            break
        image_id = sample["image_id"]

        # Use pre-generated caption if available, otherwise generate
        if pre_captions and image_id in pre_captions:
            caption = pre_captions[image_id]
            print(f"[{i+1}/{len(samples)}] judging {image_id} (pre-captioned) ...", flush=True)
        else:
            image = sample["image"]
            if isinstance(image, str) and os.path.exists(image):
                pil = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                pil = image
            print(f"[{i+1}/{len(samples)}] captioning {image_id} ...", flush=True)
            caption = captioner.caption(np.array(pil))
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
    # Official CapsBench judge is Claude-3.5 Sonnet. It has been replaced by DeepSeek-V3.2
    # (comparable capability) because the original judge endpoint is no longer available.
    # All delivered baselines were produced with this judge.
    p.add_argument("--judge-model", default="deepseek-v3.2",
                   help="OpenAI-compatible judge for yes/no QA (official paper judge is "
                        "Claude-3.5 Sonnet, replaced by DeepSeek-V3.2 due to API availability)")
    p.add_argument("--workers", type=int, default=32,
                   help="parallel judge threads per image (default 32)")
    p.add_argument("--device", default="cuda", help="torch device for local models")
    p.add_argument("--limit", type=int, default=0, help="evaluate first N images (0=all)")
    p.add_argument("--caption-only", action="store_true",
                   help="only generate + save captions, skip judging")
    p.add_argument("--captions-file", default="",
                   help="path to pre-generated captions .json/.jsonl (judge-only mode)")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
