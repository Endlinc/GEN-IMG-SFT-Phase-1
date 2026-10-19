"""Reproduce CAPTURE evaluation on arbitrary MLLM-generated captions.

Protocol (arXiv:2405.19092):
  - Generate one candidate caption per image in DetailCaps-4870.
  - Use the official `capture_metric` package to compute object/attribute/relation F1
    against the provided reference captions.
  - Report the overall CAPTURE score.

Two caption sources:
  default                    call the API and caption the images
  --captions-file PATH       score captions generated earlier (canonical JSONL; see
                             caption_io.py), skipping the API entirely

Install the metric first:
    pip install capture_metric
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from caption_io import (add_caption_args, count_errors, load_injected,
                        resolve_prompt, save_captions)
from captioners import build_captioner
from datasets import iter_items, load_samples, sample_key
from PIL import Image  # noqa: F401  (kept: datasets.to_array handles path strings)

import numpy as np


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    data_dir = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --subset overrides the full 4870-image set with a pre-sampled JSON
    samples = load_samples("capture", data_dir, subset=args.subset)
    if args.limit:
        samples = samples[: args.limit]

    refs: Dict[str, List[str]] = {}
    ids: List[str] = []
    for i, sample in enumerate(samples):
        key = sample_key("capture", sample, i)
        ids.append(key)
        refs[key] = sample.get("references", [])

    captions_path = out_path.with_suffix(".captions.jsonl")

    # --- caption source: injected file, or the API ---
    injected = load_injected(args)
    if injected is not None:
        missing = [k for k in ids if k not in injected]
        if missing:
            raise SystemExit(
                f"[CAPTURE] --captions-file covers {len(ids) - len(missing)}/{len(ids)} "
                f"images; {len(missing)} missing (first: {missing[:3]})")
        captions = {k: injected[k] for k in ids}
    else:
        captioner = build_captioner(args.captioner, device=args.device,
                                    prompt_override=resolve_prompt(args),
                                    max_tokens=args.max_tokens)
        from captioners import caption_many
        items = list(iter_items("capture", data_dir, subset=args.subset, limit=args.limit))
        cap_map = caption_many(captioner, items, max_workers=args.workers, label="CAPTURE")
        captions = {k: cap_map.get(k, "") for k, _ in items}

    # Always leave a canonical caption file behind, whichever source it came from,
    # so a run directory is self-describing and re-scorable without the original input.
    n_err = count_errors(captions)
    save_captions(captions_path, captions,
                  meta={"benchmark": "CAPTURE", "captioner": args.captioner,
                        "n_images": len(captions), "n_errors": n_err,
                        "source": args.captions_file or "generated"})
    print(f"[CAPTURE] captions saved: {len(captions)} ({n_err} errors) -> "
          f"{captions_path}", flush=True)

    preds: Dict[str, List[str]] = {k: [captions[k]] for k in ids}
    per_image: List[Dict[str, Any]] = [
        {"image_id": k, "caption": captions[k], "references": refs[k]} for k in ids
    ]

    if args.caption_only:
        print("[CAPTURE] --caption-only set; skipping local CAPTURE scoring.", flush=True)
        return 0

    # run official CAPTURE metric
    try:
        from capture_metric.capture import CAPTURE
    except ImportError as e:
        raise ImportError("`capture_metric` not installed. Run: pip install capture_metric") from e

    print(f"[CAPTURE] scoring {len(preds)} captions ...", flush=True)
    evaluator = CAPTURE()
    score = evaluator.compute_score(refs, preds)

    def _to_py(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (list, tuple)):
            return [_to_py(v) for v in x]
        if isinstance(x, dict):
            return {k: _to_py(v) for k, v in x.items()}
        return x

    summary = {
        "benchmark": "CAPTURE",
        "dataset": "DetailCaps-4870",
        "captioner": args.captioner,
        "n_images": len(preds),
        "captions_file": args.captions_file or str(captions_path),
        "capture_score": _to_py(float(score) if not isinstance(score, (list, tuple)) else score[0]),
        "raw_score": _to_py(score),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # also save per-image details
    with open(out_path.with_suffix(".per_image.jsonl"), "w", encoding="utf-8") as f:
        for row in per_image:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nCAPTURE score: {summary['capture_score']:.4f}")
    print(f"summary -> {out_path}")
    print(f"[CAPTURE_DONE] elapsed={time.time() - t0:.1f}s", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reproduce CAPTURE with an MLLM captioner")
    p.add_argument("--data", required=True, help="DetailCaps data directory (contains detailcaps_4870.json)")
    p.add_argument("--captioner", required=True, help="captioner name (see captioners.py)")
    p.add_argument("--out", required=True, help="output .json summary path")
    p.add_argument("--subset", default="", help="path to pre-sampled subset JSON (overrides full dataset)")
    p.add_argument("--device", default="cuda", help="torch device for local models")
    p.add_argument("--limit", type=int, default=0, help="evaluate first N images (0=all)")
    p.add_argument("--workers", type=int, default=8, help="concurrent caption API workers")
    p.add_argument("--caption-only", action="store_true",
                   help="only generate + save captions, skip local scoring")
    add_caption_args(p)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
