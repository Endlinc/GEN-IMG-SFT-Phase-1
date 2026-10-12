"""Reproduce CAPTURE evaluation on arbitrary MLLM-generated captions.

Protocol (arXiv:2405.19092):
  - Generate one candidate caption per image in DetailCaps-4870.
  - Use the official `capture_metric` package to compute object/attribute/relation F1
    against the provided reference captions.
  - Report the overall CAPTURE score.

Install the metric first:
    pip install capture_metric
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


def load_detailcaps(data_dir: Path) -> List[Dict[str, Any]]:
    path = data_dir / "detailcaps_4870.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_datasets.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _image_to_array(image: Any) -> np.ndarray:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.array(image)
    return image


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --subset overrides the full 4870-image set with a pre-sampled JSON
    if args.subset:
        with open(args.subset, encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = load_detailcaps(data_dir)
    captioner = build_captioner(args.captioner, device=args.device)

    refs: Dict[str, List[str]] = {}
    preds: Dict[str, List[str]] = {}
    per_image: List[Dict[str, Any]] = []

    items = []
    for i, sample in enumerate(samples):
        if args.limit and i >= args.limit:
            break
        key = str(sample.get("image_id", i))
        refs[key] = sample.get("references", [])
        image = _image_to_array(sample.get("image"))
        items.append((key, image))

    from captioners import caption_many
    cap_map = caption_many(captioner, items, max_workers=args.workers, label="CAPTURE")

    for key, _img in items:
        caption = cap_map.get(key, "")
        preds[key] = [caption]
        per_image.append({"image_id": key, "caption": caption, "references": refs[key]})

    # save captions for audit
    with open(out_path.with_suffix(".captions.json"), "w", encoding="utf-8") as f:
        json.dump({k: v[0] for k, v in preds.items()}, f, indent=2, ensure_ascii=False)

    # also save per-image details (captions + refs) so scoring can run later
    with open(out_path.with_suffix(".per_image.jsonl"), "w", encoding="utf-8") as f:
        for row in per_image:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_err = sum(1 for v in preds.values() if v[0].startswith("__ERROR__"))
    print(f"\n[CAPTURE] captions saved: {len(preds)} ({n_err} errors) -> "
          f"{out_path.with_suffix('.captions.json')}", flush=True)

    if args.caption_only:
        print("[CAPTURE] --caption-only set; skipping local CAPTURE scoring.", flush=True)
        return 0

    # run official CAPTURE metric
    try:
        from capture_metric.capture import CAPTURE
    except ImportError as e:
        raise ImportError("`capture_metric` not installed. Run: pip install capture_metric") from e

    evaluator = CAPTURE()
    score = evaluator.compute_score(refs, preds)

    def _to_py(x):
        import numpy as np
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
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
