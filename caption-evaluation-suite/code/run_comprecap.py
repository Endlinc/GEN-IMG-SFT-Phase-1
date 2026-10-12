"""Reproduce CompreCap evaluation on arbitrary MLLM-generated captions.

Protocol (arXiv:2412.08614):
  - Generate one detailed caption per CompreCap image.
  - Run the official `evaluate.py` from the CompreCap repo:
        python evaluate.py --data_root CompreCap_dataset/ --anno anno.json \
            --soft_coverage --llm PATH/to/Llama-3-8B-Instruct \
            --bert PATH/to/all-MiniLM-L6-v2 --longcap_file captions.json \
            --output-dir results/

This script generates the caption JSON, then either runs the official evaluator
(if the repo path is supplied) or prints the exact command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from captioners import build_captioner


def load_comprecap_images(data_dir: Path) -> List[Dict[str, Any]]:
    """Return list of {image_id, image_path} from CompreCap_dataset/images."""
    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"CompreCap images dir not found: {images_dir}")
    rows = []
    for p in sorted(images_dir.glob("*.jpg")):
        rows.append({"image_id": p.name, "image_path": str(p)})
    return rows


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --anno-subset overrides the full anno.json with a pre-sampled subset
    if args.anno_subset:
        with open(args.anno_subset, encoding="utf-8") as f:
            anno_data = json.load(f)
        rows = [{"image_id": ex["image"], "image_path": str(data_dir / "images" / ex["image"])}
                for ex in anno_data]
    else:
        rows = load_comprecap_images(data_dir)
    captioner = build_captioner(args.captioner, device=args.device)

    items = []
    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        image_id = row["image_id"]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        items.append((image_id, image))

    from captioners import caption_many
    cap_map = caption_many(captioner, items, max_workers=args.workers, label="CompreCap")
    captions: Dict[str, str] = {k: cap_map.get(k, "") for k, _ in items}

    captions_path = out_dir / "captions.json"
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    n_err = sum(1 for v in captions.values() if v.startswith("__ERROR__"))
    print(f"[CompreCap] captions saved: {len(captions)} ({n_err} errors) -> {captions_path}", flush=True)

    if getattr(args, "caption_only", False):
        print("[CompreCap] --caption-only set; skipping official evaluation.", flush=True)
        return 0

    if args.comprecap_repo and not args.caption_only:
        if not args.llm_path:
            raise ValueError("--llm-path (or COMPRECAP_LLM_PATH env var) is required "
                             "when running the official CompreCap evaluator.")
        if not args.bert_path:
            raise ValueError("--bert-path (or COMPRECAP_BERT_PATH env var) is required "
                             "when running the official CompreCap evaluator.")

    # Run official CompreCap evaluate.py if repo path is given.
    if args.comprecap_repo:
        eval_script = Path(args.comprecap_repo) / "evaluate.py"
        if eval_script.exists():
            official_out = out_dir / "comprecap_official"
            official_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(eval_script),
                "--data_root", str(data_dir),
                "--anno", str(data_dir / "anno.json"),
                "--soft_coverage",
                "--llm", args.llm_path,
                "--bert", args.bert_path,
                "--longcap_file", str(captions_path),
                "--output-dir", str(official_out),
            ]
            print("\nRunning official CompreCap evaluator:\n  ", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            print(f"\nWARNING: official evaluate.py not found at {eval_script}")
            print_official_command(data_dir, captions_path, out_dir, args)
    else:
        print_official_command(data_dir, captions_path, out_dir, args)

    return 0


def print_official_command(data_dir: Path, captions_path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    print("\nTo run the official CompreCap evaluator, clone the repo and execute:")
    print("  git clone https://github.com/LuFan31/CompreCap.git /path/to/CompreCap")
    print("  cd /path/to/CompreCap && pip install -r requirements.txt && pip install spacy && python -m spacy download en_core_web_lg")
    print(
        f"  python evaluate.py --data_root {data_dir} --anno {data_dir / 'anno.json'} "
        f"--soft_coverage --llm {args.llm_path} --bert {args.bert_path} "
        f"--longcap_file {captions_path} --output-dir {out_dir / 'comprecap_official'}"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reproduce CompreCap with an MLLM captioner")
    p.add_argument("--data", required=True, help="CompreCap_dataset directory (contains images/ + anno.json)")
    p.add_argument("--captioner", required=True, help="captioner name (see captioners.py)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--anno-subset", default="", help="path to pre-sampled anno subset JSON")
    p.add_argument("--comprecap-repo", default="", help="path to cloned CompreCap repo (optional)")
    p.add_argument("--llm-path", default=os.environ.get("COMPRECAP_LLM_PATH", ""),
                   help="path to Llama-3.1-8B-Instruct (or compatible) for official CompreCap "
                        "evaluator. Can also be set via COMPRECAP_LLM_PATH env var. "
                        "Required when running official evaluation (--comprecap-repo).")
    p.add_argument("--bert-path", default=os.environ.get("COMPRECAP_BERT_PATH", ""),
                   help="path to sentence-transformers all-MiniLM-L6-v2 checkpoint. "
                        "Can also be set via COMPRECAP_BERT_PATH env var. "
                        "Required when running official evaluation (--comprecap-repo).")
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument("--limit", type=int, default=0, help="evaluate first N images (0=all)")
    p.add_argument("--workers", type=int, default=8, help="concurrent caption API workers")
    p.add_argument("--caption-only", action="store_true",
                   help="only generate + save captions, skip official evaluation")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
