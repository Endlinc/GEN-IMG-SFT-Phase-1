"""Reproduce CompreCap evaluation on arbitrary MLLM-generated captions.

Protocol (arXiv:2412.08614):
  - Generate one detailed caption per CompreCap image.
  - Run the official `evaluate.py` from the CompreCap repo:
        python evaluate.py --data_root CompreCap_dataset/ --anno anno.json \
            --soft_coverage --llm PATH/to/Llama-3-8B-Instruct \
            --bert PATH/to/all-MiniLM-L6-v2 --longcap_file captions.json \
            --output-dir results/

Two caption sources:
  default                    call the API and caption the images
  --captions-file PATH       score captions generated earlier (canonical JSONL; see
                             caption_io.py), skipping the API entirely

Captions are always written twice: `<out>/captions.jsonl` in the canonical format
shared by all four benchmarks, and `<out>/captions.json` as the plain
`{image_id: caption}` object the vendored evaluator's `--longcap_file` expects.

This script generates the caption JSON, then either runs the official evaluator
(if the repo path is supplied) or prints the exact command.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

from PIL import Image  # noqa: F401  (kept: datasets.to_array handles path strings)

sys.path.insert(0, str(Path(__file__).parent))
from caption_io import (add_caption_args, count_errors, load_injected,
                        resolve_prompt, save_captions)
from captioners import build_captioner
from datasets import iter_items, load_comprecap_rows


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --anno-subset overrides the full anno.json with a pre-sampled subset
    rows = load_comprecap_rows(data_dir, args.anno_subset)
    if args.limit:
        rows = rows[: args.limit]
    ids = [row["image_id"] for row in rows]

    captions_jsonl = out_dir / "captions.jsonl"   # canonical, shared format
    captions_path = out_dir / "captions.json"     # legacy dict, for the evaluator

    # --- caption source: injected file, or the API ---
    injected = load_injected(args)
    if injected is not None:
        missing = [k for k in ids if k not in injected]
        if missing:
            raise SystemExit(
                f"[CompreCap] --captions-file covers {len(ids) - len(missing)}/{len(ids)} "
                f"images; {len(missing)} missing (first: {missing[:3]})")
        captions: Dict[str, str] = {k: injected[k] for k in ids}
    else:
        captioner = build_captioner(args.captioner, device=args.device,
                                    prompt_override=resolve_prompt(args),
                                    max_tokens=args.max_tokens)
        from captioners import caption_many
        items = list(iter_items("comprecap", data_dir, subset=args.anno_subset,
                                limit=args.limit))
        cap_map = caption_many(captioner, items, max_workers=args.workers, label="CompreCap")
        captions = {k: cap_map.get(k, "") for k, _ in items}

    n_err = count_errors(captions)
    meta = {"benchmark": "CompreCap", "captioner": args.captioner,
            "n_images": len(captions), "n_errors": n_err,
            "source": args.captions_file or "generated"}
    save_captions(captions_jsonl, captions, meta=meta)
    save_captions(captions_path, captions)  # plain dict for --longcap_file
    print(f"[CompreCap] captions saved: {len(captions)} ({n_err} errors) -> "
          f"{captions_jsonl} + {captions_path}", flush=True)

    if getattr(args, "caption_only", False):
        print("[CompreCap] --caption-only set; skipping official evaluation.", flush=True)
        return 0

    if args.comprecap_repo:
        if not args.llm_path:
            raise ValueError("--llm-path (or COMPRECAP_LLM_PATH env var) is required "
                             "when running the official CompreCap evaluator.")
        if not args.bert_path:
            raise ValueError("--bert-path (or COMPRECAP_BERT_PATH env var) is required "
                             "when running the official CompreCap evaluator.")
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

    print(f"[COMPRECAP_DONE] elapsed={time.time() - t0:.1f}s", flush=True)
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
    add_caption_args(p)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
