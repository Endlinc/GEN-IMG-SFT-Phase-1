"""Reproduce PerceptionRubrics evaluation on arbitrary MLLM-generated captions.

Protocol (arXiv:2606.28322):
  - Generate one detailed caption per image.
  - An LLM-as-judge verifies every Must-Right and Easy-Wrong rubric as boolean.
  - Gated score: fail any Must-Right -> 0; else average Easy-Wrong accuracy.

Two caption sources:
  default                    call the API and caption the images
  --captions-file PATH       judge captions generated earlier (canonical JSONL; see
                             caption_io.py), skipping the API entirely

This script writes the predictions in the official format expected by the judge
    {"id": "...", "prediction": "..."}
to `<out>/predictions.jsonl`, plus the canonical caption file shared by all four
benchmarks to `<out>/captions.jsonl`.

Then it either:
  (a) runs the official `scripts/judge.py` from the PerceptionRubrics repo if available,
  (b) or prints the exact command to run it.

Install the official evaluation package first:
    git clone https://github.com/M1chaelPeng/PerceptionRubrics.git
    cd PerceptionRubrics
    pip install -e ".[examples]"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from caption_io import (add_caption_args, count_errors, load_injected,
                        resolve_prompt, save_captions)
from captioners import build_captioner
from datasets import iter_items, load_samples, sample_key

# PerceptionRubrics official predict.py uses this exact prompt
# (scripts/predict.py:CAPTION_PROMPT) -- it is this benchmark's default.
PR_PROMPT = "Describe the image in detail."


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --subset overrides full 1038-image set with pre-sampled JSON
    samples = load_samples("perceptionrubrics", data_dir, subset=args.subset)
    if args.limit:
        samples = samples[: args.limit]
    ids = [sample_key("perceptionrubrics", s, i) for i, s in enumerate(samples)]

    preds_path = out_dir / "predictions.jsonl"   # official judge input format
    captions_path = out_dir / "captions.jsonl"   # canonical, shared format

    # --- caption source: injected file, or the API ---
    injected = load_injected(args)
    if injected is not None:
        missing = [k for k in ids if k not in injected]
        if missing:
            raise SystemExit(
                f"[PR] --captions-file covers {len(ids) - len(missing)}/{len(ids)} "
                f"images; {len(missing)} missing (first: {missing[:3]})")
        captions: Dict[str, str] = {k: injected[k] for k in ids}
    else:
        captioner = build_captioner(args.captioner, device=args.device,
                                    prompt_override=resolve_prompt(args, default=PR_PROMPT),
                                    max_tokens=args.max_tokens)
        from captioners import caption_many
        items = list(iter_items("perceptionrubrics", data_dir, subset=args.subset,
                                limit=args.limit))
        cap_map = caption_many(captioner, items, max_workers=args.workers, label="PR")
        captions = {k: cap_map.get(k, "") for k, _ in items}

    n_err = count_errors(captions)
    save_captions(captions_path, captions,
                  meta={"benchmark": "PerceptionRubrics", "captioner": args.captioner,
                        "n_images": len(captions), "n_errors": n_err,
                        "source": args.captions_file or "generated"})
    records: List[Dict[str, str]] = [{"id": k, "prediction": captions[k]} for k in ids]
    with open(preds_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[PR] predictions saved: {len(records)} ({n_err} errors) -> {preds_path}",
          flush=True)
    print(f"[PR] captions -> {captions_path}", flush=True)

    if getattr(args, "caption_only", False):
        print("[PR] --caption-only set; skipping official judge.", flush=True)
        return 0

    # Try to run official judge if repo is provided / installed.
    if args.judge_repo:
        judge_script = Path(args.judge_repo) / "scripts" / "judge.py"
        if judge_script.exists():
            cmd = [
                sys.executable, str(judge_script),
                "--dataset", "M1chaelPeng/PerceptionRubrics",
                "--split", "test",
                "--predictions", str(preds_path),
                "--output", str(out_dir / "results.jsonl"),
                "--model", args.judge_model,
                "--workers", str(args.workers),
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            print("\nRunning official judge:\n  ", " ".join(cmd))
            env = os.environ.copy()
            env.setdefault("JUDGE_MODEL", args.judge_model)
            subprocess.run(cmd, env=env, check=True)
        else:
            print(f"\nWARNING: official judge script not found at {judge_script}")
            print_official_command(preds_path, out_dir, args)
    else:
        print_official_command(preds_path, out_dir, args)

    return 0


def print_official_command(preds_path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    print("\nTo run the official judge, clone the PerceptionRubrics repo and execute:")
    print("  git clone https://github.com/M1chaelPeng/PerceptionRubrics.git /path/to/PerceptionRubrics")
    print("  cd /path/to/PerceptionRubrics && pip install -e '.[examples]'")
    limit_flag = f" --limit {args.limit}" if args.limit else ""
    print(
        f"  python scripts/judge.py --dataset M1chaelPeng/PerceptionRubrics --split test "
        f"--predictions {preds_path} --output {out_dir / 'results.jsonl'} "
        f"--model {args.judge_model} --workers {args.workers}{limit_flag}"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reproduce PerceptionRubrics with an MLLM captioner")
    p.add_argument("--data", required=True, help="PerceptionRubrics data directory (contains perceptionrubrics.json)")
    p.add_argument("--captioner", required=True, help="captioner name (see captioners.py)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--subset", default="", help="path to pre-sampled subset JSON")
    p.add_argument("--judge-repo", default="", help="path to cloned PerceptionRubrics repo (optional)")
    # All delivered baselines were judged with gpt-oss-120b (see baselines/*/judge_meta.json).
    p.add_argument("--judge-model", default="gpt-oss-120b", help="OpenAI-compatible judge model")
    p.add_argument("--workers", type=int, default=4, help="parallel workers for official judge")
    p.add_argument("--device", default="cuda", help="torch device for local models")
    p.add_argument("--limit", type=int, default=0, help="evaluate first N images (0=all)")
    p.add_argument("--caption-only", action="store_true",
                   help="only generate + save predictions, skip official judge")
    add_caption_args(p)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
