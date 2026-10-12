"""Reproduce PerceptionRubrics evaluation on arbitrary MLLM-generated captions.

Protocol (arXiv:2606.28322):
  - Generate one detailed caption per image.
  - An LLM-as-judge verifies every Must-Right and Easy-Wrong rubric as boolean.
  - Gated score: fail any Must-Right -> 0; else average Easy-Wrong accuracy.

This script generates predictions in the official format:
    {"id": "...", "prediction": "...", "reasoning_content": "..."}
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
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from captioners import build_captioner


def load_perceptionrubrics(data_dir: Path) -> List[Dict[str, Any]]:
    path = data_dir / "perceptionrubrics.json"
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --subset overrides full 1038-image set with pre-sampled JSON
    if args.subset:
        with open(args.subset, encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = load_perceptionrubrics(data_dir)
    # PerceptionRubrics official predict.py uses this exact prompt (scripts/predict.py:CAPTION_PROMPT)
    captioner = build_captioner(args.captioner, device=args.device,
                                prompt_override="Describe the image in detail.")

    preds_path = out_dir / "predictions.jsonl"
    items = []
    for i, sample in enumerate(samples):
        if args.limit and i >= args.limit:
            break
        sid = str(sample.get("id", i))
        image = _image_to_array(sample.get("image"))
        items.append((sid, image))

    from captioners import caption_many
    cap_map = caption_many(captioner, items, max_workers=args.workers, label="PR")
    records = [{"id": sid, "prediction": cap_map.get(sid, "")} for sid, _ in items]

    with open(preds_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_err = sum(1 for r in records if r["prediction"].startswith("__ERROR__"))
    print(f"[PR] predictions saved: {len(records)} ({n_err} errors) -> {preds_path}", flush=True)

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
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
