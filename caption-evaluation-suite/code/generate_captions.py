"""Generate captions for any of the four benchmarks, independent of scoring.

This is the seam that decouples caption generation from evaluation. It loads a
benchmark's image set through `datasets.iter_items()` -- the same loader the
runner uses -- and writes the canonical caption format from `caption_io`, so the
file it produces is directly consumable by every runner:

    python generate_captions.py --benchmark capture \\
        --data ~/capeval-data/DetailCaps-4870 \\
        --captioner gpt-5.6-sol \\
        --prompt-file arms/p3_exhaustive.txt \\
        --workers 16 --out runs/gpt/p3_exhaustive/capture/captions.jsonl

    python run_capture.py --data ~/capeval-data/DetailCaps-4870 \\
        --captioner gpt-5.6-sol \\
        --captions-file runs/gpt/p3_exhaustive/capture/captions.jsonl \\
        --out runs/gpt/p3_exhaustive/capture/score.json

Generation is the only step that costs API calls; scoring can then be re-run
freely against the frozen captions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

from caption_io import (add_prompt_args, count_errors, is_error, load_captions,
                        resolve_prompt, save_captions)
from captioners import build_captioner, caption_many
from datasets import BENCHMARKS, iter_items


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    out_path = Path(args.out)

    prompt = resolve_prompt(args)
    captioner = build_captioner(args.captioner, device=args.device,
                                prompt_override=prompt, max_tokens=args.max_tokens)
    system_prompt = getattr(captioner, "system", "")

    print(f"[gen] benchmark={args.benchmark} captioner={args.captioner} "
          f"max_tokens={args.max_tokens}", flush=True)
    print(f"[gen] system prompt: {system_prompt[:120]!r}"
          f"{'...' if len(system_prompt) > 120 else ''}", flush=True)

    items = list(iter_items(args.benchmark, Path(args.data),
                            subset=args.subset, limit=args.limit))
    if not items:
        raise SystemExit(f"[gen] no images found for {args.benchmark} under {args.data}")
    print(f"[gen] {len(items)} images to caption", flush=True)

    # --resume: keep captions already on disk, redo only the missing/failed ones.
    # On a flaky endpoint this turns a partial failure into a cheap top-up pass
    # instead of a full regeneration.
    previous: Dict[str, str] = {}
    if args.resume and out_path.exists():
        try:
            previous = load_captions(out_path)
        except Exception as e:  # noqa: BLE001 - a bad existing file must not block a fresh run
            print(f"[gen] --resume: cannot read {out_path} ({e}); starting fresh", flush=True)
            previous = {}
    todo = [(k, img) for k, img in items
            if not previous.get(k) or is_error(previous[k])]
    if previous:
        print(f"[gen] --resume: {len(items) - len(todo)}/{len(items)} already captioned",
              flush=True)

    # Caption generation is the only step that costs API calls, and at full scale it
    # runs for hours -- long enough to outlive a batch job's wall-clock limit. Write
    # progress to disk as it arrives so a kill costs at most --flush-every images
    # instead of the whole run. Each checkpoint goes through a temp file and an
    # atomic rename: a kill mid-write leaves the last good file intact rather than a
    # truncated one, which `--resume` would otherwise choke on.
    flush_every = args.flush_every  # <= 0 disables mid-run checkpointing

    def checkpoint() -> None:
        # Only the finished captions, in dataset order: the file has to grow as the
        # run progresses. Padding the unfinished ones with "" would make the row
        # count a constant, and a runner reading a half-written file would see every
        # key present rather than the missing ones it reports on.
        done = {k: previous[k] for k, _ in items if k in previous}
        # The ".tmp" suffix also keeps save_captions on its JSONL branch (only a
        # literal ".json" suffix selects the legacy object form).
        tmp = out_path.with_name(out_path.name + ".tmp")
        save_captions(tmp, done)
        os.replace(tmp, out_path)

    def on_result(key, caption, done, total):
        previous[key] = caption
        if flush_every > 0 and (done % flush_every == 0 or done == total):
            checkpoint()

    if todo:
        cap_map = caption_many(captioner, todo, max_workers=args.workers,
                               label=args.benchmark, on_result=on_result)
        for k, _ in todo:
            previous[k] = cap_map.get(k, "")
    captions = {k: previous.get(k, "") for k, _ in items}

    n_err = count_errors(captions)
    meta = {
        "benchmark": args.benchmark,
        "data": str(args.data),
        "subset": args.subset,
        "captioner": args.captioner,
        "system_prompt": system_prompt,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "max_tokens": args.max_tokens,
        "temperature": getattr(captioner, "temperature", None),
        "n_images": len(captions),
        "n_errors": n_err,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    save_captions(out_path, captions, meta=meta)

    print(f"[gen] wrote {len(captions)} captions ({n_err} errors) -> {out_path}",
          flush=True)
    print(f"[gen] meta -> {out_path.parent / (out_path.name + '.meta.json')}", flush=True)
    print(f"[GEN_DONE] elapsed={time.time() - t0:.1f}s errors={n_err}", flush=True)

    if n_err and not args.allow_errors:
        print(f"[gen] refusing to treat {n_err} failed captions as usable; "
              f"re-run to fill them, or pass --allow-errors", file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate canonical caption files for any benchmark")
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    p.add_argument("--data", required=True, help="benchmark data directory")
    p.add_argument("--captioner", required=True, help="captioner/model name (see captioners.py)")
    p.add_argument("--out", required=True, help="output caption path (.jsonl canonical)")
    p.add_argument("--subset", default="", help="pre-sampled subset JSON for this benchmark")
    p.add_argument("--device", default="cuda", help="torch device for local models")
    p.add_argument("--limit", type=int, default=0, help="caption only the first N images")
    p.add_argument("--workers", type=int, default=16, help="concurrent caption API workers")
    p.add_argument("--allow-errors", action="store_true",
                   help="exit 0 even when some captions failed")
    p.add_argument("--resume", action="store_true",
                   help="keep captions already in --out and only regenerate the "
                        "missing or failed ones")
    p.add_argument("--flush-every", type=int, default=50,
                   help="checkpoint --out every N finished captions, so a kill mid-run "
                        "costs at most N images instead of the whole run (default 50; "
                        "0 disables)")
    add_prompt_args(p)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
