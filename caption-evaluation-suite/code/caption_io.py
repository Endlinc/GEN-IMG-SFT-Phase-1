"""Canonical caption file format, shared by all four benchmarks and the generator.

A caption file is a JSONL stream, one object per line:

    {"image_id": "<dataset-native id>", "caption": "<text>"}

`image_id` is whatever key the benchmark's own dataset uses:

    CAPTURE            DetailCaps `image_id`      e.g. "detailcaps_00007"
    CompreCap          image filename             e.g. "000000287649.jpg"
    CapsBench          CapsBench `image_id`
    PerceptionRubrics  PerceptionRubrics `id`

So a single caption file can be handed to any runner via `--captions-file`, and
`generate_captions.py` writes exactly this format.

Two legacy dialects are still *read*, because the delivered baselines ship in them:

    {image_id: caption}         a plain JSON object   (baselines/capture, baselines/comprecap)
    {id, prediction}            JSONL rows            (baselines/perceptionrubrics, which is
                                                      also the official PR judge's input format)

Everything is *written* as canonical JSONL, so a file produced by any runner or by
`generate_captions.py` is directly consumable by all four.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# caption_many() records a failure as this sentinel rather than an empty string.
ERROR_PREFIX = "__ERROR__"


def is_error(caption: str) -> bool:
    return caption.startswith(ERROR_PREFIX)


def count_errors(captions: Dict[str, str]) -> int:
    return sum(1 for v in captions.values() if is_error(v))


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def _row_keys(row: Dict[str, Any], path: Path, lineno: int) -> tuple:
    """Pick the (id, caption) pair out of a JSONL row, accepting both dialects."""
    for id_key, cap_key in (("image_id", "caption"), ("id", "prediction")):
        if id_key in row and cap_key in row:
            return row[id_key], row[cap_key]
    raise ValueError(
        f"{path}:{lineno}: caption rows need 'image_id'+'caption' (canonical) or "
        f"'id'+'prediction' (PerceptionRubrics); got keys {sorted(row)}"
    )


def load_captions(path: str | os.PathLike) -> Dict[str, str]:
    """Read a caption file in canonical JSONL or either legacy dialect."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"captions file not found: {p}")

    if p.suffix == ".jsonl":
        captions: Dict[str, str] = {}
        with open(p, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                image_id, caption = _row_keys(json.loads(line), p, lineno)
                captions[str(image_id)] = caption
        return captions

    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{p}: legacy caption JSON must be an object mapping "
                f"image_id -> caption (got {type(data).__name__})"
            )
        return {str(k): v for k, v in data.items()}

    raise ValueError(
        f"unsupported captions format {p.suffix!r} (expected .jsonl or .json)"
    )


def save_captions(
    path: str | os.PathLike,
    captions: Dict[str, str],
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write captions as canonical JSONL, plus an optional `<path>.meta.json` sidecar.

    A `.json` path is honoured for callers that need the legacy object form
    (e.g. the vendored CompreCap evaluator's `--longcap_file`).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".json":
        with open(p, "w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
    else:
        with open(p, "w", encoding="utf-8") as f:
            for image_id, caption in captions.items():
                f.write(json.dumps({"image_id": image_id, "caption": caption},
                                   ensure_ascii=False) + "\n")

    if meta is not None:
        meta_path = p.parent / (p.name + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return p


# ---------------------------------------------------------------------------
# Shared CLI plumbing
# ---------------------------------------------------------------------------

def add_prompt_args(p: argparse.ArgumentParser) -> None:
    """Add the prompt/token flags shared by the runners and the generator."""
    p.add_argument("--prompt", default="",
                   help="override the captioner's system prompt (default: CAPTIONER_PROMPT, "
                        "or the benchmark's official prompt where one is defined)")
    p.add_argument("--prompt-file", default="",
                   help="read the system prompt from a file (takes precedence over --prompt)")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="max tokens for the generated caption (default 1024)")


def add_caption_args(p: argparse.ArgumentParser) -> None:
    """Add every caption-source flag a runner accepts: prompt flags + `--captions-file`.

    `--captions-file` switches the runner from "call the API" to "score these
    pre-generated captions".
    """
    add_prompt_args(p)
    p.add_argument("--captions-file", default="",
                   help="score pre-generated captions (canonical JSONL, or a legacy "
                        "{image_id: caption} JSON object) instead of calling the API")


def resolve_prompt(args: argparse.Namespace, default: Optional[str] = None) -> Optional[str]:
    """Return the system prompt to use, or `default` when neither flag was given."""
    prompt_file = getattr(args, "prompt_file", "")
    if prompt_file:
        with open(prompt_file, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError(f"--prompt-file {prompt_file} is empty")
        return text
    prompt = getattr(args, "prompt", "")
    return prompt or default


def load_injected(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    """Return pre-generated captions when `--captions-file` was given, else None."""
    path = getattr(args, "captions_file", "")
    if not path:
        return None
    captions = load_captions(path)
    print(f"[captions] loaded {len(captions)} pre-generated captions from {path}",
          flush=True)
    return captions


def missing_keys(captions: Dict[str, str], keys) -> list:
    """Keys the caption file does not cover -- surfaced instead of silently captioning."""
    return [k for k in keys if k not in captions]
