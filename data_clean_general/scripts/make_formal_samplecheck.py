#!/usr/bin/env python3
"""从各数据集 formal 的 images/ 随机抽 10%，写成 *_formal_samplecheck 版本。"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
FRACTION = 0.10
SEED = 42
SKIP_DIR_NAMES = {
    "checkpoint",
    ".cache",
    "__pycache__",
    "deleted_after_check",
    "images_after_check",
    "检查后删除",
    "检查后保留",
}

# (output_dir, source_run_name or None meaning output root, dest_run_name)
JOBS = [
    (PROJECT / "output_alchemist", "alchemist_formal", "alchemist_formal_samplecheck"),
    (
        PROJECT / "output_densefusion_4v_100k",
        "densefusion_4v_100k_formal",
        "densefusion_4v_100k_formal_samplecheck",
    ),
    (PROJECT / "output_wikiart", "wikiart_formal", "wikiart_formal_samplecheck"),
    (
        PROJECT / "output_diffusiondb_2m_random_50k",
        "diffusiondb_2m_random_50k_formal",
        "diffusiondb_2m_random_50k_formal_samplecheck",
    ),
    (PROJECT / "output_conceptual_captions", "conceptual_captions_formal", "conceptual_captions_formal_samplecheck"),
    (PROJECT / "output_finet2i", "finet2i_formal", "finet2i_formal_samplecheck"),
]


def list_image_names(images_dir: Path) -> list[str]:
    names: list[str] = []
    with os.scandir(images_dir) as it:
        for entry in it:
            suffix = Path(entry.name).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            names.append(entry.name)
    names.sort()
    return names


def find_image_groups(source_run: Path) -> list[tuple[str, Path]]:
    """Return (subset_or_empty, images_dir)."""
    images = source_run / "images"
    if images.is_dir():
        return [("", images)]
    groups: list[tuple[str, Path]] = []
    try:
        children = sorted(source_run.iterdir(), key=lambda p: p.name)
    except OSError:
        return []
    for child in children:
        if not child.is_dir() or child.name in SKIP_DIR_NAMES:
            continue
        if child.name.endswith("_samplecheck"):
            continue
        sub_images = child / "images"
        if sub_images.is_dir():
            groups.append((child.name, sub_images))
    return groups


def link_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return "exists"
    try:
        os.link(src, dest)
        return "link"
    except FileExistsError:
        return "exists"
    except OSError:
        try:
            shutil.copy2(src, dest)
            return "copy"
        except shutil.SameFileError:
            return "exists"


def copy_with_sidecars(src_image: Path, dest_dir: Path) -> None:
    link_or_copy(src_image, dest_dir / src_image.name)
    for ext in (".json", ".txt"):
        side = src_image.with_suffix(ext)
        if side.is_file():
            link_or_copy(side, dest_dir / side.name)


def sample_one_dir(src_images: Path, dest_images: Path, rng: random.Random) -> dict:
    names = list_image_names(src_images)
    source_count = len(names)
    n = int(round(source_count * FRACTION)) if source_count else 0
    n = min(max(n, 1 if source_count else 0), source_count)
    chosen = sorted(rng.sample(names, n)) if n else []
    dest_images.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, name in enumerate(chosen, 1):
        copy_with_sidecars(src_images / name, dest_images)
        if i % 2000 == 0 or i == len(chosen):
            elapsed = time.time() - t0
            print(f"    linked {i}/{len(chosen)} ({elapsed:.1f}s)", flush=True)
    return {
        "source_images": str(src_images),
        "dest_images": str(dest_images),
        "source_count": source_count,
        "sampled_count": len(chosen),
        "files": chosen,
    }


def run_job(output_dir: Path, source_name: str | None, dest_name: str) -> None:
    source_run = output_dir if source_name is None else output_dir / source_name
    dest_run = output_dir / dest_name
    print(f"\n=== {dest_name} ===", flush=True)
    print(f"source: {source_run}", flush=True)
    if (dest_run / "sample_manifest.json").is_file():
        print("  SKIP already sampled", flush=True)
        return
    if not source_run.is_dir():
        print("  SKIP missing source", flush=True)
        return
    groups = find_image_groups(source_run)
    if not groups:
        print("  SKIP no images/", flush=True)
        return

    rng = random.Random(SEED)
    subset_reports = []
    total_src = 0
    total_dst = 0
    for subset, src_images in groups:
        dest_images = dest_run / subset / "images" if subset else dest_run / "images"
        print(f"  subset={subset or '(root)'} listing {src_images} ...", flush=True)
        report = sample_one_dir(src_images, dest_images, rng)
        print(
            f"  sampled {report['sampled_count']} / {report['source_count']}",
            flush=True,
        )
        subset_reports.append({"subset": subset, **report})
        total_src += report["source_count"]
        total_dst += report["sampled_count"]

    manifest = {
        "source_run": str(source_run),
        "dest_run": str(dest_run),
        "fraction": FRACTION,
        "seed": SEED,
        "source_count": total_src,
        "sampled_count": total_dst,
        "groups": [
            {
                "subset": g["subset"],
                "source_images": g["source_images"],
                "dest_images": g["dest_images"],
                "source_count": g["source_count"],
                "sampled_count": g["sampled_count"],
                "files": g["files"],
            }
            for g in subset_reports
        ],
    }
    dest_run.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_run / "sample_manifest.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    print(f"  wrote {manifest_path}  total {total_dst}/{total_src}", flush=True)


def main() -> int:
    for output_dir, source_name, dest_name in JOBS:
        run_job(output_dir, source_name, dest_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
