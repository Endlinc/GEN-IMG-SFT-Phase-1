"""Dataset loading shared by the four benchmark runners and generate_captions.py.

Single source of truth for "which images does this benchmark evaluate, and under
what keys". Both the standalone caption generator and the runners go through
`iter_items()`, so a caption file produced for a benchmark is guaranteed to key
onto exactly the images that benchmark will look up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image

BENCHMARKS = ("capture", "comprecap", "capsbench", "perceptionrubrics")

# Dataset file each benchmark expects inside --data
DATASET_FILES = {
    "capture": "detailcaps_4870.json",
    "capsbench": "capsbench.json",
    "perceptionrubrics": "perceptionrubrics.json",
}


def to_array(image: Any) -> np.ndarray:
    """Normalise a dataset `image` field (path string, PIL image, or array) to ndarray."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.array(image)
    return image


def resolve_image(data_dir: Path, sample: Dict[str, Any]) -> Any:
    """The sample's image reference, falling back to the co-located `images/` copy.

    The shipped DetailCaps and CapsBench JSONs record absolute paths from the
    machine that built them (`/mnt/shared_resources/datasets/baselines/...`).
    Those do not exist on any other host, but the images themselves ship next to
    the JSON in `<data_dir>/images/`. When the recorded path is absent, fall back
    to that copy by basename.

    The fallback is only taken when the local file really is there -- a genuinely
    missing image still raises from Image.open rather than being papered over.
    """
    ref = sample.get("image")
    if isinstance(ref, str) and not Path(ref).is_file():
        alt = Path(data_dir) / "images" / Path(ref).name
        if alt.is_file():
            return str(alt)
    return ref


# ---------------------------------------------------------------------------
# Per-benchmark loaders
# ---------------------------------------------------------------------------

def _load_json_list(path: Path, hint: str) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_detailcaps(data_dir: Path) -> List[Dict[str, Any]]:
    return _load_json_list(data_dir / DATASET_FILES["capture"],
                           "Run download_datasets.py first.")


def load_capsbench(data_dir: Path) -> List[Dict[str, Any]]:
    return _load_json_list(data_dir / DATASET_FILES["capsbench"],
                           "Run download_datasets.py first.")


def load_perceptionrubrics(data_dir: Path) -> List[Dict[str, Any]]:
    return _load_json_list(data_dir / DATASET_FILES["perceptionrubrics"],
                           "Run download_datasets.py first.")


def load_comprecap_images(data_dir: Path) -> List[Dict[str, str]]:
    """Return [{image_id, image_path}] from CompreCap_dataset/images."""
    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"CompreCap images dir not found: {images_dir}")
    return [{"image_id": p.name, "image_path": str(p)}
            for p in sorted(images_dir.glob("*.jpg"))]


def load_comprecap_rows(data_dir: Path, anno_subset: str = "") -> List[Dict[str, str]]:
    """CompreCap rows, either from a pre-sampled anno subset or the images dir."""
    if anno_subset:
        with open(anno_subset, encoding="utf-8") as f:
            anno_data = json.load(f)
        return [{"image_id": ex["image"],
                 "image_path": str(data_dir / "images" / ex["image"])}
                for ex in anno_data]
    return load_comprecap_images(data_dir)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def sample_key(benchmark: str, sample: Dict[str, Any], index: int) -> str:
    """The id a benchmark keys this sample by -- must match what the runner looks up."""
    if benchmark == "capture":
        return str(sample.get("image_id", index))
    if benchmark == "capsbench":
        return str(sample["image_id"])
    return str(sample.get("id", index))  # perceptionrubrics


def load_samples(benchmark: str, data_dir: Path, subset: str = "") -> List[Dict[str, Any]]:
    """Raw dataset samples for a benchmark.

    For CompreCap this returns `{image_id, image_path}` rows; the other three
    return the dataset's own sample list (from `subset` when one is given).
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r} (expected one of {BENCHMARKS})")
    data_dir = Path(data_dir)
    if benchmark == "comprecap":
        return load_comprecap_rows(data_dir, subset)
    if subset:
        return _load_json_list(Path(subset), "")
    return {
        "capture": load_detailcaps,
        "capsbench": load_capsbench,
        "perceptionrubrics": load_perceptionrubrics,
    }[benchmark](data_dir)


def iter_items(
    benchmark: str,
    data_dir: Path,
    subset: str = "",
    limit: int = 0,
) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (image_id, image_array) for a benchmark, honouring subset + limit.

    `subset` is the benchmark's own pre-sampled JSON: a DetailCaps/PerceptionRubrics
    sample list, or a CompreCap anno list. CapsBench has no subset file.
    """
    for i, sample in enumerate(load_samples(benchmark, data_dir, subset)):
        if limit and i >= limit:
            break
        if benchmark == "comprecap":
            yield sample["image_id"], np.array(Image.open(sample["image_path"]).convert("RGB"))
        else:
            yield sample_key(benchmark, sample, i), to_array(resolve_image(data_dir, sample))
