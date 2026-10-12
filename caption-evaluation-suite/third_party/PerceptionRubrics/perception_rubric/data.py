"""Dataset loading helpers shared by the tutorial scripts."""

from __future__ import annotations

import os
from typing import Any


_FILE_EXT_TO_LOADER = {
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".arrow": "arrow",
}


def load_eval_dataset(source: str, split: str) -> Any:
    """Load a dataset from a Hub repo id, a local dataset directory, or a single data file.

    - Hub repo id (``org/name``) or local ``datasets``-format directory: dispatched to
      ``load_dataset(source, split=split)``.
    - Single file (``.parquet``, ``.json``, ``.jsonl``, ``.csv``, ``.arrow``): dispatched to
      ``load_dataset(<loader>, data_files=source, split="train")`` since file-based loads
      always materialise a single ``train`` split.

    If the ``image`` column is stored as a raw dict (``{bytes, path}``) rather than
    a decoded PIL Image, it is automatically cast to ``datasets.Image()``.
    """
    from datasets import load_dataset, Image as DatasetsImage

    ext = os.path.splitext(source)[1].lower()
    if os.path.isfile(source) and ext in _FILE_EXT_TO_LOADER:
        ds = load_dataset(_FILE_EXT_TO_LOADER[ext], data_files=source, split="train")
    else:
        ds = load_dataset(source, split=split)

    # Auto-cast image column if it's stored as a raw dict (bytes/path) instead of PIL Image
    if "image" in ds.features:
        feat = ds.features["image"]
        # If the feature is a dict-like struct (not already datasets.Image), cast it
        if not isinstance(feat, DatasetsImage):
            ds = ds.cast_column("image", DatasetsImage())

    return ds
