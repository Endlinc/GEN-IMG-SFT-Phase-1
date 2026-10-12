"""Export filter results: report JSON and optional image copies."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional

from .core import Sample


SAVE_MODES = ("report", "kept", "split")

# Loader-added keys; strip when writing sidecars so exports match source records.
_INTERNAL_META_KEYS = frozenset({"index_line", "_load_error"})


def _image_basename(sample: Sample) -> str:
    if sample.image_path:
        return os.path.basename(sample.image_path)
    source_image = sample.source.get("image") if isinstance(sample.source, dict) else None
    if source_image:
        return os.path.basename(str(source_image))
    return f"{sample.id}.jpg"


def _unique_dest(dest_dir: str, filename: str, used: set) -> str:
    """Avoid overwrite when multiple samples share the same basename."""
    if filename not in used and not os.path.exists(os.path.join(dest_dir, filename)):
        used.add(filename)
        return filename
    stem, ext = os.path.splitext(filename)
    idx = 1
    while True:
        candidate = f"{stem}__{idx}{ext}"
        if candidate not in used and not os.path.exists(os.path.join(dest_dir, candidate)):
            used.add(candidate)
            return candidate
        idx += 1


def _clean_metadata(sample: Sample) -> Dict[str, Any]:
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    return {k: v for k, v in meta.items() if k not in _INTERNAL_META_KEYS}


def _write_json_sidecar(path: str, payload: Dict[str, Any]) -> None:
    if not payload or os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_txt_sidecar(path: str, text: str) -> None:
    if not text or os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_caption_sidecar(sample: Sample, dest_dir: str, image_filename: str) -> None:
    """Write caption/metadata beside the image in a format close to the source dataset.

    - fine-t2i (tar): ``.txt`` (+ ``.json`` when present in the shard)
    - densefusion-4v-100k / conceptual-captions / wikiart / alchemist / diffusiondb_2m_random_50k: ``.json`` record (not plain ``.txt``)
    - fallback: ``.txt`` if only free-form text is available
    """
    stem, _ = os.path.splitext(image_filename)
    source = sample.source if isinstance(sample.source, dict) else {}
    meta = _clean_metadata(sample)
    text = sample.text or ""

    is_tar = bool(source.get("tar_path") or source.get("tar_file"))
    is_zip = bool(source.get("zip_path") or source.get("zip_file"))
    is_cc = source.get("subset") == "cc_images" or bool(source.get("index"))
    is_wikiart = bool(source.get("parquet_file") or source.get("parquet_path"))
    is_diffusiondb = source.get("loader") == "diffusiondb_2m_random_50k"

    if is_tar:
        # Fine-T2I shards are image + .txt (+ optional .json).
        if text:
            _write_txt_sidecar(os.path.join(dest_dir, stem + ".txt"), text)
        if meta and source.get("json_member"):
            _write_json_sidecar(os.path.join(dest_dir, stem + ".json"), meta)
        elif meta and not text:
            _write_json_sidecar(os.path.join(dest_dir, stem + ".json"), meta)
        return

    if is_diffusiondb:
        payload = dict(meta) if meta else {}
        if payload:
            _write_json_sidecar(os.path.join(dest_dir, stem + ".json"), payload)
        if text:
            _write_txt_sidecar(os.path.join(dest_dir, stem + ".txt"), text)
        return

    if is_wikiart:
        payload = dict(meta) if meta else {}
        if payload:
            _write_json_sidecar(os.path.join(dest_dir, stem + ".json"), payload)
            return

    if is_zip or is_cc or meta:
        payload = dict(meta) if meta else {}
        if text and "caption" not in payload and "prompt" not in payload and "text" not in payload:
            payload["caption"] = text
        if is_zip and source.get("zip_file") and "zip_file" not in payload:
            payload["zip_file"] = source.get("zip_file")
        if payload:
            _write_json_sidecar(os.path.join(dest_dir, stem + ".json"), payload)
            return

    if text:
        _write_txt_sidecar(os.path.join(dest_dir, stem + ".txt"), text)


def write_image_file(sample: Sample, dest_dir: str, filename: str) -> bool:
    """Write sample image to ``dest_dir/filename``. Returns False if no image.

    Prefers byte-exact copies (``image_path`` / ``image_bytes``) over PIL
    re-encoding, which is much slower for large curated photos.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    wrote = False
    if sample.image_path and os.path.exists(sample.image_path):
        shutil.copy2(sample.image_path, dest_path)
        wrote = True
    else:
        if sample.image_bytes is None:
            # May populate image_bytes from tar/path while decoding.
            sample.load_image()
        if sample.image_bytes is not None:
            with open(dest_path, "wb") as f:
                f.write(sample.image_bytes)
            wrote = True
        else:
            img = sample.image
            if img is None:
                return False
            rgb = img.convert("RGB") if img.mode not in ("RGB", "RGBA", "L") else img
            ext = os.path.splitext(filename)[1].lower()
            if ext in (".jpg", ".jpeg"):
                if rgb.mode == "RGBA":
                    rgb = rgb.convert("RGB")
                rgb.save(dest_path, format="JPEG", quality=95)
            elif ext == ".png":
                rgb.save(dest_path, format="PNG")
            elif ext == ".webp":
                rgb.save(dest_path, format="WEBP")
            else:
                rgb.save(dest_path)
            wrote = True

    _write_caption_sidecar(sample, dest_dir, filename)
    return wrote


def _copy_image(sample: Sample, dest_dir: str, used_names: set) -> Optional[str]:
    """Write sample image to dest_dir (from disk path, bytes, or tar archive)."""
    filename = _unique_dest(dest_dir, _image_basename(sample), used_names)
    if not write_image_file(sample, dest_dir, filename):
        return None
    return filename


def _write_deleted_summary(
    summary_path: str,
    deleted_dir: str,
    rows: List[Dict[str, str]],
) -> None:
    lines = [
        "Deleted images summary",
        f"Total: {len(rows)}",
        f"Deleted dir: {deleted_dir}",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['filename']}\t{row['subset']}\t{row['filter_name']}\t{row['reason']}"
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if rows else ""))


def export_results(
    *,
    output_dir: str,
    report_filename: str,
    report: Dict[str, Any],
    kept_samples: Iterable[Sample],
    id_to_sample: Dict[str, Sample],
    save_mode: str = "report",
) -> Dict[str, Any]:
    """Persist report and optionally copy kept/deleted images.

    save_mode:
      - report: only JSON report
      - kept: JSON + images/ (kept only)
      - split: JSON + images/ (kept) + deleted/ (dropped) + deleted_summary.txt
    """
    if save_mode not in SAVE_MODES:
        raise ValueError(f"Unknown save_mode={save_mode!r}; expected one of {SAVE_MODES}")

    os.makedirs(output_dir, exist_ok=True)
    artifacts: Dict[str, Any] = {"save_mode": save_mode}
    run_dir = output_dir
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    artifacts["report_path"] = report_path
    artifacts["run_dir"] = run_dir

    if save_mode == "report":
        return artifacts

    images_dir = os.path.join(run_dir, "images")
    used_kept: set = set()
    kept_files = []
    for sample in kept_samples:
        copied = _copy_image(sample, images_dir, used_kept)
        if copied:
            kept_files.append(copied)
    artifacts["images_dir"] = images_dir
    artifacts["images_copied"] = len(kept_files)

    if save_mode == "kept":
        return artifacts

    deleted_dir = os.path.join(run_dir, "deleted")
    used_deleted: set = set()
    summary_rows: List[Dict[str, str]] = []
    for filt in report.get("filters", []):
        for dropped in filt.get("dropped", []):
            sample_id = dropped.get("sample_id")
            sample = id_to_sample.get(sample_id)
            if sample is None:
                continue
            filename = _copy_image(sample, deleted_dir, used_deleted)
            if not filename:
                filename = _image_basename(sample)
            source = dropped.get("source") or {}
            subset = (
                source.get("subset")
                or source.get("tar_file")
                or source.get("zip_file")
                or "."
            )
            summary_rows.append(
                {
                    "filename": filename,
                    "subset": str(subset),
                    "filter_name": str(dropped.get("filter_name") or filt.get("name")),
                    "reason": str(dropped.get("reason") or "filtered"),
                }
            )

    summary_path = os.path.join(run_dir, "deleted_summary.txt")
    _write_deleted_summary(summary_path, deleted_dir, summary_rows)
    artifacts["deleted_dir"] = deleted_dir

    artifacts["deleted_copied"] = len(summary_rows)
    artifacts["deleted_summary_path"] = summary_path
    return artifacts
