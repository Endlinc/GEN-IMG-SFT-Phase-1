#!/usr/bin/env python3
"""轻量筛选结果看图器：一次只加载一张图，避免直接打开大目录时卡死。

    cd data_clean_general/webview
    python app.py
    # 浏览器打开 http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STATIC = ROOT / "static"
CACHE_DIR = ROOT / ".cache"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
REVIEW_DELETE_DIR = "deleted_after_check"
REVIEW_KEEP_DIR = "images_after_check"
KIND_FOLDERS = {
    "images": "images",
    "deleted": "deleted",
    "review_delete": REVIEW_DELETE_DIR,
    "review_keep": REVIEW_KEEP_DIR,
}
SKIP_DIR_NAMES = {
    "checkpoint",
    ".cache",
    "__pycache__",
    REVIEW_DELETE_DIR,
    REVIEW_KEEP_DIR,
    "检查后删除",
    "检查后保留",
}
CAPTION_KEYS = ("prompt", "caption", "text")

DATASETS = [
    {
        "id": "finet2i",
        "label": "Fine-T2I",
        "output": PROJECT / "output_finet2i",
    },
    {
        "id": "conceptual_captions",
        "label": "Conceptual Captions",
        "output": PROJECT / "output_conceptual_captions",
    },
    {
        "id": "densefusion_4v_100k",
        "label": "DenseFusion-4V-100K",
        "output": PROJECT / "output_densefusion_4v_100k",
    },
    {
        "id": "wikiart",
        "label": "WikiArt",
        "output": PROJECT / "output_wikiart",
    },
    {
        "id": "alchemist",
        "label": "Alchemist",
        "output": PROJECT / "output_alchemist",
    },
    {
        "id": "diffusiondb",
        "label": "DiffusionDB 2M random 50k",
        "output": PROJECT / "output_diffusiondb_2m_random_50k",
    },
]

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_review_lock = threading.Lock()


def _dataset_by_id(dataset_id: str) -> Dict[str, Any]:
    for item in DATASETS:
        if item["id"] == dataset_id:
            return item
    raise KeyError(dataset_id)


def _is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "images").is_dir() or (path / "deleted").is_dir():
        return True
    if (path / "filter_report.json").is_file() or (path / "checkpoint").is_dir():
        return True
    try:
        for child in path.iterdir():
            if child.name in SKIP_DIR_NAMES or not child.is_dir():
                continue
            if (child / "images").is_dir() or (child / "deleted").is_dir():
                return True
    except OSError:
        return False
    return False


ROOT_VERSION = "__root__"


def list_versions(output_dir: Path) -> List[str]:
    if not output_dir.is_dir():
        return []
    names = []
    try:
        for child in output_dir.iterdir():
            if child.name.startswith(".") or child.name in SKIP_DIR_NAMES:
                continue
            if _is_run_dir(child):
                names.append(child.name)
    except OSError:
        return []
    return sorted(names)


def _subset_label(name: str) -> str:
    return name[7:] if name.startswith("output_") else name


def _subset_entries(names: List[str]) -> List[Dict[str, str]]:
    return [{"id": name, "label": _subset_label(name)} for name in names]


def catalog() -> Dict[str, Any]:
    datasets = []
    for item in DATASETS:
        output = item["output"]
        if item.get("run_at_output_root"):
            versions = [{"id": ROOT_VERSION, "label": "当前结果"}]
            if output.is_dir():
                for name in list_versions(output):
                    if name.endswith("_samplecheck"):
                        versions.append({"id": name, "label": name})
            subsets = _subset_entries(
                [n for n in list_subsets(output) if not n.endswith("_samplecheck")]
                if output.is_dir()
                else []
            )
            hide_version = len(versions) <= 1
        else:
            names = list_versions(output)
            if item["id"] == "finet2i":
                names = [n for n in names if not n.startswith("output_")]
            versions = [{"id": name, "label": name} for name in names]
            subsets = []
            hide_version = False
        datasets.append(
            {
                "id": item["id"],
                "label": item["label"],
                "exists": output.is_dir(),
                "versions": versions,
                "subsets": subsets,
                "hide_version": hide_version,
            }
        )
    return {"datasets": datasets}


def _run_dir(dataset_id: str, version: str) -> Path:
    item = _dataset_by_id(dataset_id)
    output = item["output"].resolve()
    if item.get("run_at_output_root") and version in {"", ROOT_VERSION}:
        if not output.is_dir():
            raise FileNotFoundError(str(output))
        return output
    if not version or "/" in version or version in (".", ".."):
        raise ValueError("invalid version")
    run_dir = (output / version).resolve()
    if not str(run_dir).startswith(str(output) + os.sep):
        raise ValueError("version escapes output dir")
    if not run_dir.is_dir():
        raise FileNotFoundError(version)
    return run_dir


def list_subsets(run_dir: Path) -> List[str]:
    if (run_dir / "images").is_dir() or (run_dir / "deleted").is_dir():
        return []
    names = []
    try:
        for child in run_dir.iterdir():
            if child.name in SKIP_DIR_NAMES or child.name.endswith("_samplecheck") or not child.is_dir():
                continue
            if (child / "images").is_dir() or (child / "deleted").is_dir():
                names.append(child.name)
    except OSError:
        return []
    return sorted(names)


def _kind_dirs(run_dir: Path, kind: str, subset: str) -> List[Tuple[str, str, Path]]:
    """Return (kind, subset_or_empty, directory) to scan."""
    if kind not in {"all", *KIND_FOLDERS}:
        raise ValueError("kind must be all/images/deleted/review_delete/review_keep")
    kinds = ["images", "deleted"] if kind == "all" else [kind]
    subsets = [subset] if subset else list_subsets(run_dir)
    rows: List[Tuple[str, str, Path]] = []
    if subsets:
        for name in subsets:
            if "/" in name or name in (".", ".."):
                continue
            for k in kinds:
                d = run_dir / name / KIND_FOLDERS[k]
                if d.is_dir():
                    rows.append((k, name, d))
        return rows
    for k in kinds:
        d = run_dir / KIND_FOLDERS[k]
        if d.is_dir():
            rows.append((k, "", d))
    return rows


def _fingerprint(dirs: List[Path]) -> List[List[Any]]:
    rows = []
    for d in dirs:
        try:
            st = d.stat()
            rows.append([str(d), int(st.st_mtime), int(st.st_nlink)])
        except OSError:
            rows.append([str(d), 0, 0])
    return rows


def _cache_path(key: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
    return CACHE_DIR / f"{safe}.json"


def _scan_images(directory: Path) -> List[str]:
    names: List[str] = []
    with os.scandir(directory) as it:
        for entry in it:
            if not entry.name.endswith(tuple(IMAGE_EXTS)):
                # faster reject: check suffix via Path
                if Path(entry.name).suffix.lower() not in IMAGE_EXTS:
                    continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            names.append(entry.name)
    names.sort()
    return names


def _parse_deleted_summary(path: Path) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    if not path.is_file():
        return mapping
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return mapping
    for line in text.splitlines():
        if not line or line.startswith("Deleted ") or line.startswith("Total:") or line.startswith("Deleted dir:"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        filename, subset, filter_name, reason = parts[0], parts[1], parts[2], parts[3]
        mapping[filename] = (filter_name, reason)
        if subset:
            mapping[f"{subset}/{filename}"] = (filter_name, reason)
    return mapping


def _load_deleted_map(run_dir: Path, subsets: List[str]) -> Dict[str, Tuple[str, str]]:
    mapping = _parse_deleted_summary(run_dir / "deleted_summary.txt")
    for name in subsets:
        mapping.update(_parse_deleted_summary(run_dir / name / "deleted_summary.txt"))
    return mapping


def _job_key(dataset_id: str, version: str, kind: str, subset: str) -> str:
    return f"{dataset_id}|{version}|{kind}|{subset}"


def _invalidate_index(dataset_id: str, version: str, kind: str, subset: str) -> None:
    key = _job_key(dataset_id, version, kind, subset or "")
    with _jobs_lock:
        _jobs.pop(key, None)
    cache_file = _cache_path(key)
    if cache_file.is_file():
        try:
            cache_file.unlink()
        except OSError:
            pass


def _review_dir(run_dir: Path, subset: str, folder: str) -> Path:
    if subset:
        return run_dir / subset / folder
    return run_dir / folder


def _check_reason_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + ".check_reason.txt")


def _read_check_reason(run_dir: Path, row: Dict[str, str]) -> str:
    if row.get("k") == "images":
        dest = _review_dir(run_dir, row.get("s") or "", REVIEW_DELETE_DIR) / row["n"]
    elif row.get("k") == "review_delete":
        dest = run_dir / row["p"]
    else:
        return ""
    path = _check_reason_path(dest)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def save_check_reason(
    dataset_id: str, version: str, kind: str, subset: str, index: int, note: str
) -> Dict[str, Any]:
    """Write human check reason next to the copied file in deleted_after_check."""
    item = item_payload(dataset_id, version, kind, subset, index)
    if item.get("empty"):
        raise ValueError("没有可写原因的图片")
    src_kind = item.get("kind")
    if src_kind not in {"images", "review_delete"}:
        raise ValueError("只有从保留标到检查后删除的图才能写原因")
    if src_kind == "images" and not item.get("reviewed"):
        raise ValueError("请先点「该图需删除」")
    run_dir = _run_dir(dataset_id, version)
    if src_kind == "review_delete":
        dest = (run_dir / item["relpath"]).resolve()
    else:
        dest = (_review_dir(run_dir, item.get("subset") or "", REVIEW_DELETE_DIR) / item["filename"]).resolve()
    if not str(dest).startswith(str(run_dir.resolve()) + os.sep):
        raise ValueError("path escapes run dir")
    if not dest.is_file():
        raise ValueError("还没有复制到 deleted_after_check")
    path = _check_reason_path(dest)
    text = note if isinstance(note, str) else ""
    path.write_text(text, encoding="utf-8")
    _append_review_log(
        run_dir,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "action": "set_check_reason",
            "subset": item.get("subset") or "",
            "filename": item["filename"],
            "dest": str(dest.relative_to(run_dir)),
            "check_reason": text,
        },
    )
    return {"ok": True, "check_reason": text, "path": str(path.relative_to(run_dir))}


def _copy_with_sidecars(src_image: Path, dest_dir: Path) -> List[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    dest_image = dest_dir / src_image.name
    shutil.copy2(src_image, dest_image)
    copied.append(src_image.name)
    for ext in (".json", ".txt"):
        side = src_image.with_suffix(ext)
        if side.is_file():
            shutil.copy2(side, dest_dir / side.name)
            copied.append(side.name)
    return copied


def _append_review_log(run_dir: Path, record: Dict[str, Any]) -> None:
    path = run_dir / "review.jsonl"
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _review_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def review_item(dataset_id: str, version: str, kind: str, subset: str, index: int) -> Dict[str, Any]:
    """Copy current image + sidecars into a review folder. Does not touch filter_report."""
    item = item_payload(dataset_id, version, kind, subset, index)
    if item.get("empty"):
        raise ValueError("没有可复核的图片")
    src_kind = item.get("kind")
    if src_kind == "images":
        review_kind, folder = "review_delete", REVIEW_DELETE_DIR
        action = "mark_delete"
    elif src_kind == "deleted":
        review_kind, folder = "review_keep", REVIEW_KEEP_DIR
        action = "mark_keep"
    else:
        raise ValueError("只能从「保留」或「删除」里复核")
    if not item.get("has_image"):
        raise ValueError("这条记录没有图片文件")

    run_dir = _run_dir(dataset_id, version)
    src = (run_dir / item["relpath"]).resolve()
    if not str(src).startswith(str(run_dir.resolve()) + os.sep) or not src.is_file():
        raise ValueError("invalid source image")

    dest_dir = _review_dir(run_dir, item.get("subset") or "", folder)
    dest = dest_dir / src.name
    already = dest.is_file()
    copied: List[str] = []
    if not already:
        copied = _copy_with_sidecars(src, dest_dir)

    dest_rel = str((dest_dir / src.name).relative_to(run_dir))
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "action": action,
        "from_kind": src_kind,
        "to_folder": folder,
        "subset": item.get("subset") or "",
        "filename": item["filename"],
        "src": item["relpath"],
        "dest": dest_rel,
        "filter_name": item.get("filter_name") or "",
        "reason": item.get("reason") or "",
        "copied": copied,
        "already_existed": already,
    }
    if not already:
        _append_review_log(run_dir, record)
        _invalidate_index(dataset_id, version, review_kind, item.get("subset") or "")
        _invalidate_index(dataset_id, version, review_kind, "")
    return {"ok": True, **record}


def build_index(dataset_id: str, version: str, kind: str, subset: str) -> Dict[str, Any]:
    run_dir = _run_dir(dataset_id, version)
    subset = subset or ""
    dirs = _kind_dirs(run_dir, kind, subset)
    subsets = list_subsets(run_dir)
    fingerprint = _fingerprint([p for _, _, p in dirs])
    cache_key = _job_key(dataset_id, version, kind, subset)
    cache_file = _cache_path(cache_key)
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                cached["status"] = "ready"
                cached["from_cache"] = True
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    items: List[Dict[str, str]] = []
    for k, group, directory in dirs:
        for name in _scan_images(directory):
            rel = f"{group}/{directory.name}/{name}" if group else f"{directory.name}/{name}"
            items.append({"p": rel, "k": k, "s": group, "n": name})

    payload = {
        "status": "ready",
        "from_cache": False,
        "dataset": dataset_id,
        "version": version,
        "kind": kind,
        "subset": subset,
        "count": len(items),
        "subsets": subsets,
        "fingerprint": fingerprint,
        "built_at": int(time.time()),
        "deleted_map": {k: list(v) for k, v in _load_deleted_map(run_dir, subsets).items()},
        "items": items,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(cache_file)
    return payload


def _ensure_job(dataset_id: str, version: str, kind: str, subset: str, refresh: bool) -> Dict[str, Any]:
    key = _job_key(dataset_id, version, kind, subset or "")
    with _jobs_lock:
        job = _jobs.get(key)
        if refresh and job and job.get("status") == "ready":
            cache_file = _cache_path(key)
            if cache_file.is_file():
                try:
                    cache_file.unlink()
                except OSError:
                    pass
            job = None
            _jobs.pop(key, None)
        if job and job.get("status") == "error":
            job = None
            _jobs.pop(key, None)
        if job is None:
            job = {
                "status": "building",
                "count": 0,
                "subsets": [],
                "error": None,
                "items": None,
                "from_cache": False,
            }
            _jobs[key] = job

            def worker() -> None:
                try:
                    built = build_index(dataset_id, version, kind, subset or "")
                    with _jobs_lock:
                        job.update(
                            {
                                "status": "ready",
                                "count": built["count"],
                                "subsets": built.get("subsets") or [],
                                "items": built["items"],
                                "deleted_map": built.get("deleted_map") or {},
                                "from_cache": bool(built.get("from_cache")),
                                "error": None,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    with _jobs_lock:
                        job.update({"status": "error", "error": str(exc)})

            threading.Thread(target=worker, daemon=True).start()
        public = {
            "status": job["status"],
            "count": job.get("count") or 0,
            "subsets": job.get("subsets") or [],
            "from_cache": bool(job.get("from_cache")),
            "error": job.get("error"),
        }
    return public


def _get_ready_items(dataset_id: str, version: str, kind: str, subset: str) -> List[Dict[str, str]]:
    key = _job_key(dataset_id, version, kind, subset or "")
    with _jobs_lock:
        job = _jobs.get(key)
        if job and job.get("status") == "ready" and job.get("items") is not None:
            return job["items"]
    built = build_index(dataset_id, version, kind, subset or "")
    with _jobs_lock:
        _jobs[key] = {
            "status": "ready",
            "count": built["count"],
            "subsets": built.get("subsets") or [],
            "items": built["items"],
            "deleted_map": built.get("deleted_map") or {},
            "from_cache": bool(built.get("from_cache")),
            "error": None,
        }
    return built["items"]


def _get_ready_job(dataset_id: str, version: str, kind: str, subset: str) -> Dict[str, Any]:
    _get_ready_items(dataset_id, version, kind, subset)
    key = _job_key(dataset_id, version, kind, subset or "")
    with _jobs_lock:
        job = _jobs.get(key)
        if not job or job.get("status") != "ready":
            raise RuntimeError("index not ready")
        return job


def _read_sidecars(image_path: Path) -> Tuple[str, Dict[str, Any]]:
    stem = image_path.with_suffix("")
    caption = ""
    meta: Dict[str, Any] = {}
    txt = Path(str(stem) + ".txt")
    js = Path(str(stem) + ".json")
    if txt.is_file():
        try:
            caption = txt.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass
    if js.is_file():
        try:
            loaded = json.loads(js.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except (OSError, json.JSONDecodeError):
            pass
    if not caption:
        for key in CAPTION_KEYS:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                caption = value.strip()
                break
    return caption, meta


def item_payload(dataset_id: str, version: str, kind: str, subset: str, index: int) -> Dict[str, Any]:
    job = _get_ready_job(dataset_id, version, kind, subset)
    items = job.get("items") or []
    if not items:
        return {"count": 0, "index": 0, "empty": True}
    if index < 0 or index >= len(items):
        raise IndexError(f"index {index} out of range 0..{len(items) - 1}")
    row = items[index]
    run_dir = _run_dir(dataset_id, version)
    rel = row["p"]
    image_path = (run_dir / rel).resolve()
    if not str(image_path).startswith(str(run_dir.resolve()) + os.sep):
        raise ValueError("path escapes run dir")
    caption, meta = _read_sidecars(image_path) if image_path.is_file() else ("", {})
    deleted_map = job.get("deleted_map") or {}
    reason_key = f"{row['s']}/{row['n']}" if row.get("s") else row["n"]
    hit = deleted_map.get(reason_key) or deleted_map.get(row["n"]) or ["", ""]
    filter_name, reason = (hit + ["", ""])[:2]
    reviewed = False
    if row["k"] in {"images", "deleted"}:
        folder = REVIEW_DELETE_DIR if row["k"] == "images" else REVIEW_KEEP_DIR
        reviewed = (_review_dir(run_dir, row.get("s") or "", folder) / row["n"]).is_file()
    slim_meta = {}
    for key, value in meta.items():
        if key in CAPTION_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            slim_meta[key] = value
        else:
            slim_meta[key] = str(value)[:200]
    return {
        "empty": False,
        "index": index,
        "count": len(items),
        "filename": row["n"],
        "relpath": rel,
        "kind": row["k"],
        "subset": row.get("s") or "",
        "caption": caption,
        "metadata": slim_meta,
        "filter_name": filter_name if row["k"] == "deleted" else "",
        "reason": reason if row["k"] == "deleted" else "",
        "has_image": image_path.is_file(),
        "reviewed": reviewed,
        "check_reason": _read_check_reason(run_dir, row),
    }


def safe_file(dataset_id: str, version: str, relpath: str) -> Path:
    run_dir = _run_dir(dataset_id, version)
    rel = Path(unquote(relpath))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("invalid relpath")
    path = (run_dir / rel).resolve()
    if not str(path).startswith(str(run_dir.resolve()) + os.sep):
        raise ValueError("path escapes run dir")
    if not path.is_file():
        raise FileNotFoundError(relpath)
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError("not an image")
    return path


def _json_bytes(payload: Any, status: int = 200) -> Tuple[int, bytes, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, data, "application/json; charset=utf-8"


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "FilterViewer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path.startswith("/api/") or self.path.startswith("/file"):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> Dict[str, str]:
        q = parse_qs(urlparse(self.path).query)
        return {k: (v[0] if v else "") for k, v in q.items()}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in {"/", "/index.html"}:
                html = (STATIC / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
                return
            if path == "/api/catalog":
                self._send(*_json_bytes(catalog()))
                return
            if path == "/api/index":
                q = self._query()
                payload = _ensure_job(
                    q.get("dataset", ""),
                    q.get("version", ""),
                    q.get("kind", "images"),
                    q.get("subset", ""),
                    refresh=q.get("refresh") == "1",
                )
                self._send(*_json_bytes(payload))
                return
            if path == "/api/item":
                q = self._query()
                index = int(q.get("i", "0") or 0)
                payload = item_payload(
                    q.get("dataset", ""),
                    q.get("version", ""),
                    q.get("kind", "images"),
                    q.get("subset", ""),
                    index,
                )
                self._send(*_json_bytes(payload))
                return
            if path == "/file":
                q = self._query()
                file_path = safe_file(q.get("dataset", ""), q.get("version", ""), q.get("relpath", ""))
                data = file_path.read_bytes()
                ctype = CONTENT_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(data)
                return
            self._send(*_json_bytes({"error": "not found"}, 404))
        except FileNotFoundError as exc:
            self._send(*_json_bytes({"error": str(exc)}, 404))
        except (KeyError, ValueError, IndexError) as exc:
            self._send(*_json_bytes({"error": str(exc)}, 400))
        except Exception as exc:  # noqa: BLE001
            self._send(*_json_bytes({"error": str(exc)}, 500))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("invalid json")
            if parsed.path == "/api/review":
                payload = review_item(
                    str(body.get("dataset") or ""),
                    str(body.get("version") or ""),
                    str(body.get("kind") or "images"),
                    str(body.get("subset") or ""),
                    int(body.get("i") or 0),
                )
                self._send(*_json_bytes(payload))
                return
            if parsed.path == "/api/review_reason":
                payload = save_check_reason(
                    str(body.get("dataset") or ""),
                    str(body.get("version") or ""),
                    str(body.get("kind") or "images"),
                    str(body.get("subset") or ""),
                    int(body.get("i") or 0),
                    str(body.get("note") or ""),
                )
                self._send(*_json_bytes(payload))
                return
            self._send(*_json_bytes({"error": "not found"}, 404))
        except FileNotFoundError as exc:
            self._send(*_json_bytes({"error": str(exc)}, 404))
        except (KeyError, ValueError, IndexError, json.JSONDecodeError) as exc:
            self._send(*_json_bytes({"error": str(exc)}, 400))
        except Exception as exc:  # noqa: BLE001
            self._send(*_json_bytes({"error": str(exc)}, 500))


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选结果轻量看图器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"查看器已启动： http://{args.host}:{args.port}")
    print("一次只加载当前图片。Ctrl+C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
        server.server_close()


if __name__ == "__main__":
    main()
