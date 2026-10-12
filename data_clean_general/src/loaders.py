"""Dataset loaders that normalize different formats into Sample objects.

Supported formats (wired in ``src/cli.py``):
- conceptual-captions: ``cc_clean_index.jsonl`` + ``cc_images/``
- fine-t2i: flat ``*.tar`` shards under the dataset root (or ``data/``)
- densefusion-4v-100k: ``DenseFusion-4V-100k/DenseFusion-4V-100k.jsonl`` + ``images/DenseFusion-4V-100K/*.zip``
- wikiart: HuggingFace parquet shards under ``data/`` (image + artist/genre/style)
- alchemist: parquet/csv index (``img_key``, ``url``, ``prompt``) + ``images/``
- diffusiondb_2m_random_50k: parquet metadata + extracted ``images/{stem}_idx{row}.png``
"""

import glob
import json
import os
import tarfile
import zipfile
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .core import Sample

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_TEXT_KEYS = ("caption", "prompt", "text", "enhanced_prompt")


class ConceptualCaptionsLoader:
    """Stream Conceptual Captions records from a JSONL index and image folder."""

    def __init__(
        self,
        root_dir: str,
        index_name: str = "cc_clean_index.jsonl",
        image_dir_name: str = "cc_images",
    ):
        self.root_dir = root_dir
        self.index_path = os.path.join(root_dir, index_name)
        self.image_dir = os.path.join(root_dir, image_dir_name)

    def load(self) -> Iterable[Sample]:
        if not os.path.exists(self.index_path):
            return

        with open(self.index_path, "r", encoding="utf-8") as index_file:
            for line_number, line in enumerate(index_file, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict) or not item.get("file_name"):
                    continue

                file_name = str(item["file_name"])
                image_path = os.path.join(self.image_dir, file_name)
                sample_id = item.get("original_id") or os.path.splitext(file_name)[0]
                metadata = dict(item)
                metadata["index_line"] = line_number
                yield Sample(
                    id=str(sample_id),
                    image_path=image_path,
                    text=item.get("caption"),
                    metadata=metadata,
                    source={
                        "subset": "cc_images",
                        "image": file_name,
                        "index": os.path.basename(self.index_path),
                    },
                )


def _discover_archive_files(root_dir: str, pattern: str) -> List[str]:
    """Find tar shards under ``root_dir`` or ``root_dir/data`` (non-recursive)."""
    candidates = [
        os.path.join(root_dir, "data"),
        root_dir,
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        files = sorted(glob.glob(os.path.join(base, pattern)))
        if files:
            return files
    return []


def _extract_text_from_json_bytes(data: bytes) -> Optional[str]:
    try:
        loaded = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    for key in _TEXT_KEYS:
        value = loaded.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


class TarShardLoader:
    """Load Fine-T2I-style samples from tar shards without full extraction.

    Each tar is expected to contain flat member names such as::

        {uuid}.jpg
        {uuid}.txt
        {uuid}.json   # optional

    Point ``root_dir`` at a *subset* that contains ``*.tar`` directly
    (e.g. ``fine-t2i/curated``), not the Fine-T2I parent that only has
    subset folders.

    Images are read from the archive on demand via :meth:`Sample.load_image`.
    Use ``tar_offset`` / ``tar_limit`` for a fixed slice, or ``tar_batch_size``
    with checkpoint ``completed_tar_names`` to process the next N unfinished shards.
    """

    def __init__(
        self,
        root_dir: str,
        tar_glob: str = "*.tar",
        require_text: bool = False,
        progress_every: int = 5000,
        id_prefix_with_tar: bool = True,
        tar_offset: int = 0,
        tar_limit: Optional[int] = None,
        tar_batch_size: Optional[int] = None,
        completed_tar_names: Optional[Set[str]] = None,
        on_shard_complete: Optional[Callable[[str], None]] = None,
    ):
        self.root_dir = root_dir
        self.tar_glob = tar_glob
        self.require_text = require_text
        self.progress_every = progress_every
        self.id_prefix_with_tar = id_prefix_with_tar
        self.tar_offset = max(0, tar_offset)
        self.tar_limit = tar_limit
        self.tar_batch_size = tar_batch_size
        self.completed_tar_names = completed_tar_names
        self.on_shard_complete = on_shard_complete

    def list_tar_shards(self) -> List[str]:
        return _discover_archive_files(self.root_dir, self.tar_glob)

    def _resolve_tar_files(self) -> List[str]:
        all_files = self.list_tar_shards()
        if not all_files:
            return []

        if self.tar_batch_size:
            if self.completed_tar_names is None:
                raise ValueError(
                    "tar_batch_size requires checkpoint mode (completed_tar_names)."
                )
            pending = [
                path
                for path in all_files
                if os.path.basename(path) not in self.completed_tar_names
            ]
            selected = pending[: self.tar_batch_size]
            print(
                f"[TarShardLoader] tar batch: {len(selected)} shard(s) this run, "
                f"{len(self.completed_tar_names)} shard(s) already finished, "
                f"{len(pending)} shard(s) pending in dataset",
                flush=True,
            )
            if selected:
                names = ", ".join(os.path.basename(p) for p in selected[:5])
                if len(selected) > 5:
                    names += ", ..."
                print(f"[TarShardLoader] shards: {names}", flush=True)
            return selected

        if self.tar_limit is not None:
            end = self.tar_offset + self.tar_limit
            selected = all_files[self.tar_offset : end]
        elif self.tar_offset:
            selected = all_files[self.tar_offset :]
        else:
            selected = all_files

        if self.tar_offset or self.tar_limit is not None:
            print(
                f"[TarShardLoader] tar slice offset={self.tar_offset} "
                f"limit={self.tar_limit} -> {len(selected)} shard(s)",
                flush=True,
            )
        return selected

    def _iter_tar_samples(self, tar_path: str) -> Iterable[Sample]:
        tar_name = os.path.basename(tar_path)
        tar_stem = os.path.splitext(tar_name)[0]
        image_exts = set(_IMAGE_EXTENSIONS)

        with tarfile.open(tar_path, "r") as tf:
            by_base: Dict[str, Dict[str, str]] = defaultdict(dict)
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                base, ext = os.path.splitext(member.name)
                ext = ext.lower()
                if ext in image_exts or ext in (".txt", ".json"):
                    by_base[base][ext] = member.name

            for base in sorted(by_base.keys()):
                parts = by_base[base]
                img_ext = next((e for e in image_exts if e in parts), None)
                if not img_ext:
                    continue

                text = None
                metadata: Dict[str, object] = {}
                if ".txt" in parts:
                    fileobj = tf.extractfile(parts[".txt"])
                    if fileobj is not None:
                        text = fileobj.read().decode("utf-8", errors="replace")
                if ".json" in parts:
                    fileobj = tf.extractfile(parts[".json"])
                    if fileobj is not None:
                        raw = fileobj.read()
                        try:
                            loaded = json.loads(raw.decode("utf-8"))
                            if isinstance(loaded, dict):
                                metadata = loaded
                        except Exception:
                            metadata = {"_load_error": True}
                        if not text:
                            text = _extract_text_from_json_bytes(raw)

                if self.require_text and (not text or not str(text).strip()):
                    continue

                image_member = parts[img_ext]
                sample_id = f"{tar_stem}__{base}" if self.id_prefix_with_tar else base
                image_fname = f"{sample_id}{img_ext}"

                image_bytes = None
                img_obj = tf.extractfile(image_member)
                if img_obj is not None:
                    image_bytes = img_obj.read()

                yield Sample(
                    id=sample_id,
                    image_path=None,
                    text=text,
                    metadata=metadata,
                    image_bytes=image_bytes,
                    source={
                        "subset": tar_name,
                        "tar_path": tar_path,
                        "tar_file": tar_name,
                        "image": image_fname,
                        "image_member": image_member,
                        "txt_member": parts.get(".txt"),
                        "json_member": parts.get(".json"),
                        "sample_base": base,
                    },
                )

    def load(self) -> Iterable[Sample]:
        tar_files = self._resolve_tar_files()
        if not tar_files:
            print("[TarShardLoader] no tar shards to process", flush=True)
            return

        total = 0
        for tar_path in tar_files:
            tar_name = os.path.basename(tar_path)
            shard_count = 0
            for sample in self._iter_tar_samples(tar_path):
                yield sample
                total += 1
                shard_count += 1
                if self.progress_every and total % self.progress_every == 0:
                    print(
                        f"[TarShardLoader] indexed {total} samples "
                        f"(latest tar: {tar_name})",
                        flush=True,
                    )
            if self.on_shard_complete:
                self.on_shard_complete(tar_name)
            print(
                f"[TarShardLoader] finished shard {tar_name} "
                f"({shard_count} samples indexed this shard)",
                flush=True,
            )
        print(f"[TarShardLoader] indexed {total} samples in this run", flush=True)


class DenseFusion4V100kLoader:
    """Stream DenseFusion-4V-100K from JSONL captions + per-shard zip images.

    Expected layout under ``root_dir`` (the DenseFusion dataset root)::

        DenseFusion-4V-100k/DenseFusion-4V-100k.jsonl
        images/DenseFusion-4V-100K/000000.zip
        images/DenseFusion-4V-100K/000001.zip
        ...

    Each JSONL row has ``image_id``, ``caption``, and ``image_path`` like
    ``DenseFusion-4V-100K/000000/<image_id>.png``. Images live as flat members inside
    the matching zip (``images/DenseFusion-4V-100K/000000.zip``).

    JSONL rows are interleaved across zips, so a run opens the selected zip
    handles and streams the JSONL once. Use ``zip_offset`` / ``zip_limit`` or
    ``zip_batch_size`` (wired from CLI ``--tar-*`` flags) for shard batches.
    """

    def __init__(
        self,
        root_dir: str,
        jsonl_relpath: str = "DenseFusion-4V-100k/DenseFusion-4V-100k.jsonl",
        images_subdir: str = "images/DenseFusion-4V-100K",
        progress_every: int = 5000,
        zip_offset: int = 0,
        zip_limit: Optional[int] = None,
        zip_batch_size: Optional[int] = None,
        completed_zip_names: Optional[Set[str]] = None,
        on_shard_complete: Optional[Callable[[str], None]] = None,
    ):
        self.root_dir = root_dir
        self.jsonl_path = os.path.join(root_dir, jsonl_relpath)
        self.images_dir = os.path.join(root_dir, images_subdir)
        self.progress_every = progress_every
        self.zip_offset = max(0, zip_offset)
        self.zip_limit = zip_limit
        self.zip_batch_size = zip_batch_size
        self.completed_zip_names = completed_zip_names
        self.on_shard_complete = on_shard_complete

    def list_zip_shards(self) -> List[str]:
        if not os.path.isdir(self.images_dir):
            return []
        return sorted(glob.glob(os.path.join(self.images_dir, "*.zip")))

    # Alias so CLI shard reporting can reuse the fine-t2i path.
    def list_tar_shards(self) -> List[str]:
        return self.list_zip_shards()

    def _resolve_zip_files(self) -> List[str]:
        all_files = self.list_zip_shards()
        if not all_files:
            return []

        if self.zip_batch_size:
            if self.completed_zip_names is None:
                raise ValueError(
                    "zip_batch_size requires checkpoint mode (completed_zip_names)."
                )
            pending = [
                path
                for path in all_files
                if os.path.basename(path) not in self.completed_zip_names
            ]
            selected = pending[: self.zip_batch_size]
            print(
                f"[DenseFusion4V100kLoader] zip batch: {len(selected)} shard(s) this run, "
                f"{len(self.completed_zip_names)} shard(s) already finished, "
                f"{len(pending)} shard(s) pending in dataset",
                flush=True,
            )
            if selected:
                names = ", ".join(os.path.basename(p) for p in selected[:5])
                if len(selected) > 5:
                    names += ", ..."
                print(f"[DenseFusion4V100kLoader] shards: {names}", flush=True)
            return selected

        if self.zip_limit is not None:
            end = self.zip_offset + self.zip_limit
            selected = all_files[self.zip_offset : end]
        elif self.zip_offset:
            selected = all_files[self.zip_offset :]
        else:
            selected = all_files

        if self.zip_offset or self.zip_limit is not None:
            print(
                f"[DenseFusion4V100kLoader] zip slice offset={self.zip_offset} "
                f"limit={self.zip_limit} -> {len(selected)} shard(s)",
                flush=True,
            )
        return selected

    @staticmethod
    def _parse_image_path(image_path: str) -> Optional[Dict[str, str]]:
        """Parse ``DenseFusion-4V-100K/000000/<id>.png`` into zip basename + member."""
        parts = str(image_path).replace("\\", "/").split("/")
        if len(parts) < 3:
            return None
        shard = parts[-2]
        member = parts[-1]
        if not shard or not member:
            return None
        return {
            "shard": shard,
            "zip_name": f"{shard}.zip",
            "member": member,
        }

    def load(self) -> Iterable[Sample]:
        if not os.path.isfile(self.jsonl_path):
            print(
                f"[DenseFusion4V100kLoader] missing jsonl: {self.jsonl_path}",
                flush=True,
            )
            return

        zip_files = self._resolve_zip_files()
        if not zip_files:
            print("[DenseFusion4V100kLoader] no zip shards to process", flush=True)
            return

        zip_by_name = {os.path.basename(path): path for path in zip_files}
        selected_names = set(zip_by_name.keys())
        handles: Dict[str, zipfile.ZipFile] = {}
        total = 0
        missing_image = 0

        try:
            for name, path in zip_by_name.items():
                handles[name] = zipfile.ZipFile(path, "r")

            with open(self.jsonl_path, "r", encoding="utf-8") as index_file:
                for line_number, line in enumerate(index_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue

                    image_path = item.get("image_path")
                    if not image_path:
                        continue
                    parsed = self._parse_image_path(str(image_path))
                    if parsed is None:
                        continue
                    zip_name = parsed["zip_name"]
                    if zip_name not in selected_names:
                        continue

                    member = parsed["member"]
                    zf = handles[zip_name]
                    try:
                        image_bytes = zf.read(member)
                    except Exception:
                        # Missing member OR unreadable/corrupt member
                        # (e.g. zlib "Error -3 while decompressing data"):
                        # degrade to missing image instead of aborting
                        # the whole stream.
                        missing_image += 1
                        image_bytes = None

                    sample_id = str(item.get("image_id") or os.path.splitext(member)[0])
                    metadata = dict(item)
                    metadata["index_line"] = line_number
                    zip_path = zip_by_name[zip_name]

                    yield Sample(
                        id=sample_id,
                        image_path=None,
                        text=item.get("caption"),
                        metadata=metadata,
                        image_bytes=image_bytes,
                        source={
                            "subset": zip_name,
                            "zip_path": zip_path,
                            "zip_file": zip_name,
                            "image": member,
                            "image_member": member,
                            "image_path": str(image_path),
                            "shard": parsed["shard"],
                        },
                    )
                    total += 1
                    if self.progress_every and total % self.progress_every == 0:
                        print(
                            f"[DenseFusion4V100kLoader] indexed {total} samples "
                            f"(latest zip: {zip_name})",
                            flush=True,
                        )
        finally:
            for zf in handles.values():
                try:
                    zf.close()
                except Exception:
                    pass

        if self.on_shard_complete:
            for zip_name in sorted(selected_names):
                self.on_shard_complete(zip_name)

        if missing_image:
            print(
                f"[DenseFusion4V100kLoader] warning: {missing_image} jsonl rows "
                f"had no matching zip member",
                flush=True,
            )
        print(
            f"[DenseFusion4V100kLoader] indexed {total} samples in this run "
            f"({len(selected_names)} zip shard(s))",
            flush=True,
        )


def _guess_image_ext(image_bytes: Optional[bytes], path_hint: Optional[str] = None) -> str:
    if path_hint:
        ext = os.path.splitext(path_hint)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            return ext
    data = image_bytes or b""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"BM":
        return ".bmp"
    return ".jpg"


def _load_wikiart_label_names(root_dir: str) -> Dict[str, List[str]]:
    """Read artist/genre/style class names from HuggingFace ``dataset_infos.json``."""
    path = os.path.join(root_dir, "dataset_infos.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            infos = json.load(f)
    except Exception:
        return {}
    if not isinstance(infos, dict):
        return {}
    for cfg in infos.values():
        if not isinstance(cfg, dict):
            continue
        features = cfg.get("features")
        if not isinstance(features, dict):
            continue
        names: Dict[str, List[str]] = {}
        for key in ("artist", "genre", "style"):
            feat = features.get(key)
            raw = feat.get("names") if isinstance(feat, dict) else None
            if isinstance(raw, list) and raw:
                names[key] = [str(item) for item in raw]
        if len(names) == 3:
            return names
    return {}


def _decode_class_label(value: object, names: List[str]) -> Tuple[Optional[int], str]:
    if isinstance(value, bool):
        return int(value), names[int(value)] if 0 <= int(value) < len(names) else str(value)
    if isinstance(value, int):
        if 0 <= value < len(names):
            return value, names[value]
        return value, str(value)
    if isinstance(value, str) and value.strip():
        return None, value
    return None, ""


def _extract_hf_image(image_val: object) -> Tuple[Optional[bytes], Optional[str]]:
    if image_val is None:
        return None, None
    if isinstance(image_val, (bytes, bytearray, memoryview)):
        return bytes(image_val), None
    if isinstance(image_val, dict):
        raw = image_val.get("bytes")
        data = bytes(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else None
        path = image_val.get("path")
        hint = str(path) if isinstance(path, str) and path else None
        return data, hint
    return None, None


class WikiArtLoader:
    """Stream WikiArt HuggingFace parquet shards without extracting to disk.

    Expected layout under ``root_dir``::

        dataset_infos.json
        data/train-00000-of-00072.parquet
        ...

    Each row has ``image`` (HF ``{bytes, path}``), plus integer class labels
    ``artist`` / ``genre`` / ``style``. Those labels are the original text
    fields and are kept in ``Sample.metadata`` for export; no English caption
    is synthesized.

    Use ``parquet_offset`` / ``parquet_limit`` or ``parquet_batch_size``
    (wired from CLI ``--tar-*`` flags) for shard batches.
    """

    def __init__(
        self,
        root_dir: str,
        parquet_glob: str = "*.parquet",
        progress_every: int = 5000,
        batch_size: int = 64,
        parquet_offset: int = 0,
        parquet_limit: Optional[int] = None,
        parquet_batch_size: Optional[int] = None,
        completed_parquet_names: Optional[Set[str]] = None,
        on_shard_complete: Optional[Callable[[str], None]] = None,
    ):
        self.root_dir = root_dir
        self.parquet_glob = parquet_glob
        self.progress_every = progress_every
        self.batch_size = max(1, batch_size)
        self.parquet_offset = max(0, parquet_offset)
        self.parquet_limit = parquet_limit
        self.parquet_batch_size = parquet_batch_size
        self.completed_parquet_names = completed_parquet_names
        self.on_shard_complete = on_shard_complete
        self.label_names = _load_wikiart_label_names(root_dir)

    def list_parquet_shards(self) -> List[str]:
        return _discover_archive_files(self.root_dir, self.parquet_glob)

    def list_tar_shards(self) -> List[str]:
        return self.list_parquet_shards()

    def _resolve_parquet_files(self) -> List[str]:
        all_files = self.list_parquet_shards()
        if not all_files:
            return []

        if self.parquet_batch_size:
            if self.completed_parquet_names is None:
                raise ValueError(
                    "parquet_batch_size requires checkpoint mode "
                    "(completed_parquet_names)."
                )
            pending = [
                path
                for path in all_files
                if os.path.basename(path) not in self.completed_parquet_names
            ]
            selected = pending[: self.parquet_batch_size]
            print(
                f"[WikiArtLoader] parquet batch: {len(selected)} shard(s) this run, "
                f"{len(self.completed_parquet_names)} shard(s) already finished, "
                f"{len(pending)} shard(s) pending in dataset",
                flush=True,
            )
            if selected:
                names = ", ".join(os.path.basename(p) for p in selected[:5])
                if len(selected) > 5:
                    names += ", ..."
                print(f"[WikiArtLoader] shards: {names}", flush=True)
            return selected

        if self.parquet_limit is not None:
            end = self.parquet_offset + self.parquet_limit
            selected = all_files[self.parquet_offset : end]
        elif self.parquet_offset:
            selected = all_files[self.parquet_offset :]
        else:
            selected = all_files

        if self.parquet_offset or self.parquet_limit is not None:
            print(
                f"[WikiArtLoader] parquet slice offset={self.parquet_offset} "
                f"limit={self.parquet_limit} -> {len(selected)} shard(s)",
                flush=True,
            )
        return selected

    def _iter_parquet_samples(self, parquet_path: str) -> Iterable[Sample]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "WikiArtLoader requires pyarrow. "
                "Install with: pip install pyarrow"
            ) from e

        parquet_name = os.path.basename(parquet_path)
        parquet_stem = os.path.splitext(parquet_name)[0]
        artist_names = self.label_names.get("artist") or []
        genre_names = self.label_names.get("genre") or []
        style_names = self.label_names.get("style") or []

        pf = pq.ParquetFile(parquet_path)
        row_index = 0
        for batch in pf.iter_batches(
            batch_size=self.batch_size,
            columns=["image", "artist", "genre", "style"],
        ):
            images = batch.column("image")
            artists = batch.column("artist")
            genres = batch.column("genre")
            styles = batch.column("style")
            for i in range(batch.num_rows):
                image_bytes, path_hint = _extract_hf_image(images[i].as_py())
                artist_id, artist = _decode_class_label(artists[i].as_py(), artist_names)
                genre_id, genre = _decode_class_label(genres[i].as_py(), genre_names)
                style_id, style = _decode_class_label(styles[i].as_py(), style_names)
                ext = _guess_image_ext(image_bytes, path_hint)
                sample_id = f"{parquet_stem}__{row_index}"
                image_fname = f"{sample_id}{ext}"
                yield Sample(
                    id=sample_id,
                    image_path=None,
                    text=None,
                    metadata={
                        "source_parquet": parquet_name,
                        "row_index": row_index,
                        "artist_id": artist_id,
                        "genre_id": genre_id,
                        "style_id": style_id,
                        "artist": artist,
                        "genre": genre,
                        "style": style,
                    },
                    image_bytes=image_bytes,
                    source={
                        "subset": parquet_name,
                        "parquet_path": parquet_path,
                        "parquet_file": parquet_name,
                        "image": image_fname,
                        "row_index": row_index,
                    },
                )
                row_index += 1

    def load(self) -> Iterable[Sample]:
        if not self.label_names:
            print(
                "[WikiArtLoader] warning: could not load artist/genre/style names "
                f"from {os.path.join(self.root_dir, 'dataset_infos.json')}",
                flush=True,
            )

        parquet_files = self._resolve_parquet_files()
        if not parquet_files:
            print("[WikiArtLoader] no parquet shards to process", flush=True)
            return

        total = 0
        for parquet_path in parquet_files:
            parquet_name = os.path.basename(parquet_path)
            shard_count = 0
            for sample in self._iter_parquet_samples(parquet_path):
                yield sample
                total += 1
                shard_count += 1
                if self.progress_every and total % self.progress_every == 0:
                    print(
                        f"[WikiArtLoader] indexed {total} samples "
                        f"(latest parquet: {parquet_name})",
                        flush=True,
                    )
            if self.on_shard_complete:
                self.on_shard_complete(parquet_name)
            print(
                f"[WikiArtLoader] finished shard {parquet_name} "
                f"({shard_count} samples indexed this shard)",
                flush=True,
            )
        print(f"[WikiArtLoader] indexed {total} samples in this run", flush=True)


def _index_images_by_stem(image_dir: str) -> Dict[str, str]:
    """Map filename stem -> path for image files under ``image_dir`` (non-recursive)."""
    index: Dict[str, str] = {}
    if not os.path.isdir(image_dir):
        return index
    image_exts = set(_IMAGE_EXTENSIONS)
    for name in os.listdir(image_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in image_exts:
            continue
        index[stem] = os.path.join(image_dir, name)
    return index


class AlchemistLoader:
    """Stream Alchemist samples from a parquet/csv index and a local ``images/`` folder.

    Expected layout under ``root_dir``::

        data/train-00000-of-00001.parquet
        data/alchemist_3k_final.csv          # fallback if parquet is missing
        images/{img_key}.jpg|.webp|.png|...

    Original ``prompt`` / ``url`` / ``img_key`` are kept as-is (no recaptioning).
    Rows whose image was not downloaded still yield a Sample so BrokenFilesFilter
    can drop them.
    """

    def __init__(
        self,
        root_dir: str,
        image_dir_name: str = "images",
        data_dir_name: str = "data",
        progress_every: int = 500,
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir_name)
        self.data_dir = os.path.join(root_dir, data_dir_name)
        self.progress_every = progress_every

    def _parquet_path(self) -> Optional[str]:
        candidates = [
            os.path.join(self.data_dir, "train-00000-of-00001.parquet"),
        ]
        if os.path.isdir(self.data_dir):
            candidates.extend(
                sorted(glob.glob(os.path.join(self.data_dir, "*.parquet")))
            )
        seen = set()
        for path in candidates:
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            return path
        return None

    def _csv_path(self) -> Optional[str]:
        preferred = os.path.join(self.data_dir, "alchemist_3k_final.csv")
        if os.path.isfile(preferred):
            return preferred
        if not os.path.isdir(self.data_dir):
            return None
        csvs = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        return csvs[0] if csvs else None

    def _iter_parquet_rows(self, parquet_path: str) -> Iterable[Dict[str, object]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "AlchemistLoader requires pyarrow to read parquet. "
                "Install with: pip install pyarrow"
            ) from e
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=256):
            names = list(batch.schema.names)
            for i in range(batch.num_rows):
                yield {name: batch.column(name)[i].as_py() for name in names}

    def _iter_csv_rows(self, csv_path: str) -> Iterable[Dict[str, object]]:
        import csv

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                yield {
                    (key if key else "index"): (value if value != "" else None)
                    for key, value in row.items()
                }

    def _iter_records(self) -> Iterable[Dict[str, object]]:
        parquet_path = self._parquet_path()
        if parquet_path:
            print(f"[AlchemistLoader] index: {parquet_path}", flush=True)
            yield from self._iter_parquet_rows(parquet_path)
            return
        csv_path = self._csv_path()
        if csv_path:
            print(f"[AlchemistLoader] index (csv fallback): {csv_path}", flush=True)
            yield from self._iter_csv_rows(csv_path)
            return
        print(
            f"[AlchemistLoader] missing parquet/csv under {self.data_dir}",
            flush=True,
        )

    def load(self) -> Iterable[Sample]:
        image_index = _index_images_by_stem(self.image_dir)
        print(
            f"[AlchemistLoader] found {len(image_index)} images in {self.image_dir}",
            flush=True,
        )
        total = 0
        missing_image = 0
        for line_number, item in enumerate(self._iter_records(), start=1):
            img_key = item.get("img_key")
            if not img_key:
                continue
            img_key = str(img_key)
            image_path = image_index.get(img_key)
            if image_path is None:
                missing_image += 1
            image_name = (
                os.path.basename(image_path) if image_path else f"{img_key}.jpg"
            )
            prompt = item.get("prompt")
            if prompt is not None:
                prompt = str(prompt)
            metadata = dict(item)
            metadata["index_line"] = line_number
            yield Sample(
                id=img_key,
                image_path=image_path,
                text=prompt,
                metadata=metadata,
                source={
                    "subset": "images",
                    "image": image_name,
                    "img_key": img_key,
                },
            )
            total += 1
            if self.progress_every and total % self.progress_every == 0:
                print(f"[AlchemistLoader] indexed {total} samples", flush=True)
        if missing_image:
            print(
                f"[AlchemistLoader] warning: {missing_image} rows have no local image",
                flush=True,
            )
        print(f"[AlchemistLoader] indexed {total} samples in this run", flush=True)


def _jsonable_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return str(value)
    return value


class DiffusionDB2MRandom50kLoader:
    """Stream DiffusionDB 2M random-50k from parquet metadata + extracted images.

    Expected layout under ``root_dir``::

        data/train-00000-of-00066-*.parquet
        images/{parquet_stem}_idx{row:06d}.png
        images/{parquet_stem}_idx{row:06d}.txt

    Images must be extracted from the parquet ``image`` column beforehand; the
    parquet ``image`` bytes are not read here, and rows without an extracted PNG
    are reported as missing images. Original ``prompt`` and generation fields
    (seed, cfg, sampler, ...) are kept as-is.
    """

    _META_COLUMNS = (
        "prompt",
        "seed",
        "step",
        "cfg",
        "sampler",
        "width",
        "height",
        "user_name",
        "timestamp",
        "image_nsfw",
        "prompt_nsfw",
    )

    def __init__(
        self,
        root_dir: str,
        image_dir_name: str = "images",
        parquet_glob: str = "*.parquet",
        progress_every: int = 2000,
        batch_size: int = 256,
        parquet_offset: int = 0,
        parquet_limit: Optional[int] = None,
        parquet_batch_size: Optional[int] = None,
        completed_parquet_names: Optional[Set[str]] = None,
        on_shard_complete: Optional[Callable[[str], None]] = None,
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir_name)
        self.parquet_glob = parquet_glob
        self.progress_every = progress_every
        self.batch_size = max(1, batch_size)
        self.parquet_offset = max(0, parquet_offset)
        self.parquet_limit = parquet_limit
        self.parquet_batch_size = parquet_batch_size
        self.completed_parquet_names = completed_parquet_names
        self.on_shard_complete = on_shard_complete

    def list_parquet_shards(self) -> List[str]:
        return _discover_archive_files(self.root_dir, self.parquet_glob)

    def list_tar_shards(self) -> List[str]:
        return self.list_parquet_shards()

    def _resolve_parquet_files(self) -> List[str]:
        all_files = self.list_parquet_shards()
        if not all_files:
            return []

        if self.parquet_batch_size:
            if self.completed_parquet_names is None:
                raise ValueError(
                    "parquet_batch_size requires checkpoint mode "
                    "(completed_parquet_names)."
                )
            pending = [
                path
                for path in all_files
                if os.path.basename(path) not in self.completed_parquet_names
            ]
            selected = pending[: self.parquet_batch_size]
            print(
                f"[DiffusionDB2MRandom50kLoader] parquet batch: {len(selected)} "
                f"shard(s) this run, {len(self.completed_parquet_names)} shard(s) "
                f"already finished, {len(pending)} shard(s) pending in dataset",
                flush=True,
            )
            if selected:
                names = ", ".join(os.path.basename(p) for p in selected[:5])
                if len(selected) > 5:
                    names += ", ..."
                print(f"[DiffusionDB2MRandom50kLoader] shards: {names}", flush=True)
            return selected

        if self.parquet_limit is not None:
            end = self.parquet_offset + self.parquet_limit
            selected = all_files[self.parquet_offset : end]
        elif self.parquet_offset:
            selected = all_files[self.parquet_offset :]
        else:
            selected = all_files

        if self.parquet_offset or self.parquet_limit is not None:
            print(
                f"[DiffusionDB2MRandom50kLoader] parquet slice "
                f"offset={self.parquet_offset} limit={self.parquet_limit} "
                f"-> {len(selected)} shard(s)",
                flush=True,
            )
        return selected

    def _iter_parquet_samples(self, parquet_path: str) -> Iterable[Sample]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "DiffusionDB2MRandom50kLoader requires pyarrow. "
                "Install with: pip install pyarrow"
            ) from e

        parquet_name = os.path.basename(parquet_path)
        parquet_stem = os.path.splitext(parquet_name)[0]
        pf = pq.ParquetFile(parquet_path)
        available = set(pf.schema_arrow.names)
        columns = [c for c in self._META_COLUMNS if c in available]
        row_index = 0
        for batch in pf.iter_batches(batch_size=self.batch_size, columns=columns):
            for i in range(batch.num_rows):
                record = {
                    name: _jsonable_value(batch.column(name)[i].as_py())
                    for name in columns
                }
                prompt = record.get("prompt")
                if prompt is not None:
                    prompt = str(prompt)
                image_stem = f"{parquet_stem}_idx{row_index:06d}"
                image_fname = image_stem + ".png"
                image_path = os.path.join(self.image_dir, image_fname)
                if not os.path.isfile(image_path):
                    image_path = None
                metadata = {
                    "source_parquet": parquet_name,
                    "row_index": row_index,
                    **record,
                }
                yield Sample(
                    id=image_stem,
                    image_path=image_path,
                    text=prompt,
                    metadata=metadata,
                    source={
                        "loader": "diffusiondb_2m_random_50k",
                        "subset": parquet_name,
                        "parquet_path": parquet_path,
                        "parquet_file": parquet_name,
                        "image": image_fname,
                        "row_index": row_index,
                    },
                )
                row_index += 1

    def load(self) -> Iterable[Sample]:
        parquet_files = self._resolve_parquet_files()
        if not parquet_files:
            print(
                "[DiffusionDB2MRandom50kLoader] no parquet shards to process",
                flush=True,
            )
            return

        total = 0
        missing_image = 0
        for parquet_path in parquet_files:
            parquet_name = os.path.basename(parquet_path)
            shard_count = 0
            for sample in self._iter_parquet_samples(parquet_path):
                if not sample.image_path:
                    missing_image += 1
                yield sample
                total += 1
                shard_count += 1
                if self.progress_every and total % self.progress_every == 0:
                    print(
                        f"[DiffusionDB2MRandom50kLoader] indexed {total} samples "
                        f"(latest parquet: {parquet_name})",
                        flush=True,
                    )
            if self.on_shard_complete:
                self.on_shard_complete(parquet_name)
            print(
                f"[DiffusionDB2MRandom50kLoader] finished shard {parquet_name} "
                f"({shard_count} samples indexed this shard)",
                flush=True,
            )
        if missing_image:
            print(
                f"[DiffusionDB2MRandom50kLoader] warning: {missing_image} rows "
                f"have no extracted PNG under {self.image_dir}",
                flush=True,
            )
        print(
            f"[DiffusionDB2MRandom50kLoader] indexed {total} samples in this run",
            flush=True,
        )
