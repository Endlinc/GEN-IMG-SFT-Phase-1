"""Concrete filter implementations."""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .core import DroppedRecord, Filter, FilterResult, Sample


class BrokenFilesFilter(Filter):
    """Drop samples with missing/broken images or missing/empty text."""

    def __init__(
        self,
        enabled: bool = True,
        check_missing_image: bool = True,
        check_missing_text: bool = True,
        check_image_loadable: bool = True,
        min_text_length: int = 0,
    ):
        super().__init__(enabled)
        self.name = "BrokenFilesFilter"
        self.check_missing_image = check_missing_image
        self.check_missing_text = check_missing_text
        self.check_image_loadable = check_image_loadable
        self.min_text_length = min_text_length

    def filter(self, sample: Sample) -> FilterResult:
        if self.check_missing_image:
            path_ok = bool(
                sample.image_path and os.path.exists(sample.image_path)
            )
            source = sample.source if isinstance(sample.source, dict) else {}
            tar_ok = bool(
                source.get("tar_path")
                and source.get("image_member")
                and os.path.exists(str(source["tar_path"]))
            )
            zip_ok = bool(
                source.get("zip_path")
                and source.get("image_member")
                and os.path.exists(str(source["zip_path"]))
            )
            bytes_ok = bool(sample.image_bytes)
            if not path_ok and not tar_ok and not zip_ok and not bytes_ok:
                return FilterResult(
                    False,
                    "missing_image",
                    {"image_path": sample.image_path, "source": source},
                )

        if self.check_image_loadable:
            img = sample.load_image()
            if img is None:
                return FilterResult(
                    False, "broken_image", {"image_path": sample.image_path}
                )

        if self.check_missing_text:
            text = sample.text or ""
            text_len = len(text.strip())
            if text_len == 0:
                # WikiArt keeps original labels in metadata instead of a caption.
                meta = sample.metadata if isinstance(sample.metadata, dict) else {}
                has_labels = any(
                    isinstance(meta.get(key), str) and str(meta.get(key)).strip()
                    for key in ("artist", "genre", "style", "caption", "prompt")
                )
                if not has_labels:
                    return FilterResult(
                        False, "missing_or_empty_text", {"text_length": text_len}
                    )
            elif self.min_text_length > 0 and text_len < self.min_text_length:
                return FilterResult(
                    False,
                    "text_too_short",
                    {"text_length": text_len, "min_text_length": self.min_text_length},
                )

        return FilterResult(True)


class ResolutionFilter(Filter):
    """Drop images whose resolution is outside the configured bounds."""

    def __init__(
        self,
        enabled: bool = True,
        min_width: int = 0,
        min_height: int = 0,
        max_width: int = 0,
        max_height: int = 0,
        min_pixels: int = 0,
        max_pixels: int = 0,
    ):
        super().__init__(enabled)
        self.name = "ResolutionFilter"
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width or None
        self.max_height = max_height or None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels or None

    def filter(self, sample: Sample) -> FilterResult:
        img = sample.load_image()
        if img is None:
            return FilterResult(False, "cannot_load_image", {})

        w, h = img.size
        pixels = w * h
        details = {"width": w, "height": h, "pixels": pixels}

        if self.min_width and w < self.min_width:
            return FilterResult(
                False, "width_too_small", {**details, "min_width": self.min_width}
            )
        if self.min_height and h < self.min_height:
            return FilterResult(
                False, "height_too_small", {**details, "min_height": self.min_height}
            )
        if self.max_width and w > self.max_width:
            return FilterResult(
                False, "width_too_large", {**details, "max_width": self.max_width}
            )
        if self.max_height and h > self.max_height:
            return FilterResult(
                False, "height_too_large", {**details, "max_height": self.max_height}
            )
        if self.min_pixels and pixels < self.min_pixels:
            return FilterResult(
                False, "pixels_too_small", {**details, "min_pixels": self.min_pixels}
            )
        if self.max_pixels and pixels > self.max_pixels:
            return FilterResult(
                False, "pixels_too_large", {**details, "max_pixels": self.max_pixels}
            )

        return FilterResult(True)


class AspectRatioFilter(Filter):
    """Drop images whose aspect ratio (width/height) is outside configured bounds."""

    def __init__(
        self,
        enabled: bool = True,
        min_ratio: float = 0.0,
        max_ratio: float = float("inf"),
        allowed_ratios: Optional[List[str]] = None,
        tolerance: float = 1e-3,
    ):
        super().__init__(enabled)
        self.name = "AspectRatioFilter"
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.allowed_ratios = allowed_ratios or []
        self.tolerance = tolerance

    def filter(self, sample: Sample) -> FilterResult:
        img = sample.load_image()
        if img is None:
            return FilterResult(False, "cannot_load_image", {})

        w, h = img.size
        if h == 0:
            return FilterResult(False, "invalid_height", {"width": w, "height": h})
        ratio = w / h
        details = {"width": w, "height": h, "aspect_ratio": ratio}

        if ratio < self.min_ratio:
            return FilterResult(
                False, "aspect_ratio_too_small", {**details, "min_ratio": self.min_ratio}
            )
        if ratio > self.max_ratio:
            return FilterResult(
                False, "aspect_ratio_too_large", {**details, "max_ratio": self.max_ratio}
            )

        if self.allowed_ratios:
            allowed_normalized = set()
            for ar in self.allowed_ratios:
                try:
                    a, b = ar.split(":")
                    allowed_normalized.add(float(a) / float(b))
                except Exception:
                    continue
            if allowed_normalized and not any(
                abs(ratio - r) <= self.tolerance for r in allowed_normalized
            ):
                return FilterResult(
                    False,
                    "aspect_ratio_not_allowed",
                    {**details, "allowed_ratios": self.allowed_ratios},
                )

        return FilterResult(True)


class DuplicateFilter(Filter):
    """Drop exact or conservative near-duplicate images using perceptual hashing.

    ``average_hash`` is deliberately not the default: it is only a 64-bit global
    brightness signature, so unrelated images with similar light/dark layouts can
    collide even at distance zero.  The ``hybrid`` mode requires both pHash and
    dHash to agree, which makes this filter precision-oriented (a better default
    for destructive data cleaning).

    Hash computation (decode + pHash/dHash) can run in a thread pool via
    ``hash_workers``; index lookup/update stays serial so "first occurrence wins".
    """

    def __init__(
        self,
        enabled: bool = True,
        hash_size: int = 16,
        threshold: int = 6,
        method: str = "hybrid",
        secondary_threshold: Optional[int] = None,
        hash_workers: int = 1,
        hash_batch_size: int = 32,
    ):
        super().__init__(enabled)
        if hash_size < 8:
            raise ValueError("hash_size must be at least 8")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if hash_workers < 1:
            raise ValueError("hash_workers must be >= 1")
        if hash_batch_size < 1:
            raise ValueError("hash_batch_size must be >= 1")

        self.hash_size = hash_size
        self.threshold = threshold
        self.secondary_threshold = (
            threshold if secondary_threshold is None else secondary_threshold
        )
        self.method = method.lower()
        self.hash_workers = int(hash_workers)
        self.hash_batch_size = int(hash_batch_size)
        self.name = (
            f"DuplicateFilter(method={method}, threshold={threshold}, "
            f"secondary_threshold={self.secondary_threshold})"
        )
        self._hashes: Dict[str, Any] = {}
        # In hybrid mode this index avoids the previous O(N^2) all-pairs scan.
        # Eight 32-bit pHash bands are used.  With the default distance <= 6,
        # a true candidate must share at least one unchanged band.
        self._band_index: Dict[str, List[str]] = {}
        self._hash_executor: Optional[ThreadPoolExecutor] = None

    def _get_hash_executor(self) -> Optional[ThreadPoolExecutor]:
        if self.hash_workers <= 1:
            return None
        if self._hash_executor is None:
            self._hash_executor = ThreadPoolExecutor(
                max_workers=self.hash_workers,
                thread_name_prefix="dup-hash",
            )
        return self._hash_executor

    def close(self) -> None:
        if self._hash_executor is not None:
            self._hash_executor.shutdown(wait=False)
            self._hash_executor = None

    def _compute_hash(self, img) -> Any:
        import imagehash

        if self.method == "hybrid":
            return (
                imagehash.phash(img, hash_size=self.hash_size),
                imagehash.dhash(img, hash_size=self.hash_size),
            )
        if self.method == "average":
            return imagehash.average_hash(img, hash_size=self.hash_size)
        if self.method == "difference":
            return imagehash.dhash(img, hash_size=self.hash_size)
        if self.method in ("perception", "phash"):
            return imagehash.phash(img, hash_size=self.hash_size)
        if self.method == "wavelet":
            return imagehash.whash(img, hash_size=self.hash_size)
        raise ValueError(f"Unknown hash method: {self.method}")

    def _hash_sample(self, sample: Sample) -> Tuple[str, Any, Optional[str]]:
        """Return ``(sample_id, hash_or_None, skip_reason)``.

        ``skip_reason`` is set when the sample should be kept without indexing
        (broken image / hash failure); BrokenFilesFilter owns true rejects.
        """
        img = sample.load_image()
        if img is None:
            return sample.id, None, "no_image"
        try:
            return sample.id, self._compute_hash(img), None
        except Exception:
            return sample.id, None, "hash_error"
    @staticmethod
    def _band_keys(perceptual_hash: Any) -> List[str]:
        """Return eight 32-bit LSH bands for a 16x16 perceptual hash."""
        encoded = str(perceptual_hash)
        band_width = max(1, len(encoded) // 8)
        return [
            f"{i}:{encoded[i : i + band_width]}"
            for i in range(0, len(encoded), band_width)
        ]

    def _find_hybrid_duplicate(self, h: Tuple[Any, Any]) -> Tuple[Optional[str], Dict[str, int]]:
        primary_hash, secondary_hash = h
        candidate_ids = set()
        for key in self._band_keys(primary_hash):
            candidate_ids.update(self._band_index.get(key, []))

        for ref_id in candidate_ids:
            ref_primary, ref_secondary = self._hashes[ref_id]
            primary_distance = primary_hash - ref_primary
            secondary_distance = secondary_hash - ref_secondary
            if (
                primary_distance <= self.threshold
                and secondary_distance <= self.secondary_threshold
            ):
                return ref_id, {
                    "phash_distance": int(primary_distance),
                    "dhash_distance": int(secondary_distance),
                }
        return None, {}

    def _index_hybrid_hash(self, sample_id: str, h: Tuple[Any, Any]) -> None:
        self._hashes[sample_id] = h
        for key in self._band_keys(h[0]):
            self._band_index.setdefault(key, []).append(sample_id)

    def _compute_hashes(
        self, samples: List[Sample]
    ) -> List[Tuple[str, Any, Optional[str]]]:
        """Compute perceptual hashes, optionally in parallel."""
        if len(samples) <= 1 or self.hash_workers <= 1:
            return [self._hash_sample(s) for s in samples]
        workers = min(self.hash_workers, len(samples))
        executor = self._get_hash_executor()
        if executor is None:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(self._hash_sample, samples))
        return list(executor.map(self._hash_sample, samples))

    def apply(
        self, samples: Iterable[Sample]
    ) -> Tuple[List[Sample], List[DroppedRecord]]:
        if not self.enabled:
            return list(samples), []

        sample_list = list(samples)
        kept: List[Sample] = []
        dropped: List[DroppedRecord] = []
        hash_rows = self._compute_hashes(sample_list)

        for sample, (_sid, h, skip_reason) in zip(sample_list, hash_rows):
            if skip_reason is not None or h is None:
                # Let BrokenFilesFilter handle broken images; keep here to avoid false duplicates.
                kept.append(sample)
                continue

            distance_details: Dict[str, int] = {}
            if self.method == "hybrid":
                duplicate_of, distance_details = self._find_hybrid_duplicate(h)
            else:
                # Kept for backwards compatibility.  ``hybrid`` should be used
                # for a production run because the legacy methods are less
                # discriminative and still require an all-pairs comparison.
                if self.threshold > 0:
                    warnings.warn(
                        "Legacy duplicate hash methods are collision-prone; "
                        "prefer method='hybrid'.",
                        RuntimeWarning,
                    )
                duplicate_of = None
                for ref_id, ref_h in self._hashes.items():
                    distance = h - ref_h
                    if distance <= self.threshold:
                        duplicate_of = ref_id
                        distance_details = {"hash_distance": int(distance)}
                        break

            if duplicate_of is None:
                if self.method == "hybrid":
                    self._index_hybrid_hash(sample.id, h)
                else:
                    self._hashes[sample.id] = h
                kept.append(sample)
            else:
                dropped.append(
                    DroppedRecord(
                        sample_id=sample.id,
                        filter_name=self.name,
                        reason="duplicate_or_near_duplicate",
                        details={
                            "duplicate_of": duplicate_of,
                            "hash": str(h[0]) if self.method == "hybrid" else str(h),
                            "threshold": self.threshold,
                            "secondary_threshold": self.secondary_threshold,
                            "method": self.method,
                            **distance_details,
                        },
                        source=sample.source,
                    )
                )

        return kept, dropped

    def export_state(self) -> Dict[str, Any]:
        """Serialize duplicate index for checkpoint resume."""
        if not self.enabled:
            return {"enabled": False, "method": self.method}
        if self.method == "hybrid":
            return {
                "enabled": True,
                "method": "hybrid",
                "hash_size": self.hash_size,
                "hashes": {
                    sid: [str(h[0]), str(h[1])]
                    for sid, h in self._hashes.items()
                },
                "band_index": self._band_index,
            }
        return {
            "enabled": True,
            "method": self.method,
            "hash_size": self.hash_size,
            "hashes": {sid: str(h) for sid, h in self._hashes.items()},
            "band_index": {},
        }

    def load_state(self, data: Dict[str, Any]) -> None:
        if not data or not data.get("enabled", True):
            return
        import imagehash

        method = data.get("method", self.method)
        if method == "hybrid":
            self._hashes = {}
            for sid, pair in (data.get("hashes") or {}).items():
                self._hashes[sid] = (
                    imagehash.hex_to_hash(pair[0]),
                    imagehash.hex_to_hash(pair[1]),
                )
        else:
            self._hashes = {
                sid: imagehash.hex_to_hash(hstr)
                for sid, hstr in (data.get("hashes") or {}).items()
            }
        self._band_index = {
            key: list(ids) for key, ids in (data.get("band_index") or {}).items()
        }


class LanguageFilter(Filter):
    """Drop captions whose detected language is not in the allowed list."""

    def __init__(
        self,
        enabled: bool = True,
        allowed_languages: Optional[List[str]] = None,
        fallback_keep: bool = True,
        min_text_length: int = 5,
    ):
        super().__init__(enabled)
        self.name = "LanguageFilter"
        self.allowed_languages = set(allowed_languages or ["en", "zh", "zh-cn", "zh-tw"])
        self.fallback_keep = fallback_keep
        self.min_text_length = min_text_length
        self._detector = None
        try:
            from langdetect import detect

            self._detector = detect
        except ImportError:
            pass

    def _normalize_lang(self, lang: str) -> str:
        lang = lang.lower()
        if lang.startswith("zh"):
            return "zh"
        return lang

    def filter(self, sample: Sample) -> FilterResult:
        text = sample.text or ""
        text_len = len(text.strip())
        if text_len < self.min_text_length:
            if self.fallback_keep:
                return FilterResult(True)
            return FilterResult(
                False, "text_too_short", {"text_length": text_len}
            )

        if self._detector is None:
            # No language detector available; keep sample unless configured otherwise.
            return FilterResult(True)

        try:
            detected = self._detector(text)
            detected = self._normalize_lang(detected)
        except Exception as e:
            if self.fallback_keep:
                return FilterResult(True)
            return FilterResult(
                False, "language_detection_failed", {"error": str(e)}
            )

        if detected in self.allowed_languages:
            return FilterResult(True)

        return FilterResult(
            False,
            "language_not_allowed",
            {
                "detected_language": detected,
                "allowed_languages": sorted(self.allowed_languages),
            },
        )


class NSFWFilter(Filter):
    """Drop images classified as NSFW using a pretrained image classifier."""

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "Falconsai/nsfw_image_detection",
        threshold: float = 0.5,
        device: str = "auto",
        batch_size: int = 8,
    ):
        super().__init__(enabled)
        self.name = "NSFWFilter"
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.batch_size = batch_size
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as e:
            raise ImportError(
                "NSFWFilter requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from e

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_name)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    def apply(
        self, samples: Iterable[Sample]
    ) -> Tuple[List[Sample], List[DroppedRecord]]:
        if not self.enabled:
            return list(samples), []
        self._load()
        import torch

        samples = list(samples)
        kept: List[Sample] = []
        dropped: List[DroppedRecord] = []

        id2label = self._model.config.id2label
        nsfw_idx = next(
            (int(k) for k, v in id2label.items() if "nsfw" in v.lower()), 1
        )

        batch_images: List[Image.Image] = []
        batch_samples: List[Sample] = []

        def flush_batch() -> None:
            if not batch_images:
                return
            inputs = self._processor(batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            for j, sample in enumerate(batch_samples):
                nsfw_prob = float(probs[j, nsfw_idx].cpu())
                if nsfw_prob > self.threshold:
                    dropped.append(
                        DroppedRecord(
                            sample_id=sample.id,
                            filter_name=self.name,
                            reason="nsfw_detected",
                            details={
                                "nsfw_score": round(nsfw_prob, 6),
                                "threshold": self.threshold,
                                "model": self.model_name,
                            },
                            source=sample.source,
                        )
                    )
                else:
                    kept.append(sample)
                sample.image = None
            batch_images.clear()
            batch_samples.clear()

        for sample in samples:
            img = sample.load_image()
            if img is None:
                kept.append(sample)
                continue
            batch_images.append(img.convert("RGB"))
            batch_samples.append(sample)
            if len(batch_images) >= self.batch_size:
                flush_batch()
        flush_batch()

        return kept, dropped


class WatermarkFilter(Filter):
    """Drop images that contain a visible watermark using a SigLIP2 classifier.

    The default model is prithivMLmods/Watermark-Detection-SigLIP2, which outputs
    a binary classification: "No Watermark" / "Watermark".
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "prithivMLmods/Watermark-Detection-SigLIP2",
        threshold: float = 0.98,
        min_margin: float = 0.2,
        device: str = "auto",
        batch_size: int = 8,
    ):
        super().__init__(enabled)
        self.name = "WatermarkFilter"
        self.model_name = model_name
        self.threshold = threshold
        self.min_margin = min_margin
        self.device = device
        self.batch_size = batch_size
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as e:
            raise ImportError(
                "WatermarkFilter requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from e

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_name)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _resolve_label_indices(id2label: Dict[Any, str]) -> Tuple[int, Optional[int]]:
        """Resolve positive/negative classes without matching ``No Watermark``.

        A substring-only lookup considers both labels positive.  For the current
        model, that silently selected label 0 (``No Watermark``) and inverted
        the filter decision.
        """
        negative_markers = ("no ", "non-", "non ", "without", "not ")
        normalized = {
            int(index): str(label).strip().lower().replace("_", " ")
            for index, label in id2label.items()
        }
        positive = [
            index
            for index, label in normalized.items()
            if "watermark" in label
            and not any(marker in label for marker in negative_markers)
        ]
        if len(positive) != 1:
            raise ValueError(
                "Could not unambiguously resolve the positive watermark class "
                f"from id2label={id2label}"
            )
        negative = [
            index
            for index, label in normalized.items()
            if "watermark" in label and index != positive[0]
        ]
        return positive[0], negative[0] if len(negative) == 1 else None

    def apply(
        self, samples: Iterable[Sample]
    ) -> Tuple[List[Sample], List[DroppedRecord]]:
        if not self.enabled:
            return list(samples), []
        self._load()
        import torch

        samples = list(samples)
        kept: List[Sample] = []
        dropped: List[DroppedRecord] = []

        id2label = self._model.config.id2label
        watermark_idx, no_watermark_idx = self._resolve_label_indices(id2label)

        batch_images: List[Image.Image] = []
        batch_samples: List[Sample] = []

        def flush_batch() -> None:
            if not batch_images:
                return
            inputs = self._processor(batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            for j, sample in enumerate(batch_samples):
                watermark_prob = float(probs[j, watermark_idx].cpu())
                no_watermark_prob = (
                    float(probs[j, no_watermark_idx].cpu())
                    if no_watermark_idx is not None
                    else None
                )
                margin = (
                    watermark_prob - no_watermark_prob
                    if no_watermark_prob is not None
                    else watermark_prob
                )
                if watermark_prob >= self.threshold and margin >= self.min_margin:
                    dropped.append(
                        DroppedRecord(
                            sample_id=sample.id,
                            filter_name=self.name,
                            reason="watermark_detected",
                            details={
                                "watermark_score": round(watermark_prob, 6),
                                "no_watermark_score": (
                                    round(no_watermark_prob, 6)
                                    if no_watermark_prob is not None
                                    else None
                                ),
                                "score_margin": round(margin, 6),
                                "threshold": self.threshold,
                                "min_margin": self.min_margin,
                                "watermark_label": id2label[watermark_idx],
                                "model": self.model_name,
                            },
                            source=sample.source,
                        )
                    )
                else:
                    kept.append(sample)
                sample.image = None
            batch_images.clear()
            batch_samples.clear()

        for sample in samples:
            img = sample.load_image()
            if img is None:
                kept.append(sample)
                continue
            batch_images.append(img.convert("RGB"))
            batch_samples.append(sample)
            if len(batch_images) >= self.batch_size:
                flush_batch()
        flush_batch()

        return kept, dropped


class CompressionQualityFilter(Filter):
    """Drop images with severe compression artifacts.

    Metrics computed per image:
      - blockiness: strength of 8x8 grid boundary artifacts (JPEG blockiness)
      - high_freq_ratio: ratio of high-frequency FFT energy to low-frequency energy,
        which tends to rise when ringing or block noise is strong
    """

    def __init__(
        self,
        enabled: bool = True,
        max_blockiness: float = 0.0,
        max_high_freq_ratio: float = 0.0,
    ):
        super().__init__(enabled)
        self.name = "CompressionQualityFilter"
        self.max_blockiness = max_blockiness
        self.max_high_freq_ratio = max_high_freq_ratio

    def _compute_metrics(self, img: Image.Image) -> Dict[str, float]:
        gray = np.array(img.convert("L")).astype(np.float32)

        # 1. Blockiness: compare gradients at 8x8 block boundaries with inner gradients.
        h_grad = np.abs(np.diff(gray, axis=1))
        v_grad = np.abs(np.diff(gray, axis=0))

        h_bound_mask = np.zeros_like(h_grad, dtype=bool)
        h_bound_mask[:, 7::8] = True
        v_bound_mask = np.zeros_like(v_grad, dtype=bool)
        v_bound_mask[7::8, :] = True

        if h_bound_mask.any() and v_bound_mask.any():
            bound_grad = (
                np.mean(h_grad[h_bound_mask]) + np.mean(v_grad[v_bound_mask])
            ) / 2.0
            non_bound_grad = (
                np.mean(h_grad[~h_bound_mask]) + np.mean(v_grad[~v_bound_mask])
            ) / 2.0
            blockiness = float(bound_grad / (non_bound_grad + 1e-6))
        else:
            blockiness = 0.0

        # 2. High-frequency energy ratio via FFT.
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        fh, fw = magnitude.shape
        cy, cx = fh // 2, fw // 2
        y, x = np.ogrid[:fh, :fw]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        low_energy = float(np.sum(magnitude[r < min(fh, fw) * 0.1]))
        high_energy = float(np.sum(magnitude[r > min(fh, fw) * 0.3]))
        high_freq_ratio = high_energy / (low_energy + 1e-6)

        return {
            "blockiness": round(blockiness, 4),
            "high_freq_ratio": round(high_freq_ratio, 4),
        }

    def filter(self, sample: Sample) -> FilterResult:
        img = sample.load_image()
        if img is None:
            return FilterResult(True)

        metrics = self._compute_metrics(img)
        details = {**metrics}

        if self.max_blockiness and metrics["blockiness"] > self.max_blockiness:
            return FilterResult(
                False, "too_blocky", {**details, "max_blockiness": self.max_blockiness}
            )
        if self.max_high_freq_ratio and metrics["high_freq_ratio"] > self.max_high_freq_ratio:
            return FilterResult(
                False, "too_much_artifacts", {**details, "max_high_freq_ratio": self.max_high_freq_ratio}
            )

        return FilterResult(True)


class BlurFilter(Filter):
    """Drop overly blurry images using Laplacian variance.

    The image is converted to grayscale, optionally resized to a fixed square
    size for resolution-invariant comparison, then a 3x3 Laplacian is applied.
    Low variance indicates weak edges / blur.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_laplacian_var: float = 200.0,
        resize_to: int = 512,
    ):
        super().__init__(enabled)
        self.name = "BlurFilter"
        self.min_laplacian_var = min_laplacian_var
        self.resize_to = resize_to

    def _compute_laplacian_var(self, img: Image.Image) -> float:
        gray = img.convert("L")
        if self.resize_to and self.resize_to > 0:
            gray = gray.resize((self.resize_to, self.resize_to), Image.BILINEAR)
        arr = np.array(gray, dtype=np.float64)
        padded = np.pad(arr, 1, mode="edge")
        lap = (
            -4 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        return float(lap.var())

    def filter(self, sample: Sample) -> FilterResult:
        img = sample.load_image()
        if img is None:
            return FilterResult(True)

        laplacian_var = self._compute_laplacian_var(img)
        details = {
            "laplacian_var": round(laplacian_var, 4),
            "min_laplacian_var": self.min_laplacian_var,
            "resize_to": self.resize_to,
        }
        if laplacian_var < self.min_laplacian_var:
            return FilterResult(False, "too_blurry", details)
        return FilterResult(True)


class EntropyFilter(Filter):
    """Drop images whose grayscale entropy is outside the desired range.

    Shannon entropy is computed from the grayscale histogram. Very low entropy
    often indicates overly uniform/blank/over-blurred images, while very high
    entropy can indicate excessive noise or random texture.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_entropy: float = 0.0,
        max_entropy: float = 8.0,
    ):
        super().__init__(enabled)
        self.name = "EntropyFilter"
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def _compute_entropy(self, img: Image.Image) -> float:
        arr = np.array(img.convert("L")).astype(np.float32)
        hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))
        return entropy

    def filter(self, sample: Sample) -> FilterResult:
        img = sample.load_image()
        if img is None:
            return FilterResult(True)

        entropy = self._compute_entropy(img)
        details = {"entropy": round(entropy, 4)}

        if self.min_entropy and entropy < self.min_entropy:
            return FilterResult(
                False, "too_low_entropy", {**details, "min_entropy": self.min_entropy}
            )
        if self.max_entropy and entropy > self.max_entropy:
            return FilterResult(
                False, "too_high_entropy", {**details, "max_entropy": self.max_entropy}
            )

        return FilterResult(True)
