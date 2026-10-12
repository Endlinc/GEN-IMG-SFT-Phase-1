"""Core data structures and pipeline logic."""

import io
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


def read_bytes_from_tar_archive(tar_path: str, member_name: str) -> bytes:
    """Read a single member from a tar archive."""
    import tarfile

    with tarfile.open(tar_path, "r") as tf:
        member = tf.getmember(member_name)
        fileobj = tf.extractfile(member)
        if fileobj is None:
            raise OSError(f"Cannot extract {member_name!r} from {tar_path}")
        return fileobj.read()


def read_bytes_from_zip_archive(zip_path: str, member_name: str) -> bytes:
    """Read a single member from a zip archive."""
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(member_name)


@dataclass
class Sample:
    """A single dataset sample normalized for filtering."""

    id: str
    image_path: Optional[str] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    image: Optional[Image.Image] = None
    image_bytes: Optional[bytes] = field(default=None, repr=False)

    def load_image(self) -> Optional[Image.Image]:
        """Load and return the PIL image, caching it on the sample."""
        if self.image is not None:
            return self.image
        if self.image_bytes:
            try:
                self.image = Image.open(io.BytesIO(self.image_bytes))
                self.image.load()
                return self.image
            except Exception:
                return None
        if self.image_path and os.path.exists(self.image_path):
            try:
                with open(self.image_path, "rb") as f:
                    self.image_bytes = f.read()
                self.image = Image.open(io.BytesIO(self.image_bytes))
                self.image.load()
                return self.image
            except Exception:
                return None
        source = self.source if isinstance(self.source, dict) else {}
        tar_path = source.get("tar_path")
        image_member = source.get("image_member")
        if tar_path and image_member and os.path.exists(tar_path):
            try:
                self.image_bytes = read_bytes_from_tar_archive(tar_path, image_member)
                self.image = Image.open(io.BytesIO(self.image_bytes))
                self.image.load()
                return self.image
            except Exception:
                return None
        zip_path = source.get("zip_path")
        if zip_path and image_member and os.path.exists(zip_path):
            try:
                self.image_bytes = read_bytes_from_zip_archive(zip_path, image_member)
                self.image = Image.open(io.BytesIO(self.image_bytes))
                self.image.load()
                return self.image
            except Exception:
                return None
        return None

    def can_reload_image(self) -> bool:
        """True if the image can be read again from a path or archive."""
        if self.image_path and os.path.exists(self.image_path):
            return True
        source = self.source if isinstance(self.source, dict) else {}
        member = source.get("image_member")
        tar_path = source.get("tar_path")
        zip_path = source.get("zip_path")
        if tar_path and member and os.path.exists(str(tar_path)):
            return True
        if zip_path and member and os.path.exists(str(zip_path)):
            return True
        return False

    def release_image_cache(self, *, drop_bytes: bool = False) -> None:
        """Drop decoded PIL cache; drop raw bytes only if they can be re-read."""
        self.image = None
        if drop_bytes and self.can_reload_image():
            self.image_bytes = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "text": self.text,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class FilterResult:
    """Result of applying a single filter to one sample."""

    keep: bool
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DroppedRecord:
    """Record explaining why a sample was dropped."""

    sample_id: str
    filter_name: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)


class Filter:
    """Base class for all filters."""

    name: str = "Filter"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def apply(
        self, samples: Iterable[Sample]
    ) -> Tuple[List[Sample], List[DroppedRecord]]:
        """Apply the filter to a collection of samples.

        Returns a tuple of (kept_samples, dropped_records).
        """
        if not self.enabled:
            return list(samples), []
        kept: List[Sample] = []
        dropped: List[DroppedRecord] = []
        for sample in samples:
            result = self.filter(sample)
            if result.keep:
                kept.append(sample)
            else:
                dropped.append(
                    DroppedRecord(
                        sample_id=sample.id,
                        filter_name=self.name,
                        reason=result.reason or "filtered",
                        details=result.details,
                        source=sample.source,
                    )
                )
            sample.release_image_cache(drop_bytes=True)
        return kept, dropped

    def filter(self, sample: Sample) -> FilterResult:
        """Evaluate a single sample. Subclasses must override this."""
        raise NotImplementedError


class Pipeline:
    """Run a sequence of filters and collect statistics."""

    def __init__(
        self,
        filters: List[Filter],
        output_dir: Optional[str] = None,
        save_report: bool = True,
        report_filename: str = "filter_report.json",
        save_mode: str = "report",
    ):
        self.filters = filters
        self.output_dir = output_dir
        self.save_report = save_report
        self.report_filename = report_filename
        self.save_mode = save_mode

    def run(self, samples: Iterable[Sample]) -> Dict[str, Any]:
        """Run all enabled filters sequentially and return a report."""
        current = list(samples)
        id_to_sample = {sample.id: sample for sample in current}
        report: Dict[str, Any] = {
            "initial_count": len(current),
            "filters": [],
            "save_mode": self.save_mode,
        }

        for filt in self.filters:
            if not getattr(filt, "enabled", True):
                continue
            kept, dropped = filt.apply(current)
            report["filters"].append(
                {
                    "name": filt.name,
                    "input_count": len(current),
                    "kept_count": len(kept),
                    "dropped_count": len(dropped),
                    "dropped": [asdict(d) for d in dropped],
                }
            )
            current = kept
            for sample in current:
                sample.release_image_cache(drop_bytes=True)

        report["final_count"] = len(current)
        report["total_dropped"] = report["initial_count"] - report["final_count"]

        if self.save_report and self.output_dir:
            from .export import export_results

            artifacts = export_results(
                output_dir=self.output_dir,
                report_filename=self.report_filename,
                report=report,
                kept_samples=current,
                id_to_sample=id_to_sample,
                save_mode=self.save_mode,
            )
            report.update(artifacts)

        return report
