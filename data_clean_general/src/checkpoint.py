"""Checkpoint / resume support for long filter runs."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from .core import DroppedRecord, Sample
from .export import (
    _image_basename,
    _unique_dest,
    _write_deleted_summary,
    write_image_file,
)
from .filters import DuplicateFilter, NSFWFilter, WatermarkFilter


CHECKPOINT_SUBDIR = "checkpoint"
PROGRESS_FILENAME = "progress.jsonl"
META_FILENAME = "meta.json"
DUPLICATE_STATE_FILENAME = "duplicate_state.json"
DROPPED_DETAIL_FILENAME = "dropped.jsonl"


_PERF_TUNING_KEYS = frozenset(
    {"batch_size", "device", "hash_workers", "hash_batch_size"}
)


def _fingerprint_filters_config(
    filters_config: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy filter configs without perf-only keys (resume after batch_size tweaks)."""
    normalized: List[Dict[str, Any]] = []
    for entry in filters_config:
        row = {k: v for k, v in entry.items() if k not in _PERF_TUNING_KEYS}
        normalized.append(row)
    return normalized


def filters_config_fingerprint(filters_config: List[Dict[str, Any]]) -> str:
    payload = json.dumps(
        _fingerprint_filters_config(filters_config),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_run_dir(output_dir: str, run_name: str) -> str:
    """Return the run directory. Use run_name '.' or '' to write directly under output_dir."""
    if not run_name or run_name in (".", "/"):
        return output_dir
    return os.path.join(output_dir, run_name)


class CheckpointManager:
    """Track per-sample progress and incremental exports under ``run_dir``."""

    def __init__(
        self,
        run_dir: str,
        *,
        save_mode: str,
        filters_fingerprint: str,
        resume: bool,
        flush_every: int = 100,
        write_workers: int = 8,
    ):
        self.run_dir = run_dir
        self.save_mode = save_mode
        self.filters_fingerprint = filters_fingerprint
        self.resume = resume
        self.flush_every = max(1, flush_every)
        self.write_workers = max(1, int(write_workers))
        self.write_kept = save_mode in ("kept", "split")
        self.write_deleted = save_mode == "split"
        self.ckpt_dir = os.path.join(run_dir, CHECKPOINT_SUBDIR)
        self.progress_path = os.path.join(self.ckpt_dir, PROGRESS_FILENAME)
        self.meta_path = os.path.join(self.ckpt_dir, META_FILENAME)
        self.duplicate_state_path = os.path.join(
            self.ckpt_dir, DUPLICATE_STATE_FILENAME
        )
        self.dropped_detail_path = os.path.join(
            self.ckpt_dir, DROPPED_DETAIL_FILENAME
        )
        self._processed_ids: Set[str] = set()
        self._completed_tars: Set[str] = set()
        self._since_flush = 0
        self._used_by_dir: Dict[str, Set[str]] = {}
        self.images_dir: Optional[str] = None
        self.deleted_dir: Optional[str] = None
        self._lock = threading.Lock()
        self._write_executor: Optional[ThreadPoolExecutor] = None
        self._write_futures: List[Any] = []
        self._write_futures_lock = threading.Lock()
        # Bound in-flight exports so image_bytes memory stays under control.
        self._write_sema = threading.Semaphore(max(16, self.write_workers * 4))
        if self.write_workers > 1:
            self._write_executor = ThreadPoolExecutor(
                max_workers=self.write_workers,
                thread_name_prefix="ckpt-write",
            )

        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.write_kept:
            self.images_dir = os.path.join(run_dir, "images")
            os.makedirs(self.images_dir, exist_ok=True)
            self._used_by_dir[self.images_dir] = self._list_basenames(self.images_dir)
        if self.write_deleted:
            self.deleted_dir = os.path.join(run_dir, "deleted")
            os.makedirs(self.deleted_dir, exist_ok=True)
            self._used_by_dir[self.deleted_dir] = self._list_basenames(
                self.deleted_dir
            )

        if resume:
            self._load_existing()
        else:
            self._write_meta()

    @property
    def processed_count(self) -> int:
        return len(self._processed_ids)

    @property
    def completed_tar_names(self) -> Set[str]:
        return set(self._completed_tars)

    @staticmethod
    def _list_basenames(directory: str) -> Set[str]:
        if not os.path.isdir(directory):
            return set()
        return set(os.listdir(directory))

    def _dest_dir(self, kind: str) -> str:
        return os.path.join(self.run_dir, kind)

    def _used_names(self, dest_dir: str) -> Set[str]:
        if dest_dir not in self._used_by_dir:
            os.makedirs(dest_dir, exist_ok=True)
            self._used_by_dir[dest_dir] = self._list_basenames(dest_dir)
        return self._used_by_dir[dest_dir]

    def _load_existing(self) -> None:
        if not os.path.isfile(self.meta_path):
            raise FileNotFoundError(
                f"Cannot --resume: missing checkpoint meta at {self.meta_path}"
            )
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        prev_fp = meta.get("filters_fingerprint")
        if prev_fp and prev_fp != self.filters_fingerprint:
            raise ValueError(
                "Config filters changed since this run was started "
                f"(checkpoint {prev_fp!r} != current {self.filters_fingerprint!r}). "
                "Use a new --run-name or revert the --config file."
            )
        self._completed_tars = set(meta.get("completed_tar_files") or [])
        if not os.path.isfile(self.progress_path):
            return
        with open(self.progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = row.get("id")
                if sample_id:
                    self._processed_ids.add(sample_id)

    def _write_meta(self) -> None:
        meta = {
            "filters_fingerprint": self.filters_fingerprint,
            "save_mode": self.save_mode,
            "completed_tar_files": sorted(self._completed_tars),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def mark_tar_completed(self, tar_basename: str) -> None:
        if tar_basename in self._completed_tars:
            return
        self._completed_tars.add(tar_basename)
        self._write_meta()

    def is_processed(self, sample_id: str) -> bool:
        return sample_id in self._processed_ids

    def load_duplicate_state(self, duplicate_filter: DuplicateFilter) -> None:
        if not duplicate_filter.enabled:
            return
        if not os.path.isfile(self.duplicate_state_path):
            return
        with open(self.duplicate_state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        duplicate_filter.load_state(data)

    def save_duplicate_state(self, duplicate_filter: DuplicateFilter) -> None:
        if not duplicate_filter.enabled:
            return
        data = duplicate_filter.export_state()
        tmp = self.duplicate_state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.duplicate_state_path)

    def _submit_image_write(self, job: Any) -> None:
        """Run export job inline or on the background write pool."""
        if self._write_executor is None:
            job()
            return

        self._write_sema.acquire()

        def _wrapped() -> None:
            try:
                job()
            finally:
                self._write_sema.release()

        fut = self._write_executor.submit(_wrapped)
        with self._write_futures_lock:
            self._write_futures.append(fut)
            if len(self._write_futures) > 256:
                self._write_futures = [f for f in self._write_futures if not f.done()]

    def flush_writes(self) -> None:
        """Wait for all outstanding image export jobs to finish."""
        with self._write_futures_lock:
            futures = list(self._write_futures)
            self._write_futures.clear()
        errors: List[BaseException] = []
        for fut in futures:
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 - surface write failures
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} async image write(s) failed; first={errors[0]!r}"
            )

    def close_write_pool(self) -> None:
        self.flush_writes()
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)
            self._write_executor = None

    def record_kept(self, sample: Sample, duplicate_filter: DuplicateFilter) -> None:
        need_flush = False
        dest_dir: Optional[str] = None
        filename: Optional[str] = None
        with self._lock:
            row = {"id": sample.id, "status": "kept"}
            self._append_progress(row)
            self._processed_ids.add(sample.id)
            if self.write_kept:
                dest_dir = self._dest_dir("images")
                filename = _unique_dest(
                    dest_dir, _image_basename(sample), self._used_names(dest_dir)
                )
            need_flush = self._bump_flush_counter()

        if filename and dest_dir:
            write_dir = dest_dir
            dest_name = filename

            def _write_job() -> None:
                try:
                    write_image_file(sample, write_dir, dest_name)
                finally:
                    sample.release_image_cache(drop_bytes=True)

            self._submit_image_write(_write_job)
        else:
            sample.release_image_cache(drop_bytes=True)

        if need_flush:
            self.save_duplicate_state(duplicate_filter)

    def record_dropped(
        self,
        sample: Sample,
        dropped: DroppedRecord,
        duplicate_filter: DuplicateFilter,
    ) -> None:
        exported_filename: Optional[str] = None
        need_flush = False
        dest_dir: Optional[str] = None
        reserved_name: Optional[str] = None
        with self._lock:
            if self.write_deleted:
                dest_dir = self._dest_dir("deleted")
                reserved_name = _unique_dest(
                    dest_dir, _image_basename(sample), self._used_names(dest_dir)
                )
            self._append_progress(
                {
                    "id": sample.id,
                    "status": "dropped",
                    "filter_name": dropped.filter_name,
                    "reason": dropped.reason,
                    "details": dropped.details,
                }
            )
            self._processed_ids.add(sample.id)
            need_flush = self._bump_flush_counter()

        if self.write_deleted and dest_dir and reserved_name:
            exported_filename = reserved_name
            write_dir = dest_dir
            dest_name = reserved_name

            def _write_job() -> None:
                try:
                    if not write_image_file(sample, write_dir, dest_name):
                        # Keep progress consistent even if bytes were missing.
                        pass
                finally:
                    sample.release_image_cache(drop_bytes=True)

            self._submit_image_write(_write_job)
        else:
            if self.write_deleted:
                exported_filename = _image_basename(sample)
            sample.release_image_cache(drop_bytes=True)

        detail_row = {
            "id": sample.id,
            "status": "dropped",
            "filter_name": dropped.filter_name,
            "reason": dropped.reason,
            "details": dropped.details,
            "exported_filename": exported_filename,
            "source": dropped.source,
        }
        with self._lock:
            with open(self.dropped_detail_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail_row, ensure_ascii=False))
                f.write("\n")

        if need_flush:
            self.save_duplicate_state(duplicate_filter)

    def _bump_flush_counter(self) -> bool:
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._since_flush = 0
            return True
        return False

    def _append_progress(self, row: Dict[str, Any]) -> None:
        with open(self.progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    def finalize_duplicate(self, duplicate_filter: DuplicateFilter) -> None:
        self.save_duplicate_state(duplicate_filter)

    def finalize_split_summary(self) -> None:
        if not self.write_deleted:
            return
        rows: List[Dict[str, str]] = []
        if os.path.isfile(self.dropped_detail_path):
            with open(self.dropped_detail_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    source = obj.get("source") or {}
                    dummy = Sample(id=str(obj.get("id", "")), source=source)
                    subset = (
                        source.get("subset")
                        or source.get("tar_file")
                        or source.get("zip_file")
                        or "."
                    )
                    filename = obj.get("exported_filename") or _image_basename(dummy)
                    rows.append(
                        {
                            "filename": str(filename),
                            "subset": str(subset),
                            "filter_name": str(obj.get("filter_name")),
                            "reason": str(obj.get("reason") or "filtered"),
                        }
                    )
        summary_path = os.path.join(self.run_dir, "deleted_summary.txt")
        _write_deleted_summary(
            summary_path, self.deleted_dir or self.run_dir, rows
        )

    def build_report(
        self,
        *,
        filter_names: List[str],
        kept_count: int,
        dropped_count: int,
    ) -> Dict[str, Any]:
        """Assemble a report dict compatible with existing export summary."""
        per_filter: List[Dict[str, Any]] = []
        dropped_by_filter: Dict[str, int] = {}
        if os.path.isfile(self.progress_path):
            with open(self.progress_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("status") != "dropped":
                        continue
                    fn = str(row.get("filter_name") or "unknown")
                    dropped_by_filter[fn] = dropped_by_filter.get(fn, 0) + 1

        total_seen = kept_count + dropped_count
        running_input = total_seen
        for name in filter_names:
            dropped_n = dropped_by_filter.get(name, 0)
            per_filter.append(
                {
                    "name": name,
                    "input_count": running_input,
                    "kept_count": running_input - dropped_n,
                    "dropped_count": dropped_n,
                    "dropped": [],
                    "dropped_detail_path": self.dropped_detail_path
                    if dropped_n
                    else None,
                }
            )
            running_input -= dropped_n

        return {
            "initial_count": total_seen,
            "final_count": kept_count,
            "total_dropped": dropped_count,
            "save_mode": self.save_mode,
            "checkpoint": True,
            "progress_path": self.progress_path,
            "filters": per_filter,
        }


def _split_pre_gpu_stages(
    pre_gpu: List[Any],
) -> Tuple[List[Any], Optional[DuplicateFilter], List[Any]]:
    """Split pre-GPU filters into before-Duplicate / Duplicate / after-Duplicate."""
    before: List[Any] = []
    dup: Optional[DuplicateFilter] = None
    after: List[Any] = []
    seen_dup = False
    for filt in pre_gpu:
        if isinstance(filt, DuplicateFilter):
            dup = filt
            seen_dup = True
        elif not seen_dup:
            before.append(filt)
        else:
            after.append(filt)
    return before, dup, after


def _flush_duplicate_batch(
    batch: List[Sample],
    duplicate_filter: Optional[DuplicateFilter],
    after_dup_filters: List[Any],
    checkpoint: CheckpointManager,
    dup_for_flush: DuplicateFilter,
    gpu_pending: List[Sample],
    counters: Dict[str, int],
    *,
    after_workers: int = 1,
    after_executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[int, int]:
    """Run Duplicate (parallel hash) + post-dup CPU filters; feed GPU queue.

    Returns ``(kept_to_gpu_or_final, dropped_count)`` for this flush.
    """
    if not batch:
        return 0, 0
    samples = list(batch)
    batch.clear()
    dropped_n = 0
    kept_n = 0

    if duplicate_filter is not None and getattr(duplicate_filter, "enabled", True):
        kept_list, dropped_list = duplicate_filter.apply(samples)
        dropped_map = {d.sample_id: d for d in dropped_list}
        for sample in samples:
            if sample.id in dropped_map:
                checkpoint.record_dropped(
                    sample, dropped_map[sample.id], dup_for_flush
                )
                dropped_n += 1
                counters["dropped"] += 1
        current = kept_list
    else:
        current = samples

    if not current:
        return kept_n, dropped_n

    def _after_one(
        sample: Sample,
    ) -> Tuple[Sample, Optional[Sample], Optional[DroppedRecord]]:
        kept_sample, dropped = apply_filters_after_gpu(sample, after_dup_filters)
        return sample, kept_sample, dropped

    worker_n = max(1, after_workers)
    if (
        after_dup_filters
        and worker_n > 1
        and len(current) > 1
    ):
        if after_executor is not None:
            outcomes = list(after_executor.map(_after_one, current))
        else:
            with ThreadPoolExecutor(
                max_workers=min(worker_n, len(current))
            ) as pool:
                outcomes = list(pool.map(_after_one, current))
    else:
        outcomes = [_after_one(s) for s in current]

    for sample, kept_sample, dropped in outcomes:
        if dropped is not None:
            checkpoint.record_dropped(sample, dropped, dup_for_flush)
            dropped_n += 1
            counters["dropped"] += 1
        else:
            assert kept_sample is not None
            gpu_pending.append(kept_sample)
            kept_n += 1
    return kept_n, dropped_n



def _flush_pre_dup_batch(
    batch: List[Sample],
    pre_dup_filters: List[Any],
    *,
    workers: int = 1,
    executor: Optional[ThreadPoolExecutor] = None,
    warm_images: bool = True,
) -> List[Tuple[Sample, Optional[Sample], Optional[DroppedRecord]]]:
    """Run Broken/Resolution/Aspect (etc.) on a batch, optionally in parallel.

    Returns triples ``(original, kept_or_None, dropped_or_None)`` in input order.
    """
    if not batch:
        return []
    samples = list(batch)
    batch.clear()

    def _one(sample: Sample) -> Tuple[Sample, Optional[Sample], Optional[DroppedRecord]]:
        if warm_images:
            _warm_sample_bytes(sample)
        kept, dropped = apply_filters_before_gpu(sample, pre_dup_filters)
        return sample, kept, dropped

    worker_n = max(1, workers)
    if worker_n == 1 or len(samples) == 1:
        return [_one(s) for s in samples]
    if executor is not None:
        return list(executor.map(_one, samples))
    with ThreadPoolExecutor(max_workers=min(worker_n, len(samples))) as pool:
        return list(pool.map(_one, samples))


def apply_filters_before_gpu(
    sample: Sample,
    filters: Iterable[Any],
) -> Tuple[Optional[Sample], Optional[DroppedRecord]]:
    """Filters before NSFW/Watermark (includes DuplicateFilter state)."""
    current = sample
    for filt in filters:
        if not getattr(filt, "enabled", True):
            continue
        if isinstance(filt, DuplicateFilter):
            kept, dropped = filt.apply([current])
            if dropped:
                return None, dropped[0]
            current = kept[0]
            continue
        result = filt.filter(current)
        if not result.keep:
            return None, DroppedRecord(
                sample_id=current.id,
                filter_name=filt.name,
                reason=result.reason or "filtered",
                details=result.details,
                source=current.source,
            )
        current.release_image_cache(drop_bytes=False)
    return current, None


def apply_filters_after_gpu(
    sample: Sample,
    filters: Iterable[Any],
) -> Tuple[Optional[Sample], Optional[DroppedRecord]]:
    """CPU filters after NSFW/Watermark."""
    current = sample
    for filt in filters:
        if not getattr(filt, "enabled", True):
            continue
        result = filt.filter(current)
        if not result.keep:
            return None, DroppedRecord(
                sample_id=current.id,
                filter_name=filt.name,
                reason=result.reason or "filtered",
                details=result.details,
                source=current.source,
            )
        current.release_image_cache(drop_bytes=False)
    return current, None


def apply_filters_to_sample(
    sample: Sample,
    filters: Iterable[Any],
) -> Tuple[Optional[Sample], Optional[DroppedRecord]]:
    """Run the filter chain on a single sample (order matches batch pipeline)."""
    current = sample
    for filt in filters:
        if not getattr(filt, "enabled", True):
            continue
        if isinstance(filt, (DuplicateFilter, NSFWFilter, WatermarkFilter)):
            kept, dropped = filt.apply([current])
            if dropped:
                return None, dropped[0]
            current = kept[0]
            continue
        result = filt.filter(current)
        if not result.keep:
            return None, DroppedRecord(
                sample_id=current.id,
                filter_name=filt.name,
                reason=result.reason or "filtered",
                details=result.details,
                source=current.source,
            )
        current.release_image_cache(drop_bytes=False)
    return current, None


def _split_filter_stages(
    filters: List[Any],
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split into pre-GPU, GPU batch (NSFW/Watermark), post-GPU."""
    pre: List[Any] = []
    gpu: List[Any] = []
    post: List[Any] = []
    stage = "pre"
    for filt in filters:
        if not getattr(filt, "enabled", True):
            continue
        if isinstance(filt, (NSFWFilter, WatermarkFilter)):
            gpu.append(filt)
            stage = "post"
        elif stage == "post":
            post.append(filt)
        else:
            pre.append(filt)
    return pre, gpu, post


def _gpu_microbatch_size(gpu_filters: List[Any]) -> int:
    sizes = [
        int(getattr(f, "batch_size", 1))
        for f in gpu_filters
        if getattr(f, "enabled", True)
    ]
    return max(sizes) if sizes else 1


def _default_workers() -> int:
    """Post-GPU parallelism; tar sources usually stay at 1 (shared NFS tar reads)."""
    cpus = os.cpu_count() or 8
    return max(1, min(8, cpus // 16))


def _warm_sample_bytes(sample: Sample) -> None:
    if sample.image_bytes is None:
        sample.load_image()


def _limit_cpu_threading() -> None:
    """Avoid nested BLAS/OpenMP threads when using a thread pool."""
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _prefetch_iterator(
    iterator: Iterable[Sample],
    prefetch: int,
    *,
    warm_images: bool = False,
) -> Iterator[Sample]:
    """Read ahead samples from a slow loader (e.g. tar) on a background thread."""
    if prefetch <= 0:
        for item in iterator:
            if warm_images:
                _warm_sample_bytes(item)
            yield item
        return
    queue: Queue = Queue(maxsize=prefetch)
    sentinel = object()
    producer_error: List[BaseException] = []

    def _producer() -> None:
        try:
            for item in iterator:
                if warm_images:
                    _warm_sample_bytes(item)
                queue.put(item)
        except BaseException as exc:  # noqa: BLE001
            # Record the error instead of dying silently; the consumer
            # re-raises it after the sentinel so the run fails loudly
            # instead of being truncated without warning.
            producer_error.append(exc)
        finally:
            queue.put(sentinel)

    Thread(target=_producer, daemon=True).start()
    while True:
        item = queue.get()
        if item is sentinel:
            break
        yield item  # type: ignore[misc]
    if producer_error:
        raise producer_error[0]


def _post_gpu_task(
    sample: Sample,
    post_gpu_filters: List[Any],
) -> Tuple[Optional[Sample], Optional[DroppedRecord]]:
    return apply_filters_after_gpu(sample, post_gpu_filters)


def _flush_gpu_microbatch(
    batch: List[Sample],
    gpu_filters: List[Any],
    post_gpu_filters: List[Any],
    checkpoint: CheckpointManager,
    dup_for_flush: DuplicateFilter,
    counters: Dict[str, int],
    *,
    workers: int = 1,
    executor: Optional[ThreadPoolExecutor] = None,
) -> None:
    if not batch:
        return
    current_list = list(batch)
    batch.clear()
    for gf in gpu_filters:
        kept, dropped = gf.apply(current_list)
        dropped_map = {d.sample_id: d for d in dropped}
        kept_ids = {s.id for s in kept}
        for sample in current_list:
            if sample.id in dropped_map:
                checkpoint.record_dropped(
                    sample, dropped_map[sample.id], dup_for_flush
                )
                counters["dropped"] += 1
            else:
                sample.release_image_cache(drop_bytes=False)
        current_list = [s for s in kept if s.id in kept_ids]

    if not current_list:
        return

    worker_n = max(1, workers)
    if worker_n == 1 or len(current_list) == 1:
        outcomes = [
            apply_filters_after_gpu(sample, post_gpu_filters)
            for sample in current_list
        ]
    else:
        if executor is not None:
            outcomes = list(
                executor.map(
                    lambda s: _post_gpu_task(s, post_gpu_filters),
                    current_list,
                )
            )
        else:
            with ThreadPoolExecutor(max_workers=worker_n) as pool:
                outcomes = list(
                    pool.map(
                        lambda s: _post_gpu_task(s, post_gpu_filters),
                        current_list,
                    )
                )

    for sample, (final_sample, dropped) in zip(current_list, outcomes):
        if dropped is not None:
            checkpoint.record_dropped(sample, dropped, dup_for_flush)
            counters["dropped"] += 1
        else:
            assert final_sample is not None
            checkpoint.record_kept(final_sample, dup_for_flush)
            counters["kept"] += 1


def run_with_checkpoint(
    samples: Iterable[Sample],
    filters: List[Any],
    checkpoint: CheckpointManager,
    *,
    progress_every: int = 500,
    workers: int = 0,
    prefetch: int = 0,
    warm_images: bool = True,
    pre_workers: int = 0,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    duplicate_filter = next(
        (f for f in filters if isinstance(f, DuplicateFilter)), None
    )
    disabled_dup = DuplicateFilter(enabled=False)
    dup_for_flush = duplicate_filter or disabled_dup
    if duplicate_filter is not None:
        checkpoint.load_duplicate_state(duplicate_filter)

    pre_gpu, gpu_filters, post_gpu = _split_filter_stages(filters)
    pre_dup, dup_in_pre, after_dup = _split_pre_gpu_stages(pre_gpu)
    use_gpu_microbatch = bool(gpu_filters)
    microbatch_size = _gpu_microbatch_size(gpu_filters) if use_gpu_microbatch else 1
    worker_n = workers if workers > 0 else _default_workers()
    prefetch_n = prefetch if prefetch > 0 else max(32, worker_n * 4)
    dup_batch_size = 1
    if dup_in_pre is not None and getattr(dup_in_pre, "enabled", True):
        configured = int(getattr(dup_in_pre, "hash_batch_size", 32) or 32)
        workers_h = int(getattr(dup_in_pre, "hash_workers", 1) or 1)
        dup_batch_size = max(configured, microbatch_size, workers_h)
    # Parallel Broken/Resolution/Aspect: default to Duplicate hash_workers or 8.
    if pre_workers > 0:
        pre_worker_n = pre_workers
    elif dup_in_pre is not None and int(getattr(dup_in_pre, "hash_workers", 1) or 1) > 1:
        pre_worker_n = int(dup_in_pre.hash_workers)
    else:
        pre_worker_n = 8 if pre_dup else 1
    pre_batch_size = max(dup_batch_size, pre_worker_n, microbatch_size)
    _limit_cpu_threading()
    if use_gpu_microbatch:
        print(
            f"[checkpoint] GPU micro-batching enabled "
            f"(batch_size={microbatch_size}, filters={len(gpu_filters)}, "
            f"post_gpu_workers={worker_n}, prefetch={prefetch_n})",
            flush=True,
        )
    elif worker_n > 1:
        print(
            f"[checkpoint] parallel post-GPU workers={worker_n}, "
            f"prefetch={prefetch_n}",
            flush=True,
        )
    if pre_dup and pre_worker_n > 1:
        print(
            f"[checkpoint] pre-Duplicate parallel workers={pre_worker_n}, "
            f"batch_size={pre_batch_size}",
            flush=True,
        )
    if after_dup and pre_worker_n > 1:
        print(
            f"[checkpoint] post-Duplicate CPU parallel workers={pre_worker_n} "
            f"(LanguageFilter etc.)",
            flush=True,
        )
    if dup_in_pre is not None and getattr(dup_in_pre, "enabled", True):
        print(
            f"[checkpoint] DuplicateFilter hash_workers="
            f"{getattr(dup_in_pre, 'hash_workers', 1)}, "
            f"hash_batch_size={dup_batch_size}",
            flush=True,
        )
    print(
        f"[checkpoint] async image write_workers={checkpoint.write_workers}",
        flush=True,
    )

    # Prefetch only reads ahead; decode happens in pre-filter workers.
    sample_iter: Iterable[Sample] = _prefetch_iterator(samples, prefetch_n)

    filter_names = [f.name for f in filters if getattr(f, "enabled", True)]
    kept_this_run = 0
    dropped_this_run = 0
    processed_this_run = 0
    pre_pending: List[Sample] = []
    gpu_pending: List[Sample] = []
    dup_pending: List[Sample] = []
    counters = {"kept": 0, "dropped": 0}
    post_executor: Optional[ThreadPoolExecutor] = None
    pre_executor: Optional[ThreadPoolExecutor] = None
    if worker_n > 1:
        post_executor = ThreadPoolExecutor(max_workers=worker_n)
    if pre_worker_n > 1 and (pre_dup or after_dup):
        pre_executor = ThreadPoolExecutor(max_workers=pre_worker_n)

    def flush_gpu_queue() -> None:
        nonlocal kept_this_run, dropped_this_run
        if not gpu_pending:
            return
        before_k, before_d = counters["kept"], counters["dropped"]
        _flush_gpu_microbatch(
            gpu_pending,
            gpu_filters,
            post_gpu,
            checkpoint,
            dup_for_flush,
            counters,
            workers=worker_n,
            executor=post_executor,
        )
        kept_this_run += counters["kept"] - before_k
        dropped_this_run += counters["dropped"] - before_d

    def flush_dup_into_gpu() -> None:
        nonlocal dropped_this_run
        if not dup_pending:
            return
        _kept, dropped_n = _flush_duplicate_batch(
            dup_pending,
            dup_in_pre,
            after_dup,
            checkpoint,
            dup_for_flush,
            gpu_pending,
            counters,
            after_workers=pre_worker_n,
            after_executor=pre_executor,
        )
        dropped_this_run += dropped_n
        if use_gpu_microbatch and len(gpu_pending) >= microbatch_size:
            flush_gpu_queue()

    def finalize_non_gpu_survivors(survivors: List[Sample]) -> None:
        nonlocal kept_this_run, dropped_this_run
        for final_sample in survivors:
            kept2, dropped2 = apply_filters_after_gpu(final_sample, post_gpu)
            if dropped2 is not None:
                checkpoint.record_dropped(final_sample, dropped2, dup_for_flush)
                dropped_this_run += 1
                counters["dropped"] += 1
            else:
                assert kept2 is not None
                checkpoint.record_kept(kept2, dup_for_flush)
                kept_this_run += 1
                counters["kept"] += 1

    def route_after_pre(
        kept_sample: Optional[Sample],
        dropped: Optional[DroppedRecord],
        original: Sample,
    ) -> None:
        nonlocal dropped_this_run
        if dropped is not None:
            checkpoint.record_dropped(original, dropped, dup_for_flush)
            dropped_this_run += 1
            counters["dropped"] += 1
            return
        assert kept_sample is not None
        if dup_in_pre is not None and getattr(dup_in_pre, "enabled", True):
            dup_pending.append(kept_sample)
            if len(dup_pending) >= dup_batch_size:
                if use_gpu_microbatch:
                    flush_dup_into_gpu()
                else:
                    survivors: List[Sample] = []
                    _kept, dropped_n = _flush_duplicate_batch(
                        dup_pending,
                        dup_in_pre,
                        after_dup,
                        checkpoint,
                        dup_for_flush,
                        survivors,
                        counters,
                        after_workers=pre_worker_n,
                        after_executor=pre_executor,
                    )
                    dropped_this_run += dropped_n
                    finalize_non_gpu_survivors(survivors)
            return
        kept2, dropped2 = apply_filters_after_gpu(kept_sample, after_dup)
        if dropped2 is not None:
            checkpoint.record_dropped(kept_sample, dropped2, dup_for_flush)
            dropped_this_run += 1
            counters["dropped"] += 1
        elif use_gpu_microbatch:
            assert kept2 is not None
            gpu_pending.append(kept2)
            if len(gpu_pending) >= microbatch_size:
                flush_gpu_queue()
        else:
            assert kept2 is not None
            finalize_non_gpu_survivors([kept2])

    def flush_pre_batch() -> None:
        if not pre_pending:
            return
        outcomes = _flush_pre_dup_batch(
            pre_pending,
            pre_dup,
            workers=pre_worker_n,
            executor=pre_executor,
            warm_images=warm_images,
        )
        for original, kept_sample, dropped in outcomes:
            route_after_pre(kept_sample, dropped, original)

    try:
        for sample in sample_iter:
            if checkpoint.is_processed(sample.id):
                continue
            if sample_limit is not None and processed_this_run >= sample_limit:
                print(
                    f"[checkpoint] reached --limit={sample_limit}, stopping this run",
                    flush=True,
                )
                break
            processed_this_run += 1
            if pre_dup and pre_worker_n > 1:
                pre_pending.append(sample)
                if len(pre_pending) >= pre_batch_size:
                    flush_pre_batch()
            else:
                if warm_images:
                    _warm_sample_bytes(sample)
                kept_sample, dropped = apply_filters_before_gpu(sample, pre_dup)
                route_after_pre(kept_sample, dropped, sample)

            if progress_every and processed_this_run % progress_every == 0:
                print(
                    f"[checkpoint] total_done={checkpoint.processed_count} "
                    f"(+{processed_this_run} this run, "
                    f"kept={kept_this_run}, dropped={dropped_this_run})",
                    flush=True,
                )

        flush_pre_batch()
        if dup_pending:
            if use_gpu_microbatch:
                flush_dup_into_gpu()
            else:
                survivors = []
                _kept, dropped_n = _flush_duplicate_batch(
                    dup_pending,
                    dup_in_pre,
                    after_dup,
                    checkpoint,
                    dup_for_flush,
                    survivors,
                    counters,
                    after_workers=pre_worker_n,
                    after_executor=pre_executor,
                )
                dropped_this_run += dropped_n
                finalize_non_gpu_survivors(survivors)
        if use_gpu_microbatch and gpu_pending:
            flush_gpu_queue()
    finally:
        if post_executor is not None:
            post_executor.shutdown(wait=True)
        if pre_executor is not None:
            pre_executor.shutdown(wait=True)
        try:
            checkpoint.flush_writes()
        except Exception as exc:
            print(f"[checkpoint] flush_writes warning: {exc}", flush=True)
        checkpoint.close_write_pool()

    checkpoint.finalize_duplicate(dup_for_flush)
    checkpoint.finalize_split_summary()

    kept_total = 0
    dropped_total = 0
    if os.path.isfile(checkpoint.progress_path):
        with open(checkpoint.progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") == "kept":
                    kept_total += 1
                elif row.get("status") == "dropped":
                    dropped_total += 1

    return checkpoint.build_report(
        filter_names=filter_names,
        kept_count=kept_total,
        dropped_count=dropped_total,
    )
