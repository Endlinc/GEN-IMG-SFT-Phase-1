"""Command-line entry point for running the cleaning pipeline."""

import argparse
import json
import os
import sys
from itertools import islice

from .config import build_filters
from .core import Pipeline
from .export import SAVE_MODES
from .loaders import (
    AlchemistLoader,
    ConceptualCaptionsLoader,
    DenseFusion4V100kLoader,
    DiffusionDB2MRandom50kLoader,
    TarShardLoader,
    WikiArtLoader,
)

_SHARDED_FORMATS = (
    "fine-t2i",
    "densefusion-4v-100k",
    "wikiart",
    "diffusiondb_2m_random_50k",
)


def resolve_config_path(path: str) -> str:
    """Resolve a config path, preferring ``<repo>/config/`` for bare filenames."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(repo_dir, "config")
    basename = os.path.basename(path)
    candidates = []
    if os.path.dirname(path):
        candidates.append(path)
    candidates.extend(
        (
            os.path.join(config_dir, basename),
            os.path.join("config", basename),
            path,
        )
    )
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return path


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml

                return yaml.safe_load(f)
            except ImportError as e:
                raise ImportError(
                    "PyYAML is required for YAML config files. Install with: pip install pyyaml"
                ) from e
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run general dataset cleaning filters on a dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to JSON/YAML filter config. Dataset-specific files live in "
            "config/, e.g. config/config_conceptual_captions.yaml "
            "(bare filenames are also resolved under config/)."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Root directory of the dataset to clean.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the filter report will be saved.",
    )
    parser.add_argument(
        "--report-name",
        default="filter_report.json",
        help="Filename of the generated report (default: filter_report.json).",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=[
            "conceptual-captions",
            "fine-t2i",
            "densefusion-4v-100k",
            "wikiart",
            "alchemist",
            "diffusiondb_2m_random_50k",
        ],
        help=(
            "Dataset format: conceptual-captions (cc_clean_index.jsonl + "
            "cc_images/), fine-t2i (tar shards), "
            "densefusion-4v-100k (DenseFusion-4V-100k.jsonl + "
            "images/DenseFusion-4V-100K/*.zip), "
            "wikiart (HuggingFace parquet shards under data/), "
            "alchemist (parquet/csv index + images/), "
            "or diffusiondb_2m_random_50k (parquet metadata + extracted images/)."
        ),
    )
    parser.add_argument(
        "--tar-glob",
        default="*.tar",
        help="For --format fine-t2i: glob for shard files under dataset root or data/.",
    )
    parser.add_argument(
        "--tar-offset",
        type=int,
        default=0,
        help=(
            "For fine-t2i / densefusion-4v-100k / wikiart / "
            "diffusiondb_2m_random_50k: skip the first N archive "
            "shards (sorted tar, zip, or parquet). Use with --tar-limit for "
            "manual batches."
        ),
    )
    parser.add_argument(
        "--tar-limit",
        type=int,
        default=None,
        help=(
            "For fine-t2i / densefusion-4v-100k / wikiart / "
            "diffusiondb_2m_random_50k: process at most N "
            "archive shards (after --tar-offset)."
        ),
    )
    parser.add_argument(
        "--tar-batch-size",
        type=int,
        default=None,
        help=(
            "For fine-t2i / densefusion-4v-100k / wikiart / "
            "diffusiondb_2m_random_50k with --resume: process "
            "the next N archive shards (tar, zip, or parquet) that are not yet "
            "in checkpoint meta. Repeat the same command until all shards are "
            "done. DuplicateFilter state is kept across batches."
        ),
    )
    parser.add_argument(
        "--save-mode",
        default=None,
        choices=list(SAVE_MODES),
        help=(
            "What to persist after filtering: "
            "'report' = JSON only; "
            "'kept' = JSON + copy kept images to images/; "
            "'split' = JSON + kept images/ + deleted images/ (+ deleted_summary.txt). "
            "Default: config.save_mode if set, otherwise 'report'."
        ),
    )
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help=(
            "This run only: process at most N samples (small-sample test). "
            "Does not write a checkpoint. Mutually exclusive with --resume."
        ),
    )
    run_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Run remaining samples to completion with checkpointing. "
            "First run and later restarts use the same flag: continues from "
            "<output>/<run-name>/ if a checkpoint exists, otherwise starts "
            "fresh. Mutually exclusive with --limit. Requires --run-name."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Subdirectory under --output for this run's results "
            "(e.g. wikiart_smoke). Works with both --limit and --resume. "
            "Required with --resume."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush DuplicateFilter state every N processed samples (default: 100).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "For --resume: parallel CPU workers for post-GPU filters "
            "(Compression/Blur/Entropy). 0 = auto (default: min(32, cpu//8))."
        ),
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=0,
        help=(
            "For --resume: tar/sample read-ahead queue size. "
            "0 = auto (default: max(32, workers*4))."
        ),
    )
    parser.add_argument(
        "--dup-hash-workers",
        type=int,
        default=None,
        help=(
            "Override DuplicateFilter.hash_workers for parallel perceptual-hash "
            "compute (index update stays serial). Default: value from --config."
        ),
    )
    parser.add_argument(
        "--pre-workers",
        type=int,
        default=0,
        help=(
            "For --resume: parallel workers for pre-Duplicate CPU filters "
            "(Broken/Resolution/Aspect). 0 = auto (use DuplicateFilter.hash_workers "
            "or 8)."
        ),
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=8,
        help=(
            "For --resume: parallel workers for kept/deleted image export "
            "(async write). 1 = synchronous writes. Default: 8."
        ),
    )
    args = parser.parse_args(argv)

    args.config = resolve_config_path(args.config)
    if not os.path.isfile(args.config):
        raise SystemExit(f"Config file not found: {args.config}")
    config = load_config(args.config)
    filters = build_filters(config.get("filters", []))
    if args.dup_hash_workers is not None:
        if args.dup_hash_workers < 1:
            raise SystemExit("--dup-hash-workers must be >= 1")
        from .filters import DuplicateFilter as _DuplicateFilter

        for filt in filters:
            if isinstance(filt, _DuplicateFilter):
                filt.hash_workers = args.dup_hash_workers
    save_mode = args.save_mode or config.get("save_mode") or "report"
    if save_mode not in SAVE_MODES:
        raise SystemExit(
            f"Invalid save_mode={save_mode!r}; expected one of {SAVE_MODES}"
        )

    if args.tar_batch_size is not None and args.tar_batch_size < 1:
        raise SystemExit("--tar-batch-size must be >= 1")
    if args.tar_batch_size and not args.resume:
        raise SystemExit("--tar-batch-size requires --resume")
    if args.tar_batch_size and (args.tar_offset or args.tar_limit is not None):
        raise SystemExit(
            "Use either --tar-batch-size or --tar-offset/--tar-limit, not both"
        )
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    use_checkpoint = args.resume
    if use_checkpoint and not args.run_name:
        raise SystemExit("--resume requires --run-name (stable run id).")

    from .checkpoint import resolve_run_dir

    run_dir = resolve_run_dir(args.output, args.run_name or "")

    def build_loader(checkpoint_mgr=None):
        if args.format == "conceptual-captions":
            return ConceptualCaptionsLoader(args.dataset)
        if args.format == "fine-t2i":
            return TarShardLoader(
                args.dataset,
                tar_glob=args.tar_glob,
                tar_offset=args.tar_offset,
                tar_limit=args.tar_limit,
                tar_batch_size=args.tar_batch_size,
                completed_tar_names=(
                    checkpoint_mgr.completed_tar_names if checkpoint_mgr else None
                ),
                on_shard_complete=(
                    checkpoint_mgr.mark_tar_completed if checkpoint_mgr else None
                ),
            )
        if args.format == "densefusion-4v-100k":
            return DenseFusion4V100kLoader(
                args.dataset,
                zip_offset=args.tar_offset,
                zip_limit=args.tar_limit,
                zip_batch_size=args.tar_batch_size,
                completed_zip_names=(
                    checkpoint_mgr.completed_tar_names if checkpoint_mgr else None
                ),
                on_shard_complete=(
                    checkpoint_mgr.mark_tar_completed if checkpoint_mgr else None
                ),
            )
        if args.format == "wikiart":
            return WikiArtLoader(
                args.dataset,
                parquet_offset=args.tar_offset,
                parquet_limit=args.tar_limit,
                parquet_batch_size=args.tar_batch_size,
                completed_parquet_names=(
                    checkpoint_mgr.completed_tar_names if checkpoint_mgr else None
                ),
                on_shard_complete=(
                    checkpoint_mgr.mark_tar_completed if checkpoint_mgr else None
                ),
            )
        if args.format == "alchemist":
            return AlchemistLoader(args.dataset)
        if args.format == "diffusiondb_2m_random_50k":
            return DiffusionDB2MRandom50kLoader(
                args.dataset,
                parquet_offset=args.tar_offset,
                parquet_limit=args.tar_limit,
                parquet_batch_size=args.tar_batch_size,
                completed_parquet_names=(
                    checkpoint_mgr.completed_tar_names if checkpoint_mgr else None
                ),
                on_shard_complete=(
                    checkpoint_mgr.mark_tar_completed if checkpoint_mgr else None
                ),
            )
        raise SystemExit(f"Unsupported dataset format: {args.format}")

    if use_checkpoint:
        from .checkpoint import (
            CheckpointManager,
            filters_config_fingerprint,
            run_with_checkpoint,
        )

        meta = os.path.join(run_dir, "checkpoint", "meta.json")
        resume_existing = os.path.isfile(meta)

        os.makedirs(run_dir, exist_ok=True)
        report_filename = args.report_name
        if args.write_workers is not None and args.write_workers < 1:
            raise SystemExit("--write-workers must be >= 1")
        ckpt = CheckpointManager(
            run_dir,
            save_mode=save_mode,
            filters_fingerprint=filters_config_fingerprint(
                config.get("filters", [])
            ),
            resume=resume_existing,
            flush_every=args.checkpoint_every,
            write_workers=args.write_workers,
        )
        if resume_existing:
            print(
                f"Resuming checkpoint run at {run_dir} "
                f"({ckpt.processed_count} samples already done).",
                flush=True,
            )
        else:
            print(f"Checkpoint run -> {run_dir}", flush=True)

        loader = build_loader(ckpt)
        print(f"Streaming samples with format={args.format} ...", flush=True)
        report = run_with_checkpoint(
            loader.load(),
            filters,
            ckpt,
            progress_every=500,
            workers=args.workers,
            prefetch=args.prefetch,
            warm_images=(args.format in _SHARDED_FORMATS),
            pre_workers=args.pre_workers,
            sample_limit=None,
        )
        report_path = os.path.join(run_dir, report_filename)
        import json as json_mod

        with open(report_path, "w", encoding="utf-8") as f:
            json_mod.dump(report, f, ensure_ascii=False, indent=2)
        report["report_path"] = report_path
        report["run_dir"] = run_dir
        if save_mode in ("kept", "split"):
            report["images_dir"] = os.path.join(run_dir, "images")
        if save_mode == "split":
            report["deleted_dir"] = os.path.join(run_dir, "deleted")
            report["deleted_summary_path"] = os.path.join(
                run_dir, "deleted_summary.txt"
            )
        if args.format in _SHARDED_FORMATS:
            report["completed_tar_shards"] = sorted(ckpt.completed_tar_names)
            loader_for_count = build_loader(ckpt)
            all_shards = loader_for_count.list_tar_shards()
            report["total_tar_shards"] = len(all_shards)
            report["pending_tar_shards"] = max(
                0,
                len(all_shards) - len(ckpt.completed_tar_names),
            )
    else:
        loader = build_loader()
        print(f"Loading samples with format={args.format} ...", flush=True)
        sample_iter = loader.load()
        if args.limit is not None:
            print(f"Sample limit this run: {args.limit}", flush=True)
            sample_iter = islice(sample_iter, args.limit)
        samples = list(sample_iter)
        print(f"Loaded {len(samples)} samples. Running filters...", flush=True)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Writing results -> {run_dir}", flush=True)
        pipeline = Pipeline(
            filters,
            output_dir=run_dir,
            report_filename=args.report_name,
            save_mode=save_mode,
        )
        report = pipeline.run(samples)
        report["run_dir"] = run_dir

    summary = {
        "report_path": report.get("report_path"),
        "run_dir": report.get("run_dir"),
        "save_mode": report.get("save_mode"),
        "images_dir": report.get("images_dir"),
        "deleted_dir": report.get("deleted_dir"),
        "initial_count": report["initial_count"],
        "final_count": report["final_count"],
        "total_dropped": report["total_dropped"],
        "completed_tar_shards": report.get("completed_tar_shards"),
        "pending_tar_shards": report.get("pending_tar_shards"),
        "total_tar_shards": report.get("total_tar_shards"),
        "filter_summary": [
            {
                "name": f["name"],
                "input_count": f["input_count"],
                "kept_count": f["kept_count"],
                "dropped_count": f["dropped_count"],
            }
            for f in report["filters"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
