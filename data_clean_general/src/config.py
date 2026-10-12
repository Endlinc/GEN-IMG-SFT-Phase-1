"""Build filters from a declarative configuration."""

from typing import Any, Dict, List

from .filters import (
    AspectRatioFilter,
    BlurFilter,
    BrokenFilesFilter,
    CompressionQualityFilter,
    DuplicateFilter,
    EntropyFilter,
    LanguageFilter,
    NSFWFilter,
    ResolutionFilter,
    WatermarkFilter,
)


FILTER_MAP = {
    "BrokenFilesFilter": BrokenFilesFilter,
    "ResolutionFilter": ResolutionFilter,
    "AspectRatioFilter": AspectRatioFilter,
    "DuplicateFilter": DuplicateFilter,
    "LanguageFilter": LanguageFilter,
    "NSFWFilter": NSFWFilter,
    "WatermarkFilter": WatermarkFilter,
    "CompressionQualityFilter": CompressionQualityFilter,
    "BlurFilter": BlurFilter,
    "EntropyFilter": EntropyFilter,
}


def build_filter(config: Dict[str, Any]) -> Any:
    """Instantiate a single filter from a configuration dict."""
    cls_name = config.get("name")
    if cls_name not in FILTER_MAP:
        raise ValueError(f"Unknown filter: {cls_name}")
    cls = FILTER_MAP[cls_name]
    params = {k: v for k, v in config.items() if k not in ("name",)}
    return cls(**params)


def build_filters(configs: List[Dict[str, Any]]) -> List[Any]:
    """Instantiate a list of filters from a configuration list."""
    return [build_filter(cfg) for cfg in configs]
