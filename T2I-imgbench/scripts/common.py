"""公共配置加载与路径工具（scripts/ 下所有编排脚本共用）。

- PROJECT_ROOT：项目根目录（本文件的上上级），可用环境变量 T2I_IMGBENCH_ROOT 覆盖
- load_config()：base.yaml + 模型 yaml 递归 merge，相对路径解析为基于 PROJECT_ROOT 的绝对路径
- out_dir()：outputs/<model_name>/<benchmark>/<tag>，tag 缺省为时间戳
"""
import os
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("T2I_IMGBENCH_ROOT")
                    or Path(__file__).resolve().parent.parent).resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _merge(base: dict, override: dict) -> dict:
    """递归 merge：override 覆盖 base（字典递归合并，其余类型直接替换）。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(p) -> str:
    """相对路径基于 PROJECT_ROOT 展开为绝对路径（str），兼容 os.path.join 用法。"""
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else str(PROJECT_ROOT / p)


def _resolve_tree(node):
    """递归把 dict/list 中的路径字符串解析为绝对路径。"""
    if isinstance(node, dict):
        return {k: _resolve_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_tree(v) for v in node]
    if isinstance(node, str):
        return resolve_path(node)
    return node


def load_config(model_config_path=None) -> dict:
    """读取 configs/base.yaml 与指定模型 yaml，递归 merge（模型 yaml 覆盖 base）。

    model_config_path 支持两种写法：
      - configs/ 下的文件名："z_image" 或 "z_image.yaml"（等价于 configs/z_image.yaml）
      - yaml 文件的任意路径
    为 None 时只加载 base.yaml（make_prompts_manifest 等脚本使用）。

    - configs/base.yaml 的 paths/judges 相对项目根解析；
    - benchmarks 注册表内的子路径（repo/prompts/workdir/released_images）
      相对 paths.benchmarks_root（即 Benchmarks/）解析。
    """
    with open(CONFIGS_DIR / "base.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if model_config_path:
        p = Path(model_config_path)
        if not p.is_file():
            # 按 configs/ 下文件名解析（z_image / z_image.yaml），再按项目根解析
            cand = CONFIGS_DIR / model_config_path
            if cand.suffix != ".yaml":
                cand = cand.with_suffix(".yaml")
            if not cand.is_file():
                cand = PROJECT_ROOT / model_config_path
            if cand.is_file():
                p = cand
        p = p.resolve()
        with open(p, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
        cfg["_config_path"] = str(p)
    if cfg.get("paths"):
        cfg["paths"] = _resolve_tree(cfg["paths"])
    if cfg.get("judges"):
        cfg["judges"] = _resolve_tree(cfg["judges"])
    if cfg.get("benchmarks"):
        # 注册表子路径相对 benchmarks_root 解析（绝对路径原样保留）
        bench_root = cfg["paths"]["benchmarks_root"]
        cfg["benchmarks"] = {
            name: {k: (resolve_path(os.path.join(bench_root, v)) if isinstance(v, str) else v)
                   for k, v in entry.items()}
            for name, entry in cfg["benchmarks"].items()
        }
    return cfg


def out_dir(cfg: dict, benchmark: str, tag=None) -> str:
    """返回 outputs/<model_name>/<benchmark>/<tag>；tag 缺省为当前时间 YYYYMMDD-HHMMSS。"""
    if tag is None:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(cfg["paths"]["output_root"], cfg["model_name"], benchmark, tag)
