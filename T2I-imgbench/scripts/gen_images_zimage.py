"""本地 Z-Image（Models/Z-Image 原 repo 代码）生成 benchmark 图片。

与 gen_images.py（API 版）协议对齐：
- 同一套 prompt 加载（load_prompts）与确定性 seed（make_seed），跨模型可比
- 不做 prompt 改写，原样直传；negative_prompt 从配置读取
- 每张图落盘 raw/processed + meta.json；失败写 <tag>/failures.jsonl（口径 A）
- 产物布局：outputs/<model>/<benchmark>/<tag>/images/<prompt_id>/seed{k}_raw/processed.png
- 断点续跑：已齐 n_samples 张 processed 图的 prompt 跳过

模型加载与推理尽量复用 Models/Z-Image 原 repo：
- sys.path 注入 <repo>/src 后调用 utils.load_from_local_dir / zimage.generate

用法示例：
  python scripts/gen_images_zimage.py --config z_image --benchmark geneval [--tag smoke] [--limit 20]
"""
import argparse
import io
import json
import os
import sys
import time

from PIL import Image

# 同目录脚本可互相 import（python scripts/xxx.py 时脚本目录在 sys.path[0]）
from common import load_config, out_dir, resolve_path
from gen_images import center_crop_square, load_prompts, make_seed, sha256_bytes, utcnow


def load_zimage_components(model_cfg):
    """按原 repo inference.py 的方式加载全部组件。"""
    sys.path.insert(0, model_cfg["repo_src"])
    import torch
    from utils import AttentionBackend, load_from_local_dir, set_attention_backend

    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    components = load_from_local_dir(
        model_cfg["ckpt_dir"], device=device, dtype=dtype,
        verbose=True, compile=model_cfg.get("compile", False),
    )
    attn = os.environ.get("ZIMAGE_ATTENTION", model_cfg.get("attn_backend", "_native_flash"))
    AttentionBackend.print_available_backends()
    set_attention_backend(attn)
    print(f"device={device}, dtype={dtype}, attention backend: {attn}")
    return components, torch, device


def generate_one(components, torch, device, prompt, gen_cfg, seed):
    """调用原 repo 的 zimage.generate，返回 (PIL.Image, latency_ms)。"""
    from zimage import generate

    t0 = time.time()
    images = generate(
        prompt=prompt,
        negative_prompt=gen_cfg.get("negative_prompt", ""),
        **components,
        height=gen_cfg["height"],
        width=gen_cfg["width"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=gen_cfg["guidance_scale"],
        cfg_normalization=gen_cfg.get("cfg_normalization", False),
        generator=torch.Generator(device).manual_seed(seed),
    )
    return images[0], int((time.time() - t0) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="configs/ 下模型 yaml：可写文件名（如 z_image）或路径")
    ap.add_argument("--benchmark", required=True,
                    choices=["geneval", "dpg", "longtext_en", "longtext_zh",
                             "qwenimagebench_cn", "qwenimagebench_en"])
    ap.add_argument("--tag", default=None,
                    help="输出 tag（缺省为时间戳 YYYYMMDD-HHMMSS）；评分脚本用同一 tag 找到本批产物")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="覆盖 base.yaml 注册表中该 benchmark 的官方 n_samples")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条 prompt（smoke test 用）")
    ap.add_argument("--prompt-ids", default=None,
                    help="逗号分隔的 prompt_id 白名单，只生成这些 prompt（与 --limit 叠加：先截取再过滤）")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="按列表下标取模分片（0 起），用于多作业并行；与 --limit/--prompt-ids 叠加")
    ap.add_argument("--enhanced-prompts", default=None, metavar="JSONL",
                    help="PE 增强 prompt 文件（每行 {id, original, enhanced}），"
                         "按 prompt_id 匹配后替换生成文本（判分仍用原始 metadata）")
    args = ap.parse_args()
    cfg = load_config(args.config)

    model_name = cfg["model_name"]
    gen_cfg = cfg["generation"]
    n_samples = args.n_samples or cfg["benchmarks"][args.benchmark]["n_samples"]
    run_dir = out_dir(cfg, args.benchmark, args.tag)
    out_images = os.path.join(run_dir, "images")
    os.makedirs(out_images, exist_ok=True)
    failures_path = os.path.join(run_dir, "failures.jsonl")
    seen_failures = set()
    if os.path.exists(failures_path):
        with open(failures_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_failures.add((args.benchmark, str(d.get("prompt_id")), d.get("sample_index")))

    prompts = load_prompts(args.benchmark, args.limit, cfg)
    if args.prompt_ids:
        keep = {s.strip() for s in args.prompt_ids.split(",") if s.strip()}
        prompts = [p for p in prompts if p["prompt_id"] in keep]
        missing = keep - {p["prompt_id"] for p in prompts}
        if missing:
            raise SystemExit(f"prompt_ids 未在 benchmark {args.benchmark} 中找到: {sorted(missing)}")
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        if not (0 <= shard_i < shard_n):
            raise SystemExit(f"非法 --shard {args.shard}，要求 0 <= i < n")
        prompts = [p for idx, p in enumerate(prompts) if idx % shard_n == shard_i]
        print(f"shard {shard_i}/{shard_n}: {len(prompts)} prompts")
    print(f"[{args.benchmark}] {len(prompts)} prompts x {n_samples} samples -> {run_dir}")

    if args.enhanced_prompts:
        # PE 增强文本覆盖：id 兼容 geneval 的 0 填充（00000）与 dpg 的原始 item_id
        emap = {}
        with open(args.enhanced_prompts) as f:
            for line in f:
                d = json.loads(line)
                sid = str(d.get("id", d.get("item_id")))  # geneval 用 id，dpg 用 item_id
                emap[sid] = d["enhanced"]
                if sid.isdigit():
                    emap[f"{int(sid):05d}"] = d["enhanced"]
        missing = [p["prompt_id"] for p in prompts if p["prompt_id"] not in emap]
        if missing:
            raise SystemExit(f"增强 prompt 文件缺少 {len(missing)} 条: {missing[:5]}...")
        for p in prompts:
            p["original_prompt"] = p["prompt"]
            p["prompt"] = emap[p["prompt_id"]]
        print(f"prompt 已替换为 PE 增强文本（{args.enhanced_prompts}，{len(prompts)} 条）")

    # 先过滤已完成/部分完成的 prompt，再决定是否需要加载模型
    todo = []
    for item in prompts:
        pdir = os.path.join(out_images, item["prompt_id"])
        if os.path.isdir(pdir):
            existing = [f for f in os.listdir(pdir) if f.endswith("_processed.png")]
            if len(existing) >= n_samples:
                continue
        todo.append(item)
    if not todo:
        print("all done, nothing to generate.")
        return
    print(f"{len(todo)} prompts to generate ({len(prompts) - len(todo)} already complete)")

    # 模型权重/repo 路径相对项目根（model yaml 中为相对路径，这里展开为绝对路径）
    model_cfg = dict(cfg["model"])
    for k in ("ckpt_dir", "repo_src"):
        if model_cfg.get(k):
            model_cfg[k] = resolve_path(model_cfg[k])
    components, torch, device = load_zimage_components(model_cfg)

    from tqdm import tqdm
    for item in tqdm(todo):
        pid = item["prompt_id"]
        pdir = os.path.join(out_images, pid)
        os.makedirs(pdir, exist_ok=True)
        for k in range(n_samples):
            meta_path = os.path.join(pdir, f"seed{k}_meta.json")
            proc_path = os.path.join(pdir, f"seed{k}_processed.png")
            if os.path.exists(meta_path) and os.path.exists(proc_path):
                with open(meta_path) as f:
                    if json.load(f).get("ok"):
                        continue  # 该样本已成功且产物齐全，跳过
            seed = make_seed(args.benchmark, pid, k)
            meta = {
                "prompt_id": pid,
                "prompt_text": item["prompt"],
                "original_prompt": item.get("original_prompt"),
                "benchmark": args.benchmark,
                "model": model_name,
                "generation_params": {
                    "height": gen_cfg["height"], "width": gen_cfg["width"],
                    "num_inference_steps": gen_cfg["num_inference_steps"],
                    "guidance_scale": gen_cfg["guidance_scale"],
                    "cfg_normalization": gen_cfg.get("cfg_normalization", False),
                    "negative_prompt": gen_cfg.get("negative_prompt", ""),
                    "seed": seed,
                },
                "request_timestamp_utc": utcnow(),
                "sample_index": k,
            }
            try:
                img, latency = generate_one(components, torch, device,
                                            item["prompt"], gen_cfg, seed)
                meta["ok"] = True
                meta["latency_ms"] = latency
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                raw_bytes = buf.getvalue()
                with open(os.path.join(pdir, f"seed{k}_raw.png"), "wb") as f:
                    f.write(raw_bytes)
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                if img.size[0] != img.size[1]:
                    img = center_crop_square(img)  # 协议 §3.2：仅中心裁剪，不拉伸
                img.save(proc_path)
                meta["content_hash_sha256"] = sha256_bytes(raw_bytes)
                meta["processed_size"] = list(img.size)
            except Exception as e:  # 单样本失败不终止整批（口径 A：记失败样本）
                if device == "cuda":
                    torch.cuda.empty_cache()
                meta["ok"] = False
                meta["status"] = "failed"
                meta["failure_stage"] = "generation"
                meta["error_type"] = type(e).__name__
                fail_key = (args.benchmark, pid, k)
                if fail_key not in seen_failures:
                    seen_failures.add(fail_key)
                    with open(failures_path, "a") as f:
                        f.write(json.dumps({
                            "status": "failed", "failure_stage": "generation",
                            "error_type": type(e).__name__,
                            "raw_error_message": str(e)[:2000],
                            "retry_count": 0,
                            "model": model_name, "benchmark": args.benchmark,
                            "prompt_id": pid, "sample_index": k,
                            "timestamp_utc": utcnow(),
                        }, ensure_ascii=False) + "\n")
            with open(meta_path, "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    print("done.")


if __name__ == "__main__":
    main()
