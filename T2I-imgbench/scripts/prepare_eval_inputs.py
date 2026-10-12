"""将 outputs/<model>/<benchmark>/<tag>/images/ 的散图整理成各官方评测脚本的输入目录结构。

- geneval:    <evaldir>/<index:05d>/{metadata.jsonl, samples/00000..03.png}
- dpg:        <evaldir>/<item_id>.png          （4 张 2x2 拼图，单格 resolution x resolution）
- longtext_*: <evaldir>/<prompt_id>_<repeat_id>.png

用法示例：
  python scripts/prepare_eval_inputs.py --config qwen_image_2 --benchmark geneval \
      --tag 20261004-120000 --eval-dir outputs/qwen_image_2/geneval/20261004-120000/_eval/geneval/images
"""
import argparse
import json
import os
import shutil

from PIL import Image

from common import load_config, out_dir


def collect_samples(src_dir, prompt_ids, n_samples):
    """返回 {prompt_id: [processed 图路径,...]}，仅含成功样本。"""
    out = {}
    for pid in prompt_ids:
        pdir = os.path.join(src_dir, pid)
        if not os.path.isdir(pdir):
            continue
        imgs = []
        for k in range(n_samples):
            p = os.path.join(pdir, f"seed{k}_processed.png")
            if os.path.exists(p):
                imgs.append(p)
        if imgs:
            out[pid] = imgs
    return out


def export_geneval(src_dir, eval_dir, n_samples, meta_src):
    """GenEval 使用 raw 原图打分，保留完整画面内容。"""
    with open(meta_src) as f:
        records = [json.loads(l) for l in f]
    pid_list = [d.name for d in os.scandir(src_dir) if d.is_dir()]
    os.makedirs(eval_dir, exist_ok=True)
    cnt = 0
    for pid in pid_list:
        idx = int(pid)
        outpath = os.path.join(eval_dir, f"{idx:05d}")
        os.makedirs(os.path.join(outpath, "samples"), exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
            json.dump(records[idx], f)
        copied = 0
        for k in range(n_samples):
            raw_path = os.path.join(src_dir, pid, f"seed{k}_raw.png")
            if os.path.exists(raw_path):
                shutil.copy(raw_path, os.path.join(outpath, "samples", f"{k:05d}.png"))
                copied += 1
        if copied:
            cnt += 1
    return cnt


def export_dpg(src_dir, eval_dir, n_samples, resolution=1024):
    pid_list = [d.name for d in os.scandir(src_dir) if d.is_dir()]
    samples = collect_samples(src_dir, pid_list, n_samples)
    os.makedirs(eval_dir, exist_ok=True)
    for pid, imgs in samples.items():
        grid = Image.new("RGB", (resolution * 2, resolution * 2))
        for i, img_path in enumerate(imgs[:4]):
            im = Image.open(img_path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
            grid.paste(im, ((i % 2) * resolution, (i // 2) * resolution))
        grid.save(os.path.join(eval_dir, f"{pid}.png"))
    return len(samples)


def export_longtext(src_dir, eval_dir, n_samples):
    """LongText-Bench 使用 raw 原图送入 OCR。

    gpt-image-2 等模型返回的 raw 图可能是竖图（如 1024x1536），
    中心裁剪会裁掉上下文字导致 OCR 分数失真；raw 图能保留完整文本布局。
    """
    pid_list = [d.name for d in os.scandir(src_dir) if d.is_dir()]
    os.makedirs(eval_dir, exist_ok=True)
    cnt = 0
    for pid in pid_list:
        pdir = os.path.join(src_dir, pid)
        copied = 0
        for k in range(n_samples):
            raw_path = os.path.join(pdir, f"seed{k}_raw.png")
            if os.path.exists(raw_path):
                shutil.copy(raw_path, os.path.join(eval_dir, f"{pid}_{k}.png"))
                copied += 1
        if copied:
            cnt += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="configs/ 下模型 yaml：可写文件名（如 z_image）或路径")
    ap.add_argument("--benchmark", required=True,
                    choices=["geneval", "dpg", "longtext_en", "longtext_zh"])
    ap.add_argument("--tag", default=None,
                    help="生图时的 tag（缺省为时间戳）；与 gen_images.py --tag 一致")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="覆盖 base.yaml 注册表中该 benchmark 的官方 n_samples")
    ap.add_argument("--eval-dir", default=None,
                    help="导出的官方格式目录（缺省 <run_dir>/_eval/<benchmark>/images）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = out_dir(cfg, args.benchmark, args.tag)
    src_dir = os.path.join(run_dir, "images")
    eval_dir = args.eval_dir or os.path.join(run_dir, "_eval", args.benchmark, "images")
    n = args.n_samples or cfg["benchmarks"][args.benchmark]["n_samples"]

    if args.benchmark == "geneval":
        cnt = export_geneval(src_dir, eval_dir, n, cfg["benchmarks"]["geneval"]["prompts"])
    elif args.benchmark == "dpg":
        cnt = export_dpg(src_dir, eval_dir, n)
    else:
        cnt = export_longtext(src_dir, eval_dir, n)
    print(f"[{args.benchmark}] exported {cnt} prompts -> {eval_dir}")


if __name__ == "__main__":
    main()
