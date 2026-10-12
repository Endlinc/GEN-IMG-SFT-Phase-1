"""汇总 failures.jsonl：按 error_type 统计失败率（口径 A：失败率单列）。

用法:
  python scripts/summarize_failures.py --config qwen_image_2 --benchmark geneval [--tag 20261004-120000]
"""
import argparse
import json
import os
from collections import Counter

from common import load_config, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="configs/ 下模型 yaml：可写文件名（如 z_image）或路径")
    ap.add_argument("--benchmark", required=True,
                    choices=["geneval", "dpg", "longtext_en", "longtext_zh",
                             "qwenimagebench_cn", "qwenimagebench_en"])
    ap.add_argument("--tag", default=None,
                    help="生图时的 tag（缺省为时间戳）；failures.jsonl 在 <tag>/ 下")
    args = ap.parse_args()
    cfg = load_config(args.config)

    path = os.path.join(out_dir(cfg, args.benchmark, args.tag), "failures.jsonl")
    if not os.path.exists(path):
        print("无失败记录。")
        return
    stats = Counter()
    # 同一 (prompt_id, sample_index) 只计最后一次
    last = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            last[(str(r["prompt_id"]), r["sample_index"])] = r["error_type"]
    stats.update(last.values())
    total = sum(stats.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in stats.most_common())
    print(f"失败样本统计（{cfg['model_name']} / {args.benchmark} / {os.path.dirname(path)}）:")
    print(f"  共 {total} 条失败 ({breakdown})")


if __name__ == "__main__":
    main()
