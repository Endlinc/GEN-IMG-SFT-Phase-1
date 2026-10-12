"""经图像生成 API 平台（任意中转站 openai_compatible / 千问 qianwen）生成 benchmark 图片。

特性（对应评测协议）：
- 不做 prompt 改写（prompt_extend=false，原样直传）
- 网络/超时/5xx 指数退避重试，内容审核类 4xx 不重试（口径 A：记失败样本）
- 每张图落盘 raw/processed + meta.json；失败写 <tag>/failures.jsonl
- 产物布局：outputs/<model>/<benchmark>/<tag>/images/<prompt_id>/seed{k}_raw/processed.png

用法示例：
  python scripts/gen_images.py --config qwen_image_2 --benchmark geneval [--tag smoke] [--limit 5]
"""
import argparse
import base64
import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from common import load_config, out_dir


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_prompts(benchmark: str, limit: int | None, cfg):
    """返回 [{prompt_id, prompt, meta}]，prompt_id 为字符串。

    prompt 文件路径来自 configs/base.yaml 的 benchmarks 注册表（已解析为绝对路径）。
    """
    items = []
    path = cfg["benchmarks"][benchmark]["prompts"]
    if benchmark == "geneval":
        with open(path) as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                items.append({"prompt_id": f"{i:05d}", "prompt": d["prompt"], "meta": d})
    elif benchmark == "dpg":
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".txt"):
                with open(os.path.join(path, fn)) as f:
                    items.append({"prompt_id": fn[:-4], "prompt": f.read().strip(), "meta": {}})
    elif benchmark in ("longtext_en", "longtext_zh"):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                items.append({"prompt_id": str(d["prompt_id"]), "prompt": d["prompt"], "meta": d})
    elif benchmark in ("qwenimagebench_cn", "qwenimagebench_en"):
        # Qwen-Image-Bench 1000 条双语 prompt；prompt_id 为 0 填充的 ID（对齐官方 1-1000）
        lang = benchmark.rsplit("_", 1)[1]  # cn | en
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                items.append({"prompt_id": f"{d['ID']:06d}", "prompt": d[f"prompt_{lang}"], "meta": d})
    else:
        raise ValueError(benchmark)
    return items[:limit] if limit else items


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def make_seed(benchmark: str, prompt_id: str, sample_index: int) -> int:
    """确定性 seed（协议 §3.3）：同一 (benchmark, prompt_id, sample_index)
    跨模型/跨 run 恒定（不含模型名），且在不同 benchmark 间唯一。"""
    h = hashlib.sha256(f"{benchmark}::{prompt_id}::{sample_index}".encode()).hexdigest()
    return int(h, 16) % (2 ** 31)


class _BaseClient:
    # 审核类错误关键词（避免 "risk" 误伤 "frisky" 等正常词）
    _POLICY_KEYWORDS = ("content_policy", "data_inspection", "inappropriate", "censor", "敏感", "审核")

    def __init__(self, cfg):
        api_cfg = cfg["api"]
        self.model = api_cfg["model"]
        # size 统一规范化：内部存 宽*高，中转站（OpenAI 格式）用 x 分隔
        self.size = api_cfg["size"].replace("x", "*").replace("X", "*")
        self.size_star = self.size
        self.size_x = self.size.replace("*", "x")
        self.prompt_extend = api_cfg.get("prompt_extend", False)
        self.watermark = api_cfg.get("watermark", False)
        self.negative_prompt = api_cfg.get("negative_prompt", "")
        self.max_retries = api_cfg.get("max_retries", 3)
        self.timeout = api_cfg.get("timeout_s", 300)
        self.interval = api_cfg.get("request_interval_s", 1.0)
        # payload_style: legacy(带 qwen 风格 parameters 块) / openai(裸 OpenAI images 格式)
        self.payload_style = api_cfg.get("payload_style", "legacy")
        self.quality = api_cfg.get("quality")  # openai 风格时可选 quality 参数
        self.supports_seed = api_cfg.get("supports_seed", True)

    @staticmethod
    def _require_key(env_var):
        key = os.environ.get(env_var)
        if not key:
            raise RuntimeError(f"缺少 API key：请在 .env 中填写 {env_var}")
        return key

    def _fetch_url(self, url):
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.content

    @staticmethod
    def _extract_error_message(resp):
        """优先取结构化 error message（中转站 error.message / 千问 message），失败回退 raw text。"""
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message") or body.get("message") or ""
            if msg:
                return str(msg)[:2000]
        except Exception:
            pass
        return resp.text[:2000]

    @classmethod
    def _classify_error(cls, status, msg):
        m = msg.lower()
        if "额度已用尽" in msg or "quota" in m or "insufficient" in m:
            return "quota_exhausted"
        if status in (400, 422) and any(k in m for k in cls._POLICY_KEYWORDS):
            return "content_policy_violation"
        if status == 429:
            return "rate_limited"
        if status and 500 <= status < 600:
            return "server_error"
        if status and 400 <= status < 500:
            return "invalid_request"
        return "unknown_error"


class OpenAICompatImageClient(_BaseClient):
    """任意中转站平台（OpenAI images 兼容格式，示例平台 yunwu）。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", cfg["api"]["base_url"]).rstrip("/")
        self.model = os.environ.get("OPENAI_COMPAT_MODEL", self.model)
        self.api_key = self._require_key("OPENAI_COMPAT_API_KEY")

    def _endpoint(self):
        return f"{self.base_url}/v1/images/generations"

    def _payload(self, prompt, seed):
        if self.payload_style == "openai":
            # 裸 OpenAI images 格式（gpt-image 系列）：不发 qwen 风格 parameters 块
            payload = {"model": self.model, "prompt": prompt, "n": 1, "size": self.size_x}
            if self.quality:
                payload["quality"] = self.quality
            if seed is not None and self.supports_seed:
                payload["seed"] = seed
            return payload
        parameters = {
            "prompt_extend": self.prompt_extend,
            "watermark": self.watermark,
            "negative_prompt": self.negative_prompt,
            "size": self.size_star,
        }
        if seed is not None:
            parameters["seed"] = seed
        return {
            "model": self.model, "prompt": prompt, "n": 1,
            "size": self.size_x, "seed": seed,
            "parameters": parameters,
        }

    def _extract(self, body):
        datum = body["data"][0]
        if datum.get("b64_json"):
            return base64.b64decode(datum["b64_json"]), body.get("model", self.model)
        return self._fetch_url(datum["url"]), body.get("model", self.model)

    def generate(self, prompt: str, seed: int | None = None) -> dict:
        return _generate_with_retry(self, prompt, seed)


class QianwenImageClient(_BaseClient):
    """千问 AI 平台（DashScope 原生 multimodal-generation 同步接口）。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        api_cfg = cfg["api"]
        self.base_url = os.environ.get("QIANWEN_BASE_URL", api_cfg["base_url"]).rstrip("/")
        self.model = os.environ.get("QWEN_IMAGE_MODEL", self.model)
        self.api_key = self._require_key("QIANWEN_API_KEY")

    def _endpoint(self):
        return f"{self.base_url}/services/aigc/multimodal-generation/generation"

    def _payload(self, prompt, seed):
        parameters = {
            "prompt_extend": self.prompt_extend,
            "watermark": self.watermark,
            "negative_prompt": self.negative_prompt,
            "size": self.size_star,
        }
        if seed is not None:
            parameters["seed"] = seed
        return {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": parameters,
        }

    def _extract(self, body):
        content = body["output"]["choices"][0]["message"]["content"]
        url = next(item["image"] for item in content if "image" in item)
        return self._fetch_url(url), self.model

    def generate(self, prompt: str, seed: int | None = None) -> dict:
        return _generate_with_retry(self, prompt, seed)


def _generate_with_retry(client, prompt, seed):
    """公共重试逻辑：200 抽图；审核/非法请求不重试；网络/5xx 指数退避。"""
    headers = {"Authorization": f"Bearer {client.api_key}", "Content-Type": "application/json"}
    payload = client._payload(prompt, seed)
    retry = 0
    while True:
        t0 = time.time()
        try:
            resp = requests.post(client._endpoint(), headers=headers, json=payload, timeout=client.timeout)
            latency = int((time.time() - t0) * 1000)
            status = resp.status_code
            if status == 200:
                try:
                    img_bytes, raw_model = client._extract(resp.json())
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                    raise  # 图片 URL 下载网络失败 -> 走外层重试
                except requests.exceptions.RequestException as e:
                    # 图片 URL 返回 4xx/5xx，同样按可重试网络错误处理
                    raise requests.exceptions.ConnectionError(str(e)) from e
                return {"ok": True, "image_bytes": img_bytes, "http_status": status,
                        "error_type": None, "error_msg": None, "retry_count": retry,
                        "latency_ms": latency, "raw_model": raw_model}
            err_msg = client._extract_error_message(resp)
            err_type = client._classify_error(status, err_msg)
            if err_type in ("content_policy_violation", "invalid_request") or retry >= client.max_retries:
                return {"ok": False, "image_bytes": None, "http_status": status,
                        "error_type": err_type, "error_msg": err_msg,
                        "retry_count": retry, "latency_ms": latency, "raw_model": None}
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            latency = int((time.time() - t0) * 1000)
            if retry >= client.max_retries:
                return {"ok": False, "image_bytes": None, "http_status": None,
                        "error_type": "network_or_timeout", "error_msg": str(e)[:2000],
                        "retry_count": retry, "latency_ms": latency, "raw_model": None}
        time.sleep(2 ** (retry + 1))  # 2s / 4s / 8s
        retry += 1


CLIENTS = {"openai_compatible": OpenAICompatImageClient, "qianwen": QianwenImageClient}


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
    ap.add_argument("--provider", default=None, choices=["openai_compatible", "qianwen"],
                    help="覆盖 config 中的 api.provider")
    args = ap.parse_args()
    # .env 固定从项目根（脚本上一级）加载，不受 cwd 影响
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    cfg = load_config(args.config)

    model_name = cfg["model_name"]
    n_samples = args.n_samples or cfg["benchmarks"][args.benchmark]["n_samples"]
    run_dir = out_dir(cfg, args.benchmark, args.tag)
    out_images = os.path.join(run_dir, "images")
    os.makedirs(out_images, exist_ok=True)
    failures_path = os.path.join(run_dir, "failures.jsonl")
    # 已记录的失败样本（断点重跑时避免同一失败反复追加）
    seen_failures = set()
    if os.path.exists(failures_path):
        with open(failures_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_failures.add((args.benchmark, str(d.get("prompt_id")), d.get("sample_index")))

    provider = args.provider or cfg["api"].get("provider", "openai_compatible")
    client = CLIENTS[provider](cfg)
    prompts = load_prompts(args.benchmark, args.limit, cfg)
    if args.prompt_ids:
        keep = {s.strip() for s in args.prompt_ids.split(",") if s.strip()}
        prompts = [p for p in prompts if p["prompt_id"] in keep]
        missing = keep - {p["prompt_id"] for p in prompts}
        if missing:
            raise SystemExit(f"prompt_ids 未在 benchmark {args.benchmark} 中找到: {sorted(missing)}")
    print(f"[{args.benchmark}] {len(prompts)} prompts x {n_samples} samples -> {run_dir}")

    for item in tqdm(prompts):
        pid = item["prompt_id"]
        pdir = os.path.join(out_images, pid)
        os.makedirs(pdir, exist_ok=True)
        # 断点续跑：已齐 n_samples 张 processed 图则跳过
        existing = [f for f in os.listdir(pdir) if f.endswith("_processed.png")]
        if len(existing) >= n_samples:
            continue
        for k in range(n_samples):
            meta_path = os.path.join(pdir, f"seed{k}_meta.json")
            proc_path = os.path.join(pdir, f"seed{k}_processed.png")
            if os.path.exists(meta_path) and os.path.exists(proc_path):
                with open(meta_path) as f:
                    if json.load(f).get("http_status") == 200:
                        continue  # 该样本已成功且产物齐全，跳过
            seed = make_seed(args.benchmark, pid, k)
            result = client.generate(item["prompt"], seed=seed)
            meta = {
                "prompt_id": pid,
                "prompt_text": item["prompt"],
                "benchmark": args.benchmark,
                "model": model_name,
                "api_model_id_returned": result["raw_model"],
                "request_params": {
                    "size": client.size_star, "n": 1,
                    "prompt_extend": client.prompt_extend,
                    "watermark": client.watermark,
                    "negative_prompt": client.negative_prompt,
                    "seed_requested": seed if client.supports_seed else None,
                    "seed_supported": client.supports_seed,
                },
                "request_timestamp_utc": utcnow(),
                "latency_ms": result["latency_ms"],
                "http_status": result["http_status"],
                "retry_count": result["retry_count"],
                "refused": result["error_type"] == "content_policy_violation",
                "refusal_reason": result["error_msg"] if result["error_type"] == "content_policy_violation" else None,
                "sample_index": k,
            }
            if result["ok"]:
                raw_path = os.path.join(pdir, f"seed{k}_raw.png")
                with open(raw_path, "wb") as f:
                    f.write(result["image_bytes"])
                img = Image.open(io.BytesIO(result["image_bytes"])).convert("RGB")
                if img.size[0] != img.size[1]:
                    img = center_crop_square(img)  # 协议 §3.2：仅中心裁剪，不拉伸
                img.save(proc_path)
                meta["content_hash_sha256"] = sha256_bytes(result["image_bytes"])
                meta["processed_size"] = list(img.size)
            else:
                meta["status"] = "failed"
                meta["failure_stage"] = "generation"
                meta["error_type"] = result["error_type"]
                fail_key = (args.benchmark, pid, k)
                if fail_key not in seen_failures:
                    seen_failures.add(fail_key)
                    with open(failures_path, "a") as f:
                        f.write(json.dumps({
                            "status": "failed", "failure_stage": "generation",
                            "error_type": result["error_type"],
                            "raw_error_message": result["error_msg"],
                            "retry_count": result["retry_count"],
                            "model": model_name, "benchmark": args.benchmark,
                            "prompt_id": pid, "sample_index": k,
                            "timestamp_utc": utcnow(),
                        }, ensure_ascii=False) + "\n")
                if meta["error_type"] == "quota_exhausted":
                    raise SystemExit(
                        f"[{args.benchmark}] 令牌额度已用尽，停止生成 "
                        f"(prompt_id={pid}, sample_index={k})"
                    )
            with open(meta_path, "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            time.sleep(client.interval)

    print("done.")


if __name__ == "__main__":
    main()
