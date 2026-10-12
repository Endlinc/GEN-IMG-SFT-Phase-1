"""Unified captioner factory for baseline reproduction.

Self-contained — uses only stdlib HTTP, so it works without extra SDKs.

Supported captioner names
--------------------------
  gpt-5.5
  claude-sonnet-4-5-20250929
  qwen3.5-397b-a17b
  doubao-seed-2-1-pro-260628   (routed to Volcano Ark, thinking disabled)
  <any other model ID> — forwarded to OpenAI-compatible endpoint via env vars

Environment variables
---------------------
  OPENAI_API_KEY     API key for OpenAI-compatible endpoint (yunwu.ai)
  OPENAI_BASE_URL    Base URL for OpenAI-compatible endpoint (default: https://api.openai.com/v1)
  ANTHROPIC_API_KEY  API key for Anthropic Messages API
  CLAUDE_MODEL       Claude model override (default: claude-sonnet-4-5-20250929)
"""

from __future__ import annotations

import abc
import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Caption-generation prompts (paper-aligned)
# ---------------------------------------------------------------------------

# Default: combines CompreCap focus + CAPTURE 150-word minimum
CAPTIONER_PROMPT = (
    "Please describe this image in detail, including all visible objects, their attributes, "
    "and the relationships among them. Focus only on what is clearly visible. Write at least "
    "150 words as flowing prose. No preamble, no bullet lists."
)

# PerceptionRubrics official predict.py uses a shorter prompt
PR_CAPTIONER_PROMPT = "Describe the image in detail."


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Captioner(abc.ABC):
    @abc.abstractmethod
    def caption(self, image: np.ndarray) -> str: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_b64(image: np.ndarray) -> str:
    from PIL import Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    # JPEG q95 is visually lossless, 3-5x smaller than PNG -> avoids API size limits
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post(url: str, headers: Dict[str, str], payload: Dict[str, Any], retries: int = 8) -> Dict[str, Any]:
    import time, random
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read().decode("utf-8")
                if not raw or not raw.strip():
                    raise ValueError("Empty response from API")
                return json.loads(raw)
        except (urllib.error.HTTPError, urllib.error.URLError,
                ConnectionError, TimeoutError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < retries - 1:
                # exponential backoff capped at 30s, plus jitter to de-sync
                # concurrent workers all hitting a saturated upstream (HTTP 429)
                wait = min(2 ** attempt, 30) + random.uniform(0, 2)
                time.sleep(wait)
    raise last_err


# ---------------------------------------------------------------------------
# OpenAI-compatible captioner (gpt-5.5, qwen3.5-397b-a17b, doubao, vLLM, ...)
# ---------------------------------------------------------------------------

class OpenAICaptioner(Captioner):
    def __init__(
        self,
        model: str = "gpt-5.5",
        system: str = CAPTIONER_PROMPT,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.system = system
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        # extra top-level payload params (e.g. Volcano Ark's {"thinking": {...}})
        self.extra_body = extra_body or {}

    def caption(self, image: np.ndarray) -> str:
        b64 = _to_b64(image)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Describe this image in detail."},
                ]},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra_body)
        resp = _post(
            self.base_url + "/chat/completions",
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            payload,
        )
        return resp["choices"][0]["message"].get("content", "").strip()


# ---------------------------------------------------------------------------
# Anthropic Messages API captioner (claude-sonnet-4-5-20250929)
# Note: yunwu supports Anthropic models via OpenAI-compat endpoint too,
#       but the native Anthropic API is used when ANTHROPIC_API_KEY is set.
# ---------------------------------------------------------------------------

class AnthropicCaptioner(Captioner):
    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        system: str = CAPTIONER_PROMPT,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.system = system
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def caption(self, image: np.ndarray) -> str:
        b64 = _to_b64(image)
        payload = {
            "model": self.model,
            "system": self.system,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64,
                }},
                {"type": "text", "text": "Describe this image in detail."},
            ]}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = _post(
            self.base_url + "/messages",
            {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            payload,
        )
        texts = [b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text"]
        return " ".join(texts).strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_captioner(
    name: str,
    device: str = "cuda",           # kept for signature compat, unused for API models
    prompt_override: Optional[str] = None,
) -> Captioner:
    """Build a captioner by model name.

    All API models route through yunwu.ai (OPENAI_API_KEY + OPENAI_BASE_URL).
    The `device` arg is accepted but ignored for API models.
    """
    prompt = prompt_override if prompt_override is not None else CAPTIONER_PROMPT
    name_lower = name.lower().strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key  = os.environ.get("OPENAI_API_KEY", "")

    # Volcano Ark path for Doubao/Seed models. These are thinking models whose
    # unbounded reasoning starves the visible caption (truncates or empties it),
    # so we disable thinking — captioning is a perception task, not reasoning,
    # and the Seed-2.0-Lite that topped PerceptionRubrics is itself non-thinking.
    if name_lower.startswith("doubao") or name_lower.startswith("seed"):
        ark_key = os.environ.get("ARK_API_KEY", "")
        ark_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        return OpenAICaptioner(
            model=name, system=prompt, api_key=ark_key, base_url=ark_url,
            extra_body={"thinking": {"type": "disabled"}},
        )

    # Anthropic native API path (only if explicit ANTHROPIC_API_KEY is set
    # AND no OPENAI_BASE_URL override pointing to yunwu/proxy)
    _is_proxy = "yunwu" in base_url or "openai.com" not in base_url
    _claude_via_openai = _is_proxy or bool(api_key)

    if name_lower in ("claude", "claude-sonnet-4-5-20250929"):
        model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
        if _claude_via_openai:
            return OpenAICaptioner(model=model, system=prompt, api_key=api_key, base_url=base_url)
        return AnthropicCaptioner(
            model=model, system=prompt,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    # All other names: pass the model ID straight through to the OpenAI-compatible
    # endpoint. Covers gpt-5.5, qwen3.5-397b-a17b, and any future model ID —
    # original casing is preserved for provider-specific IDs.
    return OpenAICaptioner(model=name, system=prompt, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Concurrent captioning helper
# ---------------------------------------------------------------------------

def caption_many(
    captioner: "Captioner",
    items: List[Any],
    max_workers: int = 8,
    label: str = "",
) -> Dict[Any, str]:
    """Caption a list of (key, image) pairs concurrently.

    Returns {key: caption}. Failures (after _post's own retries are exhausted)
    are recorded as empty strings so the caller can detect and re-run them.
    Order-independent: each result is keyed, not positional.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results: Dict[Any, str] = {}
    lock = threading.Lock()
    done = 0
    total = len(items)

    def _one(key: Any, image: np.ndarray):
        try:
            return key, captioner.caption(image)
        except Exception as e:  # noqa: BLE001 - record failure, keep going
            return key, f"__ERROR__: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, k, img) for k, img in items]
        for fut in as_completed(futures):
            key, cap = fut.result()
            with lock:
                results[key] = cap
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"[{label}] {done}/{total} captioned", flush=True)
    return results
