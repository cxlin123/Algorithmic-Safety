"""Unified API client wrapping xty.app (OpenAI-compatible endpoint).

All three target models (gpt-5-chat, claude-sonnet-4-6, llama-3.1-70b-instruct)
are accessible through the same /v1/chat/completions endpoint.
"""
from __future__ import annotations
import time
import json
import httpx
from typing import Optional
from config import API_BASE, API_KEY


class APIError(Exception):
    pass


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    retries: int = 5,
    timeout: float = 180.0,
    logprobs: bool = False,
) -> dict:
    """Single chat completion call. Returns dict with 'text' and 'raw'."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(
                    f"{API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                return {"text": text or "", "raw": data}
            elif r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            else:
                raise APIError(f"HTTP {r.status_code}: {r.text[:500]}")
        except (httpx.TimeoutException, httpx.NetworkError,
                httpx.RemoteProtocolError, httpx.ProtocolError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    raise APIError(f"All retries failed: {last_err}")


def chat_simple(model: str, system: str, user: str, **kwargs) -> str:
    """Convenience: returns plain text response."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return chat(model, msgs, **kwargs)["text"]


if __name__ == "__main__":
    # quick smoke test
    from config import MODELS
    for tag, m in MODELS.items():
        try:
            out = chat_simple(m, "You are concise.", "Say hi in one word.")
            print(f"[{tag}:{m}] -> {out!r}")
        except Exception as e:
            print(f"[{tag}:{m}] FAIL: {e}")
