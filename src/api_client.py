"""OpenAI-compatible chat client。

中文说明：
1. 这里仍然沿用原论文工程的统一 chat_simple() 接口。
2. API_BASE/API_KEY 来自 config.py，可指向 xty.app 或任何兼容接口。
3. Gemini/Qwen judge、GPT/Claude 被测模型都通过同一个 chat() 发送请求。
"""
from __future__ import annotations
import time
import httpx
from config import API_BASE, API_KEY


class APIError(Exception):
    """API 调用失败时抛出的统一异常类型。"""
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
    """发送一次 chat/completions 请求，返回文本和原始响应。"""
    if not API_KEY:
        raise APIError(
            "API key is not set. Export XTY_API_KEY or API_KEY before running live calls."
        )

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
    """简化入口：传 system/user 两段文本，直接返回模型回复字符串。"""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return chat(model, msgs, **kwargs)["text"]


if __name__ == "__main__":

    from config import MODELS
    for tag, m in MODELS.items():
        try:
            out = chat_simple(m, "You are concise.", "Say hi in one word.")
            print(f"[{tag}:{m}] -> {out!r}")
        except Exception as e:
            print(f"[{tag}:{m}] FAIL: {e}")
