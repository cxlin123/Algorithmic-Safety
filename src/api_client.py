import time
import httpx
from config import API_BASE, API_KEY


class APIError(Exception):
    pass


def chat(
    model,
    messages,
    temperature=0.2,
    max_tokens=1024,
    top_p=1.0,
    retries=5,
    timeout=180.0,
    logprobs=False,
):
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


def chat_simple(model, system, user, **kwargs):
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
