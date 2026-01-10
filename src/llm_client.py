import os
import requests
import time

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"

# simple in-memory cache
_CACHE = {}

def ask_llm(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return "[LLM ERROR] OPENROUTER_API_KEY not set"

    # ---- CACHE ----
    if prompt in _CACHE:
        return _CACHE[prompt]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-name.streamlit.app",
        "X-Title": "Narrative Consistency Engine"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer briefly. If asked for JSON, return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    # ---- RETRY LOGIC ----
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=60)

            # 429 = rate limit → wait & retry
            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue

            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]

            # save in cache
            _CACHE[prompt] = text
            return text

        except Exception as e:
            if attempt == 2:
                return f"[LLM ERROR] {e}"
            time.sleep(2)

    return "[LLM ERROR] Failed after retries"

