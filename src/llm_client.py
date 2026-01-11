import os
import requests
import time
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"

# -----------------------------
# Persistent Cache
# -----------------------------
CACHE_FILE = "llm_cache.json"

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            _CACHE = json.load(f)
    except Exception:
        _CACHE = {}
else:
    _CACHE = {}

# -----------------------------
# Main LLM Call
# -----------------------------
def ask_llm(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return "[LLM ERROR] OPENROUTER_API_KEY not set"

    # ---- CACHE HIT ----
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
            {
                "role": "system",
                "content": "Answer briefly. If asked for JSON, return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }

    # ---- RETRY LOGIC ----
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=60)

            # Rate limit → wait and retry
            if r.status_code == 429:
                wait = 3 * (attempt + 1)  # 3s, 6s, 9s
                time.sleep(wait)
                continue

            r.raise_for_status()

            data = r.json()
            text = data["choices"][0]["message"]["content"]

            # ---- SAVE TO CACHE ----
            _CACHE[prompt] = text
            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump(_CACHE, f)
            except Exception:
                pass  # cache save failure should not break app

            return text

        except Exception as e:
            if attempt == 2:
                return f"[LLM ERROR] {e}"
            time.sleep(2)

    return "[LLM ERROR] Failed after retries"

# -----------------------------
# OPTIONAL: Batch Reasoning Helper
# (use later if you want 3–5x speed)
# -----------------------------
def ask_llm_batch(claims, evidence_map):
    """
    claims: list[str]
    evidence_map: dict[str, list[str]]
    """
    prompt = "You are checking story consistency.\n\n"

    for i, c in enumerate(claims, 1):
        ev = "\n".join(evidence_map.get(c, [])[:3])
        prompt += f"""
Claim {i}: {c}
Evidence:
{ev}
Decide SUPPORT / CONTRADICT / UNKNOWN.
"""

    prompt += """
Return JSON:
{
  "results": [
    {"label": "SUPPORT|CONTRADICT|UNKNOWN", "reason": "short explanation"}
  ]
}
"""
    return ask_llm(prompt)

