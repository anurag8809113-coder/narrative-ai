import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"  # free tier model

def ask_llm(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return "[LLM ERROR] OPENROUTER_API_KEY not set"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-name.streamlit.app",
        "X-Title": "Narrative Consistency Engine"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return clear text. If asked for JSON, return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM ERROR] {e}"

