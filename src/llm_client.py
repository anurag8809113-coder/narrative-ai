import os
import requests

HF_API_KEY = os.getenv("HF_API_KEY")

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def ask_llm(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.3
        }
    }

    try:
        res = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()

        # HF returns list
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        # fallback
        return str(data)

    except Exception as e:
        return f"[LLM ERROR] {e}"

