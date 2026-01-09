import os
import requests
import json

# ================================
# HuggingFace Inference API Client
# ================================

HF_API_KEY = os.getenv("HF_API_KEY")

# You can change model if you want
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def ask_llm(prompt: str) -> str:
    """
    Sends prompt to HuggingFace Inference API.
    Works on:
    - Local
    - Streamlit Cloud
    - Any server
    """

    if not HF_API_KEY:
        return '{"label":"UNKNOWN","reason":"HF_API_KEY not set"}'

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.2
        }
    }

    try:
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()

        data = r.json()

        # HF usually returns list
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        # Some models return dict
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]

        # Fallback
        return json.dumps(data)

    except Exception as e:
        return f'{{"label":"UNKNOWN","reason":"LLM error: {e}"}}'

