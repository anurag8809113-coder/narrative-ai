from src.llm_client import ask_llm
import json, re

PROMPT = """
You are an AI that MUST return ONLY valid JSON.

Extract factual claims from the text below.

Text:
{text}

Return ONLY in this exact JSON format (no extra words):

{{
  "claims": [
    "claim 1",
    "claim 2"
  ]
}}
"""

def extract_claims(text):
    raw = ask_llm(PROMPT.format(text=text))

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group(0))
        return data.get("claims", [])
    except Exception:
        return []

