from src.llm_client import ask_llm
import json
import re

PROMPT = """
Extract atomic factual claims from this backstory.
Return ONLY JSON in this format:
{{ "claims": ["...", "..."] }}

Backstory:
\"\"\"{text}\"\"\"
"""

def _extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def extract_claims(text):
    raw = ask_llm(PROMPT.format(text=text))
    data = _extract_json(raw)

    if not data or "claims" not in data:
        # Fallback: use first few sentences as weak claims
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return sentences[:3]

    return data["claims"]

