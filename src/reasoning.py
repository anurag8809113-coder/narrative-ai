from src.llm_client import ask_llm
import json, re

TEMPLATE = """
You must answer ONLY in JSON.

Claim:
{claim}

Evidence:
{evidence}

Return ONLY this JSON:

{{
 "label": "SUPPORT | CONTRADICT | UNKNOWN",
 "reason": "short explanation"
}}
"""

def _extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def classify(claim, evidence_chunks):
    if not evidence_chunks:
        return "UNKNOWN", "No evidence."

    prompt = TEMPLATE.format(
        claim=claim,
        evidence="\n---\n".join(evidence_chunks[:5])
    )

    raw = ask_llm(prompt)
    data = _extract_json(raw)

    if not data:
        return "UNKNOWN", "Model did not return structured output."

    label = data.get("label", "UNKNOWN").upper()
    reason = data.get("reason", "No explanation provided.")

    if label not in {"SUPPORT", "CONTRADICT", "UNKNOWN"}:
        label = "UNKNOWN"

    return label, reason


def decide(labels, reasons):
    total = len(labels)
    support = labels.count("SUPPORT")
    contradict = labels.count("CONTRADICT")
    unknown = labels.count("UNKNOWN")

    if total == 0:
        return 0, "No valid predictions."

    if contradict >= 2:
        conf = round((contradict / total) * 100, 2)
        i = labels.index("CONTRADICT")
        return 0, f"{reasons[i]} | Confidence: {conf}%"

    if support >= contradict:
        conf = round((support / total) * 100, 2)
        i = labels.index("SUPPORT")
        return 1, f"{reasons[i]} | Confidence: {conf}%"

    conf = round((unknown / total) * 100, 2)
    i = labels.index("UNKNOWN")
    return 2, f"{reasons[i]} | Confidence: {conf}%"

def confidence_score(part, total):
    if total == 0:
        return 0
    return round((part / total) * 100, 2)
