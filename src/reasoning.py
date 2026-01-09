from src.llm_client import ask_llm
import json, re

TEMPLATE = """
Claim:
{claim}

Evidence:
{evidence}

Decide:
SUPPORT / CONTRADICT / UNKNOWN

Return JSON:
{{
 "label": "...",
 "reason": "..."
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

    if contradict >= 2:
        conf = round((contradict / total) * 100, 2)
        i = labels.index("CONTRADICT")
        return 0, f"{reasons[i]} | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown} | Confidence:{conf}%"

    if support >= contradict:
        conf = round((support / total) * 100, 2)
        i = labels.index("SUPPORT")
        return 1, f"{reasons[i]} | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown} | Confidence:{conf}%"

    conf = round((unknown / total) * 100, 2)
    return 1, f"No strong contradictions | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown} | Confidence:{conf}%"


def confidence_score(labels):
    if not labels:
        return 50
    m = {"SUPPORT": 1, "UNKNOWN": 0, "CONTRADICT": -1}
    raw = sum(m.get(l, 0) for l in labels)
    N = len(labels)
    return int(((raw + N) / (2 * N)) * 100)

