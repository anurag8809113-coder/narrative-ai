from src.llm_client import ask_llm
import json, re

TEMPLATE = """
Claim:
{claim}

Evidence:
{evidence}

Decide:
SUPPORT / CONTRADICT / UNKNOWN

Return JSON exactly in this format:
{{
  "label": "SUPPORT|CONTRADICT|UNKNOWN",
  "reason": "short explanation"
}}
"""

def _extract_json(text: str):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ---------- CLASSIFY ----------
def classify(claim, evidence_chunks):
    if not evidence_chunks:
        return "UNKNOWN", "No evidence found in story."

    prompt = TEMPLATE.format(
        claim=claim,
        evidence="\n---\n".join(evidence_chunks[:5])
    )

    raw = ask_llm(prompt)

    data = _extract_json(raw)

    # STRICT fallback — never auto-support
    if not data:
        return "UNKNOWN", "LLM could not judge this claim."

    label = str(data.get("label", "UNKNOWN")).upper()
    reason = str(data.get("reason", "No explanation provided."))

    if label not in {"SUPPORT", "CONTRADICT", "UNKNOWN"}:
        label = "UNKNOWN"

    return label, reason


# ---------- DECIDE ----------
def decide(labels, reasons):
    support = labels.count("SUPPORT")
    contradict = labels.count("CONTRADICT")
    unknown = labels.count("UNKNOWN")

    # Clear contradiction wins
    if contradict > support:
        i = labels.index("CONTRADICT")
        return 0, f"{reasons[i]} | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown}"

    # Clear support wins
    if support > contradict:
        i = labels.index("SUPPORT")
        return 1, f"{reasons[i]} | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown}"

    # If tie or all UNKNOWN → treat as not proven (inconsistent)
    return 0, f"No strong evidence | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown}"


# ---------- CONFIDENCE ----------
def confidence_score(labels):
    if not labels:
        return 0

    total = len(labels)
    support = labels.count("SUPPORT")
    contradict = labels.count("CONTRADICT")

    strongest = max(support, contradict)

    # all UNKNOWN → low confidence
    if strongest == 0:
        return 20

    conf = (strongest / total) * 100

    # never show fake 100%
    conf = min(conf, 90)

    return round(conf, 2)

