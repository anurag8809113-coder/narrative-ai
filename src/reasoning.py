from src.llm_client import ask_llm
import json, re

TEMPLATE = """
Claim:
{claim}

Evidence:
{evidence}

Decide:
SUPPORT / CONTRADICT / UNKNOWN

If possible return JSON:
{
 "label": "...",
 "reason": "..."
}
Otherwise return plain text starting with:
SUPPORT / CONTRADICT / UNKNOWN
"""

# -------------------------
# Helpers
# -------------------------
def _extract_json(text: str):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _detect_label_from_text(text: str):
    if not text:
        return "UNKNOWN", "No response from model."

    t = text.lower()

    if "contradict" in t:
        return "CONTRADICT", text.strip()
    if "support" in t:
        return "SUPPORT", text.strip()
    if "unknown" in t:
        return "UNKNOWN", text.strip()

    # fallback
    return "UNKNOWN", text.strip()


# -------------------------
# Core functions
# -------------------------
def classify(claim, evidence_chunks):
    if not evidence_chunks:
        return "UNKNOWN", "No evidence."

    prompt = TEMPLATE.format(
        claim=claim,
        evidence="\n---\n".join(evidence_chunks[:5])
    )

    raw = ask_llm(prompt)

    # 1️⃣ Try JSON first
    data = _extract_json(raw)
    if data:
        label = str(data.get("label", "UNKNOWN")).upper()
        reason = str(data.get("reason", "")).strip()
        if label in {"SUPPORT", "CONTRADICT", "UNKNOWN"}:
            return label, reason or "No explanation provided."

    # 2️⃣ Fallback to text detection
    label, reason = _detect_label_from_text(raw)
    return label, reason


def decide(labels, reasons):
    total = len(labels)
    if total == 0:
        return 1, "No claims evaluated."

    support = labels.count("SUPPORT")
    contradict = labels.count("CONTRADICT")
    unknown = labels.count("UNKNOWN")

    if contradict >= 2:
        i = labels.index("CONTRADICT")
        conf = round((contradict / total) * 100, 2)
        return 0, f"{reasons[i]} | Confidence:{conf}%"

    if support > contradict:
        i = labels.index("SUPPORT")
        conf = round((support / total) * 100, 2)
        return 1, f"{reasons[i]} | Confidence:{conf}%"

    return 1, "No strong contradictions found."


def confidence_score(labels):
    """
    Always returns confidence between 0–100.
    """
    if not labels:
        return 0.0

    total = len(labels)
    support = labels.count("SUPPORT")
    contradict = labels.count("CONTRADICT")
    unknown = labels.count("UNKNOWN")

    best = max(support, contradict, unknown)
    return round((best / total) * 100, 2)

