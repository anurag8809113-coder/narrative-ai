from src.llm_client import ask_llm
import json
import re

# -------------------------
# Prompt Builder
# -------------------------
def build_prompt(claim, evidence):
    return (
        "Claim:\n"
        f"{claim}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Decide:\n"
        "SUPPORT / CONTRADICT / UNKNOWN\n\n"
        "If possible return JSON:\n"
        "{\n"
        '  "label": "...",\n'
        '  "reason": "..."\n'
        "}\n"
        "Otherwise return plain text starting with:\n"
        "SUPPORT / CONTRADICT / UNKNOWN\n"
    )

# -------------------------
# Helpers
# -------------------------
def _extract_json(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _detect_label_from_text(text):
    if not text:
        return "UNKNOWN", "No response from model."

    t = text.lower()
    if "contradict" in t:
        return "CONTRADICT", "The novel contradicts this claim."
    if "support" in t:
        return "SUPPORT", "The novel supports this claim."
    if "unknown" in t or "not possible" in t or "cannot determine" in t:
        return "UNKNOWN","Not enough evidence in the novel to verify this claim."

    return "UNKNOWN", "Not enough evidence in the novel to verify this claim."

# -------------------------
# Core functions
# -------------------------
def classify(claim, evidence_chunks):
    if not evidence_chunks:
        return "UNKNOWN", "No evidence."

    evidence_text = "\n---\n".join(evidence_chunks[:5])
    prompt = build_prompt(claim, evidence_text)

    raw = ask_llm(prompt)

    # Try JSON first
    data = _extract_json(raw)
    if data:
        label = str(data.get("label", "UNKNOWN")).upper()
        reason = str(data.get("reason", "")).strip()
        if label in {"SUPPORT", "CONTRADICT", "UNKNOWN"}:
            return label, reason or "No explanation provided."

    # Fallback to text detection
    label, reason = _detect_label_from_text(raw)
    return label, reason


def decide(labels, reasons):
    total = len(labels)
    support = labels.count("SUPPORT")
    contradict = labels.count("CONTRADICT")
    unknown = labels.count("UNKNOWN")

    # Strong contradiction wins
    if contradict > support:
        i = labels.index("CONTRADICT")
        return 0, f"{reasons[i]} | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown}"

    # Support only if clearly stronger
    if support > contradict:
        i = labels.index("SUPPORT")
        return 1, f"{reasons[i]} | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown}"

    # Otherwise uncertain
    return 0, f"Unclear evidence | SUPPORT:{support}, CONTRADICT:{contradict}, UNKNOWN:{unknown}"


def confidence_score(labels):
    if not labels:
        return 0

    total = len(labels)
    support = labels.count("SUPPORT")
    contradict = labels.count("CONTRADICT")
    unknown = labels.count("UNKNOWN")

    strongest = max(support, contradict)

    # agar sab UNKNOWN hai to low confidence
    if strongest == 0:
        return 20

    conf = (strongest / total) * 100

    # cap confidence so it never looks fake-perfect
    conf = min(conf, 90)

    return round(conf, 2)

