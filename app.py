import streamlit as st
import time
import pandas as pd
import os, datetime

from src.ingest import chunk_text
from src.retrieval import retrieve
from src.claims import extract_claims
from src.reasoning import classify, decide, confidence_score
from src.report import generate_pdf

# -------------------------
# Instant Analytics
# -------------------------
USAGE_LOG = "usage_log.csv"

def log_usage(action):
    if not os.path.exists(USAGE_LOG):
        with open(USAGE_LOG, "w") as f:
            f.write("timestamp,action\n")

    with open(USAGE_LOG, "a") as f:
        f.write(f"{int(time.time())},{action}\n")

log_usage("page_open")

# -------------------------
# Google Analytics
# -------------------------
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-ECQT5LSLCM"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-ECQT5LSLCM');
</script>
""", unsafe_allow_html=True)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Narrative Consistency Engine", layout="wide")
st.title("📘 Narrative Consistency Reasoning Engine – PRO")

c1, c2 = st.columns(2)
with c1:
    story_text = st.text_area("Paste Story", height=280)
with c2:
    backstory_text = st.text_area("Paste Backstory", height=280)

st.divider()

mode = st.radio("Mode", ["Best Settings", "Manual Settings"], horizontal=True)
if mode == "Best Settings":
    k, alpha = 5, 0.65
else:
    k = st.slider("Evidence chunks", 3, 10, 5)
    alpha = st.slider("Hybrid weight", 0.0, 1.0, 0.6, step=0.05)

run = st.button("🚀 Run Analysis", type="primary")

# -------------------------
# Main Logic
# -------------------------
if run:
    log_usage("run_analysis")

    if not story_text or not backstory_text:
        st.warning("Please paste both Story and Backstory.")
    else:
        with st.spinner("Running full reasoning engine..."):
            chunks = chunk_text(story_text)
            claims = extract_claims(backstory_text)

            labels, reasons = [], []
            rows = []

            for c in claims:
                ev = retrieve(chunks, c, k=k, alpha=alpha)
                l, r = classify(c, ev)
                r = f"[Evidence {len(ev)}] {r}"

                labels.append(l)
                reasons.append(r)
                rows.append({"Claim": c, "Label": l, "Reason": r})
            # ---------- SAFETY FIX ---------- 
            if not labels:
                labels = ["UNKNOWN"]
                reasons = ["No claims could be evaluated."]


            # Final decision
            prediction, rationale = decide(labels, reasons)

            # -------- CONFIDENCE (FIXED) --------
            support = labels.count("SUPPORT")
            contradict = labels.count("CONTRADICT")
            unknown = labels.count("UNKNOWN")
            total = len(labels)

            conf = confidence_score(labels)


        # -------------------------
        # OUTPUT
        # -------------------------
        st.subheader("✅ Result")
        st.write("**Prediction:**", "Consistent" if prediction == 1 else "Inconsistent")
        st.write("**Rationale:**", rationale)

        st.subheader("📊 Confidence")
        st.progress(conf / 100)
        st.write(f"{conf}% sure")

        df = pd.DataFrame(rows)
        st.subheader("📋 Claim-wise Analysis")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name="analysis_results.csv",
            mime="text/csv"
        )

        # -------------------------
        # PDF Report
        # -------------------------
        os.makedirs("results", exist_ok=True)
        pdf_file = "results/report.pdf"
        generate_pdf(
            pdf_file,
            "Consistent" if prediction == 1 else "Inconsistent",
            conf,
            rationale,
            rows
        )

        with open(pdf_file, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                data=f,
                file_name="analysis_report.pdf",
                mime="application/pdf"
            )

        # -------------------------
        # Leaderboard
        # -------------------------
        lb_file = "results/leaderboard.csv"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        new_row = {
            "Time": now,
            "Claims": len(labels),
            "Prediction": "Consistent" if prediction == 1 else "Inconsistent",
            "Confidence": conf
        }

        if os.path.exists(lb_file) and os.path.getsize(lb_file) > 0:
            lb = pd.read_csv(lb_file)
            lb = pd.concat([lb, pd.DataFrame([new_row])], ignore_index=True)
        else:
            lb = pd.DataFrame([new_row])

        lb.to_csv(lb_file, index=False)

        st.subheader("🏆 Leaderboard (History)")
        st.dataframe(lb.tail(10))

