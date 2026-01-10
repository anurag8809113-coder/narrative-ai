import streamlit as st
import time
import pandas as pd
import os, datetime
import re

from src.ingest import chunk_text
from src.retrieval import retrieve
from src.reasoning import classify, decide, confidence_score
from src.report import generate_pdf
from src.llm_client import ask_llm

def split_by_book(df):
    books = {}
    for name in df["book_name"].dropna().unique():
        books[name] = df[df["book_name"] == name].copy()
    return books


# =========================
# CLAIM EXTRACTION
# =========================
PROMPT = """
Extract 3–5 factual claims from the text below.
Return them as simple bullet points.

Text:
{text}
"""

def extract_claims(text):
    raw = ask_llm(PROMPT.format(text=text))
    lines = raw.split("\n")
    claims = []

    for ln in lines:
        ln = ln.strip()
        ln = re.sub(r"^[-*•\d.]+\s*", "", ln)
        if len(ln) > 10:
            claims.append(ln)

    if not claims:
        claims = ["No explicit claims could be extracted from the backstory."]
    return claims


# =========================
# INSTANT ANALYTICS
# =========================
USAGE_LOG = "usage_log.csv"

def log_usage(action):
    if not os.path.exists(USAGE_LOG):
        with open(USAGE_LOG, "w") as f:
            f.write("timestamp,action\n")
    with open(USAGE_LOG, "a") as f:
        f.write(f"{int(time.time())},{action}\n")

log_usage("page_open")

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="Narrative Consistency Engine", layout="wide")
st.title("📘 Narrative Consistency Reasoning Engine – PRO")

mode_tab = st.tabs(["🔍 Single Analysis", "📦 Hackathon Batch Mode"])

# ==========================================================
# TAB 1 — SINGLE STORY + BACKSTORY MODE
# ==========================================================
with mode_tab[0]:

    c1, c2 = st.columns(2)
    with c1:
        story_text = st.text_area("Paste Story", height=280)
    with c2:
        backstory_text = st.text_area("Paste Backstory", height=280)

    st.divider()

    mode = st.radio("Mode", ["Best Settings", "Manual Settings"], horizontal=True, key="single_mode")
    if mode == "Best Settings":
        k, alpha = 5, 0.65
    else:
        k = st.slider("Evidence chunks", 3, 10, 5, key="k_single")
        alpha = st.slider("Hybrid weight", 0.0, 1.0, 0.6, step=0.05, key="a_single")

    run = st.button("🚀 Run Analysis", type="primary", key="run_single")

    if run:
        log_usage("run_single")

        if not story_text or not backstory_text:
            st.warning("Please paste both Story and Backstory.")
        else:
            with st.spinner("Running full reasoning engine..."):
                chunks = chunk_text(story_text)
                claims = extract_claims(backstory_text)

                labels, reasons, rows = [], [], []

                for c in claims:
                    ev = retrieve(chunks, c, k=k, alpha=alpha)
                    l, r = classify(c, ev)
                    r = f"[Evidence {len(ev)}] {r}"

                    labels.append(l)
                    reasons.append(r)
                    rows.append({"Claim": c, "Label": l, "Reason": r})

                if not labels:
                    labels = ["UNKNOWN"]
                    reasons = ["No claims could be evaluated."]

                prediction, rationale = decide(labels, reasons)
                conf = confidence_score(labels)

            # ---------- OUTPUT ----------
            st.subheader("✅ Result")
            st.write("**Prediction:**", "Consistent" if prediction == 1 else "Inconsistent")
            st.write("**Rationale:**", rationale)

            st.subheader("📊 Confidence")
            st.progress(conf / 100)
            st.write(f"{conf}% sure")

            df = pd.DataFrame(rows)
            st.subheader("📋 Claim-wise Analysis")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False),
                file_name="analysis_results.csv",
                mime="text/csv"
            )

            # ---------- PDF ----------
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

# ==========================================================
# TAB 2 — HACKATHON BATCH MODE
# ==========================================================

with mode_tab[1]:

    st.subheader("📦 Hackathon Smart Batch Mode")
    st.caption("Upload CSV once — app auto-detects novels")

    csv_file = st.file_uploader("Upload test.csv or train.csv", type=["csv"])

    if csv_file:
        df_all = pd.read_csv(csv_file)

        # ---- DEBUG (temporary) ----
        st.write("Columns detected:", list(df_all.columns))
        st.write("Sample rows:")
        st.dataframe(df_all.head())

        if "book_name" not in df_all.columns:
            st.error("❌ Column 'book_name' not found in CSV")
            st.stop()

        # ---- SPLIT BY NOVEL ----
        books = split_by_book(df_all)

        if not books:
            st.error("❌ No novels detected")
            st.stop()

        st.success(f"Detected {len(books)} novels")

        # ---- DROPDOWN ----
        book_selected = st.selectbox(
            "Select Novel to Analyze",
            list(books.keys())
        )

        # ---- STORY INPUT ----
        st.markdown("### Paste Story for selected novel")
        story_text = st.text_area("Story", height=200)

        # ---- SETTINGS ----
        mode2 = st.radio("Mode", ["Best Settings", "Manual Settings"], horizontal=True)
        if mode2 == "Best Settings":
            k2, alpha2 = 5, 0.65
        else:
            k2 = st.slider("Evidence chunks", 3, 10, 5)
            alpha2 = st.slider("Hybrid weight", 0.0, 1.0, 0.6, step=0.05)

        run_batch = st.button("🚀 Run for Selected Novel")

        if run_batch:

            if not story_text:
                st.warning("Please paste the story for this novel.")
                st.stop()

            with st.spinner(f"Running analysis for: {book_selected}"):

                chunks = chunk_text(story_text)
                df_book = books[book_selected]

                results = []
                progress = st.progress(0)

                for i, row in df_book.iterrows():
                    bid = row["id"]
                    backstory = str(row["content"])

                    claims = extract_claims(backstory)
                    labels, reasons = [], []

                    for c in claims:
                        ev = retrieve(chunks, c, k=k2, alpha=alpha2)
                        l, r = classify(c, ev)
                        labels.append(l)
                        reasons.append(r)

                    if not labels:
                        labels = ["UNKNOWN"]
                        reasons = ["No claims evaluated"]

                    pred, rat = decide(labels, reasons)
                    conf = confidence_score(labels)

                    results.append({
                        "Backstory ID": bid,
                        "Book": book_selected,
                        "Prediction": "consistent" if pred == 1 else "inconsistent",
                        "Confidence (%)": conf
                    })

                    total = len(df_book)

                    if total > 0:
                        value = (i + 1) / total
                        value = min(max(value, 0.0), 1.0)   # 0–1 ke beech lock
                        progress.progress(value)
                    else:
                        progress.progress(1.0)


            # ---- OUTPUT ----
            out_df = pd.DataFrame(results)

            st.success("✅ Analysis complete for selected novel")
            st.dataframe(out_df, use_container_width=True)

            csv_out = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download Results for {book_selected}",
                csv_out,
                file_name=f"results_{book_selected.replace(' ','_')}.csv",
                mime="text/csv",
            )


# ==========================================================
# LEADERBOARD (GLOBAL)
# ==========================================================
st.divider()
st.subheader("🏆 Leaderboard (History)")

lb_file = "results/leaderboard.csv"
os.makedirs("results", exist_ok=True)

if os.path.exists(lb_file) and os.path.getsize(lb_file) > 0:
    lb = pd.read_csv(lb_file)
    st.dataframe(lb.tail(10), use_container_width=True)
else:
    st.info("No history yet. Run some analyses to build leaderboard.")

