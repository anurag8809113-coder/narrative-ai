# Narrative Consistency Reasoning Engine

This project checks whether a given backstory is **consistent or inconsistent**
with a novel using:

- Claim extraction
- Hybrid retrieval (TF-IDF + embeddings)
- LLM-based reasoning
- Explainable verdicts + confidence score

---

## 🚀 Features

- ✅ Single story analysis (UI)
- ✅ Hackathon batch CSV mode (UI)
- ✅ Claim-wise explainable reasoning
- ✅ Confidence meter
- ✅ CSV export
- ✅ PDF report generation
- ✅ Leaderboard history
- ✅ Free online deployment (Streamlit Cloud)

---

## 📦 Hackathon Batch Mode

1. Upload a story `.txt` file  
2. Upload a backstories `.csv` file  
   (must contain columns: `id`, `backstory`)  
3. Click **Run Batch Analysis**  
4. Download `hackathon_results.csv`

---

## 📝 Submission Pipeline

```bash
# Step 1 — Generate submission format
python src/format_submission.py

# Step 2 — Validate
python src/validate_submission.py

