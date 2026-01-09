# Narrative Consistency Reasoning Engine

## Problem
Given a novel and a hypothetical backstory, determine whether the backstory is
consistent with the story.

## Solution
We built a reasoning pipeline that:
1. Splits the novel into chunks.
2. Extracts atomic claims from the backstory using a local LLM.
3. Retrieves relevant evidence with TF-IDF similarity.
4. Classifies each claim as SUPPORT / CONTRADICT / UNKNOWN.
5. Applies constraint-based logic to make a final decision.
6. Outputs explainable results with confidence scores.

## Tech Stack
- Python
- Pathway (core pipeline orchestration)
- Ollama + Llama3 (local LLM)
- scikit-learn (TF-IDF retrieval)
- Matplotlib (confidence visualization)

## How to Run
```bash
source venv/bin/activate
python src/pipeline.py
python plot_results.py

