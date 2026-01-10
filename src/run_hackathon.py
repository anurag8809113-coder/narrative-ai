import pandas as pd
from src.claims import extract_claims
from src.reasoning import classify, decide, confidence_score
from src.retrieval import retrieve

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

def process_story_with_backstories(story_path, backstory_csv, story_name):
    story = load_text(story_path)
    chunks = chunk_text(story)

    df = pd.read_csv(backstory_csv)
    results = []

    for _, row in df.iterrows():
        bid = row["id"]
        backstory_text = row["backstory"]

        claims = extract_claims(backstory_text)

        labels, reasons = [], []

        for c in claims:
            ev = retrieve(chunks, c, k=5, alpha=0.6)
            l, r = classify(c, ev)
            labels.append(l)
            reasons.append(r)

        # safety
        if not labels:
            labels = ["UNKNOWN"]
            reasons = ["No claims found"]

        pred, rat = decide(labels, reasons)
        conf = confidence_score(labels)

        results.append({
            "story": story_name,
            "backstory_id": bid,
            "prediction": "consistent" if pred == 1 else "inconsistent",
            "confidence": conf
        })

    return results


def main():
    all_results = []

    all_results += process_story_with_backstories(
        "data/stories/story1.txt",
        "data/backstories/backstory1.csv",
        "story1"
    )

    all_results += process_story_with_backstories(
        "data/stories/story2.txt",
        "data/backstories/backstory2.csv",
        "story2"
    )

    out = pd.DataFrame(all_results)
    out.to_csv("results/hackathon_submission.csv", index=False)

    print("✅ hackathon_submission.csv created")

if __name__ == "__main__":
    main()

