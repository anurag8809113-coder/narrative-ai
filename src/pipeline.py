import csv, os
from ingest import get_chunks
from retrieval import retrieve
from claims import extract_claims
from reasoning import classify, decide

def main():
    # Load story chunks
    story_chunks = get_chunks("data/stories/1_story.txt")

    # Load backstory
    with open("data/backstories/1_backstory.txt") as f:
        backstory = f.read()

    # Extract claims
    claims = extract_claims(backstory)

    labels, reasons = [], []

    for c in claims:
        evidence = retrieve(story_chunks, c, k=5)
        l, r = classify(c, evidence)

        # add evidence strength tag
        r = f"[Evidence chunks: {len(evidence)}] {r}"

        labels.append(l)
        reasons.append(r)

    # Final decision
    pred, rat = decide(labels, reasons)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Story ID", "Prediction", "Rationale", "Claims Checked"])
        w.writerow([1, pred, rat, len(labels)])

    print("✅ results/results.csv generated")

if __name__ == "__main__":
    main()

