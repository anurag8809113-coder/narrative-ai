import csv
import matplotlib.pyplot as plt

story_ids = []
confidences = []

with open("results/results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        story_ids.append(row["Story ID"])

        # Extract confidence number from rationale text
        text = row["Rationale"]
        conf = 0
        if "Confidence:" in text:
            try:
                conf = float(text.split("Confidence:")[1].split("%")[0])
            except:
                conf = 0
        confidences.append(conf)

plt.figure()
plt.bar(story_ids, confidences)
plt.xlabel("Story ID")
plt.ylabel("Confidence (%)")
plt.title("Narrative Consistency Confidence by Story")
plt.tight_layout()
plt.show()

