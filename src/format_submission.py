import pandas as pd

REQUIRED_COLUMNS = ["Backstory ID", "Prediction"]

def main():
    inp = "hackathon_results.csv"
    out = "final_submission.csv"

    df = pd.read_csv(inp)

    # Rename if needed
    if "backstory_id" in df.columns:
        df = df.rename(columns={"backstory_id": "Backstory ID"})
    if "prediction" in df.columns:
        df = df.rename(columns={"prediction": "Prediction"})

    # Keep only required columns
    df = df[REQUIRED_COLUMNS]

    # Normalize values
    df["Prediction"] = df["Prediction"].str.lower().map({
        "consistent": 1,
        "inconsistent": 0,
        "1": 1,
        "0": 0
    }).fillna(0).astype(int)

    df.to_csv(out, index=False)
    print(f"✅ Final submission file created: {out}")

if __name__ == "__main__":
    main()

