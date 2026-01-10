import pandas as pd

def main():
    file = "final_submission.csv"
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print("❌ Cannot read file:", e)
        return

    errors = []

    # Column check
    required = {"Backstory ID", "Prediction"}
    if not required.issubset(set(df.columns)):
        errors.append(f"Missing columns. Required: {required}")

    # Value check
    if "Prediction" in df.columns:
        bad = df[~df["Prediction"].isin([0, 1])]
        if len(bad) > 0:
            errors.append("Prediction column must contain only 0 or 1.")

    if errors:
        print("❌ VALIDATION FAILED")
        for e in errors:
            print(" -", e)
    else:
        print("✅ VALIDATION PASSED — File ready to upload!")

if __name__ == "__main__":
    main()

