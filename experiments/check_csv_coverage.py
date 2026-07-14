import sys
import pandas as pd

def check_coverage(csv_path):
    df = pd.read_csv(csv_path)
    print("File: " + csv_path)
    print("Rows: " + str(len(df)))

    if "budget" not in df.columns or "strategy" not in df.columns:
        print("Missing budget or strategy column -- can't check grid coverage.")
        return

    pivot = df.groupby(["budget", "strategy"]).size().reset_index(name="row_count")
    print("")
    print("=== budget x strategy coverage ===")
    print(pivot.to_string(index=False))

    expected_budgets = [50, 100, 200]
    expected_strategies = ["influence", "random"]
    print("")
    print("=== grid check against 50/100/200 x influence/random ===")
    for b in expected_budgets:
        for s in expected_strategies:
            match = df[(df["budget"] == b) & (df["strategy"] == s)]
            status = "OK (" + str(len(match)) + " rows)" if len(match) > 0 else "MISSING"
            print("budget=" + str(b) + " strategy=" + s + " -> " + status)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_csv_coverage.py <path_to_csv>")
        sys.exit(1)
    check_coverage(sys.argv[1])