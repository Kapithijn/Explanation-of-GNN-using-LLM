#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# ADD YOUR FILES HERE
# ============================================================
FILES = [
    "outputs/First RUN SUM.json",
    "outputs/Run4 Sum.json",
    "outputs/RUN 3 SUM.json",
    "outputs/results_summary.json",
    "outputs/Second RUN SUM.json",
]

OUTPUT_FILE = "combined_results.csv"


def parse_result_file(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for key, value in data.items():
        # Expected format:
        # "MODEL|DATASET|LLM"
        parts = key.split("|")

        if len(parts) != 3:
            print(f"Skipping invalid key in {path.name}: {key}")
            continue

        model, dataset, llm = parts

        rows.append({
            "Model": model,
            "Dataset": dataset,
            "LLM": llm,
            "Accuracy": value.get("accuracy"),
            "Count": value.get("count"),
            "Source File": path.name,
        })

    return rows


# ============================================================
# LOAD ALL FILES
# ============================================================
all_rows = []

for file in FILES:
    all_rows.extend(parse_result_file(file))


# ============================================================
# NUMBER DUPLICATE RUNS
# Same MODEL + LLM => Run 1, Run 2, ...
# ============================================================
run_counter = defaultdict(int)

for row in all_rows:
    key = (row["Model"], row["LLM"])

    run_counter[key] += 1
    run_number = run_counter[key]

    row["Run"] = f"Run {run_number}"


# ============================================================
# CREATE TABLE
# ============================================================
df = pd.DataFrame(all_rows)

# Reorder columns
df = df[
    [
        "Model",
        "Dataset",
        "LLM",
        "Run",
        "Accuracy",
        "Count",
        "Source File",
    ]
]

# Sort nicely
df = df.sort_values(
    by=["Model", "LLM", "Run"],
    kind="stable"
).reset_index(drop=True)


# ============================================================
# SAVE + PRINT
# ============================================================
df.to_csv(OUTPUT_FILE, index=False)

print("\n=== RESULTS TABLE ===\n")
print(df.to_string(index=False))

print(f"\nSaved CSV to: {OUTPUT_FILE}")