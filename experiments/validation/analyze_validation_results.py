#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd


def main():
    output_dir = Path("results/validation_experiment")
    log_file = Path("validation_experiment.log")

    print("Validation experiment result summary")
    print("=" * 80)

    if log_file.exists():
        print(f"Log file: {log_file}")
        print(f"Log size: {log_file.stat().st_size / 1024:.1f} KB")
    else:
        print(f"Log file not found: {log_file}")

    expected_files = [
        "known_results.json",
        "predicted_results.json",
        "random_results.json",
        "comparison_boxplots.png",
        "validation_summary.csv",
    ]

    print("\nGenerated files")
    print("-" * 80)
    for name in expected_files:
        path = output_dir / name
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"OK  {name:35s} {size_kb:10.1f} KB")
        else:
            print(f"MISS {name}")

    summary_path = output_dir / "validation_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print("\nSummary table")
        print("-" * 80)
        print(df.to_string(index=False))

    print("\nOutput directory")
    print(os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
