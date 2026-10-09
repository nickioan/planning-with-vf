"""Summarize result CSVs written by evaluate.py / evaluate_composition.py as mean +- std tables.

Example:
    python scripts/summarize.py results/*.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", nargs="+")
    args = parser.parse_args()

    success = defaultdict(list)
    timing = {}
    for path in args.csv:
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["task"], row["planner"], row["value"], int(row["n_plans"]))
                if row["metric"] == "time_s":
                    timing[key] = float(row["result"])
                else:
                    success[key].append(float(row["result"]))

    if success:
        print(f"{'task':<10}{'planner':<42}{'value':>8}{'n_plans':>9}   success rate")
        for key in sorted(success):
            r = np.array(success[key])
            stats = f"{r.mean():.2f} +- {r.std():.2f}  ({len(r)} trials)"
            print(f"{key[0]:<10}{key[1]:<42}{key[2]:>8}{key[3]:>9}   {stats}")
    if timing:
        print(f"\n{'task':<10}{'planner':<42}{'n_plans':>9}   seconds per planning step")
        for key in sorted(timing):
            print(f"{key[0]:<10}{key[1]:<42}{key[3]:>9}   {timing[key]:.3f}")


if __name__ == "__main__":
    main()
