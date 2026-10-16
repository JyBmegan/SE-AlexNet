#!/usr/bin/env python3
"""Validate the 21-point model-prediction archive and write a manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


FC_PATTERN = re.compile(
    r"^Location-[123]-(FaceBased|ObjectBased)-(Full|E|M|N)-(2|4|8|16|32)\.csv$"
)
BASELINE_PATTERN = re.compile(
    r"^(AlexNet|VGG-16)-(FaceBased|ObjectBased)-(Full|E|M|N)-raw\.csv$"
)


def read_probability_column(path: Path) -> list[float]:
    values = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                raise ValueError(f"{path.name}: expected at least two columns")
            try:
                values.append(float(row[1]))
            except ValueError:
                continue
    if len(values) != 21:
        raise ValueError(f"{path.name}: expected 21 probabilities, found {len(values)}")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"{path.name}: probabilities must be between 0 and 1")
    return values


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root / "2_Results" / "1_Predict21Datapoints",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "2_Results" / "Prediction_Archive_Manifest.csv",
    )
    args = parser.parse_args()

    files = sorted(args.input.glob("*.csv"))
    if len(files) != 136:
        raise ValueError(f"Expected 136 CSV files, found {len(files)}")

    fc_files = [path for path in files if FC_PATTERN.match(path.name)]
    baseline_files = [path for path in files if BASELINE_PATTERN.match(path.name)]
    if len(fc_files) != 120 or len(baseline_files) != 16:
        raise ValueError(
            f"Expected 120 SE-AlexNet and 16 baseline files; found "
            f"{len(fc_files)} and {len(baseline_files)}"
        )

    rows = []
    for path in files:
        values = read_probability_column(path)
        rows.append(
            {
                "FileName": path.name,
                "Rows": len(values),
                "MinimumProbability": min(values),
                "MaximumProbability": max(values),
                "SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Validated 136 files: 120 SE-AlexNet conditions and 16 baselines.")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
