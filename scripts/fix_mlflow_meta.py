#!/usr/bin/env python3
"""Repair MLflow meta.yaml artifact_location entries under outputs/*/mlruns/*/meta.yaml

This script will update artifact_location to point to the correct outputs/<job>/mlruns/<expid>
"""
import re
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
outputs_dir = project_root / "outputs"
count = 0
for job_dir in outputs_dir.iterdir():
    if not job_dir.is_dir():
        continue
    mlruns_dir = job_dir / "mlruns"
    if not mlruns_dir.exists():
        continue
    for exp in mlruns_dir.iterdir():
        meta = exp / "meta.yaml"
        if not meta.exists():
            continue
        correct = f"file://{(mlruns_dir / exp.name).resolve()}"
        text = meta.read_text()
        new_text = re.sub(
            r"artifact_location: .*", f"artifact_location: {correct}", text
        )
        if new_text != text:
            meta.write_text(new_text)
            print(f"Updated {meta} -> {correct}")
            count += 1

print(f"Done. Updated {count} meta.yaml files.")
