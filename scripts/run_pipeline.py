"""Run the whole project end to end.

    python scripts/run_pipeline.py                      # finds the Kaggle file in data/raw/
    python scripts/run_pipeline.py --data path/to.csv   # explicit file
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdmodel.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to accepted_2007_to_2018Q4.csv(.gz)")
    a = ap.parse_args()
    run(raw_path=a.data)
