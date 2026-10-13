"""Run the whole project end to end.

    python scripts/run_pipeline.py                      # finds the Kaggle file in data/raw/
    python scripts/run_pipeline.py --data path/to.csv   # explicit file
    python scripts/run_pipeline.py --quick              # fewer resamples/scenarios (smoke test)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdmodel import config as C  # noqa: E402
from pdmodel.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to accepted_2007_to_2018Q4.csv(.gz)")
    ap.add_argument("--reports", default=None, help="Output folder (default: reports/)")
    ap.add_argument("--quick", action="store_true",
                    help="Fewer bootstrap draws and Monte Carlo scenarios, for smoke tests only")
    a = ap.parse_args()
    run(raw_path=a.data, reports_dir=Path(a.reports) if a.reports else C.REPORTS_DIR, quick=a.quick)
