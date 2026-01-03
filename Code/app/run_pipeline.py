from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"[ERROR] Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------
    # 1) Classification (collect / skip)
    # ---------
    # Update this path to your actual classification script filename if different.
    # (Based on your Results folder: collect_rf.csv / collect_rf_decisions.csv)
    classification_script = repo_root / "Code" / "forecasting" / "forecasting_classification_rf.py"

    if classification_script.exists():
        run([sys.executable, str(classification_script)])
    else:
        print(
            "[WARN] Classification script not found at:\n"
            f"       {classification_script.as_posix()}\n"
            "[WARN] Skipping classification step. Routing will run on the prepared subset of points."
        )

    # ---------
    # 2) Prepare routing inputs (Town of Cary)
    # ---------
    prepare_script = repo_root / "Code" / "routing" / "prepare_routing_data.py"
    run([sys.executable, str(prepare_script)])

    # ---------
    # 3) Solve CVRP
    # ---------
    solve_script = repo_root / "Code" / "routing" / "solve_cvrp.py"
    run([sys.executable, str(solve_script)])

    print("\n[OK] Pipeline finished.")
    print(f"[INFO] Check routing output: {(results_dir / 'routes_optimized.csv').as_posix()}")


if __name__ == "__main__":
    main()
