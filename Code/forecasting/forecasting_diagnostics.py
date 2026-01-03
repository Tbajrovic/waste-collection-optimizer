# Code/forecasting/forecasting_diagnostics.py
# Creates simple diagnostic plots for the improved forecast model.
# Input: Results/forecast_improved_predictions.csv
# Output: Results/plot_actual_vs_pred.png, Results/plot_residuals_hist.png, Results/plot_mae_per_bin_top20.png

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic plots for improved forecasting.")
    parser.add_argument("--preds", type=str, default="Results/forecast_improved_predictions.csv")
    parser.add_argument("--out_dir", type=str, default="Results")
    args = parser.parse_args()

    preds_path = Path(args.preds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {preds_path}. "
            f"Run Code/forecasting/forecasting_improved.py first to generate it."
        )

    df = pd.read_csv(preds_path)
    df["date"] = pd.to_datetime(df["date"])

    # 1) Actual vs Predicted for ONE bin
    # Choose bin with most test points (stable plot)
    bin_counts = df["bin_id"].value_counts()
    best_bin = bin_counts.index[0]
    one = df[df["bin_id"] == best_bin].sort_values("date")

    plt.figure()
    plt.plot(one["date"], one["y_true"], label="Actual")
    plt.plot(one["date"], one["y_pred"], label="Predicted")
    plt.title(f"Actual vs Predicted (bin_id={best_bin})")
    plt.xlabel("Date")
    plt.ylabel("Fill (same scale as target)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_actual_vs_pred.png", dpi=200)
    plt.close()

    # 2) Residual histogram
    residuals = (df["y_true"] - df["y_pred"]).to_numpy(dtype=float)

    plt.figure()
    plt.hist(residuals, bins=40)
    plt.title("Residuals Histogram (y_true - y_pred)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_residuals_hist.png", dpi=200)
    plt.close()

    # 3) MAE per bin (top 20 worst)
    mae_per_bin = (
        df.groupby("bin_id")
          .apply(lambda g: np.mean(np.abs(g["y_true"].to_numpy() - g["y_pred"].to_numpy())))
          .rename("mae")
          .sort_values(ascending=False)
          .head(20)
          .reset_index()
    )

    plt.figure(figsize=(10, 5))
    plt.bar(mae_per_bin["bin_id"].astype(str), mae_per_bin["mae"])
    plt.title("MAE per Bin (Top 20 worst)")
    plt.xlabel("bin_id")
    plt.ylabel("MAE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_mae_per_bin_top20.png", dpi=200)
    plt.close()

    print("[OK] Saved plots to:")
    print(" - Results/plot_actual_vs_pred.png")
    print(" - Results/plot_residuals_hist.png")
    print(" - Results/plot_mae_per_bin_top20.png")


if __name__ == "__main__":
    main()
