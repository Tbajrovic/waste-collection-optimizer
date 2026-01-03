from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest classification for collection decision.")
    parser.add_argument("--preds", type=str, default="Results/forecast_rf_predictions.csv")
    parser.add_argument("--out", type=str, default="Results/collect_rf.csv")
    parser.add_argument("--threshold", type=float, default=0.8, help="Fill threshold for collection decision (same scale as y).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    preds_path = Path(args.preds)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {preds_path}. "
            f"Run forecasting_random_forest.py first."
        )

    df = pd.read_csv(preds_path)
    df["date"] = pd.to_datetime(df["date"])

    # Classification labels from true y (for evaluation)
    y_true_cls = (df["y_true"].to_numpy(dtype=float) >= args.threshold).astype(int)

    # Features for classifier: predicted value + simple context from prediction error space
    # Keep it simple and defensible:
    X = pd.DataFrame(
        {
            "y_pred": df["y_pred"].to_numpy(dtype=float),
            "residual": df["residual"].to_numpy(dtype=float),
        }
    )

    # Target: whether bin should be collected tomorrow
    y = y_true_cls

    # Simple chronological split by date to avoid leakage
    dates = df["date"].sort_values().unique()
    if len(dates) < 5:
        raise ValueError("Not enough unique dates for classification split.")
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = df["date"] < split_date
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_hat))
    prec = float(precision_score(y_test, y_hat, zero_division=0))
    rec = float(recall_score(y_test, y_hat, zero_division=0))
    cm = confusion_matrix(y_test, y_hat)

    results = pd.DataFrame(
        [
            {
                "model": "RandomForestClassifier",
                "threshold": args.threshold,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "n_test": int(test_mask.sum()),
            }
        ]
    )
    results.to_csv(out_path, index=False)

    # Save decisions for routing (bin_id list on each test date)
    decisions_path = out_path.with_name("collect_rf_decisions.csv")
    out_df = df.loc[test_mask, ["date", "bin_id", "y_true", "y_pred"]].copy()
    out_df["collect_true"] = y_test
    out_df["collect_pred"] = y_hat
    out_df.to_csv(decisions_path, index=False)

    print("[RESULTS]")
    print(results.to_string(index=False))
    print(f"\n[INFO] Classification results saved to: {out_path.as_posix()}")
    print(f"[INFO] Decisions saved to: {decisions_path.as_posix()}")


if __name__ == "__main__":
    main()
