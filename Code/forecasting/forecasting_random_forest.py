from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def load_timeseries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # date column
    date_col = None
    for c in ["date", "datetime", "timestamp", "time"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found. Columns: {list(df.columns)}")

    # target column (supports your dataset)
    target_col = None
    for c in ["fill_level", "fill_pct", "fill", "fill_percent", "fill_ratio"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise ValueError(f"No fill column found. Columns: {list(df.columns)}")

    if "bin_id" not in df.columns:
        raise ValueError("Missing required column 'bin_id'.")

    df = df.rename(columns={date_col: "date", target_col: "fill_level"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["bin_id", "date"]).reset_index(drop=True)

    # IMPORTANT: keep scale consistent (0–1). Do NOT multiply by 100.
    # If you later switch to percent everywhere, do it for ALL models consistently.
    return df


def make_features(df: pd.DataFrame, lags: list[int], rolling_windows: list[int]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()

    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day

    for lag in lags:
        out[f"lag_{lag}"] = out.groupby("bin_id")["fill_level"].shift(lag)

    for w in rolling_windows:
        out[f"roll_mean_{w}"] = (
            out.groupby("bin_id")["fill_level"]
               .apply(lambda s: s.shift(1).rolling(window=w, min_periods=w).mean())
               .reset_index(level=0, drop=True)
        )
        out[f"roll_std_{w}"] = (
            out.groupby("bin_id")["fill_level"]
               .apply(lambda s: s.shift(1).rolling(window=w, min_periods=w).std())
               .reset_index(level=0, drop=True)
        )

    out["y_next"] = out.groupby("bin_id")["fill_level"].shift(-1)

    feature_cols = (
        ["dow", "month", "day"]
        + [f"lag_{l}" for l in lags]
        + [f"roll_mean_{w}" for w in rolling_windows]
        + [f"roll_std_{w}" for w in rolling_windows]
    )

    out = out.dropna(subset=feature_cols + ["y_next"]).reset_index(drop=True)
    return out, feature_cols


def chronological_split(df_feat: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = df_feat["date"].sort_values().unique()
    if len(dates) < 5:
        raise ValueError("Not enough unique dates for chronological split.")
    split_idx = int(len(dates) * train_ratio)
    split_idx = max(1, min(split_idx, len(dates) - 1))
    split_date = dates[split_idx]
    train_df = df_feat[df_feat["date"] < split_date].copy()
    test_df = df_feat[df_feat["date"] >= split_date].copy()
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest regression for next-day bin fill.")
    parser.add_argument("--data", type=str, default="Dataset/raw/fill_timeseries.csv")
    parser.add_argument("--out", type=str, default="Results/forecast_rf.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--lags", type=str, default="1,2,3,7")
    parser.add_argument("--rolling", type=str, default="3,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=18)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]
    rolling_windows = [int(x.strip()) for x in args.rolling.split(",") if x.strip()]

    df = load_timeseries(data_path)
    df_feat, feature_cols = make_features(df, lags=lags, rolling_windows=rolling_windows)
    train_df, test_df = chronological_split(df_feat, train_ratio=args.train_ratio)

    X_train = train_df[feature_cols]
    y_train = train_df["y_next"].to_numpy()

    X_test = test_df[feature_cols]
    y_test = test_df["y_next"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mape = float(safe_mape(y_test, y_pred))

    results = pd.DataFrame(
        [
            {
                "model": "RandomForestRegressor",
                "MAE": mae,
                "MAPE_pct": mape,
                "train_ratio": args.train_ratio,
                "lags": args.lags,
                "rolling_windows": args.rolling,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        ]
    )
    results.to_csv(out_path, index=False)

    # Save per-row predictions for diagnostics + later thresholding
    pred_path = out_path.with_name("forecast_rf_predictions.csv")
    preds = test_df[["date", "bin_id"]].copy()
    preds["y_true"] = y_test
    preds["y_pred"] = y_pred
    preds["residual"] = preds["y_true"] - preds["y_pred"]
    preds.to_csv(pred_path, index=False)

    print("[RESULTS]")
    print(results[["model", "MAE", "MAPE_pct"]].to_string(index=False))
    print(f"\n[INFO] Results saved to: {out_path.as_posix()}")
    print(f"[INFO] Predictions saved to: {pred_path.as_posix()}")


if __name__ == "__main__":
    main()
