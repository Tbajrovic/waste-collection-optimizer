"""
Forecasting baseline for bin fill levels (OLS + LASSO)

- Učitava Dataset/raw/fill_timeseries.csv
- Priprema feature-e (lagovi + dan u sedmici + bin_id)
- Treniraju se LinearRegression (OLS) i LassoCV
- Računaju se MAE i MAPE na test skupu
- Rezultati se snimaju u Results/forecast_baseline.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------- Helpers ----------


def infer_columns(df: pd.DataFrame):
    """Pokušaj automatski pronaći kolone za vrijeme, bin_id i fill level."""
    cols = [c.lower() for c in df.columns]

    # vrijeme
    time_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if not time_candidates:
        raise ValueError("Nisam našao kolonu za vrijeme (očekujem nešto kao 'date' ili 'timestamp').")
    time_col = time_candidates[0]

    # bin_id
    bin_candidates = [c for c in df.columns if "bin" in c.lower() or "container" in c.lower()]
    if not bin_candidates:
        raise ValueError("Nisam našao kolonu za bin (očekujem nešto kao 'bin_id').")
    bin_col = bin_candidates[0]

    # fill / target
    fill_candidates = [c for c in df.columns if "fill" in c.lower() or "level" in c.lower()]
    if not fill_candidates:
        raise ValueError("Nisam našao kolonu za popunjenost (očekujem nešto sa 'fill').")
    target_col = fill_candidates[0]

    return time_col, bin_col, target_col


def add_lags(df: pd.DataFrame, group_col: str, target_col: str, n_lags: int = 7):
    """Dodaj lag feature-e po binu."""
    df = df.sort_values([group_col, "time"])
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error u %."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    eps = 1e-6
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0


# ---------- Main pipeline ----------


def main():
    repo_root = Path(".")  # pokreći skriptu iz root-a repozitorija
    data_path = repo_root / "Dataset" / "raw" / "fill_timeseries.csv"
    results_path = repo_root / "Results" / "forecast_baseline.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Učitavam podatke iz: {data_path}")
    df = pd.read_csv(data_path)

    # pronađi kolone
    time_col, bin_col, target_col = infer_columns(df)
    print(f"[INFO] Kolone prepoznate kao -> time: {time_col}, bin: {bin_col}, target: {target_col}")

    
    df["time"] = pd.to_datetime(df[time_col])
    df["dow"] = df["time"].dt.dayofweek  # day of week 0-6

    # dodaj lagove
    df = add_lags(df, group_col=bin_col, target_col=target_col, n_lags=7)

    # makni redove sa NaN (početni dani zbog lagova)
    df = df.dropna().reset_index(drop=True)

    # train / test split po vremenu (80/20)
    unique_dates = np.sort(df["time"].dt.date.unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_mask = df["time"].dt.date <= split_date
    test_mask = df["time"].dt.date > split_date

    train = df[train_mask].copy()
    test = df[test_mask].copy()

    print(f"[INFO] Train period do {split_date}, "
          f"train size = {len(train)}, test size = {len(test)}")

    
    lag_cols = [f"lag_{i}" for i in range(1, 8)]
    numeric_features = lag_cols
    categorical_features = [bin_col, "dow"]

    X_train = train[numeric_features + categorical_features]
    y_train = train[target_col]
    X_test = test[numeric_features + categorical_features]
    y_test = test[target_col]

    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ------- Model 1: OLS (Linear Regression) -------
    ols_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    print("[INFO] Trening OLS (LinearRegression)...")
    ols_model.fit(X_train, y_train)
    y_pred_ols = ols_model.predict(X_test)
    mae_ols = mean_absolute_error(y_test, y_pred_ols)
    mape_ols = mape(y_test, y_pred_ols)

    # ------- Model 2: LASSO -------
    lasso_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LassoCV(cv=5, random_state=42, n_jobs=-1)),
        ]
    )

    print("[INFO] Trening LASSO (LassoCV)...")
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    mape_lasso = mape(y_test, y_pred_lasso)

    # ------- Rezultati -------
    results = pd.DataFrame(
        [
            {"model": "OLS (LinearRegression)", "MAE": mae_ols, "MAPE_pct": mape_ols},
            {"model": "LASSO (LassoCV)", "MAE": mae_lasso, "MAPE_pct": mape_lasso},
        ]
    )

    results.to_csv(results_path, index=False)
    print("\n[RESULTS]")
    print(results)
    print(f"\n[INFO] Rezultati sačuvani u: {results_path}")


if __name__ == "__main__":
    main()
