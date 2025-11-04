# Dataset/synthetic/generate_synthetic.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

def sample_points_around_center(n, center_lat=43.8563, center_lon=18.4131, radius_km=6.0):
    # random polar coords → approx lat/lon jitter
    R = 6371.0
    r = radius_km * np.sqrt(rng.random(n))  # uniform over disk
    theta = 2 * np.pi * rng.random(n)
    dlat = (r / R) * (180 / np.pi) * np.cos(theta)
    dlon = (r / R) * (180 / np.pi) * np.sin(theta) / np.cos(np.deg2rad(center_lat))
    return center_lat + dlat, center_lon + dlon

def make_bins(n_bins: int):
    lat, lon = sample_points_around_center(n_bins)
    districts = rng.choice(["Centar", "StariGrad", "NovoSarajevo", "NoviGrad", "Ilidža"], size=n_bins, p=[.2,.2,.25,.25,.1])
    types = rng.choice(["residential","commercial","mixed"], size=n_bins, p=[.6,.25,.15])
    base_rate = rng.normal(0.60, 0.12, size=n_bins).clip(0.25, 0.9)  # average daily % fill
    capacity = rng.choice([120, 240, 660, 1100], size=n_bins, p=[.25,.35,.3,.1])  # liters
    return pd.DataFrame({
        "bin_id": np.arange(n_bins, dtype=int),
        "lat": lat.round(6),
        "lon": lon.round(6),
        "district": districts,
        "bin_type": types,
        "daily_rate": base_rate,
        "capacity_l": capacity
    })

def make_trucks(n_trucks: int):
    return pd.DataFrame({
        "truck_id": np.arange(n_trucks, dtype=int),
        "capacity_l": rng.choice([6000, 8000, 10000], size=n_trucks, p=[.4,.4,.2]),
        "shift_minutes": rng.integers(6*60, 8*60, size=n_trucks)  # 6–8h
    })

def seasonal_multiplier(day):
    # weekly pattern (Mon–Sun) + mild seasonality
    dow = day.weekday()
    weekly = {0:1.05, 1:1.00, 2:1.00, 3:1.02, 4:1.08, 5:1.20, 6:1.15}[dow]
    season = 1.0 + 0.15*np.sin(2*np.pi*(day.timetuple().tm_yday)/365.0)
    return weekly * season

def events_boost(ts):
    # a few city events cause extra waste for 1–3 days
    ts = pd.to_datetime(ts)
    boost = np.ones(len(ts))
    # pick ~4 random events
    for _ in range(4):
        start = rng.integers(5, len(ts)-5)
        dur = rng.integers(1, 4)
        boost[start:start+dur] *= rng.uniform(1.2, 1.5)
    return boost

def make_timeseries(bins_df: pd.DataFrame, start_date: str, days: int):
    idx = pd.date_range(start_date, periods=days, freq="D")
    n_bins = len(bins_df)

    # base daily rates per bin with small random walk
    rates = bins_df["daily_rate"].values.reshape(-1,1) * (1 + rng.normal(0, 0.03, size=(n_bins, days))).clip(0.85, 1.15)
    # apply global seasonal + event multipliers (same for all bins, realistic enough for start)
    seasonal = np.array([seasonal_multiplier(d) for d in idx])[None, :]
    event = events_boost(idx)[None, :]
    rates = rates * seasonal * event

    # simulate % fill with overflow reset upon collection (greedy “collect when >95%”)
    fill = np.zeros((n_bins, days))
    overflow = np.zeros((n_bins, days), dtype=bool)
    for t in range(days):
        prev = fill[:, t-1] if t > 0 else 0.0
        # noise & clamp
        inc = rates[:, t] + rng.normal(0, 0.03, size=n_bins)
        nxt = np.clip(prev + inc, 0, 1.5)  # can overflow above 1.0 until collected
        need_collect = nxt >= 0.95
        overflow[:, t] = nxt > 1.05
        # simulate “collection occurs” for ~70% of bins that need it each day (not perfect ops)
        collected_today = need_collect & (rng.random(n_bins) < 0.7)
        nxt[collected_today] = rng.uniform(0.05, 0.20, size=collected_today.sum())  # reset to near empty
        fill[:, t] = nxt

    # build long dataframe
    df = (pd.DataFrame(fill, index=bins_df.bin_id, columns=idx)
            .rename_axis(index="bin_id", columns="date")
            .stack()
            .rename("fill_ratio")
            .reset_index())
    df["overflow"] = overflow.reshape(-1)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--bins", type=int, default=500)
    ap.add_argument("--trucks", type=int, default=6)
    ap.add_argument("--start", type=str, default="2025-01-01")
    args = ap.parse_args()

    out_dir = Path("Dataset/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = make_bins(args.bins)
    trucks = make_trucks(args.trucks)
    ts = make_timeseries(bins, args.start, args.days)

    # Save CSVs
    bins.to_csv(out_dir / "bins.csv", index=False)
    trucks.to_csv(out_dir / "trucks.csv", index=False)
    ts.to_csv(out_dir / "fill_timeseries.csv", index=False)

    print(f"[OK] Wrote {len(bins)} bins, {len(trucks)} trucks, {len(ts)} rows to {out_dir}")

if __name__ == "__main__":
    main()
