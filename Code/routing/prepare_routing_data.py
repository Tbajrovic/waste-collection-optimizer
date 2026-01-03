from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _stable_unit_xy(key: str) -> tuple[float, float]:
    """
    Deterministic pseudo-random point in [0,1]x[0,1] from a string key.
    This avoids non-reproducible randomness and works even without coordinates in the dataset.
    """
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Take two 16-hex chunks => 64-bit-ish ints => map to [0,1]
    a = int(h[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    b = int(h[16:32], 16) / float(0xFFFFFFFFFFFFFFFF)
    return float(a), float(b)


def _latlon_to_xy_meters(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Equirectangular approximation:
    Converts lat/lon degrees to local x/y in meters.
    Good enough for city-scale routing baseline.
    """
    R = 6371000.0  # meters
    lat_rad = np.deg2rad(lat.astype(float))
    lon_rad = np.deg2rad(lon.astype(float))
    lat0 = float(np.mean(lat_rad))
    x = R * (lon_rad - float(np.mean(lon_rad))) * np.cos(lat0)
    y = R * (lat_rad - float(np.mean(lat_rad)))
    return x, y


def _build_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Euclidean distance matrix (meters).
    """
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Town of Cary routing inputs (centroids + distance matrix).")

    parser.add_argument(
        "--data",
        type=str,
        default="Dataset/raw/town_of_cary/solid-waste-and-recycling-collection-routes1.csv",
        help="Town of Cary routes/zones CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Results",
        help="Output directory for routing-ready files",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=120,
        help="Max number of collection points (zones) to keep (excluding depot).",
    )
    parser.add_argument(
        "--decisions",
        type=str,
        default="",
        help="Optional: Results/collect_rf_decisions.csv to filter to bins/zones to collect.",
    )
    parser.add_argument(
        "--threshold_collect_only",
        action="store_true",
        help="If set and --decisions is provided, keep only rows where collect_pred==1.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Town of Cary CSV not found: {data_path.as_posix()}")

    df = pd.read_csv(data_path)

    # Pick an ID column for zones/routes
    id_col = _find_first_col(df, ["objectid", "route_id", "route", "zone_id", "zone", "id"])
    if id_col is None:
        # Fallback: create an ID from row index
        df["zone_id"] = np.arange(len(df), dtype=int)
        id_col = "zone_id"

    df[id_col] = df[id_col].astype(str)

    # Optional filtering using classification decisions
    # Note: decisions are on "bin_id" in your synthetic world.
    # For routing with Town of Cary zones, we treat zone_id as the routing node id.
    # If you don't have a clean mapping yet, we can still:
    # - sample a subset of zones (baseline) OR
    # - filter by day/cycle, etc. (later refinement)
    if args.decisions:
        dec_path = Path(args.decisions)
        if not dec_path.exists():
            raise FileNotFoundError(f"Decisions CSV not found: {dec_path.as_posix()}")

        dec = pd.read_csv(dec_path)
        # expected columns from our earlier script:
        # date, bin_id, y_true, y_pred, collect_true, collect_pred
        if args.threshold_collect_only and "collect_pred" in dec.columns:
            dec = dec[dec["collect_pred"] == 1].copy()

        # If user has no mapping between bin_id and zone_id yet:
        # we just take N zones deterministically based on bin_id list.
        if "bin_id" in dec.columns and len(dec) > 0:
            chosen_keys = dec["bin_id"].astype(str).unique().tolist()
            # Map bin_id keys onto zone rows deterministically (wrap-around)
            df = df.sort_values(id_col).reset_index(drop=True)
            idx = [i % len(df) for i in range(min(len(chosen_keys), args.max_points))]
            df = df.iloc[idx].copy()
        else:
            df = df.sample(n=min(args.max_points, len(df)), random_state=42).copy()
    else:
        # Baseline: just take a deterministic subset of zones
        df = df.sort_values(id_col).head(min(args.max_points, len(df))).copy()

    # Try to find coordinate columns
    lon_col = _find_first_col(df, ["lon", "longitude", "x", "long"])
    lat_col = _find_first_col(df, ["lat", "latitude", "y"])

    has_latlon = False
    if lat_col is not None and lon_col is not None:
        # Validate numeric-ish
        try:
            lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
            lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
            has_latlon = np.isfinite(lat).all() and np.isfinite(lon).all()
        except Exception:
            has_latlon = False

    if has_latlon:
        x, y = _latlon_to_xy_meters(lat, lon)
        coord_source = f"latlon({lat_col},{lon_col})"
    else:
        # Deterministic synthetic centroids (still valid for MLST baseline)
        # Spread them in a 10km x 10km square
        xs, ys = [], []
        for z in df[id_col].tolist():
            ux, uy = _stable_unit_xy(str(z))
            xs.append(ux * 10000.0)
            ys.append(uy * 10000.0)
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        coord_source = "deterministic_synthetic_centroids"

    # Define depot as centroid of all points (simple, defensible baseline)
    depot_x = float(np.mean(x))
    depot_y = float(np.mean(y))

    # Build nodes: depot + points
    node_ids = ["depot"] + df[id_col].astype(str).tolist()
    x_all = np.concatenate([[depot_x], x])
    y_all = np.concatenate([[depot_y], y])

    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "x_m": x_all,
            "y_m": y_all,
            "is_depot": [1] + [0] * len(df),
            "coord_source": [coord_source] * (1 + len(df)),
        }
    )

    dist = _build_distance_matrix(x_all, y_all)

    nodes_path = out_dir / "town_of_cary_nodes.csv"
    dist_path = out_dir / "town_of_cary_distance_matrix.csv"

    nodes.to_csv(nodes_path, index=False)
    pd.DataFrame(dist, index=node_ids, columns=node_ids).to_csv(dist_path, index=True)

    print("[OK] Routing inputs prepared.")
    print(f"[INFO] Nodes saved to: {nodes_path.as_posix()}")
    print(f"[INFO] Distance matrix saved to: {dist_path.as_posix()}")
    print(f"[INFO] Points used: {len(df)} (+ depot)")
    print(f"[INFO] Coordinate source: {coord_source}")


if __name__ == "__main__":
    main()
