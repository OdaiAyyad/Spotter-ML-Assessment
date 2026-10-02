"""Shared constants, feature engineering, and model factory."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RPM_LOW, RPM_HIGH = 1.2, 4.0  # corrupted label thresholds


def make_coord_lookup(train_df):
    """Return dict mapping city name to (lat, lon), built from both pickup and delivery columns."""
    coords = {}
    for _, row in train_df.iterrows():
        for city, lat, lon in [(row["pickup"], row["pickup_lat"], row["pickup_lon"]),
                               (row["delivery"], row["delivery_lat"], row["delivery_lon"])]:
            if city not in coords:
                coords[city] = (lat, lon)
            else:
                assert abs(coords[city][0] - lat) < 1e-5 and abs(coords[city][1] - lon) < 1e-5, \
                    f"Conflicting coords for {city}"
    return coords


def build_features(df, eq_cols, coord_lookup=None):
    """Return feature-ready DataFrame with equipment dummies, dow, and coordinates.

    Train and validation rows already carry lat/lon in the file; those are used as-is.
    December rows have no lat/lon columns, so coord_lookup (built from training) is used.
    """
    df = df.copy()
    df["weight"] = df["weight"].astype(float)
    df["dow"] = df["date"].dt.dayofweek
    for e in eq_cols:
        df[e] = (df["equipment"] == e).astype(int)
    if "pickup_lat" not in df.columns and coord_lookup is not None:
        df["pickup_lat"] = df["pickup"].map(lambda c: coord_lookup[c][0])
        df["pickup_lon"] = df["pickup"].map(lambda c: coord_lookup[c][1])
        df["delivery_lat"] = df["delivery"].map(lambda c: coord_lookup[c][0])
        df["delivery_lon"] = df["delivery"].map(lambda c: coord_lookup[c][1])
    return df


def make_model(seed=0):
    return HistGradientBoostingRegressor(random_state=seed)


def feature_cols(eq_cols):
    base = ["distance", "weight"] + eq_cols
    coords = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    return {
        "base": base,
        "+dow": base + ["dow"],
        "+dow+geo": base + ["dow"] + coords,
    }
