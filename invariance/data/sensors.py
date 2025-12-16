from __future__ import annotations

from pathlib import Path

import pandas as pd

import numpy as np


REQUIRED_COLUMNS = {"t", "x", "y", "temperature"}


def load_sensor_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sensor file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")

    # Basic sanity checks
    if (df["t"] < 0).any():
        raise ValueError("Sensor times must be non-negative")

    return df.sort_values("t").reset_index(drop=True)

def map_sensors_to_grid(df: pd.DataFrame, dx: float, dy: float, nx: int, ny: int):
    ii = np.round(df["x"] / dx).astype(int)
    jj = np.round(df["y"] / dy).astype(int)

    ii = ii.clip(0, nx - 1)
    jj = jj.clip(0, ny - 1)

    df = df.copy()
    df["i"] = ii
    df["j"] = jj

    return df