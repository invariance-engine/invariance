from __future__ import annotations

import numpy as np
import pandas as pd

from invariance.physics.heat2d import simulate_heat_2d
from invariance.analysis.residuals import sample_simulation_at_sensors

from scipy.optimize import least_squares

def alpha_residuals(
    alpha: float,
    sim_config,
    sensors_df: pd.DataFrame,
) -> np.ndarray:
    """
    Compute residual vector for a given alpha.
    """
    T = simulate_heat_2d(
        nx=sim_config.grid.nx,
        ny=sim_config.grid.ny,
        dx=sim_config.grid.dx,
        dy=sim_config.grid.dy,
        dt=sim_config.time.dt,
        n_steps=sim_config.time.n_steps,
        alpha=alpha,
        initial_temperature=sim_config.initial_temperature,
        boundary_value=sim_config.boundary.value,
    )

    df = sample_simulation_at_sensors(T, sensors_df, sim_config.time.dt)
    return df["residual"].to_numpy()

def calibrate_alpha(
    initial_alpha: float,
    sim_config,
    sensors_df: pd.DataFrame,
    bounds: tuple[float, float] = (1e-6, 1e2),
):
    """
    Fit alpha by minimizing sensor residuals.
    """
    result = least_squares(
        lambda a: alpha_residuals(a[0], sim_config, sensors_df),
        x0=[initial_alpha],
        bounds=bounds,
    )

    return {
        "alpha_initial": initial_alpha,
        "alpha_fitted": float(result.x[0]),
        "success": result.success,
        "n_evals": result.nfev,
        "final_cost": float(result.cost),
    }