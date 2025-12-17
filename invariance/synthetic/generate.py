from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from invariance.physics.heat2d import simulate_heat_2d

def generate_synthetic_case(
    sim_config,
    true_alpha: float,
    n_sensors: int,
    noise_std: float,
    out_dir: Path,
) -> None:
    """
    Generate a synthetic calibration case with known ground truth.
    """
    out_dir.mkdir(parents=True, exist_ok=False)

    # Override alpha with truth
    sim_config.material.alpha = true_alpha

    # Run simulation
    T = simulate_heat_2d(
        nx=sim_config.grid.nx,
        ny=sim_config.grid.ny,
        dx=sim_config.grid.dx,
        dy=sim_config.grid.dy,
        dt=sim_config.time.dt,
        n_steps=sim_config.time.n_steps,
        alpha=true_alpha,
        initial_temperature=sim_config.initial_temperature,
        boundary_value=sim_config.boundary.value,
    )

    rng = np.random.default_rng(42)

    # Sample random sensor locations (avoid boundaries)
    ii = rng.integers(1, sim_config.grid.nx - 1, size=n_sensors)
    jj = rng.integers(1, sim_config.grid.ny - 1, size=n_sensors)

    # Sample random times
    tt = rng.uniform(
        0,
        sim_config.time.dt * sim_config.time.n_steps,
        size=n_sensors,
    )

    steps = np.round(tt / sim_config.time.dt).astype(int)
    steps = steps.clip(0, T.shape[0] - 1)

    temperatures = T[steps, ii, jj]

    noisy_temperatures = temperatures + rng.normal(
        0.0, noise_std, size=n_sensors
    )

    sensors_df = pd.DataFrame(
        {
            "t": tt,
            "x": ii * sim_config.grid.dx,
            "y": jj * sim_config.grid.dy,
            "temperature": noisy_temperatures,
        }
    )

    sensors_df.to_csv(out_dir / "sensors.csv", index=False)

    # Save sim config
    with (out_dir / "sim.json").open("w") as f:
        json.dump(sim_config.model_dump(), f, indent=2)

    # Save truth
    with (out_dir / "truth.json").open("w") as f:
        json.dump(
            {
                "alpha_true": true_alpha,
                "noise_std": noise_std,
                "n_sensors": n_sensors,
                "rng_seed": 42,
            },
            f,
            indent=2,
        )

    # README
    (out_dir / "README.txt").write_text(
        "Synthetic calibration case.\n"
        "Hidden truth parameters are stored in truth.json.\n"
    )