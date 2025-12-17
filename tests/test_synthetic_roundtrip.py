import json
from pathlib import Path

import numpy as np

from invariance.synthetic.generate import generate_synthetic_case
from invariance.config.sim import SimulationConfig


def test_synthetic_generation(tmp_path: Path):
    sim_config = SimulationConfig(
        grid={"nx": 20, "ny": 20, "dx": 1.0, "dy": 1.0},
        time={"dt": 0.01, "n_steps": 50},
        material={"alpha": 1.0},
        boundary={"type": "dirichlet", "value": 0.0},
        initial_temperature=100.0,
    )

    out = tmp_path / "case"

    generate_synthetic_case(
        sim_config=sim_config,
        true_alpha=0.5,
        n_sensors=10,
        noise_std=0.0,
        out_dir=out,
    )

    with (out / "truth.json").open() as f:
        truth = json.load(f)

    assert truth["alpha_true"] == 0.5
    assert (out / "sensors.csv").exists()