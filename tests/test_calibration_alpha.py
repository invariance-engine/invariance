import numpy as np
import pandas as pd

from invariance.calibration.alpha import calibrate_alpha
from invariance.config.sim import SimulationConfig


def test_alpha_calibration_runs():
    sim_config = SimulationConfig(
        grid={"nx": 10, "ny": 10, "dx": 1.0, "dy": 1.0},
        time={"dt": 0.01, "n_steps": 20},
        material={"alpha": 1.0},
        boundary={"type": "dirichlet", "value": 0.0},
        initial_temperature=100.0,
    )

    sensors_df = pd.DataFrame(
        {
            "t": [0.1],
            "x": [5.0],
            "y": [5.0],
            "temperature": [50.0],
            "i": [5],
            "j": [5],
        }
    )

    result = calibrate_alpha(1.0, sim_config, sensors_df)

    assert "alpha_fitted" in result