import numpy as np
import pandas as pd

from invariance.data.sensors import load_sensor_data, map_sensors_to_grid
from invariance.analysis.residuals import sample_simulation_at_sensors


def test_sensor_mapping_and_sampling():
    df = pd.DataFrame(
        {
            "t": [0.0],
            "x": [1.0],
            "y": [1.0],
            "temperature": [10.0],
        }
    )

    df = map_sensors_to_grid(df, dx=1.0, dy=1.0, nx=5, ny=5)

    T = np.zeros((2, 5, 5))
    sampled = sample_simulation_at_sensors(T, df, dt=1.0)

    assert "predicted_temperature" in sampled
    assert sampled.iloc[0]["predicted_temperature"] == 0.0