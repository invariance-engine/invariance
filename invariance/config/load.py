from __future__ import annotations

import json
from pathlib import Path

from invariance.config.sim import SimulationConfig


def load_simulation_config(path: Path) -> SimulationConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    return SimulationConfig.model_validate(data)