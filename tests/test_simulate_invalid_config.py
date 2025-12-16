import json
from pathlib import Path

from typer.testing import CliRunner

from invariance.cli import app

runner = CliRunner()


def test_invalid_config_does_not_create_run_dir(tmp_path: Path):
    bad_config = {
        "grid": {"nx": -1, "ny": 10, "dx": 1.0, "dy": 1.0},
        "time": {"dt": 0.1, "n_steps": 10},
        "material": {"alpha": 1.0},
        "boundary": {"type": "dirichlet", "value": 0.0},
        "initial_temperature": 0.0,
    }

    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps(bad_config))

    out_dir = tmp_path / "run"

    result = runner.invoke(
        app,
        [
            "simulate",
            "--config",
            str(cfg_path),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code != 0
    assert "Validation failed" in result.stderr
    assert not out_dir.exists()