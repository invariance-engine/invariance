from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich import print

from invariance import __version__
from invariance.config.load import load_simulation_config
from invariance.run.create import create_run_directory
from invariance.physics.heat2d import compute_stability_dt, simulate_heat_2d

import json
import numpy as np

from invariance.data.sensors import load_sensor_data, map_sensors_to_grid
from invariance.analysis.residuals import (
    sample_simulation_at_sensors,
    compute_error_metrics,
)

from invariance.calibration.alpha import calibrate_alpha
from invariance.data.sensors import load_sensor_data, map_sensors_to_grid

from invariance.synthetic.generate import generate_synthetic_case

app = typer.Typer(
    name="invariance",
    help="Invariance: auto-calibrated physics (starting with thermal diffusion).",
)


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """
    Invariance CLI entrypoint.
    """
    if ctx.invoked_subcommand is None:
        # If no subcommand is provided, show help.
        print(ctx.get_help())


@app.command()
def version() -> None:
    """
    Print the installed Invariance version.
    """
    print(f"[bold]invariance[/bold] v{__version__}")


@app.command()
def simulate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to simulation config JSON file",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory for run artifacts",
        ),
    ],
) -> None:
    """
    Run a 2D heat diffusion simulation and write results to disk.
    """
    # 1. Load + validate config
    try:
        sim_config = load_simulation_config(config)
    except ValidationError as e:
        typer.echo("Error: Validation failed for simulation config", err=True)
        typer.echo(e, err=True)
        raise typer.Exit(code=1) from e

    # 2. Enforce stability (NEW — must be before writing anything)
    dt_max = compute_stability_dt(
        alpha=sim_config.material.alpha,
        dx=sim_config.grid.dx,
        dy=sim_config.grid.dy,
    )

    if sim_config.time.dt > dt_max:
        typer.echo(
            (
                "Error: Time step is unstable for explicit heat solver\n"
                f"  dt      = {sim_config.time.dt:.3e}\n"
                f"  dt_max  = {dt_max:.3e}"
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    # 3. Create run directory
    try:
        create_run_directory(out, sim_config)
    except FileExistsError as e:
        typer.echo(f"Error: Output directory already exists: {out}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"✔ Loaded simulation config from {config}")
    typer.echo(f"✔ Created run directory: {out}")

    # 4. Run heat simulation
    T = simulate_heat_2d(
        nx=sim_config.grid.nx,
        ny=sim_config.grid.ny,
        dx=sim_config.grid.dx,
        dy=sim_config.grid.dy,
        dt=sim_config.time.dt,
        n_steps=sim_config.time.n_steps,
        alpha=sim_config.material.alpha,
        initial_temperature=sim_config.initial_temperature,
        boundary_value=sim_config.boundary.value,
    )

    # 5. Write outputs

    np.save(out / "field.npy", T[-1])
    np.save(out / "history.npy", T)

    metrics = {
        "t_final": sim_config.time.dt * sim_config.time.n_steps,
        "min_temperature": float(T[-1].min()),
        "max_temperature": float(T[-1].max()),
        "dt_max_stable": dt_max,
    }

    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # 6. Done
    typer.echo("✔ Heat simulation completed")
    
    
@app.command()
def validate(
    run: Annotated[
        Path,
        typer.Option(
            "--run",
            "-r",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Run directory produced by `simulate`",
        ),
    ],
    sensors: Annotated[
        Path,
        typer.Option(
            "--sensors",
            "-s",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Sensor CSV file",
        ),
    ],
) -> None:
    """
    Validate a simulation run against sensor data.
    """
    # Load artifacts
    T = np.load(run / "history.npy")

    with (run / "config.json").open() as f:
        config = json.load(f)

    dt = config["time"]["dt"]
    dx = config["grid"]["dx"]
    dy = config["grid"]["dy"]
    nx = config["grid"]["nx"]
    ny = config["grid"]["ny"]

    # Load + map sensors
    sensors_df = load_sensor_data(sensors)
    sensors_df = map_sensors_to_grid(sensors_df, dx, dy, nx, ny)

    # Sample simulation
    residual_df = sample_simulation_at_sensors(T, sensors_df, dt)

    # Metrics
    metrics = compute_error_metrics(residual_df)

    # Write results
    residual_df.to_csv(run / "sensor_residuals.csv", index=False)

    with (run / "metrics.json").open("r+") as f:
        run_metrics = json.load(f)
        run_metrics["sensor_validation"] = metrics
        f.seek(0)
        json.dump(run_metrics, f, indent=2)
        f.truncate()

    typer.echo("✔ Sensor validation completed")
    typer.echo(f"RMSE = {metrics['rmse']:.3f}")
    
    
@app.command()
def calibrate(
    run: Annotated[
        Path,
        typer.Option(
            "--run",
            "-r",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Run directory produced by `simulate`",
        ),
    ],
    sensors: Annotated[
        Path,
        typer.Option(
            "--sensors",
            "-s",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Sensor CSV file",
        ),
    ],
) -> None:
    """
    Automatically calibrate thermal diffusivity (alpha).
    """
    import json
    import numpy as np

    # Load run artifacts
    with (run / "config.json").open() as f:
        config_dict = json.load(f)

    sim_config = load_simulation_config(run / "config.json")

    sensors_df = load_sensor_data(sensors)
    sensors_df = map_sensors_to_grid(
        sensors_df,
        dx=sim_config.grid.dx,
        dy=sim_config.grid.dy,
        nx=sim_config.grid.nx,
        ny=sim_config.grid.ny,
    )

    # Initial validation
    T0 = np.load(run / "history.npy")
    initial_df = sample_simulation_at_sensors(
        T0, sensors_df, sim_config.time.dt
    )
    rmse_before = float(
        np.sqrt((initial_df["residual"] ** 2).mean())
    )

    # Calibrate
    result = calibrate_alpha(
        initial_alpha=sim_config.material.alpha,
        sim_config=sim_config,
        sensors_df=sensors_df,
    )

    # Update config
    sim_config.material.alpha = result["alpha_fitted"]

    # Re-simulate with fitted alpha
    T1 = simulate_heat_2d(
        nx=sim_config.grid.nx,
        ny=sim_config.grid.ny,
        dx=sim_config.grid.dx,
        dy=sim_config.grid.dy,
        dt=sim_config.time.dt,
        n_steps=sim_config.time.n_steps,
        alpha=sim_config.material.alpha,
        initial_temperature=sim_config.initial_temperature,
        boundary_value=sim_config.boundary.value,
    )

    fitted_df = sample_simulation_at_sensors(
        T1, sensors_df, sim_config.time.dt
    )
    rmse_after = float(
        np.sqrt((fitted_df["residual"] ** 2).mean())
    )

    # Write outputs
    np.save(run / "field_calibrated.npy", T1[-1])

    with (run / "metrics.json").open("r+") as f:
        metrics = json.load(f)
        metrics["calibration"] = {
            **result,
            "rmse_before": rmse_before,
            "rmse_after": rmse_after,
        }
        f.seek(0)
        json.dump(metrics, f, indent=2)
        f.truncate()

    typer.echo("✔ Calibration completed")
    typer.echo(f"alpha: {result['alpha_initial']:.4f} → {result['alpha_fitted']:.4f}")
    typer.echo(f"RMSE:  {rmse_before:.4f} → {rmse_after:.4f}")
    
    
@app.command()
def synth(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, help="Base simulation config"),
    ],
    alpha: Annotated[
        float,
        typer.Option("--alpha", help="True thermal diffusivity"),
    ],
    n_sensors: Annotated[
        int,
        typer.Option("--n-sensors", help="Number of sensors"),
    ],
    noise: Annotated[
        float,
        typer.Option("--noise", help="Gaussian noise std"),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory"),
    ],
) -> None:
    """
    Generate a synthetic calibration case with known ground truth.
    """
    try:
        sim_config = load_simulation_config(config)
    except ValidationError as e:
        typer.echo("Error: invalid simulation config", err=True)
        raise typer.Exit(code=1) from e

    try:
        generate_synthetic_case(
            sim_config=sim_config,
            true_alpha=alpha,
            n_sensors=n_sensors,
            noise_std=noise,
            out_dir=out,
        )
    except FileExistsError:
        typer.echo(f"Error: output directory exists: {out}", err=True)
        raise typer.Exit(code=1)

    typer.echo("✔ Synthetic case generated")
    typer.echo(f"alpha_true = {alpha}")
    typer.echo(f"sensors    = {n_sensors}")
    typer.echo(f"noise_std  = {noise}")


def main() -> None:
    app()
