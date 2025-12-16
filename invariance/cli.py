from __future__ import annotations

from pathlib import Path

from invariance.config.load import load_simulation_config
from invariance.run.create import create_run_directory

import typer
from rich import print

from pydantic import ValidationError

from invariance import __version__

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
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to simulation config JSON file",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Output directory for run artifacts",
    ),
) -> None:
    """
    Validate a simulation config and create a run directory.
    (Physics comes in the next milestone.)
    """
    try:
        sim_config = load_simulation_config(config)
    except ValidationError as e:
        typer.echo("Error: Validation failed for simulation config", err=True)
        typer.echo(e, err=True)
        raise typer.Exit(code=1)

    try:
        create_run_directory(out, sim_config)
    except FileExistsError:
        typer.echo(f"Error: Output directory already exists: {out}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"✔ Loaded simulation config from {config}")
    typer.echo(f"✔ Created run directory: {out}")


def main() -> None:
    app()
