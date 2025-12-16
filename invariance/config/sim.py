from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class GridConfig(BaseModel):
    nx: int = Field(..., gt=0, description="Number of grid points in x-direction")
    ny: int = Field(..., gt=0, description="Number of grid points in y-direction")
    dx: float = Field(..., gt=0, description="Grid spacing in x")
    dy: float = Field(..., gt=0, description="Grid spacing in y")


class TimeConfig(BaseModel):
    dt: float = Field(..., gt=0, description="Time step size")
    n_steps: int = Field(..., gt=0, description="Number of time steps")


class MaterialConfig(BaseModel):
    alpha: float = Field(
        ..., gt=0, description="Thermal diffusivity (m^2 / s)"
    )


class BoundaryConfig(BaseModel):
    type: str = Field(
        "dirichlet",
        description="Boundary condition type (dirichlet only in v1)",
    )
    value: float = Field(
        0.0, description="Fixed boundary temperature (Dirichlet)"
    )


class SimulationConfig(BaseModel):
    grid: GridConfig
    time: TimeConfig
    material: MaterialConfig
    boundary: BoundaryConfig
    initial_temperature: float = Field(
        0.0, description="Initial temperature everywhere"
    )

    @field_validator("boundary")
    @classmethod
    def validate_boundary(cls, v: BoundaryConfig) -> BoundaryConfig:
        if v.type != "dirichlet":
            raise ValueError("Only Dirichlet boundary conditions are supported in v1.")
        return v