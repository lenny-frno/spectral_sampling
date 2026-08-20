from __future__ import annotations

from typing import Protocol

from wave_sampling.density.field import DensityField
from wave_sampling.result import SamplingResult


class Sampler(Protocol):
    """Protocol for deterministic samplers over a DensityField."""

    def __call__(
        self, density_field: DensityField, n_points: int
    ) -> SamplingResult: ...
