"""Deterministic, density-based spatial sampling for 2D candidate sets."""

from wave_sampling.density.field import DensityField, normalize_shape_to_density
from wave_sampling.result import SamplingResult
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point

__all__ = [
    "DensityField",
    "SamplingResult",
    "density_weighted_farthest_point",
    "normalize_shape_to_density",
]
