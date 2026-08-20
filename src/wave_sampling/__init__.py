"""Deterministic, density-based spatial sampling for 2D candidate sets."""

from wave_sampling.density.field import DensityField, normalize_shape_to_density
from wave_sampling.result import SamplingResult
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt
from wave_sampling.samplers.optimal_transport import density_weighted_optimal_transport

__all__ = [
    "DensityField",
    "SamplingResult",
    "density_weighted_farthest_point",
    "density_weighted_lloyd_cvt",
    "density_weighted_optimal_transport",
    "normalize_shape_to_density",
]
