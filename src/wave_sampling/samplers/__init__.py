"""Sampling algorithms operating on DensityField."""

from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt
from wave_sampling.samplers.optimal_transport import density_weighted_optimal_transport

__all__ = [
    "density_weighted_farthest_point",
    "density_weighted_lloyd_cvt",
    "density_weighted_optimal_transport",
]
