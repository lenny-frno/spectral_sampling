"""Density field models and reusable smooth modifier functions."""

from wave_sampling.density.field import DensityField, normalize_shape_to_density
from wave_sampling.density.modifiers import (
    clipped_linear_transition,
    compose_positive_factors,
    exponential_distance_weight,
    gaussian_weight,
    logistic_weight,
    smootherstep,
    smoothstep,
)

__all__ = [
    "DensityField",
    "clipped_linear_transition",
    "compose_positive_factors",
    "exponential_distance_weight",
    "gaussian_weight",
    "logistic_weight",
    "normalize_shape_to_density",
    "smootherstep",
    "smoothstep",
]
