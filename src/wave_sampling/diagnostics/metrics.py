from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from wave_sampling.density.field import DensityField
from wave_sampling.result import SamplingResult
from wave_sampling.samplers.poisson_disk import density_to_nominal_radius


def nearest_neighbour_statistics(
    points: npt.ArrayLike,
) -> dict[str, float]:
    """Compute nearest-neighbour distance summary statistics."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    if pts.shape[0] < 2:
        nan = float("nan")
        return {
            "min": nan,
            "mean": nan,
            "median": nan,
            "std": nan,
            "p05": nan,
            "p95": nan,
        }

    diffs = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)

    return {
        "min": float(np.min(nn)),
        "mean": float(np.mean(nn)),
        "median": float(np.median(nn)),
        "std": float(np.std(nn)),
        "p05": float(np.percentile(nn, 5.0)),
        "p95": float(np.percentile(nn, 95.0)),
    }


def hard_constraint_violations(
    selected_indices: npt.ArrayLike,
    feasible_mask: npt.ArrayLike,
) -> int:
    """Count selected indices that violate hard feasibility."""
    idx = np.asarray(selected_indices, dtype=int)
    feasible = np.asarray(feasible_mask, dtype=bool)
    if np.any(idx < 0) or np.any(idx >= feasible.shape[0]):
        raise ValueError("selected_indices contain out-of-range values")
    return int(np.sum(~feasible[idx]))


def density_reproduction_metrics(
    selected_indices: npt.ArrayLike,
    target_density: npt.ArrayLike,
    feasible_mask: npt.ArrayLike,
    cell_areas: npt.ArrayLike | None = None,
) -> dict[str, float]:
    """Compare empirical selected-point density against target density.

    Empirical density is built on candidate locations as count_i / area_i,
    with count_i equal to the number of selected points at candidate i.
    """
    idx = np.asarray(selected_indices, dtype=int)
    rho = np.asarray(target_density, dtype=float)
    feasible = np.asarray(feasible_mask, dtype=bool)

    if rho.ndim != 1:
        raise ValueError("target_density must be 1D")
    if feasible.shape != rho.shape:
        raise ValueError("feasible_mask shape mismatch")

    m = rho.shape[0]
    if cell_areas is None:
        areas = np.ones(m, dtype=float)
    else:
        areas = np.asarray(cell_areas, dtype=float)
        if areas.shape != (m,):
            raise ValueError("cell_areas shape mismatch")
        if np.any(areas <= 0.0):
            raise ValueError("cell_areas must be positive")

    if np.any(idx < 0) or np.any(idx >= m):
        raise ValueError("selected_indices out of bounds")

    counts = np.bincount(idx, minlength=m).astype(float)
    empirical = counts / areas

    mask = feasible
    target_mass = float(np.sum(rho[mask] * areas[mask]))
    if target_mass <= 0.0:
        raise ValueError("target density has zero feasible mass")

    diff = empirical[mask] - rho[mask]

    l1 = float(np.sum(np.abs(diff) * areas[mask]) / target_mass)
    target_l2_norm = float(np.sqrt(np.sum((rho[mask] ** 2) * areas[mask])))
    if target_l2_norm > 0.0:
        l2 = float(np.sqrt(np.sum((diff**2) * areas[mask])) / target_l2_norm)
    else:
        l2 = float("nan")

    emp_vals = empirical[mask]
    rho_vals = rho[mask]
    if emp_vals.size >= 2 and np.std(emp_vals) > 0.0 and np.std(rho_vals) > 0.0:
        corr = float(np.corrcoef(emp_vals, rho_vals)[0, 1])
    else:
        corr = float("nan")

    return {
        "normalized_l1_error": l1,
        "normalized_l2_error": l2,
        "correlation": corr,
    }


def compute_sampling_diagnostics(
    result: SamplingResult,
    density_field: DensityField,
    poisson_spacing_scale: float | None = None,
    poisson_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compute baseline diagnostics for a sampling result."""
    diagnostics: dict[str, Any] = {
        "hard_constraint_violations": hard_constraint_violations(
            result.selected_indices,
            density_field.feasible_mask,
        ),
        "nearest_neighbour": nearest_neighbour_statistics(
            result.selected_coordinates,
        ),
        "density_reproduction": density_reproduction_metrics(
            result.selected_indices,
            density_field.density,
            density_field.feasible_mask,
            density_field.cell_areas,
        ),
    }

    if poisson_spacing_scale is not None:
        diagnostics["poisson_separation"] = poisson_disk_separation_metrics(
            selected_indices=result.selected_indices,
            selected_coordinates=result.selected_coordinates,
            target_density=density_field.density,
            feasible_mask=density_field.feasible_mask,
            spacing_scale=poisson_spacing_scale,
            tolerance=poisson_tolerance,
        )

    return diagnostics


def poisson_disk_separation_metrics(
    selected_indices: npt.ArrayLike,
    selected_coordinates: npt.ArrayLike,
    target_density: npt.ArrayLike,
    feasible_mask: npt.ArrayLike,
    spacing_scale: float,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Validate variable-radius Poisson pairwise separation and spacing ratios.

    Separation rule for indices i,j is
        d(i,j) >= 0.5 * (r_i + r_j)
    where r_i = spacing_scale / sqrt(rho_i) for feasible rho_i > 0.
    """
    idx = np.asarray(selected_indices, dtype=np.int64)
    pts = np.asarray(selected_coordinates, dtype=float)
    rho = np.asarray(target_density, dtype=float)
    feasible = np.asarray(feasible_mask, dtype=bool)

    if idx.ndim != 1:
        raise ValueError("selected_indices must be 1D")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("selected_coordinates must have shape (N, 2)")
    if pts.shape[0] != idx.shape[0]:
        raise ValueError("selected_indices and selected_coordinates length mismatch")
    if np.any(idx < 0) or np.any(idx >= rho.shape[0]):
        raise ValueError("selected_indices out of bounds")

    if idx.size < 2:
        nan = float("nan")
        return {
            "separation_violations": 0,
            "minimum_margin_m": nan,
            "ratio_min": nan,
            "ratio_median": nan,
            "ratio_p05": nan,
            "ratio_p95": nan,
        }

    full_radius = density_to_nominal_radius(
        density=rho,
        feasible_mask=feasible,
        spacing_scale=spacing_scale,
    )
    local_radius = full_radius[idx]

    diffs = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)

    required = 0.5 * (local_radius[:, None] + local_radius[None, :])
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(required, -np.inf)

    margin = dist - required
    violations = int(np.sum(margin < -tolerance) // 2)
    minimum_margin = float(np.min(margin[np.isfinite(margin)]))

    nn = np.min(dist, axis=1)
    ratios = nn / local_radius

    return {
        "separation_violations": violations,
        "minimum_margin_m": minimum_margin,
        "ratio_min": float(np.min(ratios)),
        "ratio_median": float(np.median(ratios)),
        "ratio_p05": float(np.percentile(ratios, 5.0)),
        "ratio_p95": float(np.percentile(ratios, 95.0)),
    }
