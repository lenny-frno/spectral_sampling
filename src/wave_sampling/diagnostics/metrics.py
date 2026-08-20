from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from wave_sampling.density.field import DensityField
from wave_sampling.result import SamplingResult


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
) -> dict[str, Any]:
    """Compute baseline diagnostics for a sampling result."""
    return {
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
