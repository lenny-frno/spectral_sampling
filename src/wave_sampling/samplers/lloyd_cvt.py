from __future__ import annotations

import numpy as np
import numpy.typing as npt

from wave_sampling.density.field import DensityField
from wave_sampling.result import SamplingResult
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point


def _nearest_available_index(
    coordinates_m: npt.NDArray[np.float64],
    target_xy_m: npt.NDArray[np.float64],
    taken_mask: npt.NDArray[np.bool_],
) -> int:
    """Return nearest untaken candidate index with deterministic tie-breaking."""
    dist2 = np.sum((coordinates_m - target_xy_m) ** 2, axis=1)
    dist2 = dist2.copy()
    dist2[taken_mask] = np.inf
    idx = int(np.argmin(dist2))
    if not np.isfinite(dist2[idx]):
        raise RuntimeError("No available candidate while assigning Lloyd centers")
    return idx


def _assign_to_nearest_center(
    coordinates_m: npt.NDArray[np.float64],
    centers_local_idx: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Assign each candidate to its nearest center using O(M) working memory."""
    n_candidates = coordinates_m.shape[0]
    labels = np.zeros(n_candidates, dtype=np.int64)
    nearest_dist2 = np.full(n_candidates, np.inf, dtype=float)

    for cluster_idx, center_idx in enumerate(centers_local_idx):
        delta = coordinates_m - coordinates_m[center_idx]
        dist2 = np.sum(delta * delta, axis=1)
        better = dist2 < nearest_dist2
        labels[better] = cluster_idx
        nearest_dist2[better] = dist2[better]

    return labels


def density_weighted_lloyd_cvt(
    density_field: DensityField,
    n_points: int,
    seed: int | None = None,
    max_iterations: int = 30,
) -> SamplingResult:
    """Deterministic density-weighted Lloyd/CVT over discrete candidates.

    This method uses a discrete approximation to density-weighted Lloyd updates.
    With candidate masses w_i = rho_i * area_i and Voronoi-like partitions, each
    center is moved toward the weighted centroid of its current cell and then
    projected back to the nearest eligible candidate location.

    Notes
    -----
    - Coordinates are treated as metric Cartesian coordinates in metres.
    - The algorithm is deterministic through deterministic initialization and
      deterministic tie-breaking.
    - This is a heuristic discrete CVT method; it does not guarantee exact
      density reproduction or global optimality.
    """
    if n_points < 0:
        raise ValueError("n_points must be >= 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    if n_points == 0:
        return SamplingResult(
            selected_indices=np.array([], dtype=np.int64),
            selected_coordinates=np.empty((0, 2), dtype=float),
            method="density_weighted_lloyd_cvt",
            requested_n_points=0,
            seed=seed,
            density_summary={
                "n_candidates": density_field.n_candidates,
                "n_feasible": int(np.sum(density_field.feasible_mask)),
                "target_mass": density_field.integrated_mass,
                "iterations": 0,
            },
        )

    eligible_global_idx = density_field.eligible_indices
    if eligible_global_idx.size < n_points:
        raise ValueError(
            "Requested n_points exceeds number of eligible candidates "
            f"({n_points} > {eligible_global_idx.size}). Eligible means feasible and "
            "strictly positive target density."
        )

    coordinates_m = density_field.coordinates[eligible_global_idx]
    masses = (
        density_field.density[eligible_global_idx]
        * density_field.areas[eligible_global_idx]
    )

    # Deterministic initialization from the baseline farthest-point sampler.
    init_result = density_weighted_farthest_point(density_field, n_points=n_points)
    centers_local_idx = np.searchsorted(
        eligible_global_idx, init_result.selected_indices
    )
    centers_local_idx = centers_local_idx.astype(np.int64)

    iterations_run = 0
    for _ in range(max_iterations):
        iterations_run += 1
        labels = _assign_to_nearest_center(coordinates_m, centers_local_idx)

        cluster_mass = np.bincount(labels, weights=masses, minlength=n_points)
        x_mass = np.bincount(
            labels,
            weights=masses * coordinates_m[:, 0],
            minlength=n_points,
        )
        y_mass = np.bincount(
            labels,
            weights=masses * coordinates_m[:, 1],
            minlength=n_points,
        )

        proposed_centroids_m = np.empty((n_points, 2), dtype=float)
        nonempty = cluster_mass > 0.0
        proposed_centroids_m[nonempty, 0] = x_mass[nonempty] / cluster_mass[nonempty]
        proposed_centroids_m[nonempty, 1] = y_mass[nonempty] / cluster_mass[nonempty]
        proposed_centroids_m[~nonempty] = coordinates_m[centers_local_idx[~nonempty]]

        new_centers_local_idx = np.empty_like(centers_local_idx)
        taken = np.zeros(coordinates_m.shape[0], dtype=bool)
        for cluster_idx in range(n_points):
            candidate_idx = _nearest_available_index(
                coordinates_m,
                proposed_centroids_m[cluster_idx],
                taken,
            )
            new_centers_local_idx[cluster_idx] = candidate_idx
            taken[candidate_idx] = True

        if np.array_equal(new_centers_local_idx, centers_local_idx):
            centers_local_idx = new_centers_local_idx
            break

        centers_local_idx = new_centers_local_idx

    selected_global_idx = eligible_global_idx[centers_local_idx]

    return SamplingResult(
        selected_indices=selected_global_idx,
        selected_coordinates=density_field.coordinates[selected_global_idx],
        method="density_weighted_lloyd_cvt",
        requested_n_points=n_points,
        seed=seed,
        density_summary={
            "n_candidates": density_field.n_candidates,
            "n_feasible": int(np.sum(density_field.feasible_mask)),
            "n_eligible": int(eligible_global_idx.size),
            "target_mass": density_field.integrated_mass,
            "iterations": iterations_run,
            "max_iterations": max_iterations,
        },
    )
