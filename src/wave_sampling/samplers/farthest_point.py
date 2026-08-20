from __future__ import annotations

import numpy as np
import numpy.typing as npt

from wave_sampling.density.field import DensityField
from wave_sampling.result import SamplingResult


def _select_first_candidate(
    density: npt.NDArray[np.float64],
) -> int:
    # np.argmax is deterministic and returns the first index on ties.
    return int(np.argmax(density))


def density_weighted_farthest_point(
    density_field: DensityField,
    n_points: int,
    seed: int | None = None,
) -> SamplingResult:
    """Deterministic density-weighted farthest-point sampling.

    The greedy criterion is score_i = d_i / s_i, where d_i is current nearest
    distance to selected points and s_i = 1 / sqrt(rho_i) is local desired spacing.

    Notes
    -----
    This is a density-adapted greedy coverage heuristic. It does not provide a
    mathematical guarantee of exact density reproduction.
    """
    if n_points < 0:
        raise ValueError("n_points must be >= 0")

    if n_points == 0:
        return SamplingResult(
            selected_indices=np.array([], dtype=np.int64),
            selected_coordinates=np.empty((0, 2), dtype=float),
            method="density_weighted_farthest_point",
            requested_n_points=0,
            seed=seed,
            density_summary={
                "n_candidates": density_field.n_candidates,
                "n_feasible": int(np.sum(density_field.feasible_mask)),
                "target_mass": density_field.integrated_mass,
            },
        )

    eligible_global_idx = density_field.eligible_indices
    if eligible_global_idx.size < n_points:
        raise ValueError(
            "Requested n_points exceeds number of eligible candidates "
            f"({n_points} > {eligible_global_idx.size}). Eligible means feasible and "
            "strictly positive target density."
        )

    coords = density_field.coordinates[eligible_global_idx]
    rho = density_field.density[eligible_global_idx]
    desired_spacing = 1.0 / np.sqrt(rho)

    selected_local: list[int] = []
    nearest_distance = np.full(coords.shape[0], np.inf, dtype=float)
    selected_mask = np.zeros(coords.shape[0], dtype=bool)

    first = _select_first_candidate(rho)
    selected_local.append(first)
    selected_mask[first] = True
    nearest_distance = np.minimum(
        nearest_distance, np.linalg.norm(coords - coords[first], axis=1)
    )

    while len(selected_local) < n_points:
        score = nearest_distance / desired_spacing
        score[selected_mask] = -np.inf

        next_local = int(np.argmax(score))
        if not np.isfinite(score[next_local]):
            raise RuntimeError("No finite candidate score available during sampling")

        selected_local.append(next_local)
        selected_mask[next_local] = True

        distances_to_new = np.linalg.norm(coords - coords[next_local], axis=1)
        nearest_distance = np.minimum(nearest_distance, distances_to_new)

    selected_local_idx = np.asarray(selected_local, dtype=np.int64)
    selected_global_idx = eligible_global_idx[selected_local_idx]

    return SamplingResult(
        selected_indices=selected_global_idx,
        selected_coordinates=density_field.coordinates[selected_global_idx],
        method="density_weighted_farthest_point",
        requested_n_points=n_points,
        seed=seed,
        density_summary={
            "n_candidates": density_field.n_candidates,
            "n_feasible": int(np.sum(density_field.feasible_mask)),
            "n_eligible": int(eligible_global_idx.size),
            "target_mass": density_field.integrated_mass,
        },
    )
