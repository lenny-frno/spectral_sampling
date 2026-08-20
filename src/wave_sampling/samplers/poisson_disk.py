from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from wave_sampling.density.field import DensityField
from wave_sampling.result import SamplingResult

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class PoissonDiskCapacityError(RuntimeError):
    """Raised when variable-radius Poisson disk cannot place the requested count."""

    requested_n_points: int
    generated_n_points: int
    spacing_scale: float
    max_attempts_per_active_point: int
    n_eligible: int

    def __str__(self) -> str:
        return (
            "Unable to generate requested number of points with variable-radius "
            "Poisson disk sampling: "
            f"requested={self.requested_n_points}, "
            f"generated={self.generated_n_points}, "
            f"spacing_scale={self.spacing_scale}, "
            f"max_attempts_per_active_point={self.max_attempts_per_active_point}, "
            f"eligible_candidates={self.n_eligible}."
        )


def density_to_nominal_radius(
    density: npt.ArrayLike,
    feasible_mask: npt.ArrayLike,
    spacing_scale: float = 1.0,
) -> FloatArray:
    """Map target density to local nominal radius on feasible candidates.

    Radius is defined as r_i = spacing_scale / sqrt(rho_i) for feasible rho_i > 0.
    Feasible candidates with rho_i <= 0 are marked as non-selectable with inf radius.
    Infeasible candidates are assigned radius 0 because they are never sampled.
    """
    rho = np.asarray(density, dtype=float)
    feasible = np.asarray(feasible_mask, dtype=bool)

    if rho.ndim != 1:
        raise ValueError("density must be 1D")
    if feasible.shape != rho.shape:
        raise ValueError("feasible_mask shape mismatch")
    if spacing_scale <= 0.0:
        raise ValueError("spacing_scale must be strictly positive")

    radius = np.zeros_like(rho, dtype=float)

    eligible = feasible & (rho > 0.0)
    if np.any(rho[feasible] < 0.0):
        raise ValueError("density must be nonnegative on feasible candidates")

    radius[eligible] = spacing_scale / np.sqrt(rho[eligible])

    zero_density_feasible = feasible & (rho <= 0.0)
    radius[zero_density_feasible] = np.inf
    radius[~feasible] = 0.0

    return radius


def _hash_cell_key(point_xy: FloatArray, cell_size: float) -> tuple[int, int]:
    return (
        int(np.floor(point_xy[0] / cell_size)),
        int(np.floor(point_xy[1] / cell_size)),
    )


def _iter_neighbor_cells(
    center_key: tuple[int, int],
    radius_m: float,
    cell_size: float,
) -> list[tuple[int, int]]:
    reach = int(np.ceil(radius_m / cell_size))
    keys: list[tuple[int, int]] = []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            keys.append((center_key[0] + dx, center_key[1] + dy))
    return keys


def _is_compatible_with_selected(
    candidate_local_idx: int,
    coords: FloatArray,
    local_radius: FloatArray,
    bins: dict[tuple[int, int], list[int]],
    cell_size: float,
    tol: float,
) -> bool:
    candidate_xy = coords[candidate_local_idx]
    candidate_radius = local_radius[candidate_local_idx]
    max_radius = float(np.max(local_radius))

    # For symmetric rule d(i,j) >= 0.5 * (r_i + r_j), any conflict must occur
    # within <= 0.5 * (r_i + max_r), so this search radius is sufficient.
    search_radius = 0.5 * (candidate_radius + max_radius)
    center_key = _hash_cell_key(candidate_xy, cell_size)

    for key in _iter_neighbor_cells(center_key, search_radius, cell_size):
        if key not in bins:
            continue
        for selected_local_idx in bins[key]:
            d = float(np.linalg.norm(candidate_xy - coords[selected_local_idx]))
            required = 0.5 * (candidate_radius + local_radius[selected_local_idx])
            if d + tol < required:
                return False

    return True


def _insert_selected(
    selected_local_idx: int,
    coords: FloatArray,
    bins: dict[tuple[int, int], list[int]],
    cell_size: float,
) -> None:
    key = _hash_cell_key(coords[selected_local_idx], cell_size)
    bins.setdefault(key, []).append(selected_local_idx)


def density_adapted_poisson_disk(
    density_field: DensityField,
    n_points: int,
    seed: int | None = None,
    spacing_scale: float = 1.0,
    max_attempts_per_active_point: int = 30,
    candidate_k_neighbors: int = 8,
    tolerance: float = 1e-12,
) -> SamplingResult:
    """Variable-radius Poisson-disk sampling on discrete candidate coordinates.

    This implementation adapts Bridson's active-list strategy to a discrete
    candidate set. Trial points are generated in continuous space around active
    samples, then mapped to nearest eligible candidate locations.

    Pairwise compatibility uses the symmetric rule:
        d(i, j) >= 0.5 * (r_i + r_j)

    with local nominal radius:
        r_i = spacing_scale / sqrt(rho_i)

    Parameters
    ----------
    density_field
        Target density field with discrete candidate coordinates.
    n_points
        Number of points to sample.
    seed
        Seed for NumPy Generator. Same seed and inputs are reproducible.
    spacing_scale
        Global scale alpha in r_i = alpha / sqrt(rho_i).
    max_attempts_per_active_point
        Number of trial attempts generated from each active point before deactivating.
    candidate_k_neighbors
        Number of nearest candidate-grid cells checked for each continuous trial point.
    tolerance
        Numerical tolerance for separation checks.
    """
    if n_points < 0:
        raise ValueError("n_points must be >= 0")
    if spacing_scale <= 0.0:
        raise ValueError("spacing_scale must be strictly positive")
    if max_attempts_per_active_point < 1:
        raise ValueError("max_attempts_per_active_point must be >= 1")
    if candidate_k_neighbors < 1:
        raise ValueError("candidate_k_neighbors must be >= 1")

    if n_points == 0:
        return SamplingResult(
            selected_indices=np.array([], dtype=np.int64),
            selected_coordinates=np.empty((0, 2), dtype=float),
            method="density_adapted_poisson_disk",
            requested_n_points=0,
            seed=seed,
            density_summary={
                "n_candidates": density_field.n_candidates,
                "n_feasible": int(np.sum(density_field.feasible_mask)),
                "target_mass": density_field.integrated_mass,
                "spacing_scale": spacing_scale,
                "max_attempts_per_active_point": max_attempts_per_active_point,
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
    local_radius = spacing_scale / np.sqrt(rho)

    if not np.all(np.isfinite(local_radius)):
        raise ValueError("Non-finite local radius computed from eligible density")

    rng = np.random.default_rng(seed)
    tree = cKDTree(coords)

    selected_mask = np.zeros(coords.shape[0], dtype=bool)
    selected_local: list[int] = []
    active_local: list[int] = []
    max_initial_seeds = 8

    # Spatial hash over selected points for fast local compatibility checks.
    max_radius = float(np.max(local_radius))
    min_radius = float(np.min(local_radius))
    cell_size = max(min_radius, 0.5 * max_radius)
    bins: dict[tuple[int, int], list[int]] = {}

    def add_seed(local_idx: int) -> None:
        selected_mask[local_idx] = True
        selected_local.append(local_idx)
        active_local.append(local_idx)
        _insert_selected(local_idx, coords, bins, cell_size)

    def pick_spread_seed() -> int | None:
        """Pick a globally spread compatible seed among remaining candidates.

        Score is distance-to-selected normalized by local radius, which promotes
        coverage while still respecting density-adapted scale.
        """
        remaining = np.flatnonzero(~selected_mask)
        if remaining.size == 0:
            return None

        selected_coords = coords[np.asarray(selected_local, dtype=np.int64)]
        selected_tree = cKDTree(selected_coords)
        nearest_dist, _ = selected_tree.query(coords[remaining], k=1)
        score = nearest_dist / local_radius[remaining]

        # Deterministic tie-breaking: stable sort on (-score, index).
        order = np.lexsort((remaining, -score))
        ranked = remaining[order]

        for local_idx in ranked:
            if _is_compatible_with_selected(
                int(local_idx),
                coords,
                local_radius,
                bins,
                cell_size,
                tolerance,
            ):
                return int(local_idx)

        return None

    def pick_new_component_seed() -> int | None:
        remaining = np.flatnonzero(~selected_mask)
        if remaining.size == 0:
            return None

        order = rng.permutation(remaining)
        for local_idx in order:
            if _is_compatible_with_selected(
                local_idx,
                coords,
                local_radius,
                bins,
                cell_size,
                tolerance,
            ):
                return int(local_idx)
        return None

    first_seed = int(rng.integers(0, coords.shape[0]))
    add_seed(first_seed)

    # Multi-seed initialization improves global coverage and avoids overgrowing
    # one local patch before other areas are explored.
    n_initial_seeds = min(max_initial_seeds, n_points)
    while len(selected_local) < n_initial_seeds:
        seed_idx = pick_spread_seed()
        if seed_idx is None:
            break
        add_seed(seed_idx)

    while len(selected_local) < n_points:
        if not active_local:
            seed_idx = pick_new_component_seed()
            if seed_idx is None:
                break
            add_seed(seed_idx)
            continue

        active_position = int(rng.integers(0, len(active_local)))
        center_local_idx = active_local[active_position]
        center_xy = coords[center_local_idx]
        center_radius = local_radius[center_local_idx]

        accepted_new = False
        for _ in range(max_attempts_per_active_point):
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            radial_scale = float(rng.uniform(1.0, 2.0))
            trial_radius = radial_scale * center_radius
            trial_xy = center_xy + trial_radius * np.array(
                [np.cos(theta), np.sin(theta)], dtype=float
            )

            k = min(candidate_k_neighbors, coords.shape[0])
            distances, candidate_indices = tree.query(trial_xy, k=k)
            candidate_indices_arr = np.atleast_1d(candidate_indices).astype(np.int64)
            distances_arr = np.atleast_1d(distances).astype(float)

            order = np.lexsort((candidate_indices_arr, distances_arr))
            candidate_indices_arr = candidate_indices_arr[order]

            for local_idx in candidate_indices_arr:
                if selected_mask[local_idx]:
                    continue
                if _is_compatible_with_selected(
                    int(local_idx),
                    coords,
                    local_radius,
                    bins,
                    cell_size,
                    tolerance,
                ):
                    add_seed(int(local_idx))
                    accepted_new = True
                    break

            if accepted_new:
                break

        if not accepted_new:
            # No valid candidates from this active center; retire it.
            active_local.pop(active_position)

    if len(selected_local) != n_points:
        raise PoissonDiskCapacityError(
            requested_n_points=n_points,
            generated_n_points=len(selected_local),
            spacing_scale=spacing_scale,
            max_attempts_per_active_point=max_attempts_per_active_point,
            n_eligible=int(eligible_global_idx.size),
        )

    selected_local_idx = np.asarray(selected_local, dtype=np.int64)
    selected_global_idx = eligible_global_idx[selected_local_idx]

    return SamplingResult(
        selected_indices=selected_global_idx,
        selected_coordinates=density_field.coordinates[selected_global_idx],
        method="density_adapted_poisson_disk",
        requested_n_points=n_points,
        seed=seed,
        density_summary={
            "n_candidates": density_field.n_candidates,
            "n_feasible": int(np.sum(density_field.feasible_mask)),
            "n_eligible": int(eligible_global_idx.size),
            "target_mass": density_field.integrated_mass,
            "spacing_scale": spacing_scale,
            "max_attempts_per_active_point": max_attempts_per_active_point,
            "candidate_k_neighbors": candidate_k_neighbors,
        },
    )
