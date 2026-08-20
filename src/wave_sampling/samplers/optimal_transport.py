from __future__ import annotations

import numpy as np
import numpy.typing as npt

from wave_sampling.density.field import DensityField
from wave_sampling.result import SamplingResult
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point


def _assign_power_cells(
    coordinates_m: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    centers_local_idx: npt.NDArray[np.int64],
    transport_potentials: npt.NDArray[np.float64],
    chunk_size: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Assign each candidate to a power cell and accumulate cell masses/centroids.

    Power distance is d_ij^2 - psi_j, where psi_j is a transport potential.
    """
    n_candidates = coordinates_m.shape[0]
    n_centers = centers_local_idx.size
    centers_xy = coordinates_m[centers_local_idx]

    labels = np.empty(n_candidates, dtype=np.int64)
    cluster_mass = np.zeros(n_centers, dtype=float)
    x_mass = np.zeros(n_centers, dtype=float)
    y_mass = np.zeros(n_centers, dtype=float)

    for start in range(0, n_candidates, chunk_size):
        end = min(start + chunk_size, n_candidates)
        chunk_xy = coordinates_m[start:end]

        delta = chunk_xy[:, None, :] - centers_xy[None, :, :]
        power_dist2 = np.sum(delta * delta, axis=2) - transport_potentials[None, :]

        chunk_labels = np.argmin(power_dist2, axis=1).astype(np.int64)
        labels[start:end] = chunk_labels

        chunk_mass = masses[start:end]
        np.add.at(cluster_mass, chunk_labels, chunk_mass)
        np.add.at(x_mass, chunk_labels, chunk_mass * chunk_xy[:, 0])
        np.add.at(y_mass, chunk_labels, chunk_mass * chunk_xy[:, 1])

    centroids_m = centers_xy.copy()
    nonzero = cluster_mass > 0.0
    centroids_m[nonzero, 0] = x_mass[nonzero] / cluster_mass[nonzero]
    centroids_m[nonzero, 1] = y_mass[nonzero] / cluster_mass[nonzero]

    return labels, cluster_mass, centroids_m


def _project_unique_centers(
    coordinates_m: npt.NDArray[np.float64],
    proposed_centroids_m: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Project centroids to unique candidate indices with deterministic ties."""
    n_centers = proposed_centroids_m.shape[0]
    new_centers = np.empty(n_centers, dtype=np.int64)
    taken = np.zeros(coordinates_m.shape[0], dtype=bool)

    for cluster_idx in range(n_centers):
        dist2 = np.sum(
            (coordinates_m - proposed_centroids_m[cluster_idx]) ** 2,
            axis=1,
        )
        dist2[taken] = np.inf

        idx = int(np.argmin(dist2))
        if not np.isfinite(dist2[idx]):
            raise RuntimeError("No available candidate while projecting OT centroids")

        new_centers[cluster_idx] = idx
        taken[idx] = True

    return new_centers


def density_weighted_optimal_transport(
    density_field: DensityField,
    n_points: int,
    seed: int | None = None,
    max_iterations: int = 20,
    mass_tolerance: float = 0.03,
    potential_step: float = 0.8,
    chunk_size: int = 16384,
    fail_on_nonconvergence: bool = False,
) -> SamplingResult:
    """Deterministic semi-discrete OT-inspired sampling on discrete candidates.

    Formulation
    -----------
    Let eligible candidate locations x_i carry target mass m_i = rho_i * area_i,
    with sum_i m_i = N. The source measure is N equal point masses (capacity 1
    each) represented by N generators y_j. We iteratively minimize a quadratic
    transport objective with power-diagram assignments:

        assign i to argmin_j ||x_i - y_j||^2 - psi_j,

    where psi_j are transport potentials updated to push each generator mass
    toward one unit. Generator locations are then updated to weighted centroids
    and projected back to unique eligible candidates.

    Notes
    -----
    - This is a robust, deterministic semi-discrete OT approximation tailored to
      candidate-grid sampling.
    - It enforces exact N unique feasible output points.
    - Like discrete CVT, it is approximate; exact global OT optimality is not
      guaranteed on coarse/indivisible candidate grids.
        - By default, the best iterate is returned even if strict mass-balance
            convergence is not reached. Set fail_on_nonconvergence=True for strict mode.
    """
    if n_points < 0:
        raise ValueError("n_points must be >= 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if mass_tolerance <= 0.0:
        raise ValueError("mass_tolerance must be > 0")
    if potential_step <= 0.0:
        raise ValueError("potential_step must be > 0")
    if chunk_size < 64:
        raise ValueError("chunk_size must be >= 64")

    if n_points == 0:
        return SamplingResult(
            selected_indices=np.array([], dtype=np.int64),
            selected_coordinates=np.empty((0, 2), dtype=float),
            method="density_weighted_optimal_transport",
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

    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Eligible target mass must be strictly positive")
    if not np.isclose(total_mass, float(n_points), rtol=1e-8, atol=1e-8):
        raise ValueError(
            "Eligible target mass must equal requested n_points "
            f"(mass={total_mass:.12g}, n_points={n_points})"
        )

    # On a discrete grid, indivisible cell masses define the attainable imbalance floor.
    indivisible_mass_floor = float(np.max(masses))
    effective_tolerance = max(mass_tolerance, indivisible_mass_floor + 1e-12)

    init_result = density_weighted_farthest_point(density_field, n_points=n_points)
    centers_local_idx = np.searchsorted(
        eligible_global_idx,
        init_result.selected_indices,
    ).astype(np.int64)

    transport_potentials = np.zeros(n_points, dtype=float)
    iterations_run = 0
    converged = False
    max_mass_imbalance = float("inf")
    best_centers_local_idx = centers_local_idx.copy()
    best_mass_imbalance = float("inf")
    best_iteration = 0

    for iteration in range(max_iterations):
        iterations_run = iteration + 1
        _, cluster_mass, centroids_m = _assign_power_cells(
            coordinates_m=coordinates_m,
            masses=masses,
            centers_local_idx=centers_local_idx,
            transport_potentials=transport_potentials,
            chunk_size=chunk_size,
        )

        mass_residual = cluster_mass - 1.0
        max_mass_imbalance = float(np.max(np.abs(mass_residual)))
        if max_mass_imbalance < best_mass_imbalance:
            best_mass_imbalance = max_mass_imbalance
            best_centers_local_idx = centers_local_idx.copy()
            best_iteration = iterations_run

        new_centers_local_idx = _project_unique_centers(
            coordinates_m=coordinates_m,
            proposed_centroids_m=centroids_m,
        )

        step = potential_step / (1.0 + 0.1 * iteration)
        transport_potentials = transport_potentials + step * mass_residual
        transport_potentials = transport_potentials - float(
            np.mean(transport_potentials)
        )

        stable_centers = np.array_equal(new_centers_local_idx, centers_local_idx)
        centers_local_idx = new_centers_local_idx

        if stable_centers and max_mass_imbalance <= effective_tolerance:
            converged = True
            break

    if not converged:
        centers_local_idx = best_centers_local_idx
        max_mass_imbalance = best_mass_imbalance

    if (
        fail_on_nonconvergence
        and (not converged)
        and max_mass_imbalance > effective_tolerance
    ):
        raise RuntimeError(
            "OT sampler did not satisfy mass-balance tolerance within max_iterations. "
            f"max_imbalance={max_mass_imbalance:.6f}, "
            f"effective_tolerance={effective_tolerance:.6f}, "
            f"iterations={iterations_run}."
        )

    selected_global_idx = eligible_global_idx[centers_local_idx]

    return SamplingResult(
        selected_indices=selected_global_idx,
        selected_coordinates=density_field.coordinates[selected_global_idx],
        method="density_weighted_optimal_transport",
        requested_n_points=n_points,
        seed=seed,
        density_summary={
            "n_candidates": density_field.n_candidates,
            "n_feasible": int(np.sum(density_field.feasible_mask)),
            "n_eligible": int(eligible_global_idx.size),
            "target_mass": density_field.integrated_mass,
            "iterations": iterations_run,
            "max_iterations": max_iterations,
            "converged": converged,
            "max_mass_imbalance": max_mass_imbalance,
            "best_iteration": best_iteration,
            "mass_tolerance": mass_tolerance,
            "effective_tolerance": effective_tolerance,
        },
    )
