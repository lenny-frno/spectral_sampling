from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _as_bool_mask(mask: npt.ArrayLike, n: int) -> BoolArray:
    raw = np.asarray(mask)
    if raw.shape != (n,):
        raise ValueError(f"feasible_mask must have shape ({n},), got {raw.shape}")
    if raw.dtype == np.bool_:
        return raw.copy()
    if not np.all(np.isin(raw, [0, 1, True, False])):
        raise ValueError("feasible_mask must be boolean or contain only 0/1 values")
    return raw.astype(bool)


def normalize_shape_to_density(
    shape_values: npt.ArrayLike,
    feasible_mask: npt.ArrayLike,
    n_points: int,
    cell_areas: npt.ArrayLike | None = None,
) -> FloatArray:
    """Normalize a nonnegative shape field into a target density with exact mass.

    Parameters
    ----------
    shape_values
        Nonnegative shape field q_i over M candidate locations. The values are
        dimensionless preference weights before normalization.
    feasible_mask
        Boolean mask of shape (M,) indicating hard-feasible candidates.
        Density on infeasible cells is forced to zero.
    n_points
        Target integrated mass in points. For a valid density rho, this enforces
        sum(rho_i * area_i) == n_points within floating-point tolerance.
    cell_areas
        Optional candidate cell areas (m^2), shape (M,). If omitted, equal area
        is assumed (area_i = 1 for all candidates).

    Returns
    -------
    numpy.ndarray
        Normalized target density rho with shape (M,) and units points / area.

    Raises
    ------
    ValueError
        If inputs are invalid, if any shape value is negative, or if total
        feasible shape mass is zero while n_points > 0.
    """
    q = np.asarray(shape_values, dtype=float)
    if q.ndim != 1:
        raise ValueError(f"shape_values must be 1D, got {q.ndim}D")
    m = q.shape[0]

    feasible = _as_bool_mask(feasible_mask, m)

    if not np.all(np.isfinite(q)):
        raise ValueError("shape_values must be finite")
    if np.any(q < 0.0):
        raise ValueError("shape_values must be nonnegative")

    if n_points < 0:
        raise ValueError("n_points must be >= 0")

    if cell_areas is None:
        areas = np.ones(m, dtype=float)
    else:
        areas = np.asarray(cell_areas, dtype=float)
        if areas.shape != (m,):
            raise ValueError(f"cell_areas must have shape ({m},), got {areas.shape}")
        if not np.all(np.isfinite(areas)):
            raise ValueError("cell_areas must be finite")
        if np.any(areas <= 0.0):
            raise ValueError("cell_areas must be strictly positive")

    q_effective = q.copy()
    q_effective[~feasible] = 0.0

    if n_points == 0:
        return np.zeros(m, dtype=float)

    total_shape_mass = float(np.sum(q_effective * areas))
    if total_shape_mass <= 0.0:
        raise ValueError(
            "Cannot normalize shape field: total feasible shape mass is zero"
        )

    rho = float(n_points) * q_effective / total_shape_mass
    rho[~feasible] = 0.0

    normalized_mass = float(np.sum(rho * areas))
    if not np.isclose(normalized_mass, float(n_points), rtol=1e-10, atol=1e-10):
        raise RuntimeError("Normalization failed to satisfy integrated mass constraint")

    return rho


@dataclass(frozen=True)
class DensityField:
    """Target density field over discrete 2D candidate locations.

    Notes
    -----
    - Coordinates are candidate locations in a metric Cartesian system, typically
      cell centers or discrete storage/evaluation sites.
    - Density values have units points / area and should satisfy
      sum(rho_i * area_i) ~= N for a requested sample count N.
    - Hard feasibility is separate from density preference. Infeasible candidates
      must have zero density and must never be sampled.
    """

    coordinates: FloatArray
    density: FloatArray
    feasible_mask: BoolArray
    cell_areas: FloatArray | None = None

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        rho = np.asarray(self.density, dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coordinates must have shape (M, 2), got {coords.shape}")

        m = coords.shape[0]
        if rho.shape != (m,):
            raise ValueError(f"density must have shape ({m},), got {rho.shape}")

        feasible = _as_bool_mask(self.feasible_mask, m)

        if not np.all(np.isfinite(coords)):
            raise ValueError("coordinates must be finite")
        if not np.all(np.isfinite(rho)):
            raise ValueError("density must be finite")
        if np.any(rho < 0.0):
            raise ValueError("density must be nonnegative")

        if np.any((~feasible) & (rho > 0.0)):
            raise ValueError("density must be zero on infeasible candidates")

        if self.cell_areas is not None:
            areas = np.asarray(self.cell_areas, dtype=float)
            if areas.shape != (m,):
                raise ValueError(
                    f"cell_areas must have shape ({m},), got {areas.shape}"
                )
            if not np.all(np.isfinite(areas)):
                raise ValueError("cell_areas must be finite")
            if np.any(areas <= 0.0):
                raise ValueError("cell_areas must be strictly positive")
            object.__setattr__(self, "cell_areas", areas)

        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "density", rho)
        object.__setattr__(self, "feasible_mask", feasible)

    @property
    def n_candidates(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def areas(self) -> FloatArray:
        if self.cell_areas is None:
            return np.ones(self.n_candidates, dtype=float)
        return self.cell_areas

    @property
    def integrated_mass(self) -> float:
        return float(np.sum(self.density * self.areas))

    @property
    def eligible_mask(self) -> BoolArray:
        return self.feasible_mask & (self.density > 0.0)

    @property
    def eligible_indices(self) -> npt.NDArray[np.int64]:
        return np.flatnonzero(self.eligible_mask)

    @classmethod
    def from_shape(
        cls,
        coordinates: npt.ArrayLike,
        shape_values: npt.ArrayLike,
        feasible_mask: npt.ArrayLike,
        n_points: int,
        cell_areas: npt.ArrayLike | None = None,
    ) -> "DensityField":
        """Create a density field by normalizing positive shape values to n_points."""
        coords = np.asarray(coordinates, dtype=float)
        q = np.asarray(shape_values, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coordinates must have shape (M, 2), got {coords.shape}")
        if q.shape != (coords.shape[0],):
            raise ValueError(
                f"shape_values must have shape ({coords.shape[0]},), got {q.shape}"
            )
        rho = normalize_shape_to_density(
            shape_values=q,
            feasible_mask=feasible_mask,
            n_points=n_points,
            cell_areas=cell_areas,
        )
        return cls(
            coordinates=coords,
            density=rho,
            feasible_mask=feasible_mask,
            cell_areas=cell_areas,
        )
