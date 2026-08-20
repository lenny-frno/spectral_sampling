import numpy as np
import pytest

from wave_sampling.density.field import DensityField, normalize_shape_to_density


def _coords(m: int = 5) -> np.ndarray:
    x = np.arange(m, dtype=float)
    return np.column_stack([x, np.zeros_like(x)])


def test_invalid_density_shape_raises() -> None:
    coords = _coords(4)
    with pytest.raises(ValueError, match="density must have shape"):
        DensityField(
            coordinates=coords,
            density=np.ones(3),
            feasible_mask=np.ones(4, dtype=bool),
        )


def test_negative_density_raises() -> None:
    coords = _coords(4)
    rho = np.array([1.0, 0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="nonnegative"):
        DensityField(
            coordinates=coords,
            density=rho,
            feasible_mask=np.ones(4, dtype=bool),
        )


def test_nonfinite_density_raises() -> None:
    coords = _coords(4)
    rho = np.array([1.0, np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        DensityField(
            coordinates=coords,
            density=rho,
            feasible_mask=np.ones(4, dtype=bool),
        )


def test_density_outside_feasible_rejected() -> None:
    coords = _coords(4)
    feasible = np.array([True, False, True, True])
    rho = np.array([1.0, 2.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="zero on infeasible"):
        DensityField(coordinates=coords, density=rho, feasible_mask=feasible)


def test_normalization_equal_area_mass_equals_n() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0])
    feasible = np.array([True, True, True, False])
    rho = normalize_shape_to_density(q, feasible, n_points=20)
    assert np.isclose(np.sum(rho), 20.0)
    assert rho[3] == 0.0


def test_normalization_nonuniform_area_mass_equals_n() -> None:
    q = np.array([1.0, 2.0, 1.0, 5.0])
    feasible = np.array([True, True, True, False])
    areas = np.array([2.0, 3.0, 4.0, 100.0])
    rho = normalize_shape_to_density(q, feasible, n_points=15, cell_areas=areas)
    assert np.isclose(np.sum(rho * areas), 15.0)
    assert rho[3] == 0.0


def test_zero_feasible_shape_mass_raises() -> None:
    q = np.zeros(4)
    feasible = np.array([True, True, False, True])
    with pytest.raises(ValueError, match="shape mass is zero"):
        normalize_shape_to_density(q, feasible, n_points=3)
