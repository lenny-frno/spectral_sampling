import numpy as np
import pytest

from wave_sampling.density.field import DensityField
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt


def make_grid(nx: int = 20, ny: int = 12, spacing_m: float = 1_000.0) -> np.ndarray:
    x = np.arange(nx, dtype=float) * spacing_m
    y = np.arange(ny, dtype=float) * spacing_m
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_returns_exactly_n_no_duplicates_and_feasible() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    feasible[:10] = False
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=40)

    result = density_weighted_lloyd_cvt(field, n_points=40)

    assert result.n_selected == 40
    assert len(np.unique(result.selected_indices)) == 40
    assert np.all(field.feasible_mask[result.selected_indices])
    assert result.method == "density_weighted_lloyd_cvt"


def test_deterministic_output() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.linspace(1.0, 2.0, m)
    field = DensityField.from_shape(coords, q, feasible, n_points=30)

    a = density_weighted_lloyd_cvt(field, n_points=30, max_iterations=20)
    b = density_weighted_lloyd_cvt(field, n_points=30, max_iterations=20)

    assert np.array_equal(a.selected_indices, b.selected_indices)
    assert np.allclose(a.selected_coordinates, b.selected_coordinates)


def test_nonuniform_density_biases_high_density_region() -> None:
    coords = make_grid(24, 12)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)

    q = np.where(coords[:, 0] < 12_000.0, 4.0, 1.0)
    field = DensityField.from_shape(coords, q, feasible, n_points=60)
    result = density_weighted_lloyd_cvt(field, n_points=60)

    left_fraction = np.mean(result.selected_coordinates[:, 0] < 12_000.0)
    assert left_fraction > 0.6


def test_zero_points_returns_empty_result() -> None:
    coords = make_grid(6, 6)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=0)

    result = density_weighted_lloyd_cvt(field, n_points=0)

    assert result.n_selected == 0
    assert result.selected_indices.shape == (0,)
    assert result.selected_coordinates.shape == (0, 2)


def test_invalid_input_raises() -> None:
    coords = make_grid(6, 6)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=5)

    with pytest.raises(ValueError, match="n_points must be >= 0"):
        density_weighted_lloyd_cvt(field, n_points=-1)

    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        density_weighted_lloyd_cvt(field, n_points=3, max_iterations=0)


def test_more_points_than_eligible_raises() -> None:
    coords = make_grid(6, 6)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.zeros(m)
    q[:5] = 1.0
    field = DensityField.from_shape(coords, q, feasible, n_points=5)

    with pytest.raises(ValueError, match="exceeds number of eligible"):
        density_weighted_lloyd_cvt(field, n_points=6)
