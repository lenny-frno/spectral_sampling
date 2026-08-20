import numpy as np
import pytest

from wave_sampling.density.field import DensityField
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point


def make_grid(nx: int = 20, ny: int = 12) -> np.ndarray:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_returns_exactly_n_no_duplicates_and_feasible() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    feasible[:10] = False
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=40)

    result = density_weighted_farthest_point(field, n_points=40)

    assert result.n_selected == 40
    assert len(np.unique(result.selected_indices)) == 40
    assert np.all(field.feasible_mask[result.selected_indices])


def test_deterministic_output() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.linspace(1.0, 2.0, m)
    field = DensityField.from_shape(coords, q, feasible, n_points=30)

    a = density_weighted_farthest_point(field, n_points=30)
    b = density_weighted_farthest_point(field, n_points=30)

    assert np.array_equal(a.selected_indices, b.selected_indices)
    assert np.allclose(a.selected_coordinates, b.selected_coordinates)


def test_uniform_density_gives_nonzero_spacing() -> None:
    coords = make_grid(10, 10)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=20)
    result = density_weighted_farthest_point(field, n_points=20)

    pts = result.selected_coordinates
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_nn = np.min(np.min(dist, axis=1))
    assert min_nn > 0.0


def test_nonuniform_density_biases_high_density_region() -> None:
    coords = make_grid(24, 12)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)

    # Left half is strongly preferred.
    q = np.where(coords[:, 0] < 12.0, 4.0, 1.0)
    field = DensityField.from_shape(coords, q, feasible, n_points=60)
    result = density_weighted_farthest_point(field, n_points=60)

    left_fraction = np.mean(result.selected_coordinates[:, 0] < 12.0)
    assert left_fraction > 0.6


def test_more_points_than_eligible_raises() -> None:
    coords = make_grid(6, 6)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.zeros(m)
    q[:5] = 1.0
    field = DensityField.from_shape(coords, q, feasible, n_points=5)

    with pytest.raises(ValueError, match="exceeds number of eligible"):
        density_weighted_farthest_point(field, n_points=6)
