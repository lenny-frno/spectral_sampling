import numpy as np
import pytest

from wave_sampling.density.field import DensityField
from wave_sampling.samplers.optimal_transport import density_weighted_optimal_transport


def make_grid(nx: int = 24, ny: int = 14, spacing_m: float = 1_000.0) -> np.ndarray:
    x = np.arange(nx, dtype=float) * spacing_m
    y = np.arange(ny, dtype=float) * spacing_m
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_returns_exactly_n_unique_and_feasible() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    feasible[:12] = False
    q = np.ones(m, dtype=float)
    field = DensityField.from_shape(coords, q, feasible, n_points=60)

    result = density_weighted_optimal_transport(field, n_points=60)

    assert result.n_selected == 60
    assert len(np.unique(result.selected_indices)) == 60
    assert np.all(field.feasible_mask[result.selected_indices])
    assert result.method == "density_weighted_optimal_transport"


def test_deterministic_output() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.linspace(0.6, 2.0, m)
    field = DensityField.from_shape(coords, q, feasible, n_points=55)

    a = density_weighted_optimal_transport(field, n_points=55)
    b = density_weighted_optimal_transport(field, n_points=55)

    assert np.array_equal(a.selected_indices, b.selected_indices)


def test_uniform_density_has_reasonable_regularity() -> None:
    coords = make_grid(30, 20)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m, dtype=float)
    field = DensityField.from_shape(coords, q, feasible, n_points=90)

    result = density_weighted_optimal_transport(field, n_points=90)
    pts = result.selected_coordinates

    diffs = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)

    cv = float(np.std(nn) / np.mean(nn))
    assert cv < 0.65


def test_nonuniform_density_biases_points_to_high_density_region() -> None:
    coords = make_grid(28, 12)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.where(coords[:, 0] < 14_000.0, 4.0, 1.0)
    field = DensityField.from_shape(coords, q, feasible, n_points=70)

    result = density_weighted_optimal_transport(field, n_points=70)

    left_fraction = np.mean(result.selected_coordinates[:, 0] < 14_000.0)
    assert left_fraction > 0.6


def test_hard_forbidden_hole_contains_no_selected_points() -> None:
    coords = make_grid(28, 18)
    cx_m, cy_m, r_m = 13_000.0, 9_000.0, 4_500.0
    dist2 = (coords[:, 0] - cx_m) ** 2 + (coords[:, 1] - cy_m) ** 2
    feasible = dist2 > r_m**2
    q = np.ones(coords.shape[0], dtype=float)
    field = DensityField.from_shape(coords, q, feasible, n_points=65)

    result = density_weighted_optimal_transport(field, n_points=65)

    assert np.all(feasible[result.selected_indices])


def test_disconnected_feasible_domains_receive_points_when_mass_exists() -> None:
    coords = make_grid(30, 10)
    left_component = coords[:, 0] <= 8_000.0
    right_component = coords[:, 0] >= 20_000.0
    feasible = left_component | right_component

    q = np.zeros(coords.shape[0], dtype=float)
    q[left_component] = 1.2
    q[right_component] = 1.0

    field = DensityField.from_shape(coords, q, feasible, n_points=50)
    result = density_weighted_optimal_transport(field, n_points=50)

    selected_x = result.selected_coordinates[:, 0]
    assert np.any(selected_x <= 8_000.0)
    assert np.any(selected_x >= 20_000.0)


def test_zero_points_returns_empty_result() -> None:
    coords = make_grid(8, 8)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=0)

    result = density_weighted_optimal_transport(field, n_points=0)

    assert result.n_selected == 0
    assert result.selected_indices.shape == (0,)
    assert result.selected_coordinates.shape == (0, 2)


def test_invalid_parameters_raise() -> None:
    coords = make_grid(6, 6)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=8)

    with pytest.raises(ValueError, match="n_points must be >= 0"):
        density_weighted_optimal_transport(field, n_points=-1)

    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        density_weighted_optimal_transport(field, n_points=8, max_iterations=0)

    with pytest.raises(ValueError, match="mass_tolerance must be > 0"):
        density_weighted_optimal_transport(field, n_points=8, mass_tolerance=0.0)


def test_more_points_than_eligible_raises() -> None:
    coords = make_grid(6, 6)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.zeros(m)
    q[:5] = 1.0
    field = DensityField.from_shape(coords, q, feasible, n_points=5)

    with pytest.raises(ValueError, match="exceeds number of eligible"):
        density_weighted_optimal_transport(field, n_points=6)


def test_mass_mismatch_raises_clear_error() -> None:
    coords = make_grid(10, 10)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m, dtype=float)
    field = DensityField.from_shape(coords, q, feasible, n_points=20)

    with pytest.raises(ValueError, match="mass must equal requested n_points"):
        density_weighted_optimal_transport(field, n_points=25)
