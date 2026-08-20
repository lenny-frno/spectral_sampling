import numpy as np

from wave_sampling.density.field import DensityField
from wave_sampling.diagnostics.metrics import (
    compute_sampling_diagnostics,
    density_reproduction_metrics,
    nearest_neighbour_statistics,
    poisson_disk_separation_metrics,
)
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.poisson_disk import density_adapted_poisson_disk


def make_grid(nx: int = 12, ny: int = 8) -> np.ndarray:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_nearest_neighbour_stats_finite() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [4.0, 5.0]])
    stats = nearest_neighbour_statistics(points)
    assert np.isfinite(stats["min"])
    assert np.isfinite(stats["mean"])
    assert np.isfinite(stats["median"])
    assert np.isfinite(stats["std"])
    assert np.isfinite(stats["p05"])
    assert np.isfinite(stats["p95"])


def test_density_reproduction_metrics_finite() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.linspace(1.0, 2.0, m)
    field = DensityField.from_shape(coords, q, feasible, n_points=20)
    result = density_weighted_farthest_point(field, n_points=20)

    metrics = density_reproduction_metrics(
        result.selected_indices,
        field.density,
        field.feasible_mask,
        field.cell_areas,
    )
    assert np.isfinite(metrics["normalized_l1_error"])
    assert np.isfinite(metrics["normalized_l2_error"])


def test_compute_sampling_diagnostics_hard_violations_zero() -> None:
    coords = make_grid()
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    feasible[:3] = False
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=20)
    result = density_weighted_farthest_point(field, n_points=20)

    diag = compute_sampling_diagnostics(result, field)
    assert diag["hard_constraint_violations"] == 0


def test_poisson_separation_metrics_zero_violations() -> None:
    coords = make_grid(16, 12)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    q = np.ones(m)
    field = DensityField.from_shape(coords, q, feasible, n_points=40)

    result = density_adapted_poisson_disk(
        field,
        n_points=40,
        seed=3,
        spacing_scale=0.7,
    )

    sep = poisson_disk_separation_metrics(
        selected_indices=result.selected_indices,
        selected_coordinates=result.selected_coordinates,
        target_density=field.density,
        feasible_mask=field.feasible_mask,
        spacing_scale=0.7,
    )

    assert sep["separation_violations"] == 0
    assert np.isfinite(sep["ratio_median"])
