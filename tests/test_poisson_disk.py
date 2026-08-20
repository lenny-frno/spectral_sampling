from __future__ import annotations

import numpy as np
import pytest

from wave_sampling.demos.compare_samplers import build_comparison_density
from wave_sampling.density.field import DensityField
from wave_sampling.diagnostics.metrics import poisson_disk_separation_metrics
from wave_sampling.samplers.poisson_disk import (
    PoissonDiskCapacityError,
    density_adapted_poisson_disk,
    density_to_nominal_radius,
)


def make_grid(nx: int, ny: int, spacing_m: float = 1_000.0) -> np.ndarray:
    x = np.arange(nx, dtype=float) * spacing_m
    y = np.arange(ny, dtype=float) * spacing_m
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def build_synthetic_gradient_domain(n_points: int = 70) -> DensityField:
    coords = make_grid(36, 24)
    m = coords.shape[0]

    # Hard forbidden island.
    cx_m, cy_m, island_r_m = 20_000.0, 12_000.0, 4_000.0
    d2_island = (coords[:, 0] - cx_m) ** 2 + (coords[:, 1] - cy_m) ** 2
    feasible = d2_island > island_r_m**2

    base = np.ones(m, dtype=float)

    # Strong Gaussian high-density peak.
    dx_peak = coords[:, 0] - 8_000.0
    dy_peak = coords[:, 1] - 18_000.0
    peak = 1.0 + 4.0 * np.exp(
        -0.5 * ((dx_peak / 3_000.0) ** 2 + (dy_peak / 4_500.0) ** 2)
    )

    # Low-density region on the lower-right.
    dx_low = coords[:, 0] - 30_000.0
    dy_low = coords[:, 1] - 3_000.0
    low_blob = np.exp(-0.5 * ((dx_low / 6_000.0) ** 2 + (dy_low / 4_000.0) ** 2))
    low = 1.0 - 0.7 * low_blob

    shape = base * peak * low
    return DensityField.from_shape(coords, shape, feasible, n_points=n_points)


def test_density_to_nominal_radius_marks_zero_density_as_inf() -> None:
    density = np.array([1.0, 0.0, 4.0, 0.0])
    feasible = np.array([True, True, True, False])

    radius = density_to_nominal_radius(density, feasible, spacing_scale=2.0)

    assert np.isclose(radius[0], 2.0)
    assert np.isclose(radius[2], 1.0)
    assert np.isinf(radius[1])
    assert radius[3] == 0.0


def test_returns_exactly_n_and_feasible_and_unique() -> None:
    field = build_synthetic_gradient_domain(n_points=70)

    result = density_adapted_poisson_disk(
        field,
        n_points=70,
        seed=11,
        spacing_scale=0.65,
    )

    assert result.n_selected == 70
    assert len(np.unique(result.selected_indices)) == 70
    assert np.all(field.feasible_mask[result.selected_indices])


def test_symmetric_separation_rule_holds() -> None:
    field = build_synthetic_gradient_domain(n_points=60)
    result = density_adapted_poisson_disk(
        field,
        n_points=60,
        seed=7,
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


def test_same_seed_reproducible_different_seed_varies() -> None:
    field = build_synthetic_gradient_domain(n_points=65)

    a = density_adapted_poisson_disk(field, n_points=65, seed=123, spacing_scale=0.7)
    b = density_adapted_poisson_disk(field, n_points=65, seed=123, spacing_scale=0.7)
    c = density_adapted_poisson_disk(field, n_points=65, seed=124, spacing_scale=0.7)

    assert np.array_equal(a.selected_indices, b.selected_indices)
    assert not np.array_equal(a.selected_indices, c.selected_indices)


def test_zero_density_candidates_not_selected() -> None:
    coords = make_grid(12, 8)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    shape = np.ones(m, dtype=float)

    # Force a feasible stripe to zero density via shape = 0.
    shape[(coords[:, 0] >= 4_000.0) & (coords[:, 0] < 6_000.0)] = 0.0
    field = DensityField.from_shape(coords, shape, feasible, n_points=25)

    result = density_adapted_poisson_disk(field, n_points=25, seed=6, spacing_scale=0.6)

    assert np.all(field.density[result.selected_indices] > 0.0)


def test_one_feasible_candidate_n1_works() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    feasible = np.array([False, True, False])
    shape = np.array([0.0, 1.0, 0.0])
    field = DensityField.from_shape(coords, shape, feasible, n_points=1)

    result = density_adapted_poisson_disk(field, n_points=1, seed=1)

    assert result.n_selected == 1
    assert result.selected_indices[0] == 1


def test_exceeds_capacity_raises_clear_error() -> None:
    coords = make_grid(10, 10)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    shape = np.ones(m, dtype=float)
    field = DensityField.from_shape(coords, shape, feasible, n_points=40)

    with pytest.raises(PoissonDiskCapacityError) as exc:
        density_adapted_poisson_disk(
            field,
            n_points=40,
            seed=9,
            spacing_scale=9_000.0,
            max_attempts_per_active_point=20,
        )

    msg = str(exc.value)
    assert "requested=40" in msg
    assert "generated=" in msg


def test_uniform_density_approximately_constant_local_ratios() -> None:
    coords = make_grid(20, 20)
    m = coords.shape[0]
    feasible = np.ones(m, dtype=bool)
    shape = np.ones(m, dtype=float)
    field = DensityField.from_shape(coords, shape, feasible, n_points=90)

    result = density_adapted_poisson_disk(
        field, n_points=90, seed=8, spacing_scale=0.75
    )
    sep = poisson_disk_separation_metrics(
        selected_indices=result.selected_indices,
        selected_coordinates=result.selected_coordinates,
        target_density=field.density,
        feasible_mask=field.feasible_mask,
        spacing_scale=0.75,
    )

    assert sep["separation_violations"] == 0
    assert sep["ratio_p95"] - sep["ratio_p05"] < 2.0


def test_disconnected_feasible_regions_can_both_receive_points() -> None:
    # Two disconnected feasible components with one eligible location each.
    coords = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [10_000.0, 0.0],
            [10_100.0, 0.0],
        ]
    )
    feasible = np.array([True, False, True, False])
    shape = np.array([1.0, 0.0, 1.0, 0.0])
    field = DensityField.from_shape(coords, shape, feasible, n_points=2)

    result = density_adapted_poisson_disk(field, n_points=2, seed=5, spacing_scale=1.0)

    assert set(result.selected_indices.tolist()) == {0, 2}


def test_comparison_case_includes_upper_left_points() -> None:
    field = build_comparison_density(n_points=400)
    result = density_adapted_poisson_disk(
        field,
        n_points=400,
        seed=42,
        spacing_scale=0.75,
    )

    xy = result.selected_coordinates
    upper_left = (xy[:, 0] < 20_000.0) & (xy[:, 1] > 65_000.0)
    assert int(np.sum(upper_left)) > 0
