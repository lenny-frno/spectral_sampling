import numpy as np
import pytest

from wave_sampling.density.field import DensityField
from wave_sampling.diagnostics.metrics import compute_sampling_diagnostics
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt


def make_grid(nx: int, ny: int, spacing_m: float = 2_000.0) -> np.ndarray:
    x = np.arange(nx, dtype=float) * spacing_m
    y = np.arange(ny, dtype=float) * spacing_m
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def synthetic_uniform_with_hole() -> tuple[DensityField, int]:
    coords = make_grid(24, 14)
    cx_m, cy_m, r_m = 24_000.0, 12_000.0, 7_000.0
    d2 = (coords[:, 0] - cx_m) ** 2 + (coords[:, 1] - cy_m) ** 2
    feasible = d2 > r_m**2
    q = np.ones(coords.shape[0], dtype=float)
    n_points = 60
    return DensityField.from_shape(coords, q, feasible, n_points=n_points), n_points


def synthetic_left_biased() -> tuple[DensityField, int]:
    coords = make_grid(24, 12)
    feasible = np.ones(coords.shape[0], dtype=bool)
    q = np.where(coords[:, 0] < 24_000.0, 4.0, 1.0)
    n_points = 60
    return DensityField.from_shape(coords, q, feasible, n_points=n_points), n_points


@pytest.mark.parametrize(
    "domain_builder",
    [synthetic_uniform_with_hole, synthetic_left_biased],
)
def test_cross_method_contracts_and_diagnostics(domain_builder) -> None:
    field, n_points = domain_builder()

    fp = density_weighted_farthest_point(field, n_points=n_points)
    cvt = density_weighted_lloyd_cvt(field, n_points=n_points)

    for result in (fp, cvt):
        assert result.n_selected == n_points
        assert len(np.unique(result.selected_indices)) == n_points
        assert np.all(field.feasible_mask[result.selected_indices])

        diag = compute_sampling_diagnostics(result, field)
        assert diag["hard_constraint_violations"] == 0
        assert np.isfinite(diag["density_reproduction"]["normalized_l1_error"])
        assert np.isfinite(diag["density_reproduction"]["normalized_l2_error"])


def test_cross_method_density_bias_on_same_domain() -> None:
    field, n_points = synthetic_left_biased()

    fp = density_weighted_farthest_point(field, n_points=n_points)
    cvt = density_weighted_lloyd_cvt(field, n_points=n_points)

    fp_left_fraction = np.mean(fp.selected_coordinates[:, 0] < 24_000.0)
    cvt_left_fraction = np.mean(cvt.selected_coordinates[:, 0] < 24_000.0)

    assert fp_left_fraction > 0.6
    assert cvt_left_fraction > 0.6
