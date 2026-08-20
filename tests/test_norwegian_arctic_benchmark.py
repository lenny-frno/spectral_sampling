from __future__ import annotations

import numpy as np

from examples.norwegian_arctic_case.config import BenchmarkConfig
from examples.norwegian_arctic_case.density import build_density
from examples.norwegian_arctic_case.domain import build_synthetic_domain
from examples.norwegian_arctic_case.run import run_benchmark
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt


def _small_cfg() -> BenchmarkConfig:
    return BenchmarkConfig(nx=140, ny=100, n_points=120, cvt_max_iterations=12)


def test_feasible_mask_contains_no_land_cells() -> None:
    cfg = _small_cfg()
    domain, _ = build_synthetic_domain(cfg)
    assert not np.any(domain.feasible_mask & domain.land_mask)


def test_hard_land_distance_exclusion_respected() -> None:
    cfg = _small_cfg()
    domain, _ = build_synthetic_domain(cfg)
    min_distance = float(np.min(domain.distance_to_all_land_m[domain.feasible_mask]))
    assert min_distance >= cfg.min_land_distance_m - 1e-6


def test_mainland_filter_identifies_large_components() -> None:
    cfg = _small_cfg()
    domain, metadata = build_synthetic_domain(cfg)
    areas = metadata["component_areas_km2"]

    assert np.any(domain.mainland_land_mask)
    assert len(areas) >= 3
    assert any(area >= cfg.mainland_min_area_km2 for area in areas.values())


def test_density_mass_equals_exact_requested_n() -> None:
    cfg = _small_cfg()
    domain, _ = build_synthetic_domain(cfg)
    built = build_density(domain, cfg, mode="baseline_hard")

    assert np.isclose(
        built.density_field.integrated_mass,
        float(cfg.n_points),
        rtol=1e-10,
        atol=1e-10,
    )


def test_baseline_hard_regional_masses_match_targets() -> None:
    cfg = _small_cfg()
    domain, _ = build_synthetic_domain(cfg)
    built = build_density(domain, cfg, mode="baseline_hard")

    for region, frac in built.region_targets.items():
        target = frac * cfg.n_points
        observed = built.region_masses[region]
        assert np.isclose(observed, target, rtol=1e-7, atol=1e-7)


def test_density_zero_on_infeasible_cells() -> None:
    cfg = _small_cfg()
    domain, _ = build_synthetic_domain(cfg)
    built = build_density(domain, cfg, mode="smooth_density")

    assert np.all(
        built.density_field.density[~built.density_field.feasible_mask] == 0.0
    )


def test_deterministic_samplers_repeat_identical_outputs() -> None:
    cfg = _small_cfg()
    domain, _ = build_synthetic_domain(cfg)
    built = build_density(domain, cfg, mode="baseline_smooth")

    fp_a = density_weighted_farthest_point(built.density_field, n_points=cfg.n_points)
    fp_b = density_weighted_farthest_point(built.density_field, n_points=cfg.n_points)
    cvt_a = density_weighted_lloyd_cvt(
        built.density_field,
        n_points=cfg.n_points,
        max_iterations=cfg.cvt_max_iterations,
    )
    cvt_b = density_weighted_lloyd_cvt(
        built.density_field,
        n_points=cfg.n_points,
        max_iterations=cfg.cvt_max_iterations,
    )

    assert np.array_equal(fp_a.selected_indices, fp_b.selected_indices)
    assert np.array_equal(cvt_a.selected_indices, cvt_b.selected_indices)


def test_selected_points_all_feasible() -> None:
    cfg = _small_cfg()
    domain, _ = build_synthetic_domain(cfg)
    built = build_density(domain, cfg, mode="baseline_smooth")

    fp = density_weighted_farthest_point(built.density_field, n_points=cfg.n_points)
    cvt = density_weighted_lloyd_cvt(
        built.density_field,
        n_points=cfg.n_points,
        max_iterations=cfg.cvt_max_iterations,
    )

    assert np.all(built.density_field.feasible_mask[fp.selected_indices])
    assert np.all(built.density_field.feasible_mask[cvt.selected_indices])


def test_benchmark_runs_without_external_data() -> None:
    cfg = _small_cfg()
    out = run_benchmark(cfg=cfg, modes=("baseline_hard",), generate_plots=False)

    assert out["domain"]["n_cells"] == cfg.nx * cfg.ny
    assert (
        out["modes"]["baseline_hard"]["methods"]["density_weighted_farthest_point"][
            "actual_n"
        ]
        == cfg.n_points
    )


def test_configuration_reproducible_across_runs() -> None:
    cfg = _small_cfg()
    run1 = run_benchmark(cfg=cfg, modes=("baseline_smooth",), generate_plots=False)
    run2 = run_benchmark(cfg=cfg, modes=("baseline_smooth",), generate_plots=False)

    m1 = run1["modes"]["baseline_smooth"]["methods"]["density_weighted_farthest_point"]
    m2 = run2["modes"]["baseline_smooth"]["methods"]["density_weighted_farthest_point"]

    assert np.isclose(m1["normalized_l1_error"], m2["normalized_l1_error"])
    assert m1["reproducible_repeat"] is True
    assert m2["reproducible_repeat"] is True
