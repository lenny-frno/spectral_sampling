import numpy as np
import pytest

# tests/test_examples.py
from wave_sampling.demos.compare_samplers import (
    build_comparison_density,
    run_sampler_comparison,
)
from wave_sampling.demos.modifier_effects import build_modifier_density_variants


def test_sampler_comparison_reports_deterministic_and_feasible() -> None:
    n_points = 40
    density_field = build_comparison_density(nx=40, ny=30, n_points=n_points)

    comparison = run_sampler_comparison(density_field=density_field, n_points=n_points)

    for method_name in (
        "density_weighted_farthest_point",
        "density_weighted_lloyd_cvt",
        "density_weighted_optimal_transport",
    ):
        values = comparison[method_name]
        assert values["n_selected"] == n_points
        assert values["deterministic_repeat"] is True
        assert values["hard_constraint_violations"] == 0
        assert np.isfinite(values["nn_mean_m"])
        assert np.isfinite(values["density_l1"])
        assert np.isfinite(values["density_l2"])


def test_modifier_variants_preserve_mass_and_hard_constraints() -> None:
    n_points = 55
    variants = build_modifier_density_variants(
        n_points=n_points,
        nx=50,
        ny=35,
        variant_names=[
            "sharp_step_d1_d2",
            "clipped_linear_transition_d1_d2",
            "smoothstep_transition_d1_d2",
            "smootherstep_transition_d1_d2",
        ],
    )

    assert len(variants) == 4

    first_mask = None
    density_arrays = []
    for field in variants.values():
        assert np.isclose(
            field.integrated_mass, float(n_points), rtol=1e-10, atol=1e-10
        )
        assert np.all(field.density[~field.feasible_mask] == 0.0)

        if first_mask is None:
            first_mask = field.feasible_mask
        else:
            assert np.array_equal(field.feasible_mask, first_mask)

        density_arrays.append(field.density)

    # Different soft modifiers should produce at least one distinct density field.
    assert any(
        not np.allclose(density_arrays[0], other) for other in density_arrays[1:]
    )


def test_modifier_variants_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown variant_names"):
        build_modifier_density_variants(
            n_points=10,
            variant_names=["sharp_step_d1_d2", "does_not_exist"],
        )


def test_modifier_variants_negative_n_points_raises() -> None:
    with pytest.raises(ValueError, match="n_points must be >= 0"):
        build_modifier_density_variants(n_points=-1)


def test_modifier_variants_zero_points_returns_zero_mass() -> None:
    variants = build_modifier_density_variants(
        n_points=0,
        variant_names=["sharp_step_d1_d2", "smoothstep_transition_d1_d2"],
    )

    for field in variants.values():
        assert field.integrated_mass == 0.0
        assert np.all(field.density == 0.0)


def test_transition_variants_sharp_vs_smooth_levels() -> None:
    variants = build_modifier_density_variants(
        n_points=40,
        nx=55,
        ny=30,
        variant_names=[
            "sharp_step_d1_d2",
            "clipped_linear_transition_d1_d2",
            "smoothstep_transition_d1_d2",
            "smootherstep_transition_d1_d2",
        ],
    )

    sharp = variants["sharp_step_d1_d2"]
    clipped = variants["clipped_linear_transition_d1_d2"]
    smooth = variants["smoothstep_transition_d1_d2"]
    smoother = variants["smootherstep_transition_d1_d2"]

    sharp_unique = np.unique(np.round(sharp.density[sharp.feasible_mask], decimals=12))
    clipped_unique = np.unique(
        np.round(clipped.density[clipped.feasible_mask], decimals=12)
    )
    smooth_unique = np.unique(
        np.round(smooth.density[smooth.feasible_mask], decimals=12)
    )
    smoother_unique = np.unique(
        np.round(smoother.density[smoother.feasible_mask], decimals=12)
    )

    # Sharp two-region transition should produce only two density levels on feasible cells.
    assert sharp_unique.size == 2
    # Smooth transition variants should create multiple intermediate levels.
    assert clipped_unique.size > 2
    assert smooth_unique.size > 2
    assert smoother_unique.size > 2


def test_modifier_variants_invalid_density_levels_raise() -> None:
    with pytest.raises(ValueError, match="d1 and d2 must be strictly positive"):
        build_modifier_density_variants(n_points=10, d1=0.0, d2=2.0)
