from __future__ import annotations

import numpy as np

from wave_sampling.density.field import DensityField
from wave_sampling.density.modifiers import (
    compose_positive_factors,
    gaussian_weight,
    smootherstep,
)
from wave_sampling.diagnostics.metrics import compute_sampling_diagnostics
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt


def build_comparison_density(
    nx: int = 120,
    ny: int = 80,
    n_points: int = 400,
) -> DensityField:
    """Build a synthetic DensityField for side-by-side sampler comparison.

    Coordinates are in metres and represent candidate locations on a discrete grid.
    """
    x_m = np.linspace(0.0, 120_000.0, nx)
    y_m = np.linspace(0.0, 80_000.0, ny)
    xx_m, yy_m = np.meshgrid(x_m, y_m)
    coordinates_m = np.column_stack([xx_m.ravel(), yy_m.ravel()])

    # Hard forbidden region (circular exclusion): cannot be sampled.
    cx_m, cy_m, radius_m = 70_000.0, 40_000.0, 12_000.0
    dist2_center_m2 = (coordinates_m[:, 0] - cx_m) ** 2 + (
        coordinates_m[:, 1] - cy_m
    ) ** 2
    feasible_mask = dist2_center_m2 > radius_m**2

    base = np.ones(coordinates_m.shape[0], dtype=float)

    dx_hi_m = coordinates_m[:, 0] - 32_000.0
    dy_hi_m = coordinates_m[:, 1] - 58_000.0
    high_preference = 1.0 + 3.0 * np.exp(
        -0.5 * ((dx_hi_m / 15_000.0) ** 2 + (dy_hi_m / 11_000.0) ** 2)
    )

    radial_lr_m = np.sqrt(
        (coordinates_m[:, 0] - 100_000.0) ** 2 + (coordinates_m[:, 1] - 14_000.0) ** 2
    )
    low_region = gaussian_weight(radial_lr_m, center=0.0, sigma=20_000.0)
    low_preference = 1.0 - 0.75 * low_region

    north_preference = 0.65 + 0.35 * smootherstep(
        coordinates_m[:, 1],
        12_000.0,
        72_000.0,
    )

    shape_values = compose_positive_factors(
        base,
        high_preference,
        low_preference,
        north_preference,
    )

    return DensityField.from_shape(
        coordinates=coordinates_m,
        shape_values=shape_values,
        feasible_mask=feasible_mask,
        n_points=n_points,
    )


def run_sampler_comparison(
    density_field: DensityField,
    n_points: int,
    max_iterations: int = 30,
) -> dict[str, dict[str, float | int | bool]]:
    """Run both samplers and return compact comparison diagnostics."""
    fp_a = density_weighted_farthest_point(density_field, n_points=n_points)
    fp_b = density_weighted_farthest_point(density_field, n_points=n_points)

    cvt_a = density_weighted_lloyd_cvt(
        density_field,
        n_points=n_points,
        max_iterations=max_iterations,
    )
    cvt_b = density_weighted_lloyd_cvt(
        density_field,
        n_points=n_points,
        max_iterations=max_iterations,
    )

    diag_fp = compute_sampling_diagnostics(fp_a, density_field)
    diag_cvt = compute_sampling_diagnostics(cvt_a, density_field)

    return {
        "density_weighted_farthest_point": {
            "n_selected": fp_a.n_selected,
            "deterministic_repeat": bool(
                np.array_equal(fp_a.selected_indices, fp_b.selected_indices)
            ),
            "hard_constraint_violations": int(diag_fp["hard_constraint_violations"]),
            "nn_mean_m": float(diag_fp["nearest_neighbour"]["mean"]),
            "density_l1": float(diag_fp["density_reproduction"]["normalized_l1_error"]),
            "density_l2": float(diag_fp["density_reproduction"]["normalized_l2_error"]),
        },
        "density_weighted_lloyd_cvt": {
            "n_selected": cvt_a.n_selected,
            "deterministic_repeat": bool(
                np.array_equal(cvt_a.selected_indices, cvt_b.selected_indices)
            ),
            "hard_constraint_violations": int(diag_cvt["hard_constraint_violations"]),
            "nn_mean_m": float(diag_cvt["nearest_neighbour"]["mean"]),
            "density_l1": float(
                diag_cvt["density_reproduction"]["normalized_l1_error"]
            ),
            "density_l2": float(
                diag_cvt["density_reproduction"]["normalized_l2_error"]
            ),
            "iterations": int(cvt_a.density_summary.get("iterations", 0)),
        },
    }


def main() -> None:
    n_points = 400
    density_field = build_comparison_density(n_points=n_points)
    comparison = run_sampler_comparison(density_field, n_points=n_points)

    print("Sampler comparison on identical DensityField")
    print(f"Target integrated mass: {density_field.integrated_mass:.6f} points")
    for method_name, values in comparison.items():
        print(f"\\n{method_name}")
        print(f"  n_selected: {values['n_selected']}")
        print(f"  deterministic_repeat: {values['deterministic_repeat']}")
        print(f"  hard_constraint_violations: {values['hard_constraint_violations']}")
        print(f"  nn_mean_m: {values['nn_mean_m']:.3f}")
        print(f"  density_l1: {values['density_l1']:.6f}")
        print(f"  density_l2: {values['density_l2']:.6f}")
        if "iterations" in values:
            print(f"  iterations: {values['iterations']}")

    fp = density_weighted_farthest_point(density_field, n_points=n_points)
    cvt = density_weighted_lloyd_cvt(density_field, n_points=n_points)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting this example. "
            "Install with: /bin/python3 -m pip install -e .[plot]"
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    for ax, result, title in (
        (axes[0], fp, "density_weighted_farthest_point"),
        (axes[1], cvt, "density_weighted_lloyd_cvt"),
    ):
        sc = ax.scatter(
            density_field.coordinates[:, 0],
            density_field.coordinates[:, 1],
            c=density_field.density,
            s=7,
            cmap="viridis",
            alpha=0.8,
        )
        ax.scatter(
            result.selected_coordinates[:, 0],
            result.selected_coordinates[:, 1],
            s=12,
            c="crimson",
            alpha=0.9,
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")

    axes[0].set_ylabel("y (m)")
    fig.colorbar(sc, ax=axes, label="Target density (points / area)")
    fig.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = "examples/compare_samplers.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
