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


def build_synthetic_density(
    nx: int = 120, ny: int = 80, n_points: int = 400
) -> DensityField:
    x = np.linspace(0.0, 120_000.0, nx)
    y = np.linspace(0.0, 80_000.0, ny)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Hard forbidden area: circular exclusion zone.
    cx, cy, r = 70_000.0, 40_000.0, 12_000.0
    d2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
    feasible = d2 > r**2

    base = np.ones(coords.shape[0], dtype=float)

    # Smooth high-density area of interest.
    dx_hi = coords[:, 0] - 30_000.0
    dy_hi = coords[:, 1] - 55_000.0
    gauss_hi = np.exp(-0.5 * ((dx_hi / 15_000.0) ** 2 + (dy_hi / 10_000.0) ** 2))
    high_factor = 1.0 + 3.0 * gauss_hi

    # Smooth low-density region near lower-right.
    radial = np.sqrt((coords[:, 0] - 98_000.0) ** 2 + (coords[:, 1] - 15_000.0) ** 2)
    low_region = gaussian_weight(radial, center=0.0, sigma=20_000.0)
    low_factor = 1.0 - 0.8 * low_region

    # Extra smooth northward preference.
    north_pref = 0.6 + 0.4 * smootherstep(coords[:, 1], 10_000.0, 70_000.0)

    q = compose_positive_factors(base, high_factor, low_factor, north_pref)

    return DensityField.from_shape(
        coordinates=coords,
        shape_values=q,
        feasible_mask=feasible,
        n_points=n_points,
    )


def main() -> None:
    n_points = 400
    field = build_synthetic_density(n_points=n_points)

    result_a = density_weighted_farthest_point(field, n_points=n_points)
    result_b = density_weighted_farthest_point(field, n_points=n_points)

    same = np.array_equal(result_a.selected_indices, result_b.selected_indices)
    print(f"Deterministic repeatable output: {same}")

    diagnostics = compute_sampling_diagnostics(result_a, field)
    print("Hard constraint violations:", diagnostics["hard_constraint_violations"])
    print("Nearest-neighbour mean:", diagnostics["nearest_neighbour"]["mean"])
    print(
        "Density normalized L1:",
        diagnostics["density_reproduction"]["normalized_l1_error"],
    )

    if not np.all(field.feasible_mask[result_a.selected_indices]):
        raise RuntimeError("Infeasible point was selected")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting this example. "
            "Install with: /bin/python3 -m pip install -e .[plot]"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        field.coordinates[:, 0],
        field.coordinates[:, 1],
        c=field.density,
        s=8,
        cmap="viridis",
        alpha=0.85,
    )
    ax.scatter(
        result_a.selected_coordinates[:, 0],
        result_a.selected_coordinates[:, 1],
        s=18,
        c="crimson",
        label="Selected points",
    )
    ax.set_title("Synthetic Target Density and Deterministic Selected Points")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, label="Target density (points / area)")
    fig.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = "examples/basic_density_sampling.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
