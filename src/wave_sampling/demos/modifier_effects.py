from __future__ import annotations

import numpy as np

from wave_sampling.density.field import DensityField
from wave_sampling.density.modifiers import (
    clipped_linear_transition,
    compose_positive_factors,
    smoothstep,
    smootherstep,
)


def build_modifier_density_variants(
    n_points: int,
    nx: int = 110,
    ny: int = 75,
    d1: float = 0.6,
    d2: float = 2.4,
    variant_names: list[str] | None = None,
) -> dict[str, DensityField]:
    """Create transition-based DensityField variants from soft modifiers.

    The hard feasible mask is shared by all variants.

    Each variant starts from the same conceptual two-region preference with
    region weights d1 and d2 and differs only by how the transition between
    regions is represented (sharp step, clipped linear, smoothstep, smootherstep)
    before normalization to the same integrated mass.
    """
    if n_points < 0:
        raise ValueError("n_points must be >= 0")
    if d1 <= 0.0 or d2 <= 0.0:
        raise ValueError("d1 and d2 must be strictly positive")

    x_m = np.linspace(0.0, 140_000.0, nx)
    y_m = np.linspace(0.0, 90_000.0, ny)
    xx_m, yy_m = np.meshgrid(x_m, y_m)
    coordinates_m = np.column_stack([xx_m.ravel(), yy_m.ravel()])

    hole_cx_m, hole_cy_m, hole_r_m = 72_000.0, 40_000.0, 11_000.0
    dist2_hole_m2 = (coordinates_m[:, 0] - hole_cx_m) ** 2 + (
        coordinates_m[:, 1] - hole_cy_m
    ) ** 2
    hard_hole = dist2_hole_m2 <= hole_r_m**2
    hard_corner = (coordinates_m[:, 0] < 12_000.0) & (coordinates_m[:, 1] < 12_000.0)
    feasible_mask = ~(hard_hole | hard_corner)

    x_coord_m = coordinates_m[:, 0]

    # Transition from region-1 weight d1 (west side) to region-2 weight d2 (east side).
    sharp_boundary_m = 70_000.0
    transition_start_m = 52_000.0
    transition_end_m = 88_000.0

    base = np.ones(coordinates_m.shape[0], dtype=float)

    sharp_step = np.where(x_coord_m < sharp_boundary_m, d1, d2)
    linear_transition = d1 + (d2 - d1) * clipped_linear_transition(
        x_coord_m,
        transition_start_m,
        transition_end_m,
    )
    smooth_transition = d1 + (d2 - d1) * smoothstep(
        x_coord_m,
        transition_start_m,
        transition_end_m,
    )
    smoother_transition = d1 + (d2 - d1) * smootherstep(
        x_coord_m,
        transition_start_m,
        transition_end_m,
    )

    variant_shape_values: dict[str, np.ndarray] = {
        "sharp_step_d1_d2": compose_positive_factors(base, sharp_step),
        "clipped_linear_transition_d1_d2": compose_positive_factors(
            base,
            linear_transition,
        ),
        "smoothstep_transition_d1_d2": compose_positive_factors(
            base,
            smooth_transition,
        ),
        "smootherstep_transition_d1_d2": compose_positive_factors(
            base,
            smoother_transition,
        ),
    }

    if variant_names is None:
        selected_names = list(variant_shape_values.keys())
    else:
        selected_names = list(variant_names)

    unknown = [name for name in selected_names if name not in variant_shape_values]
    if unknown:
        raise ValueError(f"Unknown variant_names: {unknown}")

    variants: dict[str, DensityField] = {}
    for name in selected_names:
        variants[name] = DensityField.from_shape(
            coordinates=coordinates_m,
            shape_values=variant_shape_values[name],
            feasible_mask=feasible_mask,
            n_points=n_points,
        )

    return variants


def main() -> None:
    n_points = 400
    variants = build_modifier_density_variants(n_points=n_points)

    print(
        "Density modifier variants (same hard constraints, different soft preferences)"
    )
    for name, field in variants.items():
        print(
            f"{name}: integrated_mass={field.integrated_mass:.6f}, "
            f"feasible_count={int(np.sum(field.feasible_mask))}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting this example. "
            "Install with: /bin/python3 -m pip install -e .[plot]"
        ) from exc

    n_cols = 3
    n_rows = int(np.ceil(len(variants) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4.5 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, (name, field) in zip(axes_arr, variants.items()):
        sc = ax.scatter(
            field.coordinates[:, 0],
            field.coordinates[:, 1],
            c=field.density,
            s=7,
            cmap="viridis",
            alpha=0.85,
        )
        ax.set_title(name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    for ax in axes_arr[len(variants) :]:
        ax.axis("off")

    fig.colorbar(sc, ax=axes_arr.tolist(), label="Target density (points / area)")
    fig.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = "examples/modifier_effects.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
