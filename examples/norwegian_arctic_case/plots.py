from __future__ import annotations

from typing import Any

import numpy as np

from wave_sampling.result import SamplingResult

from .density import DensityBuildOutput
from .domain import SyntheticDomain


def _empirical_density(
    selected_indices: np.ndarray,
    n_cells: int,
    cell_area_m2: float,
) -> np.ndarray:
    counts = np.bincount(selected_indices, minlength=n_cells).astype(float)
    return counts / cell_area_m2


def _nearest_neighbour_distances(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.array([], dtype=float)
    diffs = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1)


def save_composite_figure(
    domain: SyntheticDomain,
    density_build: DensityBuildOutput,
    farthest_result: SamplingResult,
    cvt_result: SamplingResult,
    reports: dict[str, dict[str, Any]],
    figure_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting this benchmark. "
            "Install with: /bin/python3 -m pip install -e .[plot]"
        ) from exc

    shape2d = domain.lon_deg.shape
    field = density_build.density_field

    rho2d = field.density.reshape(shape2d)
    feasible2d = domain.feasible_mask.astype(float)
    spacing2d = density_build.intermediates["desired_spacing_m"].reshape(shape2d)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    axs = axes.ravel()

    ax0 = axs[0]
    ax0.set_title("Synthetic Geography and Feasible Domain")
    ax0.contourf(
        domain.x_m,
        domain.y_m,
        domain.land_mask.astype(int),
        levels=[-0.5, 0.5, 1.5],
        colors=["#6fa36b", "#4f5d75"],
        alpha=0.9,
    )
    ax0.contour(
        domain.x_m, domain.y_m, feasible2d, levels=[0.5], colors="white", linewidths=1.0
    )
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")

    ax1 = axs[1]
    im1 = ax1.pcolormesh(domain.x_m, domain.y_m, rho2d, cmap="viridis", shading="auto")
    ax1.set_title("Target Density rho (points/m^2)")
    fig.colorbar(im1, ax=ax1)
    ax1.set_xlabel("x (m)")

    ax2 = axs[2]
    im2 = ax2.pcolormesh(
        domain.x_m, domain.y_m, spacing2d / 1000.0, cmap="magma_r", shading="auto"
    )
    ax2.set_title("Desired Spacing 1/sqrt(rho) (km)")
    fig.colorbar(im2, ax=ax2)
    ax2.set_xlabel("x (m)")

    ax3 = axs[3]
    ax3.pcolormesh(
        domain.x_m,
        domain.y_m,
        np.where(domain.feasible_mask, 1.0, np.nan),
        cmap="Greys",
        shading="auto",
        alpha=0.25,
    )
    ax3.scatter(
        farthest_result.selected_coordinates[:, 0],
        farthest_result.selected_coordinates[:, 1],
        s=11,
        c="#d84b2a",
    )
    ax3.set_title("Farthest Point Selection (projected m)")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = axs[4]
    ax4.pcolormesh(
        domain.x_m,
        domain.y_m,
        np.where(domain.feasible_mask, 1.0, np.nan),
        cmap="Greys",
        shading="auto",
        alpha=0.25,
    )
    ax4.scatter(
        cvt_result.selected_coordinates[:, 0],
        cvt_result.selected_coordinates[:, 1],
        s=11,
        c="#1f7a8c",
    )
    ax4.set_title("Lloyd/CVT Selection (projected m)")
    ax4.set_xlabel("x (m)")

    ax5 = axs[5]
    emp_fp = _empirical_density(
        farthest_result.selected_indices, field.n_candidates, domain.cell_area_m2
    )
    emp_cvt = _empirical_density(
        cvt_result.selected_indices, field.n_candidates, domain.cell_area_m2
    )
    rho = field.density

    ax5.scatter(rho, emp_fp, s=6, alpha=0.25, label="farthest")
    ax5.scatter(rho, emp_cvt, s=6, alpha=0.25, label="cvt")
    low = 0.0
    high = max(np.max(rho), np.max(emp_fp), np.max(emp_cvt))
    ax5.plot([low, high], [low, high], "k--", linewidth=1.0)
    ax5.set_title("Target vs Empirical Density")
    ax5.set_xlabel("target rho")
    ax5.set_ylabel("empirical rho")
    ax5.legend(loc="upper left")

    ax6 = axs[6]
    nn_fp = _nearest_neighbour_distances(farthest_result.selected_coordinates)
    nn_cvt = _nearest_neighbour_distances(cvt_result.selected_coordinates)
    ax6.hist(nn_fp / 1000.0, bins=24, alpha=0.65, label="farthest")
    ax6.hist(nn_cvt / 1000.0, bins=24, alpha=0.65, label="cvt")
    ax6.set_title("Nearest-Neighbour Distance Distributions")
    ax6.set_xlabel("distance (km)")
    ax6.legend(loc="upper right")

    ax7 = axs[7]
    keys = [
        "NorwegianSea",
        "BarentsSea",
        "NorthSea",
        "BalticSea",
        "Arctic",
        "Boundary",
        "Rest",
    ]
    targets = [density_build.region_targets[k] for k in keys]
    masses = [density_build.region_masses[k] / field.integrated_mass for k in keys]
    x = np.arange(len(keys))
    width = 0.4
    ax7.bar(x - width / 2, targets, width=width, label="target")
    ax7.bar(x + width / 2, masses, width=width, label="density")
    ax7.set_xticks(x)
    ax7.set_xticklabels(keys, rotation=30, ha="right")
    ax7.set_ylim(0.0, max(max(targets), max(masses)) * 1.35)
    ax7.set_title("Regional Mass Fractions")
    ax7.legend(loc="upper right")

    ax8 = axs[8]
    ax8.axis("off")
    text = (
        f"Hard violations fp={reports['density_weighted_farthest_point']['infeasible_selected']}\n"
        f"Hard violations cvt={reports['density_weighted_lloyd_cvt']['infeasible_selected']}\n"
        f"L1 fp={reports['density_weighted_farthest_point']['normalized_l1_error']:.3f}\n"
        f"L1 cvt={reports['density_weighted_lloyd_cvt']['normalized_l1_error']:.3f}\n"
        f"NN cv fp={reports['density_weighted_farthest_point']['nn_cv']:.3f}\n"
        f"NN cv cvt={reports['density_weighted_lloyd_cvt']['nn_cv']:.3f}"
    )
    ax8.text(0.05, 0.95, text, va="top", ha="left", fontsize=11)

    fig.savefig(figure_path, dpi=170)
