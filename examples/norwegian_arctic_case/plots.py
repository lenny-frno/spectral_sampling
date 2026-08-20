from __future__ import annotations

from typing import Any

import numpy as np

from wave_sampling.result import SamplingResult
from wave_sampling.samplers.poisson_disk import density_to_nominal_radius

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
    poisson_result: SamplingResult,
    reports: dict[str, dict[str, Any]],
    poisson_spacing_scale: float,
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

    fig, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
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
    ax5.pcolormesh(
        domain.x_m,
        domain.y_m,
        np.where(domain.feasible_mask, 1.0, np.nan),
        cmap="Greys",
        shading="auto",
        alpha=0.25,
    )
    ax5.scatter(
        poisson_result.selected_coordinates[:, 0],
        poisson_result.selected_coordinates[:, 1],
        s=11,
        c="#2a9d8f",
    )
    ax5.set_title("Poisson-Disk Selection (projected m)")
    ax5.set_xlabel("x (m)")

    ax6 = axs[6]
    emp_fp = _empirical_density(
        farthest_result.selected_indices, field.n_candidates, domain.cell_area_m2
    )
    emp_cvt = _empirical_density(
        cvt_result.selected_indices, field.n_candidates, domain.cell_area_m2
    )
    emp_poisson = _empirical_density(
        poisson_result.selected_indices, field.n_candidates, domain.cell_area_m2
    )
    rho = field.density

    ax6.scatter(rho, emp_fp, s=6, alpha=0.25, label="farthest")
    ax6.scatter(rho, emp_cvt, s=6, alpha=0.25, label="cvt")
    ax6.scatter(rho, emp_poisson, s=6, alpha=0.25, label="poisson")
    low = 0.0
    high = max(np.max(rho), np.max(emp_fp), np.max(emp_cvt), np.max(emp_poisson))
    ax6.plot([low, high], [low, high], "k--", linewidth=1.0)
    ax6.set_title("Target vs Empirical Density")
    ax6.set_xlabel("target rho")
    ax6.set_ylabel("empirical rho")
    ax6.legend(loc="upper left")

    ax7 = axs[7]
    nn_fp = _nearest_neighbour_distances(farthest_result.selected_coordinates)
    nn_cvt = _nearest_neighbour_distances(cvt_result.selected_coordinates)
    nn_poisson = _nearest_neighbour_distances(poisson_result.selected_coordinates)
    ax7.hist(nn_fp / 1000.0, bins=24, alpha=0.65, label="farthest")
    ax7.hist(nn_cvt / 1000.0, bins=24, alpha=0.65, label="cvt")
    ax7.hist(nn_poisson / 1000.0, bins=24, alpha=0.65, label="poisson")
    ax7.set_title("Nearest-Neighbour Distance Distributions")
    ax7.set_xlabel("distance (km)")
    ax7.legend(loc="upper right")

    ax8 = axs[8]
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
    ax8.bar(x - width / 2, targets, width=width, label="target")
    ax8.bar(x + width / 2, masses, width=width, label="density")
    ax8.set_xticks(x)
    ax8.set_xticklabels(keys, rotation=30, ha="right")
    ax8.set_ylim(0.0, max(max(targets), max(masses)) * 1.35)
    ax8.set_title("Regional Mass Fractions")
    ax8.legend(loc="upper right")

    ax9 = axs[9]
    ax9.axis("off")
    text = (
        f"Hard violations fp={reports['density_weighted_farthest_point']['infeasible_selected']}\n"
        f"Hard violations cvt={reports['density_weighted_lloyd_cvt']['infeasible_selected']}\n"
        f"Hard violations pd={reports['density_adapted_poisson_disk']['infeasible_selected']}\n"
        f"L1 fp={reports['density_weighted_farthest_point']['normalized_l1_error']:.3f}\n"
        f"L1 cvt={reports['density_weighted_lloyd_cvt']['normalized_l1_error']:.3f}\n"
        f"L1 pd={reports['density_adapted_poisson_disk']['normalized_l1_error']:.3f}\n"
        f"NN cv fp={reports['density_weighted_farthest_point']['nn_cv']:.3f}\n"
        f"NN cv cvt={reports['density_weighted_lloyd_cvt']['nn_cv']:.3f}\n"
        f"NN cv pd={reports['density_adapted_poisson_disk']['nn_cv']:.3f}\n"
        f"PD sep violations={reports['density_adapted_poisson_disk']['separation_violations']}"
    )
    ax9.text(0.05, 0.95, text, va="top", ha="left", fontsize=11)

    ax10 = axs[10]
    radius = density_to_nominal_radius(
        density=field.density,
        feasible_mask=field.feasible_mask,
        spacing_scale=poisson_spacing_scale,
    )
    radius_plot = np.where(
        field.feasible_mask,
        radius,
        np.nan,
    ).reshape(shape2d)
    im10 = ax10.pcolormesh(
        domain.x_m,
        domain.y_m,
        radius_plot / 1000.0,
        cmap="plasma",
        shading="auto",
    )
    ax10.set_title("Poisson Nominal Radius (km)")
    ax10.set_xlabel("x (m)")
    fig.colorbar(im10, ax=ax10)

    ax11 = axs[11]
    ax11.axis("off")

    fig.savefig(figure_path, dpi=170)
