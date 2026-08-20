from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import numpy.typing as npt

from wave_sampling.diagnostics.metrics import density_reproduction_metrics
from wave_sampling.result import SamplingResult
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt

from .config import BENCHMARK_MODES, BenchmarkConfig
from .density import DensityBuildOutput, build_density
from .domain import SyntheticDomain, build_synthetic_domain
from .plots import save_composite_figure


@dataclass(frozen=True)
class MethodReport:
    method: str
    requested_n: int
    actual_n: int
    runtime_s: float
    infeasible_selected: int
    duplicate_count: int
    normalized_l1_error: float
    normalized_l2_error: float
    density_correlation: float
    regional_mass_error_l1: float
    regional_mass_error_max: float
    nn_min_m: float
    nn_median_m: float
    nn_mean_m: float
    nn_std_m: float
    nn_cv: float
    spacing_correlation: float
    reproducible_repeat: bool


def _nearest_neighbour_distances(
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if points.shape[0] < 2:
        return np.array([], dtype=float)
    diffs = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1)


def _regional_errors(
    selected_indices: npt.NDArray[np.int64],
    density_build: DensityBuildOutput,
    density_field_mass: float,
) -> tuple[float, float]:
    m = density_build.density_field.n_candidates
    counts = np.bincount(selected_indices, minlength=m).astype(float)

    l1 = 0.0
    max_abs = 0.0
    for region, target_fraction in density_build.region_targets.items():
        membership = density_build.region_memberships[region]
        observed = float(np.sum(counts * membership))
        target = target_fraction * density_field_mass
        abs_err = abs(observed - target)
        l1 += abs_err
        max_abs = max(max_abs, abs_err)

    return l1 / density_field_mass, max_abs / density_field_mass


def _spacing_correlation(
    selected_indices: npt.NDArray[np.int64],
    density_build: DensityBuildOutput,
) -> float:
    points = density_build.density_field.coordinates[selected_indices]
    nn = _nearest_neighbour_distances(points)
    if nn.size < 2:
        return float("nan")

    rho_at_selected = density_build.density_field.density[selected_indices]
    expected = 1.0 / np.sqrt(np.maximum(rho_at_selected, 1e-15))

    if np.std(nn) == 0.0 or np.std(expected) == 0.0:
        return float("nan")
    return float(np.corrcoef(nn, expected)[0, 1])


def _build_random_result(
    indices: npt.NDArray[np.int64],
    density_build: DensityBuildOutput,
    method: str,
    seed: int,
) -> SamplingResult:
    field = density_build.density_field
    return SamplingResult(
        selected_indices=indices.astype(np.int64),
        selected_coordinates=field.coordinates[indices],
        method=method,
        requested_n_points=density_build.density_field.integrated_mass.__int__(),
        seed=seed,
        density_summary={
            "n_candidates": field.n_candidates,
            "n_feasible": int(np.sum(field.feasible_mask)),
            "n_eligible": int(np.sum(field.eligible_mask)),
            "target_mass": field.integrated_mass,
        },
    )


def _run_method(
    method_name: str,
    density_build: DensityBuildOutput,
    cfg: BenchmarkConfig,
) -> tuple[SamplingResult, MethodReport]:
    field = density_build.density_field
    start = time.perf_counter()

    if method_name == "density_weighted_farthest_point":
        result = density_weighted_farthest_point(field, n_points=cfg.n_points)
        repeat = density_weighted_farthest_point(field, n_points=cfg.n_points)
        reproducible = bool(
            np.array_equal(result.selected_indices, repeat.selected_indices)
        )
    elif method_name == "density_weighted_lloyd_cvt":
        result = density_weighted_lloyd_cvt(
            field,
            n_points=cfg.n_points,
            max_iterations=cfg.cvt_max_iterations,
        )
        repeat = density_weighted_lloyd_cvt(
            field,
            n_points=cfg.n_points,
            max_iterations=cfg.cvt_max_iterations,
        )
        reproducible = bool(
            np.array_equal(result.selected_indices, repeat.selected_indices)
        )
    elif method_name == "uniform_random":
        rng = np.random.default_rng(cfg.random_seed)
        eligible = np.flatnonzero(field.feasible_mask)
        idx = rng.choice(eligible, size=cfg.n_points, replace=False)
        result = _build_random_result(idx, density_build, method_name, cfg.random_seed)
        rng_rep = np.random.default_rng(cfg.random_seed)
        idx_rep = rng_rep.choice(eligible, size=cfg.n_points, replace=False)
        reproducible = bool(np.array_equal(np.sort(idx), np.sort(idx_rep)))
    elif method_name == "target_density_random":
        rng = np.random.default_rng(cfg.random_seed)
        eligible = np.flatnonzero(field.eligible_mask)
        probs = field.density[eligible] / np.sum(field.density[eligible])
        idx = rng.choice(eligible, size=cfg.n_points, replace=False, p=probs)
        result = _build_random_result(idx, density_build, method_name, cfg.random_seed)
        rng_rep = np.random.default_rng(cfg.random_seed)
        idx_rep = rng_rep.choice(eligible, size=cfg.n_points, replace=False, p=probs)
        reproducible = bool(np.array_equal(np.sort(idx), np.sort(idx_rep)))
    else:
        raise ValueError(f"Unknown method: {method_name}")

    runtime_s = time.perf_counter() - start

    density_metrics = density_reproduction_metrics(
        result.selected_indices,
        field.density,
        field.feasible_mask,
        field.cell_areas,
    )

    infeasible = int(np.sum(~field.feasible_mask[result.selected_indices]))
    duplicates = int(result.n_selected - np.unique(result.selected_indices).size)

    nn = _nearest_neighbour_distances(result.selected_coordinates)
    if nn.size == 0:
        nn_min = nn_median = nn_mean = nn_std = nn_cv = float("nan")
    else:
        nn_min = float(np.min(nn))
        nn_median = float(np.median(nn))
        nn_mean = float(np.mean(nn))
        nn_std = float(np.std(nn))
        nn_cv = float(nn_std / nn_mean) if nn_mean > 0.0 else float("nan")

    regional_l1, regional_max = _regional_errors(
        result.selected_indices,
        density_build,
        density_field_mass=float(cfg.n_points),
    )

    report = MethodReport(
        method=method_name,
        requested_n=cfg.n_points,
        actual_n=result.n_selected,
        runtime_s=runtime_s,
        infeasible_selected=infeasible,
        duplicate_count=duplicates,
        normalized_l1_error=float(density_metrics["normalized_l1_error"]),
        normalized_l2_error=float(density_metrics["normalized_l2_error"]),
        density_correlation=float(density_metrics["correlation"]),
        regional_mass_error_l1=regional_l1,
        regional_mass_error_max=regional_max,
        nn_min_m=nn_min,
        nn_median_m=nn_median,
        nn_mean_m=nn_mean,
        nn_std_m=nn_std,
        nn_cv=nn_cv,
        spacing_correlation=_spacing_correlation(
            result.selected_indices, density_build
        ),
        reproducible_repeat=reproducible,
    )

    return result, report


def run_benchmark(
    cfg: BenchmarkConfig | None = None,
    modes: tuple[str, ...] = BENCHMARK_MODES,
    output_dir: str | None = None,
    generate_plots: bool = True,
) -> dict[str, object]:
    cfg = cfg or BenchmarkConfig()

    domain, domain_meta = build_synthetic_domain(cfg)

    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "outputs")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, object] = {
        "config": asdict(cfg),
        "domain": {
            "n_cells": domain.n_cells,
            "feasible_count": int(np.sum(domain.feasible_mask)),
            "cell_area_m2": domain.cell_area_m2,
            "metadata": domain_meta,
        },
        "modes": {},
    }

    for mode in modes:
        density_build = build_density(domain, cfg, mode=mode)

        methods = (
            "density_weighted_farthest_point",
            "density_weighted_lloyd_cvt",
            "uniform_random",
            "target_density_random",
        )

        method_reports: dict[str, MethodReport] = {}
        method_results: dict[str, SamplingResult] = {}
        for method in methods:
            result, report = _run_method(method, density_build, cfg)
            method_results[method] = result
            method_reports[method] = report

            if report.infeasible_selected != 0:
                raise RuntimeError(
                    f"{method} selected infeasible points in mode {mode}"
                )

        figure_path = output_path / f"benchmark_{mode}.png"
        if generate_plots:
            save_composite_figure(
                domain=domain,
                density_build=density_build,
                farthest_result=method_results["density_weighted_farthest_point"],
                cvt_result=method_results["density_weighted_lloyd_cvt"],
                reports={k: asdict(v) for k, v in method_reports.items()},
                figure_path=str(figure_path),
            )

        all_results["modes"][mode] = {
            "region_targets": density_build.region_targets,
            "region_masses": density_build.region_masses,
            "methods": {k: asdict(v) for k, v in method_reports.items()},
            "figure": str(figure_path) if generate_plots else None,
        }

    return all_results


def _format_mode_summary(mode_name: str, mode_payload: dict[str, object]) -> str:
    lines = [
        f"\n[{mode_name}]",
        "method                         L1_err   NN_mean_km  NN_cv   runtime_s",
    ]
    methods = mode_payload["methods"]

    for method_name in (
        "density_weighted_farthest_point",
        "density_weighted_lloyd_cvt",
        "uniform_random",
        "target_density_random",
    ):
        rep = methods[method_name]
        lines.append(
            f"{method_name:28s} {rep['normalized_l1_error']:7.3f} "
            f"{rep['nn_mean_m'] / 1000.0:10.3f} {rep['nn_cv']:7.3f} {rep['runtime_s']:10.3f}"
        )

    return "\n".join(lines)


def main() -> None:
    cfg = BenchmarkConfig()
    results = run_benchmark(cfg=cfg)

    print("Synthetic Nordic/Arctic benchmark")
    print(f"Requested N: {cfg.n_points}")
    print(f"Total grid cells: {results['domain']['n_cells']}")
    print(f"Feasible cells: {results['domain']['feasible_count']}")

    for mode_name in BENCHMARK_MODES:
        print(_format_mode_summary(mode_name, results["modes"][mode_name]))
        figure = results["modes"][mode_name]["figure"]
        print(f"figure: {figure}")


if __name__ == "__main__":
    main()
