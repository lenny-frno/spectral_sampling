from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from wave_sampling.density.field import DensityField
from wave_sampling.density.modifiers import gaussian_weight, smootherstep

from .config import ARCTIC_MODES, BENCHMARK_MODES, DENSITY_MODES, BenchmarkConfig
from .domain import SyntheticDomain

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class DensityBuildOutput:
    density_field: DensityField
    region_memberships: dict[str, FloatArray]
    region_targets: dict[str, float]
    region_masses: dict[str, float]
    intermediates: dict[str, FloatArray]


def _point_in_polygon(
    lon: FloatArray, lat: FloatArray, polygon: list[tuple[float, float]]
) -> BoolArray:
    """Ray-casting point-in-polygon for vectorized lon/lat arrays."""
    x = lon
    y = lat
    inside = np.zeros(x.shape, dtype=bool)
    n = len(polygon)

    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]

        intersects = ((y0 > y) != (y1 > y)) & (
            x < (x1 - x0) * (y - y0) / ((y1 - y0) + 1e-15) + x0
        )
        inside ^= intersects

    return inside


def _region_polygons() -> dict[str, list[tuple[float, float]]]:
    return {
        "NorwegianSea": [
            (-10.0, 60.0),
            (10.0, 60.0),
            (20.0, 72.0),
            (8.0, 78.0),
            (-10.0, 75.0),
            (-14.0, 66.0),
        ],
        "BarentsSea": [
            (16.0, 69.0),
            (44.0, 69.0),
            (44.0, 80.5),
            (26.0, 83.0),
            (14.0, 77.0),
        ],
        "NorthSea": [(-6.0, 51.0), (9.0, 51.0), (9.0, 60.5), (2.0, 61.5), (-6.0, 58.0)],
        "BalticSea": [
            (9.0, 54.5),
            (30.0, 54.5),
            (30.0, 66.0),
            (18.0, 66.0),
            (11.0, 60.0),
        ],
        "Arctic": [(-35.0, 75.0), (45.0, 75.0), (45.0, 88.0), (-35.0, 88.0)],
    }


def _hard_region_masks(domain: SyntheticDomain) -> dict[str, BoolArray]:
    polygons = _region_polygons()
    lon = domain.lon_flat
    lat = domain.lat_flat

    masks: dict[str, BoolArray] = {}
    assigned = np.zeros_like(lon, dtype=bool)

    for name in ("NorthSea", "BalticSea", "NorwegianSea", "BarentsSea", "Arctic"):
        mask = _point_in_polygon(lon, lat, polygons[name])
        mask &= ~assigned
        masks[name] = mask
        assigned |= mask

    masks["Rest"] = ~assigned
    return masks


def _smooth_region_memberships(
    hard_masks: dict[str, BoolArray],
    feasible_flat: BoolArray,
    shape2d: tuple[int, int],
    sigma_cells: float,
) -> dict[str, FloatArray]:
    region_names = [
        "NorthSea",
        "BalticSea",
        "NorwegianSea",
        "BarentsSea",
        "Arctic",
        "Rest",
    ]

    stacked = np.zeros((len(region_names), feasible_flat.size), dtype=float)
    for i, name in enumerate(region_names):
        base = hard_masks[name].astype(float) * feasible_flat.astype(float)
        base_2d = base.reshape(shape2d)
        smoothed = ndimage.gaussian_filter(base_2d, sigma=sigma_cells, mode="nearest")
        stacked[i, :] = smoothed.ravel()

    denom = np.sum(stacked, axis=0)
    nonzero = denom > 0.0
    memberships = np.zeros_like(stacked)
    memberships[:, nonzero] = stacked[:, nonzero] / denom[nonzero]

    out = {name: memberships[i, :] for i, name in enumerate(region_names)}
    for name in out:
        out[name][~feasible_flat] = 0.0
    return out


def _boundary_split(
    rest_membership: FloatArray,
    boundary_soft: FloatArray,
    feasible_flat: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    boundary = rest_membership * boundary_soft
    rest = rest_membership * (1.0 - boundary_soft)
    boundary[~feasible_flat] = 0.0
    rest[~feasible_flat] = 0.0
    return boundary, rest


def _arctic_weight_piecewise(lat_deg: FloatArray, cfg: BenchmarkConfig) -> FloatArray:
    w = np.ones_like(lat_deg, dtype=float)
    w[(lat_deg >= cfg.lat_threshold_1_deg) & (lat_deg < cfg.lat_threshold_2_deg)] = (
        cfg.arctic_level_1
    )
    w[(lat_deg >= cfg.lat_threshold_2_deg) & (lat_deg < cfg.lat_threshold_3_deg)] = (
        cfg.arctic_level_2
    )
    w[lat_deg >= cfg.lat_threshold_3_deg] = cfg.arctic_level_3
    return w


def _arctic_weight_smooth(lat_deg: FloatArray, cfg: BenchmarkConfig) -> FloatArray:
    half = 0.5 * cfg.arctic_transition_width_deg
    s1 = smootherstep(
        lat_deg, cfg.lat_threshold_1_deg - half, cfg.lat_threshold_1_deg + half
    )
    s2 = smootherstep(
        lat_deg, cfg.lat_threshold_2_deg - half, cfg.lat_threshold_2_deg + half
    )
    s3 = smootherstep(
        lat_deg, cfg.lat_threshold_3_deg - half, cfg.lat_threshold_3_deg + half
    )

    # Start at 1.0 and smoothly apply three downward steps.
    drop1 = 1.0 - cfg.arctic_level_1
    drop2 = cfg.arctic_level_1 - cfg.arctic_level_2
    drop3 = cfg.arctic_level_2 - cfg.arctic_level_3
    return 1.0 - drop1 * s1 - drop2 * s2 - drop3 * s3


def _coast_weight(distance_to_land_m: FloatArray, cfg: BenchmarkConfig) -> FloatArray:
    scaled = 1.0 - np.exp(-distance_to_land_m / cfg.coast_length_scale_m)
    return cfg.coast_weight_floor + cfg.coast_weight_strength * scaled


def _norwegian_bump(
    distance_to_mainland_m: FloatArray,
    distance_to_all_land_m: FloatArray,
    norwegian_membership: FloatArray,
    cfg: BenchmarkConfig,
) -> FloatArray:
    gaussian = gaussian_weight(
        distance_to_mainland_m,
        center=cfg.norwegian_target_distance_m,
        sigma=cfg.norwegian_distance_spread_m,
    )
    activation = smootherstep(
        distance_to_all_land_m,
        cfg.norwegian_barrier_distance_m,
        cfg.norwegian_barrier_distance_m + cfg.norwegian_transition_width_m,
    )
    return (
        1.0
        + cfg.norwegian_boost_strength * gaussian * activation * norwegian_membership
    )


def _masked_gaussian_smoothing(
    values_flat: FloatArray,
    feasible_flat: BoolArray,
    shape2d: tuple[int, int],
    sigma_cells: float,
) -> FloatArray:
    data = values_flat.reshape(shape2d)
    mask = feasible_flat.reshape(shape2d).astype(float)

    numer = ndimage.gaussian_filter(data * mask, sigma=sigma_cells, mode="nearest")
    denom = ndimage.gaussian_filter(mask, sigma=sigma_cells, mode="nearest")

    out = np.zeros_like(numer)
    good = denom > 1e-12
    out[good] = numer[good] / denom[good]
    out[~mask.astype(bool)] = 0.0
    return out.ravel()


def _compute_targets(cfg: BenchmarkConfig) -> dict[str, float]:
    explicit = dict(cfg.region_fractions)
    explicit_sum = sum(explicit.values())
    if explicit_sum >= 1.0:
        raise ValueError("Sum of explicit region fractions must be < 1")

    rest_total = 1.0 - explicit_sum
    if cfg.boundary_fraction < 0.0 or cfg.boundary_fraction > rest_total:
        raise ValueError("boundary_fraction must be between 0 and remaining Rest mass")

    rest_after_boundary = rest_total - cfg.boundary_fraction

    targets = {
        "NorwegianSea": explicit["NorwegianSea"],
        "BarentsSea": explicit["BarentsSea"],
        "NorthSea": explicit["NorthSea"],
        "BalticSea": explicit["BalticSea"],
        "Arctic": explicit["Arctic"],
        "Boundary": cfg.boundary_fraction,
        "Rest": rest_after_boundary,
    }

    total = sum(targets.values())
    if not np.isclose(total, 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Target fractions do not sum to 1")

    return targets


def _build_region_memberships(
    domain: SyntheticDomain,
    cfg: BenchmarkConfig,
    density_mode: str,
) -> tuple[dict[str, FloatArray], FloatArray]:
    hard_masks = _hard_region_masks(domain)
    feasible_flat = domain.feasible_mask.ravel()

    xmin = float(np.min(domain.x_m))
    xmax = float(np.max(domain.x_m))
    ymin = float(np.min(domain.y_m))
    ymax = float(np.max(domain.y_m))

    x = domain.x_m.ravel()
    y = domain.y_m.ravel()
    boundary_distance = np.minimum.reduce([x - xmin, xmax - x, y - ymin, ymax - y])
    if density_mode == "hard_regions":
        boundary_soft = (boundary_distance <= cfg.boundary_band_width_m).astype(float)
        boundary_soft[~feasible_flat] = 0.0

        region_membership = {
            "NorthSea": hard_masks["NorthSea"].astype(float),
            "BalticSea": hard_masks["BalticSea"].astype(float),
            "NorwegianSea": hard_masks["NorwegianSea"].astype(float),
            "BarentsSea": hard_masks["BarentsSea"].astype(float),
            "Arctic": hard_masks["Arctic"].astype(float),
            "Rest": hard_masks["Rest"].astype(float),
        }
        for k in region_membership:
            region_membership[k][~feasible_flat] = 0.0
    elif density_mode == "smooth_regions":
        boundary_soft = 1.0 - smootherstep(
            boundary_distance, 0.0, cfg.boundary_band_width_m
        )
        boundary_soft = np.clip(boundary_soft, 0.0, 1.0)
        boundary_soft[~feasible_flat] = 0.0

        region_membership = _smooth_region_memberships(
            hard_masks,
            feasible_flat,
            domain.lon_deg.shape,
            sigma_cells=cfg.smooth_region_sigma_cells,
        )
    else:
        raise ValueError(f"Unknown density_mode: {density_mode}")

    boundary_membership, rest_membership = _boundary_split(
        region_membership["Rest"],
        boundary_soft,
        feasible_flat,
    )
    region_membership["Boundary"] = boundary_membership
    region_membership["Rest"] = rest_membership

    return region_membership, boundary_distance


def build_density(
    domain: SyntheticDomain,
    cfg: BenchmarkConfig,
    mode: str,
) -> DensityBuildOutput:
    if mode not in BENCHMARK_MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")

    if mode == "baseline_hard":
        density_mode = "hard_regions"
        arctic_mode = "piecewise"
        do_final_smoothing = False
    elif mode == "baseline_smooth":
        density_mode = "smooth_regions"
        arctic_mode = "smooth"
        do_final_smoothing = False
    else:
        density_mode = "smooth_regions"
        arctic_mode = "smooth"
        do_final_smoothing = True

    if density_mode not in DENSITY_MODES:
        raise ValueError("Invalid density mode mapping")
    if arctic_mode not in ARCTIC_MODES:
        raise ValueError("Invalid arctic mode mapping")

    feasible_flat = domain.feasible_mask.ravel()
    areas = np.full(domain.n_cells, domain.cell_area_m2, dtype=float)

    region_membership, boundary_distance = _build_region_memberships(
        domain, cfg, density_mode
    )

    arctic_weight = (
        _arctic_weight_piecewise(domain.lat_flat, cfg)
        if arctic_mode == "piecewise"
        else _arctic_weight_smooth(domain.lat_flat, cfg)
    )

    coast_weight = _coast_weight(domain.distance_to_all_land_m.ravel(), cfg)
    norwegian_bump = _norwegian_bump(
        domain.distance_to_mainland_m.ravel(),
        domain.distance_to_all_land_m.ravel(),
        region_membership["NorwegianSea"],
        cfg,
    )

    base = np.ones(domain.n_cells, dtype=float)
    raw_shape = base * coast_weight * norwegian_bump * arctic_weight
    raw_shape[~feasible_flat] = 0.0

    targets = _compute_targets(cfg)

    rho = np.zeros(domain.n_cells, dtype=float)
    eps = 1e-20
    for region_name, fraction in targets.items():
        membership = region_membership[region_name]
        region_shape = raw_shape * membership
        region_mass = float(np.sum(region_shape * areas))
        if region_mass <= eps:
            raise ValueError(
                f"Region {region_name} has zero effective shape mass under current constraints"
            )
        rho += (cfg.n_points * fraction) * region_shape / region_mass

    rho[~feasible_flat] = 0.0

    if do_final_smoothing:
        rho = _masked_gaussian_smoothing(
            rho,
            feasible_flat,
            domain.lon_deg.shape,
            sigma_cells=cfg.final_smoothing_sigma_cells,
        )
        # Renormalize to exact integrated mass N.
        current_mass = float(np.sum(rho * areas))
        if current_mass <= 0.0:
            raise RuntimeError("Smoothed density has zero mass")
        rho *= cfg.n_points / current_mass
        rho[~feasible_flat] = 0.0

    field = DensityField(
        coordinates=domain.coordinates_m,
        density=rho,
        feasible_mask=feasible_flat,
        cell_areas=areas,
    )

    region_masses = {
        name: float(np.sum(field.density * region_membership[name] * areas))
        for name in targets
    }

    desired_spacing_m = np.full(domain.n_cells, np.nan, dtype=float)
    positive = field.density > 0.0
    desired_spacing_m[positive] = 1.0 / np.sqrt(field.density[positive])

    intermediates = {
        "distance_to_all_land_m": domain.distance_to_all_land_m.ravel(),
        "distance_to_mainland_m": domain.distance_to_mainland_m.ravel(),
        "coast_weight": coast_weight,
        "norwegian_bump": norwegian_bump,
        "arctic_weight": arctic_weight,
        "raw_shape": raw_shape,
        "boundary_distance_m": boundary_distance,
        "desired_spacing_m": desired_spacing_m,
    }

    return DensityBuildOutput(
        density_field=field,
        region_memberships=region_membership,
        region_targets=targets,
        region_masses=region_masses,
        intermediates=intermediates,
    )
