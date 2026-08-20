from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the synthetic Nordic/Arctic benchmark case."""

    n_points: int = 300

    lon_min_deg: float = -35.0
    lon_max_deg: float = 45.0
    lat_min_deg: float = 50.0
    lat_max_deg: float = 88.0
    nx: int = 280
    ny: int = 180

    projection_lon0_deg: float = 10.0
    projection_lat0_deg: float = 70.0
    earth_radius_m: float = 6_371_000.0

    min_depth_m: float = 10.0
    min_land_distance_m: float = 5_000.0

    mainland_min_area_km2: float = 50_000.0

    coast_length_scale_m: float = 180_000.0
    coast_weight_floor: float = 0.35
    coast_weight_strength: float = 0.75

    norwegian_target_distance_m: float = 250_000.0
    norwegian_distance_spread_m: float = 200_000.0
    norwegian_boost_strength: float = 0.1
    norwegian_barrier_distance_m: float = 10_000.0
    norwegian_transition_width_m: float = 5_000.0

    boundary_band_width_m: float = 200_000.0
    boundary_fraction: float = 0.05

    lat_threshold_1_deg: float = 75.0
    lat_threshold_2_deg: float = 80.0
    lat_threshold_3_deg: float = 85.0
    arctic_level_1: float = 0.75
    arctic_level_2: float = 0.50
    arctic_level_3: float = 0.33
    arctic_transition_width_deg: float = 2.0

    region_fractions: dict[str, float] = field(
        default_factory=lambda: {
            "NorwegianSea": 0.25,
            "BarentsSea": 0.09,
            "NorthSea": 0.10,
            "BalticSea": 0.04,
            "Arctic": 0.08,
        }
    )

    smooth_region_sigma_cells: float = 2.0
    final_smoothing_sigma_cells: float = 1.25

    cvt_max_iterations: int = 25
    random_seed: int = 7
    poisson_spacing_scale: float = 0.75
    poisson_max_attempts_per_active_point: int = 30
    poisson_candidate_k_neighbors: int = 8


BENCHMARK_MODES = ("baseline_hard", "baseline_smooth", "smooth_density")
DENSITY_MODES = ("hard_regions", "smooth_regions")
ARCTIC_MODES = ("piecewise", "smooth")
