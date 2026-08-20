from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from .config import BenchmarkConfig

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class SyntheticDomain:
    lon_deg: FloatArray
    lat_deg: FloatArray
    x_m: FloatArray
    y_m: FloatArray
    depth_m: FloatArray
    land_mask: BoolArray
    ocean_mask: BoolArray
    mainland_land_mask: BoolArray
    distance_to_all_land_m: FloatArray
    distance_to_mainland_m: FloatArray
    feasible_mask: BoolArray
    cell_area_m2: float

    @property
    def coordinates_m(self) -> FloatArray:
        return np.column_stack([self.x_m.ravel(), self.y_m.ravel()])

    @property
    def lon_flat(self) -> FloatArray:
        return self.lon_deg.ravel()

    @property
    def lat_flat(self) -> FloatArray:
        return self.lat_deg.ravel()

    @property
    def n_cells(self) -> int:
        return int(self.lon_deg.size)


def _in_ellipse(
    lon: FloatArray,
    lat: FloatArray,
    lon0: float,
    lat0: float,
    rx: float,
    ry: float,
    angle_deg: float = 0.0,
) -> BoolArray:
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    x = lon - lon0
    y = lat - lat0
    xr = c * x + s * y
    yr = -s * x + c * y
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def _build_synthetic_land_mask(lon_deg: FloatArray, lat_deg: FloatArray) -> BoolArray:
    """Construct synthetic land with coast complexity for benchmark stress testing."""
    land = np.zeros_like(lon_deg, dtype=bool)

    # Scandinavian/Fennoscandian mainland with an irregular west coastline.
    coast_lon = 2.5 + 0.22 * (lat_deg - 52.0) + 1.7 * np.sin(0.38 * (lat_deg - 52.0))
    scandi = (lon_deg > coast_lon) & (lon_deg < 42.0) & (lat_deg > 54.0)
    land |= scandi

    # Carve a Baltic-like basin from the broad mainland block.
    baltic_basin = (
        (lon_deg > 9.5)
        & (lon_deg < 30.5)
        & (lat_deg > 55.0)
        & (lat_deg < 66.5)
        & (lon_deg < 26.0 + 0.35 * (lat_deg - 55.0))
    )
    land &= ~baltic_basin

    # Continental Europe edge and UK vicinity shaping the North Sea boundary.
    land |= (lat_deg < 54.0) & (lon_deg > 6.0)

    # British Isles (two overlapping components).
    land |= _in_ellipse(lon_deg, lat_deg, lon0=-4.0, lat0=55.4, rx=4.5, ry=6.2)
    land |= _in_ellipse(lon_deg, lat_deg, lon0=-3.4, lat0=58.0, rx=2.9, ry=3.5)

    # Iceland-like island.
    land |= _in_ellipse(lon_deg, lat_deg, lon0=-19.0, lat0=65.0, rx=4.8, ry=2.7)

    # Jan Mayen-like island.
    land |= _in_ellipse(lon_deg, lat_deg, lon0=-8.5, lat0=71.0, rx=0.9, ry=0.5)

    # Svalbard-like island.
    land |= _in_ellipse(lon_deg, lat_deg, lon0=18.0, lat0=78.5, rx=3.2, ry=2.0)

    # Baltic entrance and interior sea constraints.
    denmark_bar_west = (
        (lon_deg > 8.4) & (lon_deg < 10.4) & (lat_deg > 55.3) & (lat_deg < 57.7)
    )
    denmark_bar_east = (
        (lon_deg > 11.8) & (lon_deg < 13.6) & (lat_deg > 55.4) & (lat_deg < 57.2)
    )
    south_sweden_strip = (
        (lon_deg > 14.2) & (lon_deg < 16.6) & (lat_deg > 56.2) & (lat_deg < 58.6)
    )
    land |= denmark_bar_west | denmark_bar_east | south_sweden_strip

    # Irregular Arctic islands.
    for lon0, lat0, rx, ry, angle in (
        (-6.0, 77.4, 2.2, 1.4, 15.0),
        (4.0, 79.5, 2.0, 1.2, -20.0),
        (24.0, 80.2, 2.5, 1.6, 10.0),
        (33.0, 81.6, 2.8, 1.5, -5.0),
    ):
        land |= _in_ellipse(
            lon_deg, lat_deg, lon0=lon0, lat0=lat0, rx=rx, ry=ry, angle_deg=angle
        )

    return land


def _project_lonlat_to_m(
    lon_deg: FloatArray,
    lat_deg: FloatArray,
    cfg: BenchmarkConfig,
) -> tuple[FloatArray, FloatArray]:
    """Simple equirectangular projection for benchmark metric calculations."""
    lon0 = np.deg2rad(cfg.projection_lon0_deg)
    lat0 = np.deg2rad(cfg.projection_lat0_deg)
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    x_m = cfg.earth_radius_m * (lon - lon0) * np.cos(lat0)
    y_m = cfg.earth_radius_m * (lat - lat0)
    return x_m, y_m


def _compute_depth_m(
    lon_deg: FloatArray,
    lat_deg: FloatArray,
    ocean_mask: BoolArray,
) -> FloatArray:
    """Synthetic ocean depth field in metres (positive ocean depth)."""
    basin = 2400.0 + 900.0 * np.sin(np.deg2rad(lon_deg * 1.2)) * np.cos(
        np.deg2rad(lat_deg - 67.0)
    )
    trench = 1300.0 * np.exp(
        -0.5 * ((lon_deg + 5.0) / 9.0) ** 2 - 0.5 * ((lat_deg - 72.0) / 6.5) ** 2
    )
    shelf = 700.0 * np.exp(
        -0.5 * ((lon_deg - 10.0) / 7.5) ** 2 - 0.5 * ((lat_deg - 58.0) / 4.5) ** 2
    )
    depth = basin + trench - shelf
    depth = np.clip(depth, 2.0, None)

    out = np.zeros_like(depth)
    out[ocean_mask] = depth[ocean_mask]
    return out


def _classify_mainland(
    land_mask: BoolArray,
    cell_area_m2: float,
    mainland_min_area_km2: float,
) -> tuple[BoolArray, dict[int, float], set[int]]:
    structure = np.ones((3, 3), dtype=int)
    labels_raw, n_labels_raw = ndimage.label(land_mask, structure=structure)  # type: ignore[reportGeneralTypeIssues]
    labels = cast(npt.NDArray[np.int64], np.asarray(labels_raw, dtype=np.int64))
    n_labels = int(n_labels_raw)

    component_areas_km2: dict[int, float] = {}
    mainland_labels: set[int] = set()
    threshold_m2 = mainland_min_area_km2 * 1_000_000.0

    for label_id in range(1, n_labels + 1):
        count = int(np.sum(labels == label_id))
        area_m2 = count * cell_area_m2
        area_km2 = area_m2 / 1_000_000.0
        component_areas_km2[label_id] = area_km2
        if area_m2 >= threshold_m2:
            mainland_labels.add(label_id)

    mainland_land_mask = np.isin(labels, list(mainland_labels))
    return mainland_land_mask, component_areas_km2, mainland_labels


def _grid_spacing_m(x_m: FloatArray, y_m: FloatArray) -> tuple[float, float]:
    dx = float(np.mean(np.diff(x_m[0, :])))
    dy = float(np.mean(np.diff(y_m[:, 0])))
    return abs(dx), abs(dy)


def build_synthetic_domain(
    cfg: BenchmarkConfig,
) -> tuple[SyntheticDomain, dict[str, object]]:
    lon_axis = np.linspace(cfg.lon_min_deg, cfg.lon_max_deg, cfg.nx)
    lat_axis = np.linspace(cfg.lat_min_deg, cfg.lat_max_deg, cfg.ny)
    lon_deg, lat_deg = np.meshgrid(lon_axis, lat_axis)

    x_m, y_m = _project_lonlat_to_m(lon_deg, lat_deg, cfg)
    dx_m, dy_m = _grid_spacing_m(x_m, y_m)
    cell_area_m2 = dx_m * dy_m

    land_mask = _build_synthetic_land_mask(lon_deg, lat_deg)
    ocean_mask = ~land_mask

    depth_m = _compute_depth_m(lon_deg, lat_deg, ocean_mask)

    mainland_land_mask, component_areas_km2, mainland_labels = _classify_mainland(
        land_mask,
        cell_area_m2,
        cfg.mainland_min_area_km2,
    )

    sampling = (dy_m, dx_m)
    distance_to_all_land_m = cast(
        FloatArray,
        np.asarray(
            ndimage.distance_transform_edt(ocean_mask, sampling=sampling),
            dtype=float,
        ),
    )

    non_mainland_binary = np.ones_like(land_mask, dtype=bool)
    non_mainland_binary[mainland_land_mask] = False
    distance_to_mainland_m = cast(
        FloatArray,
        np.asarray(
            ndimage.distance_transform_edt(
                non_mainland_binary,
                sampling=sampling,
            ),
            dtype=float,
        ),
    )

    feasible_mask = (
        ocean_mask
        & (depth_m >= cfg.min_depth_m)
        & (distance_to_all_land_m >= cfg.min_land_distance_m)
    )

    domain = SyntheticDomain(
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        x_m=x_m,
        y_m=y_m,
        depth_m=depth_m,
        land_mask=land_mask,
        ocean_mask=ocean_mask,
        mainland_land_mask=mainland_land_mask,
        distance_to_all_land_m=distance_to_all_land_m,
        distance_to_mainland_m=distance_to_mainland_m,
        feasible_mask=feasible_mask,
        cell_area_m2=cell_area_m2,
    )

    metadata: dict[str, object] = {
        "dx_m": dx_m,
        "dy_m": dy_m,
        "cell_area_m2": cell_area_m2,
        "component_areas_km2": component_areas_km2,
        "mainland_labels": sorted(mainland_labels),
    }
    return domain, metadata
