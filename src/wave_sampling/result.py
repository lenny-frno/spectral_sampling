from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SamplingResult:
    """Container for deterministic sampling outputs and metadata."""

    selected_indices: npt.NDArray[np.int64]
    selected_coordinates: npt.NDArray[np.float64]
    method: str
    requested_n_points: int
    seed: int | None
    density_summary: dict[str, Any]
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        idx = np.asarray(self.selected_indices, dtype=np.int64)
        coords = np.asarray(self.selected_coordinates, dtype=float)

        if idx.ndim != 1:
            raise ValueError("selected_indices must be 1D")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("selected_coordinates must have shape (N, 2)")
        if coords.shape[0] != idx.shape[0]:
            raise ValueError("indices/coordinates length mismatch")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("selected_indices must be unique")

        object.__setattr__(self, "selected_indices", idx)
        object.__setattr__(self, "selected_coordinates", coords)

    @property
    def n_selected(self) -> int:
        return int(self.selected_indices.shape[0])
