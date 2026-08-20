# Wave Sampling (v0.1 foundation)

Wave Sampling is a Python library for deterministic, density-based 2D spatial sampling over a discrete candidate set.

The motivating use case is selecting exactly N locations where 2D wave spectra are stored from a much larger wave-model grid.

## Motivation

Storing spectra at every model cell is often too expensive. We instead construct a target density field over feasible candidates and sample exactly N points.

The design separates:

- hard constraints (feasibility)
- soft density preferences (importance)
- sampling algorithm (point selection)

## Target Density Definition

Let F be the feasible candidate set and rho_i be target density at candidate i.

For candidate areas area_i:

- rho_i >= 0
- rho_i = 0 for infeasible candidates
- sum(rho_i * area_i) = N

Equal-area candidates are the special case area_i = 1.

## Hard vs Soft Constraints

Hard constraints define whether sampling is allowed at a location.

Soft constraints modify preference weight, not feasibility. For example, distance-from-boundary can reduce probability-like preference near a boundary, but hard exclusion is still handled by feasible_mask.

## Current Sampler

Implemented sampler:

- density_weighted_farthest_point

Greedy score:

- desired spacing: s_i = 1 / sqrt(rho_i)
- selection score: score_i = d_i / s_i

where d_i is nearest distance to already selected points.

Important: this is a deterministic density-adapted greedy heuristic. It does not mathematically guarantee exact final point-process density equal to rho.

## Reproducibility

The sampler is deterministic by construction with stable tie-breaking (lower index wins ties through stable argmax behavior).

Same inputs produce identical selected indices.

## Minimal API Example

```python
import numpy as np
from wave_sampling.density.field import DensityField
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point

coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
shape = np.array([1.0, 2.0, 1.0, 1.0])
feasible = np.array([True, True, True, True])

field = DensityField.from_shape(coords, shape, feasible, n_points=2)
result = density_weighted_farthest_point(field, n_points=2)
print(result.selected_indices)
```

Full synthetic demo with plotting:

- examples/basic_density_sampling.py

## Diagnostics Included

- nearest-neighbour summary statistics (min/mean/median/std/p05/p95)
- density reproduction metrics (normalized L1, normalized L2, correlation)
- hard-constraint violation count

## Current Limitations

- Candidate set is discrete (no continuous-domain sampling yet)
- Only one sampler is implemented in v0.1
- Density reproduction diagnostics are candidate-grid based and intentionally simple

## Planned Algorithms (future)

- density-weighted Lloyd / CVT
- density-adapted Poisson-disk
- optimal transport based methods

These are intentionally not included in this first milestone.

## Development

Install package and test dependencies:

```bash
/bin/python3 -m pip install -e .[dev]
```

Run tests:

```bash
/bin/python3 -m pytest
```

Run example (with plotting):

```bash
/bin/python3 -m pip install -e .[plot]
/bin/python3 examples/basic_density_sampling.py
```
