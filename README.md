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

Implemented samplers:

- density_weighted_farthest_point
- density_weighted_lloyd_cvt
- density_adapted_poisson_disk

Greedy score:

- desired spacing: s_i = 1 / sqrt(rho_i)
- selection score: score_i = d_i / s_i

where d_i is nearest distance to already selected points.

Important: this is a deterministic density-adapted greedy heuristic. It does not mathematically guarantee exact final point-process density equal to rho.

For density-weighted Lloyd/CVT on a discrete candidate grid, the conceptual update is:

- x_k <- integral over V_k of x * rho(x) dx / integral over V_k of rho(x) dx

This library uses a discrete approximation with candidate masses w_i = rho_i * area_i,
deterministic nearest-center partitioning, weighted centroids, and projection back to
eligible candidates.

Important: this is a deterministic discrete CVT heuristic. It does not mathematically
guarantee global optimality or exact density reproduction.

For density-adapted Poisson-disk, the local nominal radius is

- r_i = alpha / sqrt(rho_i)

where alpha is a configurable spacing scale (`spacing_scale`).

Pairwise compatibility uses a symmetric separation rule:

- d(i, j) >= 0.5 * (r_i + r_j)

This avoids asymmetric acceptance rules for variable radii.

Important: this radius relation is a nominal spacing model, not an exact density
identity. The global scale alpha is application-dependent and should be calibrated.

## Reproducibility

The sampler is deterministic by construction with stable tie-breaking (lower index wins ties through stable argmax behavior).

Same inputs produce identical selected indices.

For Poisson-disk sampling, reproducibility is controlled by an explicit seed passed
to a local NumPy Generator (`np.random.default_rng(seed)`).

## Minimal API Example

```python
import numpy as np
from wave_sampling.density.field import DensityField
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt
from wave_sampling.samplers.poisson_disk import density_adapted_poisson_disk

coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
shape = np.array([1.0, 2.0, 1.0, 1.0])
feasible = np.array([True, True, True, True])

field = DensityField.from_shape(coords, shape, feasible, n_points=2)
result_fp = density_weighted_farthest_point(field, n_points=2)
result_cvt = density_weighted_lloyd_cvt(field, n_points=2)
result_pd = density_adapted_poisson_disk(field, n_points=2, seed=42, spacing_scale=0.8)
print(result_fp.selected_indices)
print(result_cvt.selected_indices)
print(result_pd.selected_indices)
```

Full synthetic demo with plotting:

- examples/basic_density_sampling.py
- examples/compare_samplers.py
- examples/modifier_effects.py
- examples/norwegian_arctic_case/README.md

## Diagnostics Included

- nearest-neighbour summary statistics (min/mean/median/std/p05/p95)
- density reproduction metrics (normalized L1, normalized L2, correlation)
- hard-constraint violation count
- Poisson separation diagnostics (symmetric-rule violations and nearest-distance to nominal-radius ratios)

## Current Limitations

- Candidate set is discrete (no continuous-domain sampling yet)
- Both implemented samplers are deterministic heuristics, so diagnostics remain
	necessary to evaluate density reproduction and regularity on each use case
- Density reproduction diagnostics are candidate-grid based and intentionally simple
- Poisson-disk can fail to place exactly N points when spacing is too restrictive;
  this raises a clear `PoissonDiskCapacityError` rather than silently returning fewer points

## Planned Algorithms (future)

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
/bin/python3 examples/compare_samplers.py
/bin/python3 examples/modifier_effects.py
/bin/python3 -m examples.norwegian_arctic_case.run
```

The Nordic/Arctic benchmark is fully synthetic and does not require external
bathymetry files or network access.
