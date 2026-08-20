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

## Samplers

Implemented samplers:

- density_weighted_farthest_point
- density_weighted_lloyd_cvt
- density_weighted_optimal_transport

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

For density-weighted optimal transport on a discrete candidate grid, the formulation is:

- source measure: N equal point masses (one per output point)
- target measure: candidate-cell masses m_i = rho_i * area_i over feasible candidates
- cost: squared Euclidean distance in projected metric coordinates

The implementation uses a deterministic semi-discrete OT approximation with power-diagram
assignments and transport potentials, then projects generators back to unique feasible
candidate indices.

Important: this is an OT-inspired discrete approximation. It does not guarantee the exact
global optimum of the continuous OT problem.

## Reproducibility

The sampler is deterministic by construction with stable tie-breaking (lower index wins ties through stable argmax behavior).

Same inputs produce identical selected indices.

## Minimal API Example

```python
import numpy as np
from wave_sampling.density.field import DensityField
from wave_sampling.samplers.farthest_point import density_weighted_farthest_point
from wave_sampling.samplers.lloyd_cvt import density_weighted_lloyd_cvt
from wave_sampling.samplers.optimal_transport import density_weighted_optimal_transport

coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
shape = np.array([1.0, 2.0, 1.0, 1.0])
feasible = np.array([True, True, True, True])

field = DensityField.from_shape(coords, shape, feasible, n_points=2)
result_fp = density_weighted_farthest_point(field, n_points=2)
result_cvt = density_weighted_lloyd_cvt(field, n_points=2)
result_ot = density_weighted_optimal_transport(field, n_points=2)
print(result_fp.selected_indices)
print(result_cvt.selected_indices)
print(result_ot.selected_indices)
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

## Current Limitations

- Candidate set is discrete (no continuous-domain sampling yet)
- Both implemented samplers are deterministic heuristics, so diagnostics remain
	necessary to evaluate density reproduction and regularity on each use case
- OT mass balancing is approximate on discrete indivisible candidate masses; attainable
	imbalance is bounded by the largest candidate mass unit
- Density reproduction diagnostics are candidate-grid based and intentionally simple

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

## Next steps

1. [HIGH] Add explicit per-generator transport-mass diagnostics in OT and persist them in benchmark outputs.
	Reason: current OT diagnostics report only aggregate mass imbalance and make convergence/failure triage harder on realistic cases.

2. [HIGH] Add disconnected-component-aware OT initialization with per-component minimum point quotas.
	Reason: density-driven initialization can still under-seed tiny but nonzero-mass components before transport balancing.

3. [MEDIUM] Profile OT assignment on large grids (>1e6 candidates) and replace brute-force chunked center distances with KD-tree candidate pruning if assignment dominates runtime.
	Reason: current assignment is robust but still evaluates many center distances per iteration.

4. [MEDIUM] Add optional local regularization pass after OT convergence (small fixed-count Lloyd refinement under fixed feasible candidates).
	Reason: this can improve nearest-neighbour regularity in steep density gradients without changing hard-constraint handling.

5. [LOW] Add benchmark export to CSV/Parquet for method-comparison tables across parameter sweeps.
	Reason: persistent tabular outputs make method regression tracking easier than console summaries alone.

### Recommended next feature

Implement disconnected-component-aware OT seeding and balancing so each feasible component with nonzero target mass receives at least one generator when geometrically possible, then benchmark the impact on regional mass and regularity diagnostics.
