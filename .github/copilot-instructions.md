# Project Instructions — 2D Wave Spectrum Sampling

## Project purpose

This repository implements deterministic, density-based spatial sampling for selecting a limited number of spatial locations at which a 2D wave spectrum will be stored or evaluated.

The motivating application is a wave model where a potentially very large spatial grid contains a 2D wave spectrum at every location. Storing spectra everywhere is too expensive, so we need to select exactly N spatial locations.

The selected locations must:

1. satisfy all hard spatial constraints;
2. approximately reproduce a prescribed spatial target density;
3. be as spatially regular/evenly distributed as possible;
4. be deterministic and reproducible;
5. support multiple sampling algorithms;
6. remain independent of any particular wave model implementation.

The primary conceptual model is:

    hard constraints
          ↓
    feasible domain F
          ↓
    soft spatial preferences
          ↓
    target density ρ(x,y)
          ↓
    sampling algorithm
          ↓
    exactly N points
          ↓
    diagnostics / validation

The density field is the central abstraction shared by all sampling algorithms.

---

# Core mathematical concepts

Let Ω be the spatial domain.

Let F ⊆ Ω be the feasible domain defined by hard constraints.

Hard constraints answer:

    "Can a point exist here?"

Soft density modifiers answer:

    "How strongly do we prefer a point here?"

The target density is:

    ρ(x,y) >= 0

and should satisfy:

    ∫_F ρ(x,y) dA = N

where N is the requested number of points.

Therefore ρ has units of points / area.

The expected local point spacing in 2D scales approximately as:

    s(x,y) ∝ 1 / sqrt(ρ(x,y))

This relationship is useful for sampling algorithms, but must not be assumed to mean that every sampling algorithm exactly realizes ρ.

The distinction between:

1. target density reproduction
2. spatial regularity

must always be maintained.

---

# Hard vs soft constraints

This distinction is fundamental.

## Hard constraints

Hard constraints define feasibility.

Examples:

- land
- outside model domain
- invalid wave-model cells
- missing data
- prohibited geographic areas
- absolute minimum distance from a coastline, if this is genuinely non-negotiable

Hard constraints must never be implemented merely as a low density.

A hard constraint should remove the location from the feasible domain.

Sampling algorithms must never return points violating a hard constraint.

A successful sampler should have:

    number_of_hard_constraint_violations == 0

## Soft constraints

Soft constraints influence density but do not make a location impossible.

Examples:

- preference for open water
- preference for distance from coast
- Arctic weighting
- Norwegian Sea coastal preference
- area-of-interest weighting
- bathymetry preference
- wave variability
- user-defined importance fields

Soft constraints should be represented as smooth functions whenever practical.

---

# Density construction

Density construction must be independent of the sampling algorithm.

A sampler receives a completed DensityField and does not know how the density was constructed.

Conceptually:

    raw features
        ↓
    smooth transformations
        ↓
    combine features
        ↓
    optional smoothing
        ↓
    regional/mass normalization
        ↓
    normalized DensityField
        ↓
    sampler

Prefer multiplicative positive shape weights when appropriate:

    q(x,y) = Π_i w_i(x,y)

followed by normalization:

    ρ(x,y) = N q(x,y) / ∫_F q(x,y) dA

Regional allocations may be imposed by allocating a specified mass to regions.

If a region fraction is f_r:

    ∫_region_r ρ(x,y) dA = N f_r

when exact regional allocation is requested.

Do not introduce discontinuities unnecessarily.

Prefer smooth region blending where possible.

---

# Smooth functions

The density system should provide reusable smooth functions rather than embedding arbitrary formulas throughout application code.

Useful functions include:

- smoothstep
- smootherstep
- logistic
- Gaussian
- exponential distance decay
- compact-support smooth functions
- spline-based mappings where appropriate

For example, smootherstep is:

    S(t) = 6t^5 - 15t^4 + 10t^3

for t ∈ [0,1].

Use smootherstep when a C2 transition is desirable.

Logistic functions are useful for soft thresholds.

Gaussian functions are useful for localized preferences.

Do not claim a function is C∞ unless it actually is.

---

# Coast and land handling

Land is a hard constraint.

If distance from land is used as a soft preference, it must not be confused with the hard land mask.

Distance fields may be used to construct smooth preferences such as:

    w(d)

where d is distance from coastline.

Parameters should have physical units, normally metres.

Avoid normalizing distances by the maximum distance in the current computational domain unless there is a compelling reason.

Prefer physically meaningful parameters such as:

- transition distance
- characteristic length
- preferred distance
- Gaussian width
- minimum hard distance

The final sampler must always operate on the feasible domain.

If a density is smoothed, ensure smoothing cannot create nonzero density on infeasible cells.

---

# Coordinate system

Distance calculations must be performed in a metric Cartesian/projected coordinate system.

Do not assume latitude/longitude degrees represent constant distances.

If input data are geographic coordinates, provide explicit conversion/projection before metric calculations.

The library should keep geographic coordinates as an input/output concern and use metric coordinates internally for geometry and spacing calculations.

---

# Candidate grids

The initial implementation may support sampling from discrete candidate locations, e.g. wave-model grid cells.

A candidate grid represents locations at which spectra can actually be stored/evaluated.

Do not unnecessarily assume that the sampler must always operate on arbitrary continuous coordinates.

Design abstractions so that continuous sampling could be added later.

---

# Sampling algorithms

The repository should support multiple methods using the same DensityField interface.

## Method 1: density-weighted farthest-point sampling

The initial/reference implementation is a deterministic greedy method.

Given desired local spacing:

    s_i = 1 / sqrt(ρ_i)

select the next candidate by maximizing a normalized distance criterion such as:

    score_i = d_i / s_i

where d_i is the distance to the nearest selected point.

This method is:

- deterministic
- easy to understand
- suitable for exactly N points
- suitable as a baseline

It should be documented carefully as a density-adapted greedy coverage algorithm.

Do NOT claim that this algorithm mathematically guarantees exact reproduction of ρ.

Its density reproduction must be measured empirically.

The implementation should not introduce region-specific loops that allow points from one region to ignore points from another.

All candidates must compete globally unless an explicitly documented constrained allocation mode is requested.

---

## Method 2: density-weighted Lloyd / centroidal Voronoi tessellation

A future method should implement density-weighted CVT/Lloyd relaxation.

The conceptual update is:

    x_i = ∫_{V_i} x ρ(x) dx / ∫_{V_i} ρ(x) dx

where V_i is the Voronoi cell of point i.

For grid-based implementation, a discrete approximation using feasible grid cells is acceptable.

This method should emphasize spatial regularity and density conformity.

It must respect hard constraints.

---

## Method 3: density-adapted Poisson-disk sampling

A future method should support spatially varying minimum separation.

Local spacing scales approximately as:

    r(x) ∝ 1 / sqrt(ρ(x))

The implementation must handle the fact that spacing varies spatially.

Randomized algorithms must accept an explicit seed.

For the same:

- input data
- parameters
- seed
- library version/algorithm

the result should be reproducible.

---

## Method 4: optimal transport

Optimal transport is a future/v2 feature.

The conceptual goal is to transform a uniform point distribution into the prescribed density distribution.

Do not implement this prematurely.

---

# Determinism and reproducibility

Determinism is a first-class requirement.

If an algorithm does not require randomness, it should be deterministic by construction.

For randomized methods:

- require or expose an explicit seed;
- never use uncontrolled global random state;
- document reproducibility behavior.

Tie-breaking must also be deterministic.

For example, if two candidates have equal score, choose the candidate with the lower stable index.

Avoid relying on unordered dictionary/set iteration for algorithmically significant decisions.

---

# Exactly N points

The primary API should request exactly N points.

Example conceptual API:

    points = sample_density(
        density,
        n_points=500,
        method="farthest_point",
        seed=None,
    )

Unless an algorithm explicitly documents otherwise, it should return exactly N feasible points.

If the feasible domain cannot support N points under a hard minimum-separation constraint, fail clearly with a useful error rather than silently returning fewer points.

---

# Diagnostics

Diagnostics are a core feature, not an afterthought.

Sampling results should be evaluable using:

## Hard constraint violations

Must be zero.

## Density reproduction

Compare observed point density against target density.

Useful metrics include:

- normalized L1 error
- normalized L2 error
- correlation
- regional mass error
- maximum regional error

## Spatial regularity

Measure nearest-neighbour distances.

Report at least:

- minimum
- median
- mean
- standard deviation
- selected percentiles

## Local spacing

Compare actual nearest-neighbour distances with:

    1 / sqrt(ρ)

appropriately scaled.

## Reproducibility

Run identical inputs twice and verify identical outputs.

---

# Testing philosophy

Tests should focus on mathematical and behavioral contracts.

Important tests:

1. Density integrates to N.
2. Density is nonnegative.
3. Hard constraints are never violated.
4. Exactly N points are returned.
5. Deterministic algorithms return identical results.
6. Seeded stochastic algorithms return identical results for identical seeds.
7. Regional mass constraints are satisfied within numerical tolerance.
8. Smoothing does not introduce density on infeasible cells.
9. Zero/invalid density is handled explicitly.
10. Impossible sampling requests fail clearly.
11. Small synthetic domains produce known/reasonable results.
12. Algorithms behave correctly for uniform density.

Use synthetic analytic density fields for most algorithmic tests.

Do not make tests dependent on large external wave-model datasets.

---

# Performance

The application is memory constrained.

Avoid algorithms with unnecessary O(N_grid × N_points) memory allocation.

It is acceptable for an algorithm to have O(N_grid) memory.

Avoid constructing full pairwise distance matrices unless the problem size is explicitly small.

For greedy farthest-point sampling, maintain a nearest-distance vector and update it incrementally.

Prefer vectorized NumPy operations where they improve clarity and performance.

For very large grids, consider chunking, spatial indexing, KD-trees, or compiled acceleration only after profiling.

Do not prematurely optimize.

---

# API design

Keep public APIs small and explicit.

Prefer:

    DensityField
    FeasibleDomain
    DensityModifier
    Sampler
    SamplingResult
    SamplingDiagnostics

over large monolithic functions.

Separate:

- domain construction
- density construction
- sampling
- diagnostics
- I/O
- plotting

Do not put wave-model-specific logic inside the core sampler.

---

# Dependencies

Prefer a lightweight scientific Python stack.

Likely core dependencies:

- numpy
- scipy

Potential optional dependencies:

- matplotlib
- xarray
- netCDF4/h5netcdf
- pyproj
- shapely
- scikit-learn or scipy spatial functionality where justified

Do not add a dependency merely for convenience if a small NumPy/SciPy implementation is clearer.

Optional dependencies should remain optional where practical.

---

# Code quality

Use modern Python.

Prefer:

- type hints
- dataclasses where appropriate
- NumPy-style docstrings
- explicit error messages
- small functions
- pure functions for mathematical transformations
- deterministic behavior

Avoid:

- hidden global state
- magic constants
- duplicated mathematical formulas
- application-specific conditionals inside generic samplers
- silent fallback behavior

All physical parameters should have explicit units in names or documentation where ambiguity is possible.

Examples:

    min_distance_m
    gaussian_width_m
    preferred_distance_m

---

# Visualization

Provide plotting utilities for:

1. feasible domain
2. raw density features
3. final target density
4. desired spacing
5. selected points
6. target vs observed density
7. nearest-neighbour distribution

Visualization is particularly important for validating density transitions and geographic constraints.

---

# Documentation philosophy

Documentation should explain both:

1. what the code does;
2. why the mathematical formulation was chosen.

Avoid presenting heuristics as exact mathematical guarantees.

When an algorithm is approximate, say so.

When a property is guaranteed, state the actual condition under which it is guaranteed.

The README should contain a simple end-to-end example before detailed API documentation.

---

# Development workflow

Implement functionality incrementally.

For each significant feature:

1. define the mathematical behavior;
2. implement the smallest clean version;
3. add unit tests;
4. add a synthetic example;
5. add diagnostics;
6. update documentation;
7. benchmark only if performance matters.

Do not implement all planned algorithms at once.

The first milestone should be:

    DensityField
    +
    hard feasible mask
    +
    smooth density modifiers
    +
    normalization to exactly N
    +
    deterministic density-weighted farthest-point sampler
    +
    diagnostics
    +
    tests
    +
    example

Only after this baseline is stable should CVT and Poisson-disk methods be added.

---

# Important project principle

Do not overfit the architecture to the current Norwegian Sea / Arctic use case.

The current application may include:

- mainland coastline distance
- Arctic weighting
- Norwegian Sea coastal bump
- regional fractions

but these must be implemented as generic density modifiers/configuration rather than hard-coded into the sampler.

The core library should be usable for any 2D spatial density field.

The intended abstraction is:

    arbitrary feasible domain
            +
    arbitrary smooth target density
            +
    sampling algorithm
            =
    reproducible spatial point set

The wave-spectrum use case is the motivating application, not the mathematical limitation of the library.