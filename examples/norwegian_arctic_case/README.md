# Synthetic Nordic/Arctic Benchmark Use Case

This benchmark is a realistic, end-to-end validation case for selecting exactly N wave-spectrum storage locations from a large ocean candidate grid.

It is intentionally close to the intended North Atlantic / Nordic / Arctic application, while remaining fully synthetic and reproducible.

## Why This Exists

This case is designed to stress the separation:

1. hard feasibility constraints;
2. soft spatial density preferences;
3. sampling algorithm behavior.

It exercises:

- irregular coastlines and islands;
- distance-to-all-land vs distance-to-mainland distinctions;
- hard coastal exclusion;
- regional mass allocations;
- boundary-band allocation;
- Arctic weighting (piecewise and smooth variants);
- optional masked density smoothing;
- sampler comparison on identical target density.

## Synthetic vs Real

The geometry is synthetic benchmark data. It is not authoritative geographic or bathymetric data.

The purpose is reproducing numerical and geometric challenges relevant to wave-spectrum location sampling.

## Benchmark Architecture

The pipeline is:

hard constraints -> feasible domain -> soft modifiers -> target density -> sampler -> diagnostics -> plots

Benchmark-specific region polygons and geography stay in this example package, not in generic library classes.

## Modes

Three benchmark modes are included:

- baseline_hard: hard region memberships + piecewise Arctic weighting
- baseline_smooth: smooth region memberships + smooth Arctic weighting
- smooth_density: baseline_smooth plus final feasible-aware smoothing and renormalization

## Samplers Compared

On each density mode, the benchmark runs:

- density_weighted_farthest_point
- density_weighted_lloyd_cvt
- uniform_random (control)
- target_density_random (control)

## Diagnostics

For each method:

- requested/actual N
- runtime
- hard-constraint violations
- duplicate count
- normalized L1/L2 density error and density correlation
- regional mass error (L1 and max)
- nearest-neighbour statistics and coefficient of variation
- spacing correlation with 1/sqrt(rho)
- reproducibility check

## Running

Install dependencies:

```bash
/home/lehuc2580/work/spectral_sampling/.venv/bin/python -m pip install -e '.[dev,plot]'
```

Run benchmark:

```bash
/home/lehuc2580/work/spectral_sampling/.venv/bin/python -m examples.norwegian_arctic_case.run
```

Outputs are written to:

- examples/norwegian_arctic_case/outputs/benchmark_baseline_hard.png
- examples/norwegian_arctic_case/outputs/benchmark_baseline_smooth.png
- examples/norwegian_arctic_case/outputs/benchmark_smooth_density.png

## Limitations

- Region polygons are approximate benchmark shapes.
- Smooth region transitions and strict regional targets require explicit compromise; this benchmark enforces target masses at density-construction time using weighted regional components.
- Candidate-grid resolution affects distances, component areas, and point placement.
