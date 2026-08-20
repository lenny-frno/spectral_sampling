from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def clipped_linear_transition(x: npt.ArrayLike, x0: float, x1: float) -> FloatArray:
    """Linear transition from 0 to 1, clipped outside [x0, x1].

    Parameters
    ----------
    x
        Input variable.
    x0
        Lower transition bound where output is exactly 0.
    x1
        Upper transition bound where output is exactly 1. Requires x1 > x0.

    Returns
    -------
    numpy.ndarray
        Values in [0, 1]. Continuous but not differentiable at x0 and x1.
    """
    if x1 <= x0:
        raise ValueError("x1 must be greater than x0")
    x_arr = np.asarray(x, dtype=float)
    t = (x_arr - x0) / (x1 - x0)
    return np.clip(t, 0.0, 1.0)


def smoothstep(x: npt.ArrayLike, x0: float, x1: float) -> FloatArray:
    """C1 smooth transition from 0 to 1 on [x0, x1].

    Uses S(t) = 3t^2 - 2t^3 for t in [0, 1].
    """
    t = clipped_linear_transition(x, x0, x1)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(x: npt.ArrayLike, x0: float, x1: float) -> FloatArray:
    """C2 smooth transition from 0 to 1 on [x0, x1].

    Uses S(t) = 6t^5 - 15t^4 + 10t^3 for t in [0, 1].
    """
    t = clipped_linear_transition(x, x0, x1)
    return t**3 * (t * (t * 6.0 - 15.0) + 10.0)


def logistic_weight(
    x: npt.ArrayLike,
    midpoint: float,
    scale: float,
    increasing: bool = True,
) -> FloatArray:
    """Logistic soft-threshold weighting.

    Parameters
    ----------
    x
        Input variable.
    midpoint
        Input value where output equals 0.5.
    scale
        Positive slope scale in x units. Smaller values produce steeper
        transitions.
    increasing
        If True, output increases with x. If False, output decreases with x.

    Returns
    -------
    numpy.ndarray
        Values in (0, 1), smooth (C-infinity).
    """
    if scale <= 0.0:
        raise ValueError("scale must be strictly positive")
    x_arr = np.asarray(x, dtype=float)
    z = (x_arr - midpoint) / scale
    y = 1.0 / (1.0 + np.exp(-z))
    if increasing:
        return y
    return 1.0 - y


def gaussian_weight(x: npt.ArrayLike, center: float, sigma: float) -> FloatArray:
    """Gaussian weighting centered at center.

    Parameters
    ----------
    x
        Input variable.
    center
        Location of maximum weight.
    sigma
        Standard deviation in same units as x. Must be > 0.

    Returns
    -------
    numpy.ndarray
        Positive smooth weights in (0, 1], with value 1 at x = center.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive")
    x_arr = np.asarray(x, dtype=float)
    z = (x_arr - center) / sigma
    return np.exp(-0.5 * z * z)


def exponential_distance_weight(
    distance_m: npt.ArrayLike,
    length_scale_m: float,
) -> FloatArray:
    """Exponential distance weighting exp(-d / L) for distances in metres.

    Parameters
    ----------
    distance_m
        Nonnegative distances in metres.
    length_scale_m
        Positive e-folding length scale in metres.

    Returns
    -------
    numpy.ndarray
        Weights in (0, 1] for d >= 0.
    """
    if length_scale_m <= 0.0:
        raise ValueError("length_scale_m must be strictly positive")
    d = np.asarray(distance_m, dtype=float)
    if np.any(d < 0.0):
        raise ValueError("distance_m must be nonnegative")
    return np.exp(-d / length_scale_m)


def compose_positive_factors(
    base: npt.ArrayLike,
    *factors: npt.ArrayLike,
) -> FloatArray:
    """Compose multiplicative nonnegative density factors.

    Computes q = base * factor_1 * ... * factor_k.
    """
    q = np.asarray(base, dtype=float).copy()
    if q.ndim != 1:
        raise ValueError("base must be a 1D array")
    if not np.all(np.isfinite(q)):
        raise ValueError("base must be finite")
    if np.any(q < 0.0):
        raise ValueError("base must be nonnegative")

    for idx, factor in enumerate(factors):
        f = np.asarray(factor, dtype=float)
        if f.shape != q.shape:
            raise ValueError(
                f"factor {idx} shape mismatch: expected {q.shape}, got {f.shape}"
            )
        if not np.all(np.isfinite(f)):
            raise ValueError(f"factor {idx} must be finite")
        if np.any(f < 0.0):
            raise ValueError(f"factor {idx} must be nonnegative")
        q *= f

    return q
