import numpy as np

from wave_sampling.density.modifiers import (
    clipped_linear_transition,
    exponential_distance_weight,
    gaussian_weight,
    logistic_weight,
    smootherstep,
    smoothstep,
)


def test_clipped_linear_expected_values() -> None:
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    y = clipped_linear_transition(x, 0.0, 1.0)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.allclose(y, expected)


def test_smoothstep_endpoints_and_monotonicity() -> None:
    x = np.linspace(0.0, 1.0, 101)
    y = smoothstep(x, 0.0, 1.0)
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[-1], 1.0)
    assert np.all(np.diff(y) >= 0.0)


def test_smootherstep_endpoints_and_monotonicity() -> None:
    x = np.linspace(0.0, 1.0, 101)
    y = smootherstep(x, 0.0, 1.0)
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[-1], 1.0)
    assert np.all(np.diff(y) >= 0.0)


def test_gaussian_center_is_one() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    y = gaussian_weight(x, center=0.0, sigma=2.0)
    assert np.isclose(y[1], 1.0)
    assert y[0] == y[2]


def test_logistic_midpoint_is_half() -> None:
    y = logistic_weight(np.array([0.0]), midpoint=0.0, scale=1.0)
    assert np.isclose(y[0], 0.5)


def test_exponential_distance_expected_values() -> None:
    d = np.array([0.0, 10.0, 20.0])
    y = exponential_distance_weight(d, length_scale_m=10.0)
    assert np.isclose(y[0], 1.0)
    assert np.isclose(y[1], np.exp(-1.0))
    assert np.all(np.diff(y) < 0.0)
