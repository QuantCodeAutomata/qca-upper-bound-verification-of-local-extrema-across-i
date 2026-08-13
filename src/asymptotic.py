"""
Asymptotic shape probability analysis under the Hull-White risk-neutral measure.

The Hull-White short rate is Gaussian:
  r(t) ~ N(m(t), v(t))
where v(t) = sigma^2 * (1-exp(-2*lambda*t)) / (2*lambda)
and m(t) = exp(-lambda*t)*r0 + integral_0^t exp(-lambda*(t-s)) * b(s) ds.

Asymptotic probabilities as t->infinity:

Yield curve:
  Q(inverse) -> 0.5
  Q(humped)  -> 0.5 - Phi(-3*sigma / sqrt(8*lambda^3))
  Q(normal)  -> Phi(-3*sigma / sqrt(8*lambda^3))

Forward curve (case lambda < min{1/tau1, 1/tau2} or special):
  Q(inverse) -> 0.5
  Q(humped)  -> 0.5 - Phi(-sqrt(2)*sigma / lambda^(3/2))
  Q(normal)  -> Phi(-sqrt(2)*sigma / lambda^(3/2))

Forward (terminal-increasing):
  Q(dipped)  -> 0.5
  Q(hd)      -> 0.5 - Phi(-sqrt(2)*sigma / lambda^(3/2))
  Q(normal)  -> Phi(-sqrt(2)*sigma / lambda^(3/2))

Forward (terminal-decreasing):
  Q(inverse) -> 0.5
  Q(humped)  -> 0.5
"""

from __future__ import annotations
import numpy as np
from scipy import integrate, stats
from typing import Dict, List, Optional, Tuple


def short_rate_variance(t: float, lam: float, sigma: float) -> float:
    """
    Exact variance of r(t) under Hull-White model:
      v(t) = sigma^2 * (1 - exp(-2*lambda*t)) / (2*lambda).
    """
    return sigma ** 2 * (1 - np.exp(-2 * lam * t)) / (2 * lam)


def short_rate_mean(
    t: float, lam: float, sigma: float, r0: float, b_func: Optional[callable] = None
) -> float:
    """
    Mean of r(t) = exp(-lambda*t)*r0 + int_0^t exp(-lambda*(t-s)) * b(s) ds.

    If b_func is None, uses b(s)=0 (zero drift for asymptotic analysis where
    initial condition effect decays).

    Parameters
    ----------
    b_func : callable(s) -> float, optional
    """
    decay = np.exp(-lam * t) * r0
    if b_func is None:
        return decay
    integral_val, _ = integrate.quad(
        lambda s: np.exp(-lam * (t - s)) * b_func(s), 0, t, limit=500
    )
    return decay + integral_val


def short_rate_moments(
    t: float, lam: float, sigma: float, r0: float = 0.0
) -> Tuple[float, float]:
    """
    Return (mean, variance) of r(t) in Hull-White model.
    Mean includes initial rate effect; variance is independent of r0.
    """
    m = np.exp(-lam * t) * r0  # asymptotic: initial condition decays
    v = short_rate_variance(t, lam, sigma)
    return m, v


# ─────────────────────────────────────────────────────────────────────────────
# Analytic asymptotic probability formulas
# ─────────────────────────────────────────────────────────────────────────────

def yield_asymptotic_probs(lam: float, sigma: float) -> Dict[str, float]:
    """
    Compute theoretical asymptotic yield-curve shape probabilities.

    Q(inverse) = 0.5
    Q(humped)  = 0.5 - Phi(-3*sigma/sqrt(8*lambda^3))
    Q(normal)  = Phi(-3*sigma/sqrt(8*lambda^3))
    """
    arg = -3 * sigma / np.sqrt(8 * lam ** 3)
    phi_val = stats.norm.cdf(arg)
    return {
        "inverse": 0.5,
        "humped": 0.5 - phi_val,
        "normal": phi_val,
    }


def forward_asymptotic_probs_generic(lam: float, sigma: float) -> Dict[str, float]:
    """
    Asymptotic forward-curve shape probabilities when
    lambda < min{1/tau1, 1/tau2} (generic case).

    Q(inverse) = 0.5
    Q(humped)  = 0.5 - Phi(-sqrt(2)*sigma / lambda^(3/2))
    Q(normal)  = Phi(-sqrt(2)*sigma / lambda^(3/2))
    """
    arg = -np.sqrt(2) * sigma / lam ** 1.5
    phi_val = stats.norm.cdf(arg)
    return {
        "inverse": 0.5,
        "humped": 0.5 - phi_val,
        "normal": phi_val,
    }


def forward_asymptotic_probs_terminal_increasing(lam: float, sigma: float) -> Dict[str, float]:
    """
    Asymptotic forward-curve shape probabilities under terminal-increasing condition.

    Q(dipped) = 0.5
    Q(hd)     = 0.5 - Phi(-sqrt(2)*sigma / lambda^(3/2))
    Q(normal) = Phi(-sqrt(2)*sigma / lambda^(3/2))
    """
    arg = -np.sqrt(2) * sigma / lam ** 1.5
    phi_val = stats.norm.cdf(arg)
    return {
        "dipped": 0.5,
        "hd": 0.5 - phi_val,
        "normal": phi_val,
    }


def forward_asymptotic_probs_terminal_decreasing() -> Dict[str, float]:
    """
    Asymptotic forward-curve shape probabilities under terminal-decreasing condition.

    Q(inverse) = 0.5
    Q(humped)  = 0.5
    """
    return {
        "inverse": 0.5,
        "humped": 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shape probability via direct Gaussian integration
# ─────────────────────────────────────────────────────────────────────────────

def compute_shape_probs_gaussian(
    t: float,
    lam: float, sigma: float,
    r0_init: float,
    hw_state,
    r_boundaries: Dict[str, float],
    r_grid: np.ndarray,
    curve_type: str = "yield",
) -> Dict[str, float]:
    """
    Compute shape probabilities via numerical Gaussian integration.

    For each r in a dense grid, classify the shape, then integrate
    the Gaussian density N(m(t), v(t)) over shape regions.

    Parameters
    ----------
    t : time
    lam, sigma : model parameters
    r0_init : initial short rate
    hw_state : HullWhiteCurveState object
    r_boundaries : dict of boundary r values (if available)
    r_grid : dense grid of r values for classification
    curve_type : "forward" or "yield"

    Returns
    -------
    Dict mapping shape -> probability
    """
    from .root_counting import count_extrema_and_classify

    m, v = short_rate_moments(t, lam, sigma, r0_init)
    std = np.sqrt(v)

    # Classify each r
    shape_intervals: Dict[str, List[Tuple[float, float]]] = {}
    prev_shape = None
    seg_start = r_grid[0]

    shapes_at_r = []
    for r in r_grid:
        if curve_type == "forward":
            def d_func(x, _t=t, _r=r):
                xa = np.atleast_1d(np.asarray(x, dtype=float))
                res = hw_state.d_forward_dx(xa, _t, _r)
                return res if xa.shape != (1,) or np.ndim(x) > 0 else float(res[0])
        else:
            def d_func(x, _t=t, _r=r):
                xa = np.atleast_1d(np.asarray(x, dtype=float))
                res = hw_state.d_yield_dx(xa, _t, _r)
                return res if xa.shape != (1,) or np.ndim(x) > 0 else float(res[0])

        n_ext, shape, _ = count_extrema_and_classify(
            d_func, x_start=1e-6, n_grid=100
        )
        shapes_at_r.append(shape)

    # Group consecutive same-shape regions
    i = 0
    while i < len(r_grid):
        current_shape = shapes_at_r[i]
        j = i + 1
        while j < len(r_grid) and shapes_at_r[j] == current_shape:
            j += 1
        # Region: [r_grid[i], r_grid[min(j, len-1)]]
        r_lo = r_grid[i]
        r_hi = r_grid[min(j, len(r_grid) - 1)]
        # Integrate N(m, v) over [r_lo, r_hi]
        prob = stats.norm.cdf(r_hi, loc=m, scale=std) - stats.norm.cdf(r_lo, loc=m, scale=std)
        if current_shape not in shape_intervals:
            shape_intervals[current_shape] = 0.0
        shape_intervals[current_shape] += prob
        i = j

    return shape_intervals


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo estimation
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_shape_probs(
    t: float,
    lam: float, sigma: float,
    r0_init: float,
    hw_state,
    curve_type: str = "yield",
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Monte Carlo estimate of shape probabilities at time t.

    Draws r(t) ~ N(m(t), v(t)), classifies shape for each draw.

    Parameters
    ----------
    n_samples : number of Monte Carlo samples

    Returns
    -------
    Dict mapping shape -> empirical frequency
    """
    from .root_counting import count_extrema_and_classify

    rng = np.random.default_rng(seed)
    m, v = short_rate_moments(t, lam, sigma, r0_init)
    r_samples = rng.normal(loc=m, scale=np.sqrt(v), size=n_samples)

    shape_counts: Dict[str, int] = {}
    for r in r_samples:
        if curve_type == "forward":
            def d_func(x, _t=t, _r=float(r)):
                xa = np.atleast_1d(np.asarray(x, dtype=float))
                res = hw_state.d_forward_dx(xa, _t, _r)
                return res if xa.shape != (1,) or np.ndim(x) > 0 else float(res[0])
        else:
            def d_func(x, _t=t, _r=float(r)):
                xa = np.atleast_1d(np.asarray(x, dtype=float))
                res = hw_state.d_yield_dx(xa, _t, _r)
                return res if xa.shape != (1,) or np.ndim(x) > 0 else float(res[0])

        _, shape, _ = count_extrema_and_classify(d_func, x_start=1e-6, n_grid=80)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    return {s: c / n_samples for s, c in shape_counts.items()}


def monte_carlo_mc_se(probs: Dict[str, float], n_samples: int) -> Dict[str, float]:
    """Compute Monte Carlo standard errors: se = sqrt(p*(1-p)/n)."""
    return {s: np.sqrt(p * (1 - p) / n_samples) for s, p in probs.items()}
