"""
State-space segmentation for Hull-White model shape classification.

Implements:
  - Flat family: explicit initial and terminal boundaries
  - Monotone family: r0(t), terminal boundaries, forward envelope
  - Nelson-Siegel special case (lambda=1/tau): r0(t), envelope existence, tracking function
  - Bliss/Svensson: r0(t), terminal boundaries, numerical envelope tracing

All boundary formulas are implemented analytically as specified in the paper.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq, fsolve, minimize_scalar
from typing import Callable, List, Optional, Tuple, Union
from .initial_curves import (
    FlatCurve, MonotoneCurve, NelsonSiegelCurve, BlissCurve, SvenssonCurve
)


# ─────────────────────────────────────────────────────────────────────────────
# Flat family boundaries
# ─────────────────────────────────────────────────────────────────────────────

def r0_flat(t: float, beta0: float, lam: float, sigma: float) -> float:
    """
    Initial-sign boundary for flat curves:
      r0(t) = beta0 + (sigma^2/(2*lambda^2))*(1-exp(-2*lambda*t)).

    For r > r0: initial forward/yield derivative is positive (normal/humped behavior).
    For r < r0: initial derivative is negative (inverse/dipped behavior).
    """
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return beta0 + C


def rinf_forward_flat(t: float, beta0: float, lam: float, sigma: float) -> float:
    """
    Terminal forward boundary for flat curves:
      r_infty^f(t) = beta0 - (sigma^2/(2*lambda^2))*(1-exp(-2*lambda*t)).

    For r > rinf_f: forward curve is eventually increasing (has no more extremum at large x).
    """
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return beta0 - C


def rinf_yield_flat(t: float, beta0: float, lam: float, sigma: float) -> float:
    """
    Terminal yield boundary for flat curves:
      r_infty^y(t) = beta0 - (sigma^2/(4*lambda^2))*(1-exp(-2*lambda*t)).
    """
    C = (sigma ** 2 / (4 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return beta0 - C


# ─────────────────────────────────────────────────────────────────────────────
# Monotone family boundaries
# ─────────────────────────────────────────────────────────────────────────────

def r0_monotone(
    t: float, tau: float, beta0: float, beta1: float, lam: float, sigma: float
) -> float:
    """
    Initial-sign boundary for monotone curves:
      r0(t) = f0(t) + f0'(t)/lambda + C(t)
    where f0(t) = beta0 + beta1*exp(-t/tau),
          f0'(t) = -(beta1/tau)*exp(-t/tau),
          C(t) = (sigma^2/(2*lambda^2))*(1-exp(-2*lambda*t)).
    """
    curve = MonotoneCurve(tau, beta0, beta1)
    f0t = float(curve.f0(t))
    f0p = float(curve.f0_prime(t))
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return f0t + f0p / lam + C


def r0_general(t: float, curve, lam: float, sigma: float) -> float:
    """
    General initial-sign boundary:
      r0(t) = f0(t) + f0'(t)/lambda + (sigma^2/(2*lambda^2))*(1-exp(-2*lambda*t)).
    Valid for all curve families.
    """
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return float(curve.f0(t)) + float(curve.f0_prime(t)) / lam + C


def rinf_forward_general(t: float, curve, lam: float, sigma: float) -> Optional[float]:
    """
    Terminal forward boundary (when well-defined):
      r_infty^f(t) = f0(t) - C(t).
    Valid when the leading exponential term in f0 dominates (e.g., 1/tau > lambda).
    """
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return float(curve.f0(t)) - C


def rinf_yield_svensson(
    t: float,
    tau1: float, tau2: float,
    beta0: float, beta1: float, beta2: float, beta3: float,
    lam: float, sigma: float
) -> float:
    """
    Terminal yield boundary for Svensson family:
      r_infty^y(t) = beta0 + beta1*(1-lambda*tau1)*exp(-t/tau1)
                   + beta2*(t/tau1 - lambda*(tau1+t))*exp(-t/tau1)
                   + beta3*(t/tau2 - lambda*(tau2+t))*exp(-t/tau2)
                   - (sigma^2/(4*lambda^2))*(1-exp(-2*lambda*t)).
    """
    e1 = np.exp(-t / tau1)
    e2 = np.exp(-t / tau2)
    C = (sigma ** 2 / (4 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return (
        beta0
        + beta1 * (1 - lam * tau1) * e1
        + beta2 * (t / tau1 - lam * (tau1 + t)) * e1
        + beta3 * (t / tau2 - lam * (tau2 + t)) * e2
        - C
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monotone forward envelope
# ─────────────────────────────────────────────────────────────────────────────

def forward_envelope_monotone(
    t: float,
    tau: float, beta0: float, beta1: float,
    lam: float, sigma: float,
) -> Optional[Tuple[float, float]]:
    """
    Forward envelope for monotone curves when 2*lambda != 1/tau.
    Envelope exists at maturity x where:
      x = alpha * log(gamma(t))
    where alpha = 1 / (2*lambda - 1/tau),
          gamma(t) = sigma^2 * tau * (1-exp(-2*lambda*t)) * exp(t/tau) / (beta1*(1/tau-lambda))

    Returns (x_env, r_env) if envelope exists, else None.
    """
    denom = 2 * lam - 1.0 / tau
    if abs(denom) < 1e-10:
        return None  # special case, handle separately

    alpha = 1.0 / denom

    denom2 = beta1 * (1.0 / tau - lam)
    if abs(denom2) < 1e-12:
        return None

    numer = sigma ** 2 * tau * (1 - np.exp(-2 * lam * t)) * np.exp(t / tau)
    gamma_t = numer / denom2

    if gamma_t <= 0:
        return None  # log not defined

    x_env = alpha * np.log(gamma_t)
    if x_env <= 0:
        return None  # must be positive maturity

    # Compute corresponding r from forward tracking function
    r_env = rx_forward_monotone(x_env, t, tau, beta0, beta1, lam, sigma)
    return (x_env, r_env)


def rx_forward_monotone(
    x: float, t: float,
    tau: float, beta0: float, beta1: float,
    lam: float, sigma: float,
) -> float:
    """
    Forward tracking function: given envelope maturity x and time t,
    compute the state r on the forward envelope.

      r_x^f(t) = f0'(x+t) * exp(lambda*x) / lambda
                 - ((sigma^2/(2*lambda^2))*(1-exp(-2*lambda*t)) - f0(t))
                 + (sigma^2/lambda^2)*(1-exp(-2*lambda*t)) * exp(-lambda*x).

    This is derived by setting df/dx = 0 and solving for K(t,r).
    """
    curve = MonotoneCurve(tau, beta0, beta1)
    f0p = float(curve.f0_prime(x + t))
    f0t = float(curve.f0(t))
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))

    # df/dx = f0'(x+t) - lambda*K*exp(-lambda*x) + 2*lambda*C*exp(-2*lambda*x) = 0
    # => lambda*K*exp(-lambda*x) = f0'(x+t) + 2*lambda*C*exp(-2*lambda*x)
    # => K = f0'(x+t)*exp(lambda*x)/lambda + 2*C*exp(-lambda*x)
    # K = C - f0(t) + r => r = K + f0(t) - C

    K = f0p * np.exp(lam * x) / lam + 2 * C * np.exp(-lam * x)
    r = K + f0t - C
    return r


def rx_forward_general(
    x: float, t: float, curve, lam: float, sigma: float
) -> float:
    """
    General forward tracking function (r on forward envelope at maturity x):
      r = f0'(x+t) * exp(lambda*x) / lambda + 2*C(t)*exp(-lambda*x) + f0(t) - C(t).
    Derived by setting df/dx = 0 and solving for r.
    """
    f0p = float(curve.f0_prime(x + t))
    f0t = float(curve.f0(t))
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    K = f0p * np.exp(lam * x) / lam + 2 * C * np.exp(-lam * x)
    r = K + f0t - C
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Nelson-Siegel special case (lambda = 1/tau)
# ─────────────────────────────────────────────────────────────────────────────

def r0_ns_special(
    t: float,
    tau: float, beta0: float, beta1: float, beta2: float,
    sigma: float,
) -> float:
    """
    Initial-sign boundary for NS with lambda=1/tau:
      r0(t) = beta0 + beta2*exp(-lambda*t) + C(t).
    """
    lam = 1.0 / tau
    C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
    return beta0 + beta2 * np.exp(-lam * t) + C


def ns_special_delta(
    t: float,
    tau: float, beta2: float, sigma: float,
) -> float:
    """
    Forward envelope existence discriminant for NS special case:
      delta(t) = -(sigma^2 * tau^2 / beta2) * (1-exp(-2*lambda*t)) * exp(lambda*t).
    Envelope exists if delta(t) > 0 and x = (1/lambda)*log(delta(t)) > 0.
    """
    lam = 1.0 / tau
    if abs(beta2) < 1e-12:
        return -np.inf
    return -(sigma ** 2 * tau ** 2 / beta2) * (1 - np.exp(-2 * lam * t)) * np.exp(lam * t)


def ns_special_envelope(
    t: float,
    tau: float, beta0: float, beta1: float, beta2: float,
    sigma: float,
) -> Optional[Tuple[float, float]]:
    """
    Forward envelope for NS with lambda=1/tau.
    Returns (x_env, r_env) if envelope exists.
    """
    lam = 1.0 / tau
    delta = ns_special_delta(t, tau, beta2, sigma)
    if delta <= 0:
        return None
    x_env = (1.0 / lam) * np.log(delta)
    if x_env <= 0:
        return None
    curve = NelsonSiegelCurve(tau, beta0, beta1, beta2)
    r_env = rx_forward_general(x_env, t, curve, lam, sigma)
    return (x_env, r_env)


def ns_special_tracking_initial_direction(
    t: float,
    tau: float, beta2: float, sigma: float,
) -> float:
    """
    Initial direction of the forward tracking function for NS special case:
      d/dt (r_x^f)|_{initial} = -beta2*exp(-lambda*t) - (sigma^2/lambda^2)*(1-exp(-2*lambda*t)).

    Positive value means tracking function is increasing in r (for increasing x).
    """
    lam = 1.0 / tau
    C = (sigma ** 2 / lam ** 2) * (1 - np.exp(-2 * lam * t))
    return -beta2 * np.exp(-lam * t) - C


# ─────────────────────────────────────────────────────────────────────────────
# Numerical envelope tracing (general)
# ─────────────────────────────────────────────────────────────────────────────

def trace_forward_envelope_numerical(
    curve,
    lam: float, sigma: float,
    t_grid: np.ndarray,
    x_search_grid: np.ndarray = None,
    xtol: float = 1e-10,
) -> List[Tuple[float, float, float]]:
    """
    Numerically trace the forward envelope by solving:
      df/dx = 0  and  d^2f/dx^2 = 0  simultaneously,
    which is equivalent to solving:
      f0''(x+t) + lambda*f0'(x+t) = sigma^2*(1-exp(-2*lambda*t))*exp(-2*lambda*x).

    For each t in t_grid, finds x solutions, then computes r.

    Returns list of (t, x_env, r_env) tuples.
    """
    if x_search_grid is None:
        x_search_grid = np.linspace(0.01, 40.0, 2000)

    results = []

    for t in t_grid:
        C = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))
        rhs_t = 2 * sigma ** 2 / (sigma ** 2 / C) if C > 1e-12 else 0.0
        # Reduced equation: f0''(x+t) + lambda*f0'(x+t) = sigma^2*(1-exp(-2*lambda*t))*exp(-2*lambda*x)
        rhs_coef = sigma ** 2 * (1 - np.exp(-2 * lam * t))

        def eq(x):
            f0pp = float(curve.f0_second(x + t))
            f0p = float(curve.f0_prime(x + t))
            rhs = rhs_coef * np.exp(-2 * lam * x)
            return f0pp + lam * f0p - rhs

        # Find sign changes
        vals = np.array([eq(xi) for xi in x_search_grid])
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                try:
                    x_sol = brentq(eq, x_search_grid[i], x_search_grid[i + 1], xtol=xtol)
                    r_sol = rx_forward_general(x_sol, t, curve, lam, sigma)
                    results.append((t, x_sol, r_sol))
                except Exception:
                    pass

    return results


def trace_yield_envelope_numerical(
    hw_state,
    t_grid: np.ndarray,
    r_search_grid: np.ndarray = None,
    x_search_grid: np.ndarray = None,
    xtol: float = 1e-10,
) -> List[Tuple[float, float, float]]:
    """
    Numerically trace yield envelope by solving dy/dx=0 and df/dx=0 simultaneously.
    For each t, search over (x, r) space.

    Returns list of (t, x_env, r_env) tuples where both conditions hold.
    """
    if r_search_grid is None:
        r_search_grid = np.linspace(-5, 5, 30)
    if x_search_grid is None:
        x_search_grid = np.linspace(0.1, 30.0, 300)

    results = []

    for t in t_grid:
        for x in x_search_grid:
            # Find r such that dy/dx=0 (i.e., f=y) and df/dx=0 (i.e., on forward envelope)
            # df/dx = 0 gives unique r for each x:
            r_f = rx_forward_general(x, t, hw_state.curve, hw_state.lam, hw_state.sigma)
            # Check if dy/dx is also 0 at this r:
            dy_val = float(hw_state.d_yield_dx(np.array([x]), t, r_f))
            if abs(dy_val) < 1e-4:
                results.append((t, x, r_f))

    return results
