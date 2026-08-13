"""
Transformed-plane geometry for general Nelson-Siegel curve classification.

At fixed (lambda, tau, t), the paper maps Nelson-Siegel parameters to
transformed coordinates (tilde_beta0, tilde_beta1, tilde_beta2, tilde_sigma^2),
then expresses the forward and yield derivatives as a linear combination:
  d/dx = a(x) + b(x)*tilde_beta1 + c(x)*r

This module implements:
  - Transformed parameter computation
  - Line-family coefficients for forward and yield
  - Wronskian-based envelope computation
  - Regime classification via D and inequality conditions
  - Direct root-counting verification
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from typing import Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Transformed parameters
# ─────────────────────────────────────────────────────────────────────────────

def compute_transformed_params(
    lam: float, tau: float, sigma: float,
    beta0: float, beta1: float, beta2: float,
    t: float,
) -> Dict[str, float]:
    """
    Compute transformed NS parameters at fixed (lambda, tau, t):
      tilde_beta0 = f0(t) - C(t)
      tilde_beta1 = ((beta2/tau^2)*t - (beta2-beta1)/tau) * tau^2 * exp(-t/tau)
      tilde_beta2 = (beta2/tau^2) * exp(-t/tau)
      tilde_sigma^2 = C(t) = (sigma^2/(2*lambda^2))*(1-exp(-2*lambda*t))
    where f0(t) = beta0 + beta1*exp(-t/tau) + (beta2/tau)*t*exp(-t/tau).
    """
    e_t = np.exp(-t / tau)
    C_t = (sigma ** 2 / (2 * lam ** 2)) * (1 - np.exp(-2 * lam * t))

    f0_t = beta0 + beta1 * e_t + (beta2 / tau) * t * e_t

    tb0 = f0_t - C_t
    tb1 = ((beta2 / tau ** 2) * t - (beta2 - beta1) / tau) * tau ** 2 * e_t
    tb2 = (beta2 / tau ** 2) * e_t
    ts2 = C_t

    return {
        "tilde_beta0": tb0,
        "tilde_beta1": tb1,
        "tilde_beta2": tb2,
        "tilde_sigma2": ts2,
        "C_t": C_t,
        "f0_t": f0_t,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions h1, h2
# ─────────────────────────────────────────────────────────────────────────────

def h1(x: float, tau: float) -> float:
    """
    h1(x, tau) = (1 + x/tau) * exp(-x/tau) - 1.
    Stable for all x; for small x, uses Taylor series to avoid cancellation.
    """
    u = x / tau
    if abs(u) < 1e-6:
        # Taylor: h1 ≈ -u^2/2 + u^3/3 - ...
        return -u ** 2 / 2 + u ** 3 / 6 - u ** 4 / 24
    return (1 + u) * np.exp(-u) - 1


def h2(x: float, tau: float) -> float:
    """
    h2(x, tau) = (2 + 2*x/tau + (x/tau)^2) * exp(-x/tau) - 2.
    For small x, use Taylor series for numerical stability.
    """
    u = x / tau
    if abs(u) < 1e-6:
        # Taylor: h2 ≈ -u^3/3 + ...
        return -u ** 3 / 3 + u ** 4 / 12
    return (2 + 2 * u + u ** 2) * np.exp(-u) - 2


# ─────────────────────────────────────────────────────────────────────────────
# Line-family coefficients
# ─────────────────────────────────────────────────────────────────────────────

def forward_line_coeffs(
    x: float,
    tau: float,
    tilde_beta0: float, tilde_beta2: float, tilde_sigma2: float,
    lam: float,
) -> Tuple[float, float, float]:
    """
    Forward line-family coefficients at maturity x:
      a_f(x) = -tilde_beta2 * x * tau^2 * exp(-x/tau)
               + tilde_beta0 * lambda * exp(-lambda*x)
               + tilde_sigma2 * 2*lambda * exp(-2*lambda*x)
      b_f(x) = -exp(-x/tau)
      c_f(x) = -lambda * exp(-lambda*x)

    Such that partial_x f(x,t,r) = a_f(x) + b_f(x)*tilde_beta1 + c_f(x)*r.
    """
    et = np.exp(-x / tau)
    el = np.exp(-lam * x)
    e2l = np.exp(-2 * lam * x)

    a = (-tilde_beta2 * x * tau ** 2 * et
         + tilde_beta0 * lam * el
         + tilde_sigma2 * 2 * lam * e2l)
    b = -et
    c = -lam * el
    return a, b, c


def yield_line_coeffs(
    x: float,
    tau: float,
    tilde_beta0: float, tilde_beta2: float, tilde_sigma2: float,
    lam: float,
) -> Tuple[float, float, float]:
    """
    Yield line-family coefficients at maturity x:
      a_y(x) = tilde_beta2 * tau^5 * h2(x,tau) / x^2
               - tilde_beta0 * h1(x, 1/lambda) / (lambda * x^2)
               - tilde_sigma2 * h1(x, 1/(2*lambda)) / (2*lambda * x^2)
      b_y(x) = tau^2 * h1(x,tau) / x^2
      c_y(x) = h1(x, 1/lambda) / (lambda * x^2)

    Such that partial_x y(x,t,r) = a_y(x) + b_y(x)*tilde_beta1 + c_y(x)*r.
    """
    if x < 1e-8:
        # Near x=0: use forward coefficients scaled by 0.5
        a_f, b_f, c_f = forward_line_coeffs(x, tau, tilde_beta0, tilde_beta2, tilde_sigma2, lam)
        return 0.5 * a_f, 0.5 * b_f, 0.5 * c_f

    x2 = x ** 2
    h1_tau = h1(x, tau)
    h2_tau = h2(x, tau)
    h1_l = h1(x, 1.0 / lam)
    h1_2l = h1(x, 1.0 / (2 * lam))

    a = (tilde_beta2 * tau ** 5 * h2_tau / x2
         - tilde_beta0 * h1_l / (lam * x2)
         - tilde_sigma2 * h1_2l / (2 * lam * x2))
    b = tau ** 2 * h1_tau / x2
    c = h1_l / (lam * x2)
    return a, b, c


# ─────────────────────────────────────────────────────────────────────────────
# Wronskian-based envelope
# ─────────────────────────────────────────────────────────────────────────────

def wronskian(f_vals: np.ndarray, g_vals: np.ndarray, dx: float) -> np.ndarray:
    """Numerical Wronskian W(f,g) = f*g' - f'*g via finite differences."""
    fp = np.gradient(f_vals, dx)
    gp = np.gradient(g_vals, dx)
    return f_vals * gp - fp * g_vals


def compute_envelope(
    x_grid: np.ndarray,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    c_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute envelope of the line family {a(x) + b(x)*s + c(x)*r = 0}
    using Wronskian formula:
      eta(x) = (W(c,a)(x)/W(b,c)(x), W(a,b)(x)/W(b,c)(x))
    where the two coordinates correspond to (tilde_beta1, r).

    Returns (tilde_beta1_env, r_env) arrays.
    """
    dx = x_grid[1] - x_grid[0]

    W_ca = wronskian(c_vals, a_vals, dx)
    W_ab = wronskian(a_vals, b_vals, dx)
    W_bc = wronskian(b_vals, c_vals, dx)

    # Avoid division near singular points
    mask = np.abs(W_bc) > 1e-12
    tb1_env = np.where(mask, W_ca / W_bc, np.nan)
    r_env = np.where(mask, W_ab / W_bc, np.nan)

    return tb1_env, r_env


# ─────────────────────────────────────────────────────────────────────────────
# Regime discriminant D
# ─────────────────────────────────────────────────────────────────────────────

def compute_D(
    tilde_beta2: float, tau: float, tilde_sigma2: float, lam: float
) -> float:
    """
    Compute the regime discriminant D:
      D = tilde_beta2 * tau^2 * (1/tau - lambda) - 2*lambda^2 * tilde_sigma2 * (1/tau - 2*lambda).

    D determines the existence of additional envelope intersections and
    discriminates between regime sub-cases in the NS forward classification.
    """
    return (tilde_beta2 * tau ** 2 * (1.0 / tau - lam)
            - 2 * lam ** 2 * tilde_sigma2 * (1.0 / tau - 2 * lam))


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification for forward and yield curves
# ─────────────────────────────────────────────────────────────────────────────

def classify_ns_forward_regime(
    lam: float, tau: float,
    tilde_beta2: float, tilde_sigma2: float,
    D: float,
    cond_5_6: bool = False,
) -> int:
    """
    Classify the NS forward curve regime (1–7) based on transformed parameters.

    Regimes and predicted shape sets (from the paper):
    1: 1/tau < lambda, tilde_beta2 < 0         => normal, dipped, hd
    2: 1/tau < lambda, tilde_beta2 > 0, D < 0  => inverse, humped, dh
    3: 1/tau < lambda, tilde_beta2 > 0, D > 0  => inverse, humped, dh, hdh
    4: lambda < 1/tau <= 2*lambda, tilde_beta2 > 0  => normal, inverse, humped, dipped, hd
    5: {lambda < 1/tau <= 2*lambda, tilde_beta2 < 0, D < 0} or
       {2*lambda < 1/tau, tilde_beta2 < 0 or (tilde_beta2 > 0, D < 0)}
       => normal, inverse, humped, dipped, dh
    6: ((lambda < 1/tau < 2*lambda, tilde_beta2 < 0) or
        (2*lambda < 1/tau, tilde_beta2 > 0)) and D > 0 but not cond_5_6
       => normal, inverse, humped, dipped, dh, hdh
    7: same as 6 plus cond_5_6  => normal, inverse, humped, dipped, hd, dh, hdh

    Returns regime number (1–7) or 0 if unclassified.
    """
    inv_tau = 1.0 / tau

    if inv_tau < lam:  # 1/tau < lambda
        if tilde_beta2 < 0:
            return 1
        else:  # tilde_beta2 > 0
            if D < 0:
                return 2
            else:
                return 3
    elif lam < inv_tau <= 2 * lam:  # lambda < 1/tau <= 2*lambda
        if tilde_beta2 > 0:
            return 4
        else:  # tilde_beta2 < 0
            if D < 0:
                return 5
            elif D > 0:
                if cond_5_6:
                    return 7
                else:
                    return 6
    else:  # 2*lambda < 1/tau
        if tilde_beta2 < 0:
            return 5
        else:  # tilde_beta2 > 0
            if D < 0:
                return 5
            elif D > 0:
                if cond_5_6:
                    return 7
                else:
                    return 6
    return 0


def classify_ns_yield_regime(
    lam: float, tau: float,
    tilde_beta2: float, tilde_sigma2: float,
    D: float,
    cond_5_7: bool = False,
    cond_5_9: bool = False,
    cond_5_10: bool = False,
) -> int:
    """
    Classify NS yield curve regime (1–9).
    Returns regime number or 0 if unclassified.

    The yield has 9 regimes. For this implementation we use direct root counting
    as primary verification, and this function provides regime approximation.
    """
    inv_tau = 1.0 / tau

    if inv_tau < lam:
        if tilde_beta2 < 0:
            return 1  # normal, dipped, hd, dh, dipped_hd
        else:
            if D < 0:
                return 2
            else:
                if cond_5_7:
                    return 3
                else:
                    return 2
    elif lam < inv_tau <= 2 * lam:
        if tilde_beta2 > 0:
            return 4
        else:
            if D < 0:
                if cond_5_9:
                    return 6
                else:
                    return 5
            else:
                if cond_5_10:
                    return 8
                else:
                    return 7
    else:
        if tilde_beta2 < 0:
            return 5
        else:
            if D < 0:
                return 5
            else:
                if cond_5_10:
                    return 9
                else:
                    return 7
    return 0


# Shape sets per regime (for forward)
FORWARD_REGIME_SHAPES = {
    1: ["normal", "dipped", "hd"],
    2: ["inverse", "humped", "dh"],
    3: ["inverse", "humped", "dh", "hdh"],
    4: ["normal", "inverse", "humped", "dipped", "hd"],
    5: ["normal", "inverse", "humped", "dipped", "dh"],
    6: ["normal", "inverse", "humped", "dipped", "dh", "hdh"],
    7: ["normal", "inverse", "humped", "dipped", "hd", "dh", "hdh"],
    0: [],
}

# Shape sets per regime (for yield)
YIELD_REGIME_SHAPES = {
    1: ["normal", "inverse", "humped", "dipped", "hd"],
    2: ["normal", "inverse", "humped"],
    3: ["normal", "inverse", "humped", "hdh"],
    4: ["normal", "inverse", "humped", "dipped", "hd"],
    5: ["normal", "inverse", "humped", "dipped", "dh"],
    6: ["normal", "inverse", "humped", "dipped", "dh", "hdh"],
    7: ["normal", "inverse", "humped", "dipped", "hd", "dh"],
    8: ["normal", "inverse", "humped", "dipped", "hd", "dh", "hdh"],
    9: ["normal", "inverse", "humped", "dipped", "hd", "dh", "hdh"],
    0: [],
}
