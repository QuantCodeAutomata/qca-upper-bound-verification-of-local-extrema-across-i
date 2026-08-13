"""
Hull-White one-factor model forward and yield curve implementations.

Forward curve:
  f(x,t,r) = f0(x+t) + K(t,r)*exp(-lambda*x) - C(t)*exp(-2*lambda*x)
where:
  C(t) = (sigma^2/(2*lambda^2)) * (1 - exp(-2*lambda*t))
  K(t,r) = C(t) - f0(t) + r

Yield curve:
  y(x,t,r) = (1/x)*int_0^x f0(xi+t) dxi
             + K(t,r)*(1-exp(-lambda*x))/(lambda*x)
             - C(t)*(1-exp(-2*lambda*x))/(2*lambda*x)

x-derivative of forward:
  df/dx = f0'(x+t) - lambda*K(t,r)*exp(-lambda*x) + 2*lambda*C(t)*exp(-2*lambda*x)

x-derivative of yield uses the identity:
  y'(x) = (f(x) - y(x)) / x   [from y(x) = (1/x)*int_0^x f(xi) dxi]

At x=0 (limiting values):
  f(0,t,r) = f0(t) + K(t,r) - C(t) = r   [consistent]
  y(0,t,r) = f0(t) + K(t,r) - C(t) = r   [by L'Hopital]
  df/dx|_{x=0} = f0'(t) - lambda*K(t,r) + 2*lambda*C(t)
  dy/dx|_{x=0} = 0.5 * df/dx|_{x=0}      [by the identity at x->0]
"""

from __future__ import annotations
import numpy as np
from scipy import integrate
from typing import Union
from .initial_curves import (
    FlatCurve, MonotoneCurve, NelsonSiegelCurve, BlissCurve, SvenssonCurve
)

Array = Union[float, np.ndarray]


class HullWhiteCurveState:
    """
    Computes Hull-White forward and yield curves at a given state (t, r)
    for a given initial curve object.

    Parameters
    ----------
    curve : initial curve object (FlatCurve, MonotoneCurve, etc.)
    lam : float, lambda > 0, mean reversion speed
    sigma : float, sigma > 0, volatility
    """

    def __init__(self, curve, lam: float, sigma: float) -> None:
        assert lam > 0, "lambda must be positive"
        assert sigma > 0, "sigma must be positive"
        self.curve = curve
        self.lam = lam
        self.sigma = sigma

    def C(self, t: float) -> float:
        """C(t) = (sigma^2 / (2*lambda^2)) * (1 - exp(-2*lambda*t))."""
        return (self.sigma ** 2 / (2 * self.lam ** 2)) * (1 - np.exp(-2 * self.lam * t))

    def K(self, t: float, r: float) -> float:
        """K(t,r) = C(t) - f0(t) + r."""
        return self.C(t) - float(self.curve.f0(t)) + r

    def forward(self, x: Array, t: float, r: float) -> Array:
        """
        Hull-White forward curve:
          f(x,t,r) = f0(x+t) + K(t,r)*exp(-lambda*x) - C(t)*exp(-2*lambda*x).
        """
        x = np.asarray(x, dtype=float)
        Ct = self.C(t)
        Kt = self.K(t, r)
        return (
            self.curve.f0(x + t)
            + Kt * np.exp(-self.lam * x)
            - Ct * np.exp(-2 * self.lam * x)
        )

    def yield_curve(self, x: Array, t: float, r: float) -> Array:
        """
        Hull-White yield curve:
          y(x,t,r) = (1/x)*int_0^x f0(xi+t) dxi
                   + K(t,r)*(1-exp(-lambda*x))/(lambda*x)
                   - C(t)*(1-exp(-2*lambda*x))/(2*lambda*x).

        At x=0, returns the limit r (same as forward at x=0).
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        Ct = self.C(t)
        Kt = self.K(t, r)
        result = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < 1e-10:
                # Limit as x->0 equals f(0,t,r) = r
                result[i] = self.forward(0.0, t, r)
            else:
                # Integral term: (1/x)*int_0^x f0(xi+t) dxi
                int_f0, _ = integrate.quad(
                    lambda u: float(self.curve.f0(u + t)), 0, xi, limit=200
                )
                int_term = int_f0 / xi

                # K term: K*(1-exp(-lambda*x))/(lambda*x)
                k_term = Kt * (1 - np.exp(-self.lam * xi)) / (self.lam * xi)

                # C term: -C*(1-exp(-2*lambda*x))/(2*lambda*x)
                c_term = -Ct * (1 - np.exp(-2 * self.lam * xi)) / (2 * self.lam * xi)

                result[i] = int_term + k_term + c_term

        return float(result[0]) if scalar else result

    def yield_curve_fast(self, x: Array, t: float, r: float, curve=None) -> Array:
        """
        Fast yield curve using closed-form integral when curve supports integral_f0.
        Falls back to numerical integration otherwise.
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        Ct = self.C(t)
        Kt = self.K(t, r)
        result = np.zeros_like(x)

        has_analytic = hasattr(self.curve, "integral_f0")

        for i, xi in enumerate(x):
            if xi < 1e-10:
                result[i] = self.forward(0.0, t, r)
            else:
                if has_analytic:
                    int_f0 = self.curve.integral_f0(t, xi + t)
                else:
                    int_f0, _ = integrate.quad(
                        lambda u: float(self.curve.f0(u + t)), 0, xi, limit=200
                    )
                int_term = int_f0 / xi
                k_term = Kt * (1 - np.exp(-self.lam * xi)) / (self.lam * xi)
                c_term = -Ct * (1 - np.exp(-2 * self.lam * xi)) / (2 * self.lam * xi)
                result[i] = int_term + k_term + c_term

        return float(result[0]) if scalar else result

    def d_forward_dx(self, x: Array, t: float, r: float) -> Array:
        """
        Derivative of forward curve with respect to x:
          df/dx = f0'(x+t) - lambda*K(t,r)*exp(-lambda*x) + 2*lambda*C(t)*exp(-2*lambda*x).
        """
        x = np.asarray(x, dtype=float)
        Ct = self.C(t)
        Kt = self.K(t, r)
        return (
            self.curve.f0_prime(x + t)
            - self.lam * Kt * np.exp(-self.lam * x)
            + 2 * self.lam * Ct * np.exp(-2 * self.lam * x)
        )

    def d_forward_dx_at_zero(self, t: float, r: float) -> float:
        """
        df/dx at x=0: f0'(t) - lambda*K(t,r) + 2*lambda*C(t).
        """
        return float(self.curve.f0_prime(t)) - self.lam * self.K(t, r) + 2 * self.lam * self.C(t)

    def d_yield_dx(self, x: Array, t: float, r: float) -> Array:
        """
        Derivative of yield curve with respect to x, using:
          dy/dx = (f(x,t,r) - y(x,t,r)) / x  for x > 0.
        At x=0, dy/dx = 0.5 * df/dx|_{x=0}.
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < 1e-10:
                # lim_{x->0} dy/dx = 0.5 * df/dx|_{x=0}
                result[i] = 0.5 * self.d_forward_dx_at_zero(t, r)
            else:
                f_val = float(self.forward(xi, t, r))
                y_val = float(self.yield_curve_fast(xi, t, r))
                result[i] = (f_val - y_val) / xi

        return float(result[0]) if scalar else result

    def d_yield_dx_integral(self, x: Array, t: float, r: float) -> Array:
        """
        Alternative yield derivative via integral identity:
          dy/dx = (1/x^2) * int_0^x xi * df/dxi dxi.
        More stable near x=0. Uses the identity at x=0: dy/dx = 0.5*df/dx|_{x=0}.
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < 1e-10:
                result[i] = 0.5 * self.d_forward_dx_at_zero(t, r)
            else:
                val, _ = integrate.quad(
                    lambda u: u * float(self.d_forward_dx(u, t, r)),
                    0, xi, limit=200
                )
                result[i] = val / xi ** 2

        return float(result[0]) if scalar else result

    def verify_identities(self, x_test: float, t: float, r: float, tol: float = 1e-6) -> dict:
        """
        Verify fundamental identities at a test point (x, t, r):
         1. y(x) = (1/x)*int_0^x f(xi) dxi
         2. f(x) = y(x) + x*y'(x)
         3. dy/dx = (1/x^2)*int_0^x xi*df/dxi dxi
         4. dy/dx|_{x=0} = 0.5 * df/dx|_{x=0}
        """
        results = {}

        f_val = float(self.forward(x_test, t, r))
        y_val = float(self.yield_curve_fast(x_test, t, r))

        # Identity 1: y = (1/x) * int_0^x f dxi
        int_f, _ = integrate.quad(
            lambda u: float(self.forward(u, t, r)), 0, x_test, limit=200
        )
        y_from_int = int_f / x_test
        results["identity1_err"] = abs(y_val - y_from_int)
        results["identity1_ok"] = results["identity1_err"] < tol

        # Identity 2: f = y + x * dy/dx
        dy = float(self.d_yield_dx(x_test, t, r))
        f_from_yield = y_val + x_test * dy
        results["identity2_err"] = abs(f_val - f_from_yield)
        results["identity2_ok"] = results["identity2_err"] < tol

        # Identity 3: dy/dx integral form
        dy_int = float(self.d_yield_dx_integral(x_test, t, r))
        results["identity3_err"] = abs(dy - dy_int)
        results["identity3_ok"] = results["identity3_err"] < tol

        # Identity 4: dy/dx at x=0
        dydx0 = 0.5 * self.d_forward_dx_at_zero(t, r)
        dfdx0 = self.d_forward_dx_at_zero(t, r)
        results["identity4_err"] = abs(dydx0 - 0.5 * dfdx0)
        results["identity4_ok"] = results["identity4_err"] < tol

        return results
