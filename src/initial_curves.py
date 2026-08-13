"""
Initial forward-curve families for Hull-White model shape experiments.

Each family is implemented as a class with methods:
  f0(x)         - initial forward curve
  f0_prime(x)   - first derivative
  f0_second(x)  - second derivative
  f0_third(x)   - third derivative (where meaningful)

Parameter naming follows the paper exactly.
"""

from __future__ import annotations
import numpy as np
from typing import Union

Array = Union[float, np.ndarray]


class FlatCurve:
    """
    Flat initial forward curve: f0(x) = beta0.
    All derivatives are identically zero.
    """

    def __init__(self, beta0: float) -> None:
        self.beta0 = beta0
        self.name = "Flat"
        self.n_params = 1

    def f0(self, x: Array) -> Array:
        """f0(x) = beta0."""
        return np.full_like(np.asarray(x, dtype=float), self.beta0)

    def f0_prime(self, x: Array) -> Array:
        """f0'(x) = 0."""
        return np.zeros_like(np.asarray(x, dtype=float))

    def f0_second(self, x: Array) -> Array:
        """f0''(x) = 0."""
        return np.zeros_like(np.asarray(x, dtype=float))

    def f0_third(self, x: Array) -> Array:
        """f0'''(x) = 0."""
        return np.zeros_like(np.asarray(x, dtype=float))

    def integral_f0(self, a: float, b: float) -> float:
        """Integral of f0 from a to b: beta0 * (b - a)."""
        return self.beta0 * (b - a)

    def __repr__(self) -> str:
        return f"FlatCurve(beta0={self.beta0})"


class MonotoneCurve:
    """
    Monotone initial forward curve: f0(x) = beta0 + beta1 * exp(-x/tau).

    Parameters
    ----------
    tau : float > 0
    beta0, beta1 : floats
    """

    def __init__(self, tau: float, beta0: float, beta1: float) -> None:
        assert tau > 0, "tau must be positive"
        self.tau = tau
        self.beta0 = beta0
        self.beta1 = beta1
        self.name = "Monotone"
        self.n_params = 3

    def f0(self, x: Array) -> Array:
        """f0(x) = beta0 + beta1 * exp(-x/tau)."""
        x = np.asarray(x, dtype=float)
        return self.beta0 + self.beta1 * np.exp(-x / self.tau)

    def f0_prime(self, x: Array) -> Array:
        """f0'(x) = -(beta1/tau) * exp(-x/tau)."""
        x = np.asarray(x, dtype=float)
        return -(self.beta1 / self.tau) * np.exp(-x / self.tau)

    def f0_second(self, x: Array) -> Array:
        """f0''(x) = (beta1/tau^2) * exp(-x/tau)."""
        x = np.asarray(x, dtype=float)
        return (self.beta1 / self.tau ** 2) * np.exp(-x / self.tau)

    def f0_third(self, x: Array) -> Array:
        """f0'''(x) = -(beta1/tau^3) * exp(-x/tau)."""
        x = np.asarray(x, dtype=float)
        return -(self.beta1 / self.tau ** 3) * np.exp(-x / self.tau)

    def integral_f0(self, a: float, b: float) -> float:
        """Integral of f0 from a to b."""
        return self.beta0 * (b - a) + self.beta1 * self.tau * (
            np.exp(-a / self.tau) - np.exp(-b / self.tau)
        )

    def __repr__(self) -> str:
        return f"MonotoneCurve(tau={self.tau}, beta0={self.beta0}, beta1={self.beta1})"


class NelsonSiegelCurve:
    """
    Nelson-Siegel initial forward curve:
      f0(x) = beta0 + beta1*exp(-x/tau) + (beta2/tau)*x*exp(-x/tau).

    Parameters
    ----------
    tau : float > 0
    beta0, beta1, beta2 : floats
    """

    def __init__(self, tau: float, beta0: float, beta1: float, beta2: float) -> None:
        assert tau > 0, "tau must be positive"
        self.tau = tau
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.name = "NelsonSiegel"
        self.n_params = 4

    def f0(self, x: Array) -> Array:
        """f0(x) = beta0 + beta1*exp(-x/tau) + (beta2/tau)*x*exp(-x/tau)."""
        x = np.asarray(x, dtype=float)
        e = np.exp(-x / self.tau)
        return self.beta0 + self.beta1 * e + (self.beta2 / self.tau) * x * e

    def f0_prime(self, x: Array) -> Array:
        """
        f0'(x) = ((beta2-beta1)/tau)*exp(-x/tau) - (beta2/tau^2)*x*exp(-x/tau).
        """
        x = np.asarray(x, dtype=float)
        e = np.exp(-x / self.tau)
        return (
            (self.beta2 - self.beta1) / self.tau * e
            - self.beta2 / self.tau ** 2 * x * e
        )

    def f0_second(self, x: Array) -> Array:
        """f0''(x), derived analytically."""
        x = np.asarray(x, dtype=float)
        e = np.exp(-x / self.tau)
        # d/dx [ (beta2-beta1)/tau * e - beta2/tau^2 * x * e ]
        # = -(beta2-beta1)/tau^2 * e - beta2/tau^2 * e + beta2/tau^3 * x * e
        # = (-(beta2-beta1) - beta2)/tau^2 * e + beta2/tau^3 * x * e
        # = (-2*beta2 + beta1)/tau^2 * e + beta2/tau^3 * x * e
        return (
            (self.beta1 - 2 * self.beta2) / self.tau ** 2 * e
            + self.beta2 / self.tau ** 3 * x * e
        )

    def f0_third(self, x: Array) -> Array:
        """f0'''(x), derived analytically."""
        x = np.asarray(x, dtype=float)
        e = np.exp(-x / self.tau)
        # d/dx [ (beta1-2*beta2)/tau^2 * e + beta2/tau^3 * x * e ]
        # = -(beta1-2*beta2)/tau^3 * e + beta2/tau^3 * e - beta2/tau^4 * x * e
        # = (-(beta1-2*beta2) + beta2)/tau^3 * e - beta2/tau^4 * x * e
        # = (3*beta2 - beta1)/tau^3 * e - beta2/tau^4 * x * e
        return (
            (3 * self.beta2 - self.beta1) / self.tau ** 3 * e
            - self.beta2 / self.tau ** 4 * x * e
        )

    def integral_f0(self, a: float, b: float) -> float:
        """
        Integral of f0 from a to b.
        int exp(-x/tau) dx = -tau*exp(-x/tau)
        int x*exp(-x/tau) dx = -tau*x*exp(-x/tau) - tau^2*exp(-x/tau)
        """
        def antideriv(x):
            e = np.exp(-x / self.tau)
            return (
                self.beta0 * x
                + self.beta1 * (-self.tau) * e
                + (self.beta2 / self.tau) * (-self.tau * x * e - self.tau ** 2 * e)
            )
        return antideriv(b) - antideriv(a)

    def __repr__(self) -> str:
        return (
            f"NelsonSiegelCurve(tau={self.tau}, beta0={self.beta0}, "
            f"beta1={self.beta1}, beta2={self.beta2})"
        )


class BlissCurve:
    """
    Bliss initial forward curve:
      f0(x) = beta0 + beta1*exp(-x/tau1) + (beta3/tau2)*x*exp(-x/tau2).

    Parameters
    ----------
    tau1, tau2 : floats > 0, tau1 != tau2
    beta0, beta1, beta3 : floats
    """

    def __init__(
        self,
        tau1: float,
        tau2: float,
        beta0: float,
        beta1: float,
        beta3: float,
    ) -> None:
        assert tau1 > 0, "tau1 must be positive"
        assert tau2 > 0, "tau2 must be positive"
        assert abs(tau1 - tau2) > 1e-12, "tau1 and tau2 must be distinct"
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta3 = beta3
        self.name = "Bliss"
        self.n_params = 5

    def f0(self, x: Array) -> Array:
        """f0(x) = beta0 + beta1*exp(-x/tau1) + (beta3/tau2)*x*exp(-x/tau2)."""
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        return self.beta0 + self.beta1 * e1 + (self.beta3 / self.tau2) * x * e2

    def f0_prime(self, x: Array) -> Array:
        """
        f0'(x) = -beta1/tau1 * exp(-x/tau1)
                 + beta3/tau2 * exp(-x/tau2)
                 - beta3/tau2^2 * x * exp(-x/tau2).
        """
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        return (
            -self.beta1 / self.tau1 * e1
            + self.beta3 / self.tau2 * e2
            - self.beta3 / self.tau2 ** 2 * x * e2
        )

    def f0_second(self, x: Array) -> Array:
        """f0''(x), derived analytically."""
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        # d/dx of f0'
        # d/dx(-beta1/tau1 * e1) = beta1/tau1^2 * e1
        # d/dx(beta3/tau2 * e2 - beta3/tau2^2 * x * e2)
        #   = -beta3/tau2^2 * e2 - beta3/tau2^2 * e2 + beta3/tau2^3 * x * e2
        #   = -2*beta3/tau2^2 * e2 + beta3/tau2^3 * x * e2
        return (
            self.beta1 / self.tau1 ** 2 * e1
            - 2 * self.beta3 / self.tau2 ** 2 * e2
            + self.beta3 / self.tau2 ** 3 * x * e2
        )

    def f0_third(self, x: Array) -> Array:
        """f0'''(x), derived analytically."""
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        return (
            -self.beta1 / self.tau1 ** 3 * e1
            + 3 * self.beta3 / self.tau2 ** 3 * e2
            - self.beta3 / self.tau2 ** 4 * x * e2
        )

    def integral_f0(self, a: float, b: float) -> float:
        """Integral of f0 from a to b."""
        def antideriv(x):
            e1 = np.exp(-x / self.tau1)
            e2 = np.exp(-x / self.tau2)
            return (
                self.beta0 * x
                - self.beta1 * self.tau1 * e1
                + (self.beta3 / self.tau2) * (-self.tau2 * x * e2 - self.tau2 ** 2 * e2)
            )
        return antideriv(b) - antideriv(a)

    def __repr__(self) -> str:
        return (
            f"BlissCurve(tau1={self.tau1}, tau2={self.tau2}, beta0={self.beta0}, "
            f"beta1={self.beta1}, beta3={self.beta3})"
        )


class SvenssonCurve:
    """
    Svensson initial forward curve:
      f0(x) = beta0 + beta1*exp(-x/tau1) + (beta2/tau1)*x*exp(-x/tau1)
                    + (beta3/tau2)*x*exp(-x/tau2).

    Parameters
    ----------
    tau1, tau2 : floats > 0, tau1 != tau2
    beta0, beta1, beta2, beta3 : floats
    """

    def __init__(
        self,
        tau1: float,
        tau2: float,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
    ) -> None:
        assert tau1 > 0, "tau1 must be positive"
        assert tau2 > 0, "tau2 must be positive"
        assert abs(tau1 - tau2) > 1e-12, "tau1 and tau2 must be distinct"
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.name = "Svensson"
        self.n_params = 6

    def f0(self, x: Array) -> Array:
        """
        f0(x) = beta0 + beta1*exp(-x/tau1) + (beta2/tau1)*x*exp(-x/tau1)
                      + (beta3/tau2)*x*exp(-x/tau2).
        """
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        return (
            self.beta0
            + self.beta1 * e1
            + (self.beta2 / self.tau1) * x * e1
            + (self.beta3 / self.tau2) * x * e2
        )

    def f0_prime(self, x: Array) -> Array:
        """
        Corrected Svensson derivative from paper plan:
        (f0^S)'(x) = ((beta2-beta1)/tau1)*exp(-x/tau1) - (beta2/tau1^2)*x*exp(-x/tau1)
                   + (beta3/tau2)*exp(-x/tau2) - (beta3/tau2^2)*x*exp(-x/tau2).
        """
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        return (
            (self.beta2 - self.beta1) / self.tau1 * e1
            - self.beta2 / self.tau1 ** 2 * x * e1
            + self.beta3 / self.tau2 * e2
            - self.beta3 / self.tau2 ** 2 * x * e2
        )

    def f0_second(self, x: Array) -> Array:
        """f0''(x), derived analytically from the corrected first derivative."""
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        # d/dx [ (beta2-beta1)/tau1 * e1 - beta2/tau1^2 * x * e1
        #        + beta3/tau2 * e2 - beta3/tau2^2 * x * e2 ]
        return (
            -(self.beta2 - self.beta1) / self.tau1 ** 2 * e1
            - self.beta2 / self.tau1 ** 2 * e1
            + self.beta2 / self.tau1 ** 3 * x * e1
            - self.beta3 / self.tau2 ** 2 * e2
            - self.beta3 / self.tau2 ** 2 * e2
            + self.beta3 / self.tau2 ** 3 * x * e2
        )

    def f0_second_simplified(self, x: Array) -> Array:
        """f0''(x) simplified."""
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        # Collect terms:
        # e1 coefficient: -(beta2-beta1)/tau1^2 - beta2/tau1^2 = (-beta2+beta1-beta2)/tau1^2 = (beta1-2*beta2)/tau1^2
        # x*e1 coefficient: beta2/tau1^3
        # e2 coefficient: -2*beta3/tau2^2
        # x*e2 coefficient: beta3/tau2^3
        return (
            (self.beta1 - 2 * self.beta2) / self.tau1 ** 2 * e1
            + self.beta2 / self.tau1 ** 3 * x * e1
            - 2 * self.beta3 / self.tau2 ** 2 * e2
            + self.beta3 / self.tau2 ** 3 * x * e2
        )

    def f0_third(self, x: Array) -> Array:
        """f0'''(x), derived analytically."""
        x = np.asarray(x, dtype=float)
        e1 = np.exp(-x / self.tau1)
        e2 = np.exp(-x / self.tau2)
        # d/dx of f0_second_simplified:
        # d/dx[ (beta1-2*beta2)/tau1^2 * e1 ] = -(beta1-2*beta2)/tau1^3 * e1
        # d/dx[ beta2/tau1^3 * x * e1 ] = beta2/tau1^3 * e1 - beta2/tau1^4 * x * e1
        # d/dx[ -2*beta3/tau2^2 * e2 ] = 2*beta3/tau2^3 * e2
        # d/dx[ beta3/tau2^3 * x * e2 ] = beta3/tau2^3 * e2 - beta3/tau2^4 * x * e2
        coeff_e1 = -(self.beta1 - 2 * self.beta2) / self.tau1 ** 3 + self.beta2 / self.tau1 ** 3
        coeff_xe1 = -self.beta2 / self.tau1 ** 4
        coeff_e2 = 2 * self.beta3 / self.tau2 ** 3 + self.beta3 / self.tau2 ** 3
        coeff_xe2 = -self.beta3 / self.tau2 ** 4
        return (
            coeff_e1 * e1
            + coeff_xe1 * x * e1
            + coeff_e2 * e2
            + coeff_xe2 * x * e2
        )

    def integral_f0(self, a: float, b: float) -> float:
        """Integral of f0 from a to b."""
        def antideriv(x):
            e1 = np.exp(-x / self.tau1)
            e2 = np.exp(-x / self.tau2)
            return (
                self.beta0 * x
                - self.beta1 * self.tau1 * e1
                + (self.beta2 / self.tau1) * (-self.tau1 * x * e1 - self.tau1 ** 2 * e1)
                + (self.beta3 / self.tau2) * (-self.tau2 * x * e2 - self.tau2 ** 2 * e2)
            )
        return antideriv(b) - antideriv(a)

    def __repr__(self) -> str:
        return (
            f"SvenssonCurve(tau1={self.tau1}, tau2={self.tau2}, beta0={self.beta0}, "
            f"beta1={self.beta1}, beta2={self.beta2}, beta3={self.beta3})"
        )


def make_curve(family: str, params: dict):
    """Factory function to create initial curve objects from family name and params dict."""
    family = family.lower()
    if family == "flat":
        return FlatCurve(params["beta0"])
    elif family == "monotone":
        return MonotoneCurve(params["tau"], params["beta0"], params["beta1"])
    elif family in ("nelsonsiegel", "nelson_siegel", "ns"):
        return NelsonSiegelCurve(params["tau"], params["beta0"], params["beta1"], params["beta2"])
    elif family == "bliss":
        return BlissCurve(params["tau1"], params["tau2"], params["beta0"], params["beta1"], params["beta3"])
    elif family == "svensson":
        return SvenssonCurve(
            params["tau1"], params["tau2"],
            params["beta0"], params["beta1"], params["beta2"], params["beta3"]
        )
    else:
        raise ValueError(f"Unknown curve family: {family}")
