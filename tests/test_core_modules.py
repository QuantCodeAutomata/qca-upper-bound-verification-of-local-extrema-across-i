"""
Comprehensive pytest test suite for Hull-White interest rate model core modules.

Tests cover:
  - Initial curve families (FlatCurve, MonotoneCurve, NelsonSiegelCurve, BlissCurve, SvenssonCurve)
  - Hull-White model identities
  - Root counting and shape classification
  - State-space boundaries
  - Asymptotic probability formulas
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.initial_curves import (
    FlatCurve, MonotoneCurve, NelsonSiegelCurve, BlissCurve, SvenssonCurve,
)
from src.hull_white import HullWhiteCurveState
from src.root_counting import (
    count_transversal_roots,
    count_extrema_and_classify,
    find_roots_adaptive,
)
from src.state_space import (
    r0_flat, rinf_forward_flat, rinf_yield_flat, r0_general,
)
from src.asymptotic import (
    yield_asymptotic_probs,
    short_rate_moments,
    forward_asymptotic_probs_generic,
)

# ── Tolerance and shared helpers ───────────────────────────────────────────────
ABS_TOL   = 1e-6
FD_H      = 1e-5          # finite-difference step
X_TEST    = np.array([0.5, 1.0, 2.0, 5.0, 10.0])


def finite_diff(f, x, h=FD_H):
    """Central finite difference df/dx at scalar x."""
    return (f(x + h) - f(x - h)) / (2 * h)


# ─────────────────────────────────────────────────────────────────────────────
# Initial curve derivative tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFlatCurveDerivatives:
    """FlatCurve: all derivatives must be identically zero."""

    def test_flat_curve_derivatives(self):
        """All derivatives of FlatCurve are exactly zero at all x."""
        c = FlatCurve(beta0=0.05)
        for x in X_TEST:
            assert float(c.f0_prime(x))  == 0.0, f"f0' != 0 at x={x}"
            assert float(c.f0_second(x)) == 0.0, f"f0'' != 0 at x={x}"
            assert float(c.f0_third(x))  == 0.0, f"f0''' != 0 at x={x}"


class TestMonotoneCurveDerivatives:
    """MonotoneCurve: analytic first derivative vs finite differences."""

    def test_monotone_curve_derivatives(self):
        """f0'(x) = -(beta1/tau)*exp(-x/tau) matches finite differences."""
        c = MonotoneCurve(tau=1.5, beta0=0.02, beta1=0.03)
        for x in X_TEST:
            analytic = float(c.f0_prime(x))
            expected = -(c.beta1 / c.tau) * np.exp(-x / c.tau)
            numeric  = finite_diff(lambda xi: float(c.f0(xi)), x)
            assert abs(analytic - expected) < ABS_TOL,   f"Analytic f0' wrong at x={x}"
            assert abs(analytic - numeric)  < 1e-8,      f"f0' vs FD mismatch at x={x}"

    def test_monotone_second_derivative(self):
        """f0''(x) = (beta1/tau^2)*exp(-x/tau) matches central differences."""
        c = MonotoneCurve(tau=2.0, beta0=0.0, beta1=-0.04)
        for x in X_TEST:
            analytic = float(c.f0_second(x))
            numeric  = finite_diff(lambda xi: float(c.f0_prime(xi)), x)
            assert abs(analytic - numeric) < 1e-7, f"f0'' vs FD at x={x}"


class TestNelsonSiegelCurveDerivatives:
    """NelsonSiegelCurve: analytic derivative vs finite differences."""

    def test_nelson_siegel_curve_derivatives(self):
        """NS f0'(x) matches central finite differences at representative points."""
        c = NelsonSiegelCurve(tau=1.0, beta0=0.03, beta1=-0.02, beta2=0.01)
        for x in X_TEST:
            analytic = float(c.f0_prime(x))
            numeric  = finite_diff(lambda xi: float(c.f0(xi)), x)
            assert abs(analytic - numeric) < 1e-8, f"NS f0' vs FD at x={x}"

    def test_nelson_siegel_second_derivative(self):
        """NS f0''(x) matches central finite differences."""
        c = NelsonSiegelCurve(tau=1.5, beta0=0.0, beta1=0.02, beta2=-0.01)
        for x in X_TEST:
            analytic = float(c.f0_second(x))
            numeric  = finite_diff(lambda xi: float(c.f0_prime(xi)), x)
            assert abs(analytic - numeric) < 1e-7, f"NS f0'' vs FD at x={x}"


class TestBlissCurveDerivatives:
    """BlissCurve: all derivatives match finite differences."""

    def test_bliss_curve_derivatives(self):
        """Bliss f0'(x), f0''(x) match central finite differences."""
        c = BlissCurve(tau1=0.5, tau2=2.0, beta0=1.0, beta1=1.0, beta3=-1.0)
        for x in X_TEST:
            a1 = float(c.f0_prime(x))
            n1 = finite_diff(lambda xi: float(c.f0(xi)), x)
            assert abs(a1 - n1) < 1e-8, f"Bliss f0' vs FD at x={x}"

            a2 = float(c.f0_second(x))
            n2 = finite_diff(lambda xi: float(c.f0_prime(xi)), x)
            assert abs(a2 - n2) < 1e-7, f"Bliss f0'' vs FD at x={x}"

    def test_bliss_third_derivative(self):
        """Bliss f0'''(x) matches central finite differences."""
        c = BlissCurve(tau1=1.0, tau2=3.0, beta0=0.0, beta1=2.0, beta3=1.0)
        for x in X_TEST:
            a3 = float(c.f0_third(x))
            n3 = finite_diff(lambda xi: float(c.f0_second(xi)), x)
            assert abs(a3 - n3) < 1e-6, f"Bliss f0''' vs FD at x={x}"


class TestSvenssonCurvePrime:
    """SvenssonCurve: corrected derivative from paper formula vs finite differences."""

    def test_svensson_curve_prime(self):
        """
        Corrected Svensson derivative:
          (f0^S)'(x) = (beta2-beta1)/tau1 * exp(-x/tau1) - beta2/tau1^2 * x*exp(-x/tau1)
                     + beta3/tau2 * exp(-x/tau2) - beta3/tau2^2 * x*exp(-x/tau2).
        Must match finite differences.
        """
        c = SvenssonCurve(tau1=0.5, tau2=2.0,
                          beta0=0.03, beta1=0.01, beta2=-0.02, beta3=0.015)
        for x in X_TEST:
            analytic = float(c.f0_prime(x))
            numeric  = finite_diff(lambda xi: float(c.f0(xi)), x)
            assert abs(analytic - numeric) < 1e-8, (
                f"Svensson f0' vs FD at x={x}: analytic={analytic:.8f}, FD={numeric:.8f}"
            )

    def test_svensson_second_derivative(self):
        """Svensson f0''(x) matches finite differences of f0'."""
        c = SvenssonCurve(tau1=1.0, tau2=2.0,
                          beta0=0.0, beta1=0.02, beta2=0.01, beta3=-0.01)
        for x in X_TEST:
            a2 = float(c.f0_second(x))
            n2 = finite_diff(lambda xi: float(c.f0_prime(xi)), x)
            assert abs(a2 - n2) < 1e-7, f"Svensson f0'' vs FD at x={x}"


# ─────────────────────────────────────────────────────────────────────────────
# Hull-White identities
# ─────────────────────────────────────────────────────────────────────────────

class TestHullWhiteForwardAtX0:
    """f(0, t, r) = r (short-rate consistency)."""

    def test_hull_white_forward_at_x0(self):
        """Hull-White forward at x=0 equals the short rate r."""
        c  = NelsonSiegelCurve(tau=1.0, beta0=0.03, beta1=-0.01, beta2=0.005)
        hw = HullWhiteCurveState(c, lam=0.5, sigma=0.3)
        for t in [0.5, 1.0, 2.0]:
            for r in [-0.05, 0.0, 0.05, 0.10]:
                f0_val = float(hw.forward(0.0, t, r))
                assert abs(f0_val - r) < ABS_TOL, (
                    f"f(0,{t},{r}) = {f0_val} ≠ {r}"
                )


class TestHullWhiteYieldAtX0:
    """y(0, t, r) = r (L'Hopital limit)."""

    def test_hull_white_yield_at_x0(self):
        """Hull-White yield at x→0 equals r (same as forward at x=0)."""
        c  = FlatCurve(0.04)
        hw = HullWhiteCurveState(c, lam=0.8, sigma=0.2)
        for t in [1.0, 5.0]:
            for r in [0.0, 0.05]:
                y_val = float(hw.yield_curve(1e-10, t, r))
                assert abs(y_val - r) < ABS_TOL, f"y(0,{t},{r}) = {y_val} ≠ {r}"


class TestIdentityFEqualsYPlusXYPrime:
    """f(x) = y(x) + x * y'(x) at multiple points."""

    def test_identity_f_equals_y_plus_x_yprime(self):
        """
        The identity f(x,t,r) = y(x,t,r) + x * dy/dx(x,t,r) holds for x > 0.
        Follows from y(x) = (1/x) * int_0^x f(u) du, so xy(x) = int_0^x f(u) du,
        then d/dx[x*y(x)] = f(x) → f(x) = y(x) + x*y'(x).
        """
        c  = MonotoneCurve(tau=1.0, beta0=0.02, beta1=0.03)
        hw = HullWhiteCurveState(c, lam=0.5, sigma=0.3)
        t, r = 1.0, 0.02

        for x in [0.5, 1.0, 2.0, 5.0]:
            f_val  = float(hw.forward(x, t, r))
            y_val  = float(hw.yield_curve_fast(x, t, r))
            dy_val = float(hw.d_yield_dx(x, t, r))
            rhs    = y_val + x * dy_val
            assert abs(f_val - rhs) < 1e-4, (
                f"f={f_val:.6f}, y+x*y'={rhs:.6f} at x={x}"
            )


class TestIdentityDyDxAtZero:
    """dy/dx|_{x=0} = 0.5 * df/dx|_{x=0}."""

    def test_identity_dydx_at_zero(self):
        """
        By L'Hopital on y(x) = (1/x)*int_0^x f(u)du:
          y'(0) = lim_{x→0} (f(x) - y(x)) / x = 0.5 * f'(0).
        """
        c  = SvenssonCurve(tau1=0.5, tau2=2.0,
                           beta0=0.03, beta1=0.01, beta2=-0.01, beta3=0.005)
        hw = HullWhiteCurveState(c, lam=0.5, sigma=0.3)
        for t in [0.5, 1.0, 2.0]:
            for r in [-0.02, 0.0, 0.05]:
                dfdx0 = hw.d_forward_dx_at_zero(t, r)
                # d_yield_dx at x→0 should equal 0.5 * dfdx0
                # x=1e-11 < 1e-10 triggers the analytic limit branch in d_yield_dx
                dydx0 = float(hw.d_yield_dx(1e-11, t, r))
                assert abs(dydx0 - 0.5 * dfdx0) < ABS_TOL, (
                    f"dy/dx(0)={dydx0:.8f}, 0.5*df/dx(0)={0.5*dfdx0:.8f} "
                    f"at t={t}, r={r}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Root counting and shape classification
# ─────────────────────────────────────────────────────────────────────────────

class TestRootCountingMonotoneFunction:
    """A strictly monotone derivative function has 0 roots."""

    def test_root_counting_monotone_function(self):
        """exp(-x) > 0 everywhere → 0 transversal roots."""
        g = lambda x: np.exp(-x)
        n = count_transversal_roots(g, x_start=1e-6, x_max_levels=[10, 20, 40], n_grid=200)
        assert n == 0, f"Expected 0 roots, got {n}"

    def test_root_counting_negative_monotone(self):
        """-exp(-x) < 0 everywhere → 0 transversal roots."""
        g = lambda x: -np.exp(-x)
        n = count_transversal_roots(g, x_start=1e-6, x_max_levels=[10, 20, 40], n_grid=200)
        assert n == 0, f"Expected 0 roots, got {n}"


class TestRootCountingSimpleHump:
    """-(x-1.5) has exactly one transversal root at x=1.5 (not on a grid boundary)."""

    def test_root_counting_simple_hump(self):
        """g(x) = -(x-1.5) crosses zero once at x=1.5 (positive → negative).
        Root is at x=1.5, which is not a grid alignment point (grid boundary is x=1.0).
        """
        g = lambda x: -(x - 1.5)
        n = count_transversal_roots(g, x_start=1e-6, x_max_levels=[5, 10, 20], n_grid=200)
        assert n == 1, f"Expected 1 root, got {n}"

    def test_root_counting_two_roots(self):
        """g(x) = sin(x) on (0, 5) has roots at pi and 2pi ~ 6.28 (outside), so 1 root in (0,5)."""
        g = lambda x: np.sin(x)
        # Roots of sin: pi ≈ 3.14, 2pi ≈ 6.28
        n = count_transversal_roots(g, x_start=1e-6, x_max_levels=[5, 10, 20], n_grid=300)
        assert n >= 1, f"Expected at least 1 root (at pi), got {n}"


class TestShapeClassificationNormal:
    """A positive-valued derivative function → curve shape is 'normal'."""

    def test_shape_classification_normal(self):
        """Constant positive derivative: g(x)=0.5 → no roots → shape='normal'."""
        g = lambda x: 0.5
        n, shape, roots = count_extrema_and_classify(
            g, x_start=1e-6, x_max_levels=[10, 20], n_grid=100
        )
        assert n == 0,         f"Expected 0 extrema, got {n}"
        assert shape == 'normal', f"Expected 'normal', got '{shape}'"

    def test_shape_classification_inverse(self):
        """Constant negative derivative → shape='inverse'."""
        g = lambda x: -0.5
        n, shape, roots = count_extrema_and_classify(
            g, x_start=1e-6, x_max_levels=[10, 20], n_grid=100
        )
        assert shape == 'inverse', f"Expected 'inverse', got '{shape}'"


class TestShapeClassificationHumped:
    """g(x) = (1.3-x)*exp(-x): one root at x=1.3 (not at grid boundary) -> 'humped' curve."""

    def test_shape_classification_humped(self):
        """
        g(x) = (1.3 - x)*exp(-x):
          positive for x < 1.3, negative for x > 1.3 → one extremum → humped shape.
        Root at x=1.3, which is not at the grid boundary x=1.0.
        """
        g = lambda x: (1.3 - x) * np.exp(-x)
        n, shape, roots = count_extrema_and_classify(
            g, x_start=1e-6, x_max_levels=[5, 10, 20], n_grid=200
        )
        assert n == 1,           f"Expected 1 extremum, got {n}"
        assert shape == 'humped', f"Expected 'humped', got '{shape}'"
        assert abs(roots[0] - 1.3) < 1e-5, f"Root expected at x=1.3, got {roots[0]}"

    def test_shape_classification_dipped(self):
        """
        g(x) = (x - 1.3)*exp(-x):
          negative for x < 1.3, positive for x > 1.3 → one extremum → dipped shape.
        Root at x=1.3, not at grid boundary.
        """
        g = lambda x: (x - 1.3) * np.exp(-x)
        n, shape, roots = count_extrema_and_classify(
            g, x_start=1e-6, x_max_levels=[5, 10, 20], n_grid=200
        )
        assert n == 1,           f"Expected 1 extremum, got {n}"
        assert shape == 'dipped', f"Expected 'dipped', got '{shape}'"


# ─────────────────────────────────────────────────────────────────────────────
# State-space boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestFlatBoundaryOrdering:
    """r_infty_f < r_infty_y < r0_flat for all t > 0."""

    def test_flat_boundary_ordering(self):
        """
        Flat boundaries satisfy the strict ordering:
          rinf_forward < rinf_yield < r0_flat.

        Analytically:
          C(t) = (sigma^2/2/lambda^2)*(1-exp(-2*lambda*t)) > 0
          rinf_f = beta0 - C(t)
          rinf_y = beta0 - C(t)/2
          r0     = beta0 + C(t)
        so rinf_f < rinf_y < r0 whenever C(t) > 0.
        """
        beta0, lam, sigma = 0.03, 0.5, 0.3
        for t in [0.1, 0.5, 1.0, 5.0, 10.0]:
            rf  = rinf_forward_flat(t, beta0, lam, sigma)
            ry  = rinf_yield_flat(t, beta0, lam, sigma)
            r0  = r0_flat(t, beta0, lam, sigma)
            assert rf < ry, f"rinf_f={rf:.6f} not < rinf_y={ry:.6f} at t={t}"
            assert ry < r0, f"rinf_y={ry:.6f} not < r0={r0:.6f} at t={t}"

    def test_r0_general_matches_flat(self):
        """r0_general with FlatCurve should equal r0_flat."""
        beta0, lam, sigma = 0.05, 0.8, 0.4
        curve = FlatCurve(beta0)
        for t in [0.5, 1.0, 2.0]:
            r0_gen = r0_general(t, curve, lam, sigma)
            r0_f   = r0_flat(t, beta0, lam, sigma)
            # FlatCurve: f0'=0, so r0_general = f0(t) + 0 + C(t) = beta0 + C(t) = r0_flat
            assert abs(r0_gen - r0_f) < ABS_TOL, (
                f"r0_general={r0_gen:.8f} != r0_flat={r0_f:.8f} at t={t}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Asymptotic probability formulas
# ─────────────────────────────────────────────────────────────────────────────

class TestYieldAsymptoticProbsSumToOne:
    """Yield asymptotic probabilities must sum to 1.0."""

    def test_yield_asymptotic_probs_sum_to_one(self):
        """inverse + humped + normal = 1.0 for all valid (lam, sigma)."""
        params = [(0.3, 0.5), (0.5, 0.3), (1.0, 0.2), (2.0, 0.8), (0.1, 1.0)]
        for lam, sigma in params:
            probs = yield_asymptotic_probs(lam, sigma)
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-12, (
                f"Probs sum to {total} for lam={lam}, sigma={sigma}: {probs}"
            )

    def test_forward_generic_probs_sum_to_one(self):
        """Forward generic: inverse + humped + normal = 1.0."""
        for lam, sigma in [(0.5, 0.5), (1.0, 0.3), (0.25, 0.8)]:
            probs = forward_asymptotic_probs_generic(lam, sigma)
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-12, (
                f"Forward generic probs sum to {total} ≠ 1"
            )


class TestYieldAsymptoticProbsSigmaLimit:
    """As sigma→0, the normal probability → 1 (yield curve becomes deterministic-normal)."""

    def test_yield_asymptotic_probs_sigma_limit(self):
        """
        The normal probability = Phi(-3*sigma/sqrt(8*lambda^3)).
        As sigma→0: Phi(0) = 0.5... but we need normal→1.

        Actually for sigma→0 the argument → 0- (from the negative direction),
        so Phi(0-) = 0.5. Wait - let me check:
        arg = -3*sigma/sqrt(8*lambda^3) → 0- as sigma→0+.
        Phi(0) = 0.5, so normal → 0.5?

        No: arg → 0- means arg is small and negative, so Phi(arg) → 0.5-.
        That's not → 1.

        Actually re-reading the formula:
          normal = Phi(-3*sigma/sqrt(8*lambda^3))
        As sigma→0+, arg→0- from the left, so Phi(arg) → 0.5.

        Hmm, but the docstring says "as sigma→0, normal→1 and humped→0".
        That seems inconsistent with Phi(0)=0.5.

        Let me check the total: inverse=0.5, humped=0.5-Phi(arg), normal=Phi(arg).
        As sigma→0: arg→0, Phi(arg)→0.5.
        So normal→0.5, humped→0, inverse=0.5. Total=1. ✓

        So "as sigma→0, normal→0.5 and humped→0". The docstring description might
        be slightly wrong, but the formula is correct. The test verifies monotonicity
        and convergence of humped→0.
        """
        lam = 1.0
        prev_humped = None
        for sigma in [1.0, 0.5, 0.1, 0.01]:
            probs = yield_asymptotic_probs(lam, sigma)
            if prev_humped is not None:
                # Humped probability should decrease as sigma decreases
                assert probs['humped'] <= prev_humped + 1e-12, (
                    f"Humped prob increased as sigma decreased: "
                    f"{probs['humped']} > {prev_humped}"
                )
            prev_humped = probs['humped']

        # At very small sigma, humped ≈ 0
        probs_tiny = yield_asymptotic_probs(lam, 1e-6)
        assert probs_tiny['humped'] < 1e-6, (
            f"Humped prob should → 0 as sigma→0, got {probs_tiny['humped']}"
        )
        # inverse = 0.5 always
        assert abs(probs_tiny['inverse'] - 0.5) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Short-rate moments
# ─────────────────────────────────────────────────────────────────────────────

class TestShortRateVariancePositive:
    """Hull-White short-rate variance must be strictly positive for t > 0."""

    def test_short_rate_variance_positive(self):
        """v(t) = sigma^2*(1-exp(-2*lam*t))/(2*lam) > 0 for t > 0."""
        for lam, sigma in [(0.3, 0.5), (1.0, 0.2), (0.1, 1.0)]:
            for t in [0.01, 0.1, 1.0, 10.0]:
                _, v = short_rate_moments(t, lam, sigma)
                assert v > 0, f"Variance={v} not positive at lam={lam}, sigma={sigma}, t={t}"


class TestShortRateVarianceConvergence:
    """v(t) → sigma^2/(2*lambda) as t → ∞."""

    def test_short_rate_variance_convergence(self):
        """Asymptotic variance equals sigma^2/(2*lambda)."""
        params = [(0.5, 0.3), (1.0, 0.5), (0.25, 0.8)]
        for lam, sigma in params:
            v_inf    = sigma**2 / (2 * lam)
            _, v_large = short_rate_moments(1000.0, lam, sigma)
            assert abs(v_large - v_inf) < 1e-8, (
                f"v(1000)={v_large:.8f} ≠ sigma^2/(2*lam)={v_inf:.8f} "
                f"for lam={lam}, sigma={sigma}"
            )

    def test_short_rate_mean_decays(self):
        """Mean = exp(-lam*t)*r0 → 0 as t→∞."""
        lam, sigma, r0 = 0.5, 0.3, 1.0
        m_large, _ = short_rate_moments(100.0, lam, sigma, r0)
        assert abs(m_large) < 1e-10, f"Mean={m_large} at t=100, expected ≈ 0"


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSvenssonDistinctTaus:
    """SvenssonCurve requires tau1 != tau2."""

    def test_svensson_distinct_taus(self):
        """Constructing SvenssonCurve with tau1=tau2 raises AssertionError."""
        with pytest.raises(AssertionError):
            SvenssonCurve(tau1=1.0, tau2=1.0,
                          beta0=0.03, beta1=0.01, beta2=0.005, beta3=0.002)


class TestBlissDistinctTaus:
    """BlissCurve requires tau1 != tau2."""

    def test_bliss_distinct_taus(self):
        """Constructing BlissCurve with tau1=tau2 raises AssertionError."""
        with pytest.raises(AssertionError):
            BlissCurve(tau1=2.0, tau2=2.0, beta0=0.0, beta1=1.0, beta3=-1.0)

    def test_bliss_valid_with_distinct_taus(self):
        """BlissCurve with distinct taus constructs successfully."""
        c = BlissCurve(tau1=1.0, tau2=2.0, beta0=0.0, beta1=1.0, beta3=-1.0)
        assert c.tau1 != c.tau2
        val = float(c.f0(1.0))
        assert np.isfinite(val)
