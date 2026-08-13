"""
Implementation verification tests for the Hull-White experiment logic.

Uses small parameter grids (2-3 values per dimension) so that tests run quickly.
Each test verifies a core claim of the research experiments.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.initial_curves import (
    FlatCurve, MonotoneCurve, NelsonSiegelCurve, BlissCurve, SvenssonCurve,
)
from src.hull_white import HullWhiteCurveState
from src.root_counting import count_extrema_and_classify
from src.asymptotic import yield_asymptotic_probs

# Shared settings for fast test scanning
T_GRID_FAST = [0.5, 2.0]
R_GRID_FAST = [-2.0, 0.0, 2.0]
X_MAX_LVL   = [10, 20, 40]
N_GRID_TEST = 100


def _count_forward_extrema(hw, t, r):
    """Count forward-curve extrema at (t, r)."""
    def g(x):
        return float(hw.d_forward_dx(float(x), t, r))
    n, shape, _ = count_extrema_and_classify(
        g, x_start=1e-6, x_max_levels=X_MAX_LVL, n_grid=N_GRID_TEST
    )
    return n, shape


def _count_yield_extrema(hw, t, r):
    """Count yield-curve extrema at (t, r)."""
    def g(x):
        return float(hw.d_yield_dx(float(x), t, r))
    n, shape, _ = count_extrema_and_classify(
        g, x_start=1e-6, x_max_levels=X_MAX_LVL, n_grid=N_GRID_TEST
    )
    return n, shape


def _max_extrema_scan(hw, t_grid, r_grid, curve_type='forward'):
    """Return max extrema count over a (t, r) grid."""
    max_n = 0
    shapes = set()
    for t in t_grid:
        for r in r_grid:
            if curve_type == 'forward':
                n, sh = _count_forward_extrema(hw, t, r)
            else:
                n, sh = _count_yield_extrema(hw, t, r)
            max_n = max(max_n, n)
            shapes.add(sh)
    return max_n, shapes


# ─────────────────────────────────────────────────────────────────────────────
# Exp 1: flat, monotone, NS upper bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestExp1FlatBound:
    """Flat initial curve: forward and yield have at most 1 extremum."""

    def test_exp1_flat_bound(self):
        """
        For FlatCurve the forward derivative is
          df/dx = -lam*K*exp(-lam*x) + 2*lam*C*exp(-2*lam*x),
        which is a difference of two exponentials and has at most 1 zero.
        """
        beta0_vals = [0.0, 0.05]
        lam_vals   = [0.5, 1.0]
        sigma_vals = [0.3, 0.8]

        max_f = 0
        max_y = 0
        for beta0 in beta0_vals:
            for lam in lam_vals:
                for sigma in sigma_vals:
                    hw = HullWhiteCurveState(FlatCurve(beta0), lam, sigma)
                    n_f, _ = _max_extrema_scan(hw, T_GRID_FAST, R_GRID_FAST, 'forward')
                    n_y, _ = _max_extrema_scan(hw, T_GRID_FAST, R_GRID_FAST, 'yield')
                    max_f = max(max_f, n_f)
                    max_y = max(max_y, n_y)

        assert max_f <= 1, f"Flat forward: max extrema = {max_f} > 1"
        assert max_y <= 1, f"Flat yield:   max extrema = {max_y} > 1"


class TestExp1MonotoneBound:
    """Monotone initial curve: forward and yield have at most 2 extrema."""

    def test_exp1_monotone_bound(self):
        """
        MonotoneCurve introduces one more exponential term.
        Theory guarantees at most 2 forward extrema.
        """
        tau_vals   = [0.5, 2.0]
        beta1_vals = [-1.0, 1.0]
        lam_vals   = [0.5, 1.0]
        sigma_vals = [0.3, 0.8]

        max_f = 0
        max_y = 0
        for tau in tau_vals:
            for beta1 in beta1_vals:
                for lam in lam_vals:
                    for sigma in sigma_vals:
                        hw = HullWhiteCurveState(
                            MonotoneCurve(tau, 0.03, beta1), lam, sigma
                        )
                        n_f, _ = _max_extrema_scan(hw, T_GRID_FAST, R_GRID_FAST, 'forward')
                        n_y, _ = _max_extrema_scan(hw, T_GRID_FAST, R_GRID_FAST, 'yield')
                        max_f = max(max_f, n_f)
                        max_y = max(max_y, n_y)

        assert max_f <= 2, f"Monotone forward: max extrema = {max_f} > 2"
        assert max_y <= 2, f"Monotone yield:   max extrema = {max_y} > 2"


class TestExp1NSBound:
    """Nelson-Siegel initial curve: forward and yield have at most 3 extrema."""

    def test_exp1_ns_bound(self):
        """
        NelsonSiegelCurve has two distinct exponential scales.
        Theory guarantees at most 3 forward extrema.
        """
        tau_vals   = [0.5, 2.0]
        beta2_vals = [-1.0, 1.0]
        lam_vals   = [0.5, 1.0]
        sigma_vals = [0.3]

        max_f = 0
        max_y = 0
        for tau in tau_vals:
            for beta2 in beta2_vals:
                for lam in lam_vals:
                    for sigma in sigma_vals:
                        curve = NelsonSiegelCurve(tau, 0.03, 0.01, beta2)
                        hw    = HullWhiteCurveState(curve, lam, sigma)
                        n_f, _ = _max_extrema_scan(hw, T_GRID_FAST, R_GRID_FAST, 'forward')
                        n_y, _ = _max_extrema_scan(hw, T_GRID_FAST, R_GRID_FAST, 'yield')
                        max_f = max(max_f, n_f)
                        max_y = max(max_y, n_y)

        assert max_f <= 3, f"NS forward: max extrema = {max_f} > 3"
        assert max_y <= 3, f"NS yield:   max extrema = {max_y} > 3"


# ─────────────────────────────────────────────────────────────────────────────
# Exp 2: flat curve – only three yield shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestExp2FlatShapesOnlyThree:
    """Flat curve forward/yield: only normal, inverse, humped shapes are attainable."""

    def test_exp2_flat_shapes_only_three(self):
        """
        Flat initial curve with one exponential term produces only:
          normal, inverse, humped  (no dipped, hd, etc.).
        """
        hw = HullWhiteCurveState(FlatCurve(0.0), lam=0.5, sigma=0.5)
        # Wider r range to cover all shape regions
        r_wide = [-5, -3, -1, 0, 1, 3, 5]
        _, shapes_f = _max_extrema_scan(hw, [0.1, 0.5, 1.0, 5.0], r_wide, 'forward')
        _, shapes_y = _max_extrema_scan(hw, [0.1, 0.5, 1.0, 5.0], r_wide, 'yield')

        allowed = {'normal', 'inverse', 'humped'}
        unexpected_f = shapes_f - allowed
        unexpected_y = shapes_y - allowed

        assert not unexpected_f, (
            f"Flat forward: unexpected shapes {unexpected_f}"
        )
        assert not unexpected_y, (
            f"Flat yield: unexpected shapes {unexpected_y}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Exp 3: NS special case shape verification
# ─────────────────────────────────────────────────────────────────────────────

class TestExp3NSSpecialBeta2NegForward:
    """NS forward with beta2 < 0: normal, dipped, hd shapes should appear."""

    def test_exp3_ns_special_beta2_neg_forward(self):
        """
        For NS with lambda = 1/tau (special case) and beta2 < 0,
        the forward can exhibit normal, dipped, hd shapes (Regime 1).
        Verify at least two distinct shapes are present.
        """
        tau   = 1.0
        lam   = 1.0 / tau   # special case
        sigma = 0.5
        beta2 = -1.0

        curve = NelsonSiegelCurve(tau, 0.03, 0.01, beta2)
        hw    = HullWhiteCurveState(curve, lam, sigma)

        # Scan wide r range to find multiple shapes
        r_wide = np.linspace(-4, 4, 9).tolist()
        _, shapes = _max_extrema_scan(hw, [0.5, 1.0, 2.0], r_wide, 'forward')

        # At minimum, normal and inverse should appear for extreme r values
        assert len(shapes) >= 2, (
            f"Expected at least 2 forward shapes for beta2<0, got {shapes}"
        )
        # normal or dipped or inverse should appear
        assert shapes & {'normal', 'inverse', 'dipped', 'hd', 'humped'}, (
            f"No expected shapes found; got {shapes}"
        )


class TestExp3NSSpecialBeta2PosForward:
    """NS forward with beta2 > 0: inverse and humped shapes should appear."""

    def test_exp3_ns_special_beta2_pos_forward(self):
        """
        For NS with lambda = 1/tau and beta2 > 0,
        the forward exhibits inverse, humped shapes (Regime 2/3).
        Verify inverse or humped appear.
        """
        tau   = 1.0
        lam   = 1.0 / tau
        sigma = 0.5
        beta2 = 1.0

        curve = NelsonSiegelCurve(tau, 0.03, -0.02, beta2)
        hw    = HullWhiteCurveState(curve, lam, sigma)

        r_wide = np.linspace(-4, 4, 9).tolist()
        _, shapes = _max_extrema_scan(hw, [0.5, 1.0, 2.0], r_wide, 'forward')

        assert {'inverse', 'humped'} & shapes, (
            f"Expected inverse or humped for beta2>0, got {shapes}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Exp 5: small-time attainability
# ─────────────────────────────────────────────────────────────────────────────

class TestExp5SmallTimeAttainability:
    """At small t with K≈0, the HW forward shape matches the initial curve shape."""

    def test_exp5_small_time_attainability(self):
        """
        For t = 1e-3 and r = f0(t) - C(t) (which makes K(t,r) ≈ 0),
        the forward f(x,t,r) ≈ f0(x+t) ≈ f0(x), so the shape should match.

        Tests several representative Bliss and Svensson parameter sets.
        """
        def classify_initial(curve):
            def g(x):
                return float(curve.f0_prime(float(x)))
            _, shape, _ = count_extrema_and_classify(
                g, x_start=1e-6, x_max_levels=[10, 20, 40], n_grid=N_GRID_TEST
            )
            return shape

        def classify_hw(hw, t, r):
            def g(x):
                return float(hw.d_forward_dx(float(x), t, r))
            _, shape, _ = count_extrema_and_classify(
                g, x_start=1e-6, x_max_levels=[10, 20, 40], n_grid=N_GRID_TEST
            )
            return shape

        t = 1e-3
        curves_and_models = [
            # (curve, lam, sigma)
            (BlissCurve(0.5, 2.0, 1.0, 1.0, 1.0),   0.5, 0.5),
            (BlissCurve(0.5, 2.0, 0.0, -1.0, 1.0),  0.5, 0.5),
            (SvenssonCurve(0.5, 2.0, 1.0, 1.0, 0.5, 1.0), 0.5, 0.5),
            (SvenssonCurve(0.5, 2.0, 0.0, -1.0, 1.0, -1.0), 0.5, 0.5),
        ]

        matches = 0
        total   = len(curves_and_models)
        for curve, lam, sigma in curves_and_models:
            init_shape = classify_initial(curve)
            C_t  = (sigma**2 / (2 * lam**2)) * (1 - np.exp(-2 * lam * t))
            r    = float(curve.f0(t)) - C_t
            hw   = HullWhiteCurveState(curve, lam, sigma)
            hw_shape = classify_hw(hw, t, r)
            if hw_shape == init_shape:
                matches += 1

        # Expect all (or almost all) to match at t=1e-3
        assert matches >= total - 1, (
            f"Small-time attainability: only {matches}/{total} matched "
            f"(expected at least {total - 1})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Exp 6: yield asymptotic probability sum
# ─────────────────────────────────────────────────────────────────────────────

class TestExp6YieldProbSum:
    """Yield asymptotic probabilities sum to 1.0 for all tested (lam, sigma)."""

    def test_exp6_yield_prob_sum(self):
        """
        The yield asymptotic formulas:
          inverse = 0.5
          humped  = 0.5 - Phi(-3*sigma/sqrt(8*lambda^3))
          normal  = Phi(-3*sigma/sqrt(8*lambda^3))
        must always sum to 1.0.
        """
        params = [
            (0.5, 0.5),
            (1.0, 0.3),
            (0.25, 0.8),
            (2.0, 0.1),
        ]
        for lam, sigma in params:
            probs = yield_asymptotic_probs(lam, sigma)
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-12, (
                f"Yield probs sum to {total} ≠ 1 for lam={lam}, sigma={sigma}"
            )
            # inverse is always exactly 0.5
            assert abs(probs['inverse'] - 0.5) < 1e-12
            # all probabilities are in [0, 1]
            for s, p in probs.items():
                assert 0.0 <= p <= 1.0 + 1e-12, (
                    f"Prob for '{s}' = {p} outside [0,1]"
                )
