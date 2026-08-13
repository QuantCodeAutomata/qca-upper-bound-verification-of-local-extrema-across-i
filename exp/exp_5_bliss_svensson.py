"""
Experiment 5: Bliss and Svensson families.

Covers:
  - Upper-bound verification (Bliss ≤ 4 extrema, Svensson ≤ 5 extrema)
  - Boundary construction r0(t) and r_inf(t)
  - Small-time attainability test
  - Terminal-behavior case study
"""

import sys
sys.path.insert(0, '/workspace/project')

import os
import csv
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.initial_curves import BlissCurve, SvenssonCurve
from src.hull_white import HullWhiteCurveState
from src.root_counting import count_extrema_and_classify
from src.state_space import (
    r0_general,
    rinf_forward_general,
    rinf_yield_svensson,
    rx_forward_general,
    trace_forward_envelope_numerical,
)

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameter grids ────────────────────────────────────────────────────────────
# Representative tau pairs (distinct) covering different decay-rate regimes
BLISS_TAU_PAIRS = [(0.25, 1.0), (0.5, 2.0), (1.0, 4.0), (2.0, 0.5), (4.0, 1.0)]

# Pre-selected Bliss beta tuples (b0, b1, b3) covering diverse shapes
BLISS_BETA_TUPLES = [
    (0, -2, -2), (0, -1, 1), (0, 1, -1), (0, 2, 2),
    (1, -2, 1), (1, 1, -2), (1, -1, -1), (1, 2, -1),
]

# Representative Svensson tau pairs
SVN_TAU_PAIRS = [(0.5, 2.0), (1.0, 4.0), (2.0, 0.5), (0.5, 1.0), (4.0, 2.0)]

# Pre-selected Svensson beta tuples (b0, b1, b2, b3) — representative diversity
SVN_BETA_TUPLES = [
    (0, -2, -1, 1), (0, 1, -2, -1), (0, 2, 1, -2), (0, -1, 2, 1),
    (1, -1, -1, 1), (1, 1, 1, -1), (1, -2, 1, -1), (1, 1, -1, 2),
    (0, -1, 1, -1), (0, 2, -1, 1),
]

LAM_VALS    = [0.25, 0.5, 1.0]
SIGMA_VALS  = [0.2, 0.5, 1.0]
T_GRID      = [0.01, 0.1, 0.5, 1.0, 5.0]
R_GRID      = [-3, -1, 0, 1, 3]
X_MAX_LVL   = [10, 20, 40]
N_GRID      = 80

CSV_FIELDNAMES = [
    'family', 'tau1', 'tau2', 'betas', 'lam', 'sigma',
    't', 'r', 'curve_type', 'extrema_count', 'shape',
    'small_time_match', 'initial_curve_shape',
]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def classify_initial_shape(curve):
    """Classify shape of f0 by root-counting f0'(x) on (0, ∞)."""
    def g(x):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        res = np.array([float(curve.f0_prime(float(xi))) for xi in xa])
        return res if xa.shape != (1,) or np.ndim(x) > 0 else float(res[0])
    _, shape, _ = count_extrema_and_classify(
        g, x_start=1e-6, x_max_levels=X_MAX_LVL, n_grid=N_GRID
    )
    return shape


def hw_forward_extrema(hw, t, r):
    """Count forward extrema and return (n, shape, roots)."""
    def g(x):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        res = hw.d_forward_dx(xa, t, r)
        return res if xa.shape != (1,) or np.ndim(x) > 0 else float(res[0])
    return count_extrema_and_classify(g, x_start=1e-6, x_max_levels=X_MAX_LVL, n_grid=N_GRID)


def hw_yield_extrema(hw, t, r):
    """Count yield extrema and return (n, shape, roots)."""
    def g(x):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        res = hw.d_yield_dx(xa, t, r)
        return res if xa.shape != (1,) or np.ndim(x) > 0 else float(res[0])
    return count_extrema_and_classify(g, x_start=1e-6, x_max_levels=X_MAX_LVL, n_grid=N_GRID)


def make_row(family, tau1, tau2, betas_str, lam, sigma, t, r,
             curve_type, n_ext, shape, small_time_match, init_shape):
    return {
        'family': family, 'tau1': tau1, 'tau2': tau2, 'betas': betas_str,
        'lam': lam, 'sigma': sigma, 't': t, 'r': r, 'curve_type': curve_type,
        'extrema_count': n_ext, 'shape': shape,
        'small_time_match': small_time_match, 'initial_curve_shape': init_shape,
    }


# ── Bliss scan ─────────────────────────────────────────────────────────────────

def run_bliss_scan():
    """Scan Bliss parameter grid; verify max extrema ≤ 4."""
    records = []
    max_f = 0
    max_y = 0
    counts_f = []
    counts_y = []

    for (tau1, tau2) in BLISS_TAU_PAIRS:
        for (beta0, beta1, beta3) in BLISS_BETA_TUPLES:
            curve = BlissCurve(tau1, tau2, beta0, beta1, beta3)
            init_shape = classify_initial_shape(curve)
            betas_str = f'{beta0},{beta1},{beta3}'

            for lam, sigma in itertools.product(LAM_VALS, SIGMA_VALS):
                hw = HullWhiteCurveState(curve, lam, sigma)
                for t in T_GRID:
                    for r in R_GRID:
                        n_f, sh_f, _ = hw_forward_extrema(hw, t, r)
                        n_y, sh_y, _ = hw_yield_extrema(hw, t, r)
                        max_f = max(max_f, n_f)
                        max_y = max(max_y, n_y)
                        counts_f.append(n_f)
                        counts_y.append(n_y)
                        records.append(make_row(
                            'Bliss', tau1, tau2, betas_str, lam, sigma, t, r,
                            'forward', n_f, sh_f, '', init_shape
                        ))
                        records.append(make_row(
                            'Bliss', tau1, tau2, betas_str, lam, sigma, t, r,
                            'yield', n_y, sh_y, '', init_shape
                        ))

    return records, max_f, max_y, counts_f, counts_y


# ── Svensson scan ──────────────────────────────────────────────────────────────

def run_svensson_scan():
    """Scan Svensson parameter grid; verify max extrema ≤ 5."""
    records = []
    max_f = 0
    max_y = 0
    counts_f = []
    counts_y = []

    for (tau1, tau2) in SVN_TAU_PAIRS:
        for (beta0, beta1, beta2, beta3) in SVN_BETA_TUPLES:
            try:
                curve = SvenssonCurve(tau1, tau2, beta0, beta1, beta2, beta3)
            except (AssertionError, ValueError):
                continue
            init_shape = classify_initial_shape(curve)
            betas_str = f'{beta0},{beta1},{beta2},{beta3}'

            for lam, sigma in itertools.product(LAM_VALS, SIGMA_VALS):
                hw = HullWhiteCurveState(curve, lam, sigma)
                for t in T_GRID:
                    for r in R_GRID:
                        n_f, sh_f, _ = hw_forward_extrema(hw, t, r)
                        n_y, sh_y, _ = hw_yield_extrema(hw, t, r)
                        max_f = max(max_f, n_f)
                        max_y = max(max_y, n_y)
                        counts_f.append(n_f)
                        counts_y.append(n_y)
                        records.append(make_row(
                            'Svensson', tau1, tau2, betas_str, lam, sigma, t, r,
                            'forward', n_f, sh_f, '', init_shape
                        ))
                        records.append(make_row(
                            'Svensson', tau1, tau2, betas_str, lam, sigma, t, r,
                            'yield', n_y, sh_y, '', init_shape
                        ))

    return records, max_f, max_y, counts_f, counts_y


# ── Boundary computation ───────────────────────────────────────────────────────

def run_boundary_study():
    """
    Compute r0(t) and r_inf(t) for representative Bliss and Svensson parameter sets.
    Returns list of (label, t_arr, r0_arr, rinf_arr) tuples.
    """
    t_arr = np.linspace(0.01, 10.0, 200)
    results = []

    # ── Bliss representative sets ──────────────────────────────────────────────
    bliss_rep = [
        dict(tau1=0.5, tau2=2.0, beta0=1, beta1=1, beta3=1),
        dict(tau1=1.0, tau2=4.0, beta0=0, beta1=-1, beta3=1),
        dict(tau1=0.25, tau2=1.0, beta0=1, beta1=2, beta3=-2),
    ]
    for p in bliss_rep:
        for lam, sigma in [(0.5, 0.5), (1.0, 0.3)]:
            curve = BlissCurve(p['tau1'], p['tau2'], p['beta0'], p['beta1'], p['beta3'])
            r0_arr   = np.array([r0_general(t, curve, lam, sigma) for t in t_arr])
            rinf_arr = np.array([rinf_forward_general(t, curve, lam, sigma) for t in t_arr])
            label = (f"Bliss tau=({p['tau1']},{p['tau2']}) "
                     f"b=({p['beta0']},{p['beta1']},{p['beta3']}) "
                     f"lam={lam} sig={sigma}")
            results.append((label, t_arr, r0_arr, rinf_arr))

    # ── Svensson representative sets ───────────────────────────────────────────
    svn_rep = [
        dict(tau1=0.5, tau2=2.0, beta0=1, beta1=1, beta2=0.5, beta3=1),
        dict(tau1=1.0, tau2=2.0, beta0=0, beta1=-1, beta2=1, beta3=-1),
    ]
    for p in svn_rep:
        for lam, sigma in [(0.5, 0.5), (1.0, 0.3)]:
            curve = SvenssonCurve(
                p['tau1'], p['tau2'], p['beta0'], p['beta1'], p['beta2'], p['beta3']
            )
            r0_arr   = np.array([r0_general(t, curve, lam, sigma) for t in t_arr])
            rinf_arr = np.array([rinf_forward_general(t, curve, lam, sigma) for t in t_arr])
            # Svensson yield boundary
            rinfy_arr = np.array([
                rinf_yield_svensson(
                    t, p['tau1'], p['tau2'],
                    p['beta0'], p['beta1'], p['beta2'], p['beta3'],
                    lam, sigma
                ) for t in t_arr
            ])
            label = (f"Svensson tau=({p['tau1']},{p['tau2']}) "
                     f"b=({p['beta0']},{p['beta1']},{p['beta2']},{p['beta3']}) "
                     f"lam={lam} sig={sigma}")
            results.append((label, t_arr, r0_arr, rinf_arr))

    return results


# ── Envelope tracing ───────────────────────────────────────────────────────────

def run_envelope_study():
    """Numerically trace forward envelopes for representative param sets."""
    t_grid = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    env_results = []

    rep_params = [
        ('Bliss',
         BlissCurve(0.5, 2.0, 1, 1, 1),
         0.5, 0.5),
        ('Bliss',
         BlissCurve(1.0, 4.0, 0, -1, 1),
         0.5, 0.3),
        ('Svensson',
         SvenssonCurve(0.5, 2.0, 1, 1, 0.5, 1),
         0.5, 0.5),
    ]

    for name, curve, lam, sigma in rep_params:
        env = trace_forward_envelope_numerical(curve, lam, sigma, t_grid)
        env_results.append((name, curve, lam, sigma, env))

    return env_results


# ── Small-time attainability ───────────────────────────────────────────────────

def run_small_time_test():
    """
    Verify that for small t with r ≈ f0(t) - C(t) (K≈0),
    the shape of f(x,t,r) matches the shape of f0(x).
    """
    records = []
    t_small = [1e-4, 1e-3, 1e-2]

    # Five representative sets per family
    bliss_sets = [
        (0.5, 2.0, 1, 1, 1,  0.5, 0.5),
        (0.5, 2.0, 0, -1, 1, 0.5, 0.5),
        (1.0, 4.0, 1, 2, -2, 1.0, 0.3),
        (0.25, 1.0, 1, -2, 2, 0.25, 0.5),
        (2.0, 4.0, 0, 1, -1, 0.5, 1.0),
    ]
    svensson_sets = [
        (0.5, 2.0, 1, 1, 0.5, 1,   0.5, 0.5),
        (0.5, 2.0, 0, -1, 1, -1,   0.5, 0.5),
        (1.0, 2.0, 1, 2, -1, 2,    1.0, 0.3),
        (0.5, 1.0, 0, 1, 1, -2,    0.25, 0.5),
        (1.0, 2.0, 1, -1, 0.5, 1,  0.5, 1.0),
    ]

    for (tau1, tau2, b0, b1, b3, lam, sigma) in bliss_sets:
        curve = BlissCurve(tau1, tau2, b0, b1, b3)
        init_shape = classify_initial_shape(curve)
        hw = HullWhiteCurveState(curve, lam, sigma)
        betas_str = f'{b0},{b1},{b3}'
        for t in t_small:
            C_t = (sigma**2 / (2*lam**2)) * (1 - np.exp(-2*lam*t))
            r = float(curve.f0(t)) - C_t
            n_f, sh_f, _ = hw_forward_extrema(hw, t, r)
            match = (sh_f == init_shape)
            records.append(make_row(
                'Bliss', tau1, tau2, betas_str, lam, sigma, t, r,
                'forward', n_f, sh_f, str(match), init_shape
            ))

    for (tau1, tau2, b0, b1, b2, b3, lam, sigma) in svensson_sets:
        curve = SvenssonCurve(tau1, tau2, b0, b1, b2, b3)
        init_shape = classify_initial_shape(curve)
        hw = HullWhiteCurveState(curve, lam, sigma)
        betas_str = f'{b0},{b1},{b2},{b3}'
        for t in t_small:
            C_t = (sigma**2 / (2*lam**2)) * (1 - np.exp(-2*lam*t))
            r = float(curve.f0(t)) - C_t
            n_f, sh_f, _ = hw_forward_extrema(hw, t, r)
            match = (sh_f == init_shape)
            records.append(make_row(
                'Svensson', tau1, tau2, betas_str, lam, sigma, t, r,
                'forward', n_f, sh_f, str(match), init_shape
            ))

    return records


# ── Terminal behavior ──────────────────────────────────────────────────────────

def _terminal_sign(curve, x_large=30.0):
    """
    Evaluate sign of f0'(x) at a large-but-finite x to determine terminal direction.

    x=30 is used so that exp(-x/tau) terms are numerically distinguishable yet
    substantially smaller than their values near the origin.  The smallest tau in
    our parameter grids is 0.25, giving exp(-30/0.25)=exp(-120)≈0, while the
    largest tau (4.0) gives exp(-30/4)=exp(-7.5)≈5e-4, which is detectable.
    """
    val = float(curve.f0_prime(x_large))
    if val > 1e-14:
        return 'increasing'
    elif val < -1e-14:
        return 'decreasing'
    return 'flat'


def run_terminal_behavior():
    """
    Case study: compare terminal-increasing vs terminal-decreasing initial curves.
    Verify the asymptotic sign of df/dx for large x matches the terminal condition.
    """
    records = []
    x_large = 30.0   # large enough for terminal behavior; small enough to be numerically detectable
    t_test = 1.0
    r_test = 0.0

    # Bliss: terminal sign determined by sign of beta3 (leading x*exp(-x/tau2) term)
    # terminal-increasing: beta3 < 0 (f0'(x) → 0+ for x→∞)
    # terminal-decreasing: beta3 > 0 (f0'(x) → 0- for x→∞)
    bliss_cases = [
        # (tau1, tau2, beta0, beta1, beta3, expected_terminal)
        (0.5, 2.0, 1, 1, -1, 'increasing'),
        (0.5, 2.0, 1, 1,  1, 'decreasing'),
        (1.0, 4.0, 0, -1, -2, 'increasing'),
        (1.0, 4.0, 0, -1,  2, 'decreasing'),
    ]

    for (tau1, tau2, b0, b1, b3, expected) in bliss_cases:
        curve = BlissCurve(tau1, tau2, b0, b1, b3)
        term = _terminal_sign(curve, x_large)
        lam, sigma = 0.5, 0.5
        hw = HullWhiteCurveState(curve, lam, sigma)
        n_f, sh_f, _ = hw_forward_extrema(hw, t_test, r_test)
        df_large = float(hw.d_forward_dx(float(x_large), t_test, r_test))
        sign_match = (term == expected)
        betas_str = f'{b0},{b1},{b3}'
        print(f"Bliss({tau1},{tau2},{b0},{b1},{b3}): terminal={term} "
              f"expected={expected} match={sign_match} "
              f"df/dx@x={x_large}: {df_large:.2e} shape={sh_f}")
        records.append(make_row(
            'Bliss', tau1, tau2, betas_str, lam, sigma, t_test, r_test,
            'forward', n_f, sh_f, str(sign_match), term
        ))

    # Svensson: terminal sign depends on which tau is larger and sign of its beta
    # With tau2 > tau1: dominant decay is exp(-x/tau2); leading term ~ -beta3/tau2^2 * x * e^{-x/tau2}
    # terminal-increasing (f0' → 0+): beta3 < 0
    # terminal-decreasing (f0' → 0-): beta3 > 0
    svensson_cases = [
        (0.5, 2.0, 1, 1, 0.5, -1, 'increasing'),
        (0.5, 2.0, 1, 1, 0.5,  1, 'decreasing'),
        (1.0, 2.0, 0, -1, 1,  -2, 'increasing'),
        (1.0, 2.0, 0, -1, 1,   2, 'decreasing'),
    ]

    for (tau1, tau2, b0, b1, b2, b3, expected) in svensson_cases:
        curve = SvenssonCurve(tau1, tau2, b0, b1, b2, b3)
        term = _terminal_sign(curve, x_large)
        lam, sigma = 0.5, 0.5
        hw = HullWhiteCurveState(curve, lam, sigma)
        n_f, sh_f, _ = hw_forward_extrema(hw, t_test, r_test)
        df_large = float(hw.d_forward_dx(float(x_large), t_test, r_test))
        sign_match = (term == expected)
        betas_str = f'{b0},{b1},{b2},{b3}'
        print(f"Svensson({tau1},{tau2},{b0},{b1},{b2},{b3}): terminal={term} "
              f"expected={expected} match={sign_match} "
              f"df/dx@x={x_large}: {df_large:.2e} shape={sh_f}")
        records.append(make_row(
            'Svensson', tau1, tau2, betas_str, lam, sigma, t_test, r_test,
            'forward', n_f, sh_f, str(sign_match), term
        ))

    return records


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_histogram(counts_f, counts_y, family, bound, out_path):
    """Histogram of forward and yield extrema counts for a family."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, counts, ctype in zip(axes, [counts_f, counts_y], ['Forward', 'Yield']):
        unique, freq = np.unique(counts, return_counts=True)
        ax.bar(unique, freq, color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(x=bound, color='red', linestyle='--', label=f'Bound = {bound}')
        ax.set_xlabel('Extrema count')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{family} {ctype} extrema')
        ax.legend()
    fig.suptitle(f'{family} family extrema distribution (max ≤ {bound})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def fig_boundaries(boundary_results, out_path):
    """Plot r0(t) and r_inf(t) for representative parameter sets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (label, t_arr, r0_arr, rinf_arr) in enumerate(boundary_results[:6]):
        ax = axes[i]
        ax.plot(t_arr, r0_arr,   label='r0(t)',      color='blue')
        ax.plot(t_arr, rinf_arr, label='rinf_f(t)', color='orange', linestyle='--')
        ax.set_xlabel('t')
        ax.set_ylabel('r')
        ax.set_title(label[:60])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('State-space boundaries r0(t) and r_inf(t)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def fig_small_time(small_time_records, out_path):
    """Plot HW forward f(x,t,r) vs initial curve f0(x) for small t."""
    x_plot = np.linspace(0.01, 10.0, 300)

    # Pick a few representative records (one per small t value)
    rep = {}
    for row in small_time_records:
        t_val = row['t']
        if t_val not in rep:
            rep[t_val] = row

    n_panels = len(rep)
    if n_panels == 0:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, (t_val, row) in zip(axes, sorted(rep.items())):
        family = row['family']
        tau1, tau2 = float(row['tau1']), float(row['tau2'])
        betas = row['betas'].split(',')
        lam, sigma = float(row['lam']), float(row['sigma'])
        r_val = float(row['r'])

        if family == 'Bliss':
            b0, b1, b3 = float(betas[0]), float(betas[1]), float(betas[2])
            curve = BlissCurve(tau1, tau2, b0, b1, b3)
        else:
            b0, b1, b2, b3 = float(betas[0]), float(betas[1]), float(betas[2]), float(betas[3])
            curve = SvenssonCurve(tau1, tau2, b0, b1, b2, b3)

        hw = HullWhiteCurveState(curve, lam, sigma)
        f0_vals = curve.f0(x_plot)
        fhw_vals = hw.forward(x_plot, t_val, r_val)

        ax.plot(x_plot, f0_vals,  label='f0(x)',         color='blue')
        ax.plot(x_plot, fhw_vals, label=f'f(x,t,r) t={t_val}', color='orange',
                linestyle='--')
        ax.set_xlabel('x (maturity)')
        ax.set_ylabel('rate')
        ax.set_title(f'{family}, t={t_val:.0e}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Small-time attainability: f(x,t,r) ≈ f0(x) for small t')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ── Save results ───────────────────────────────────────────────────────────────

def save_csv(records, path):
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} rows → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_records = []

    # ── 1. Bliss scan ──────────────────────────────────────────────────────────
    print("Running Bliss scan ...")
    b_records, b_max_f, b_max_y, b_counts_f, b_counts_y = run_bliss_scan()
    all_records.extend(b_records)
    bliss_pass_f = b_max_f <= 4
    bliss_pass_y = b_max_y <= 4
    print(f"  Bliss forward  max extrema = {b_max_f}  ({'PASS' if bliss_pass_f else 'FAIL'} ≤ 4)")
    print(f"  Bliss yield    max extrema = {b_max_y}  ({'PASS' if bliss_pass_y else 'FAIL'} ≤ 4)")

    # ── 2. Svensson scan ───────────────────────────────────────────────────────
    print("Running Svensson scan ...")
    s_records, s_max_f, s_max_y, s_counts_f, s_counts_y = run_svensson_scan()
    all_records.extend(s_records)
    svn_pass_f = s_max_f <= 5
    svn_pass_y = s_max_y <= 5
    print(f"  Svensson forward max extrema = {s_max_f}  ({'PASS' if svn_pass_f else 'FAIL'} ≤ 5)")
    print(f"  Svensson yield   max extrema = {s_max_y}  ({'PASS' if svn_pass_y else 'FAIL'} ≤ 5)")

    # ── 3. Small-time attainability ────────────────────────────────────────────
    print("Running small-time attainability test ...")
    st_records = run_small_time_test()
    all_records.extend(st_records)
    n_match   = sum(r['small_time_match'] == 'True' for r in st_records if r['small_time_match'])
    n_total   = len(st_records)
    print(f"  Small-time matches: {n_match}/{n_total}")

    # ── 4. Terminal behavior ───────────────────────────────────────────────────
    print("Running terminal behavior study ...")
    tb_records = run_terminal_behavior()
    all_records.extend(tb_records)

    # ── 5. Boundary study ─────────────────────────────────────────────────────
    print("Running boundary study ...")
    boundary_results = run_boundary_study()

    # ── 6. Envelope study ─────────────────────────────────────────────────────
    print("Running envelope study ...")
    env_results = run_envelope_study()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    save_csv(all_records, os.path.join(RESULTS_DIR, 'exp5_results.csv'))

    # ── Figures ────────────────────────────────────────────────────────────────
    print("Creating figures ...")

    fig_histogram(
        b_counts_f, b_counts_y, 'Bliss', bound=4,
        out_path=os.path.join(RESULTS_DIR, 'exp5_bliss_bounds.png'),
    )

    fig_histogram(
        s_counts_f, s_counts_y, 'Svensson', bound=5,
        out_path=os.path.join(RESULTS_DIR, 'exp5_svensson_bounds.png'),
    )

    fig_boundaries(
        boundary_results,
        out_path=os.path.join(RESULTS_DIR, 'exp5_boundaries.png'),
    )

    fig_small_time(
        st_records,
        out_path=os.path.join(RESULTS_DIR, 'exp5_smalltime.png'),
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== Experiment 5 Summary ===")
    print(f"Bliss   forward bound ≤ 4 : {'PASS' if bliss_pass_f else 'FAIL'} (max={b_max_f})")
    print(f"Bliss   yield   bound ≤ 4 : {'PASS' if bliss_pass_y else 'FAIL'} (max={b_max_y})")
    print(f"Svensson forward bound ≤ 5: {'PASS' if svn_pass_f else 'FAIL'} (max={s_max_f})")
    print(f"Svensson yield   bound ≤ 5: {'PASS' if svn_pass_y else 'FAIL'} (max={s_max_y})")
    print(f"Small-time attainability: {n_match}/{n_total} matched initial shape")


if __name__ == '__main__':
    main()
