"""
Experiment 6: Asymptotic shape probabilities under the Hull-White risk-neutral law.

Covers:
  - Analytic formula verification (sum to 1, sigma→0 limit)
  - Convergence of empirical MC probabilities to analytic limits
  - Yield curve, forward generic, terminal-increasing, terminal-decreasing cases
  - Sensitivity to initial short rate r0
"""

import sys
sys.path.insert(0, '/workspace/project')

import os
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.asymptotic import (
    yield_asymptotic_probs,
    forward_asymptotic_probs_generic,
    forward_asymptotic_probs_terminal_increasing,
    forward_asymptotic_probs_terminal_decreasing,
    short_rate_moments,
    monte_carlo_shape_probs,
    monte_carlo_mc_se,
)
from src.initial_curves import FlatCurve, MonotoneCurve, SvenssonCurve
from src.hull_white import HullWhiteCurveState

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

T_SEQUENCE = [1, 2, 5, 10, 20, 50]
N_SAMPLES  = 500        # MC samples per (lam, sigma, t) point
SEED       = 42

CSV_FIELDNAMES = [
    'case', 'lam', 'sigma', 't', 'method', 'shape',
    'prob', 'theoretical_limit', 'abs_error', 'mc_se', 'r0_init',
]

ALL_RECORDS = []


def _record(case, lam, sigma, t, method, shape, prob,
            theoretical_limit, abs_error, mc_se, r0_init):
    ALL_RECORDS.append({
        'case': case, 'lam': lam, 'sigma': sigma, 't': t,
        'method': method, 'shape': shape, 'prob': prob,
        'theoretical_limit': theoretical_limit,
        'abs_error': abs_error, 'mc_se': mc_se, 'r0_init': r0_init,
    })


# ── 1. Analytic formula verification ──────────────────────────────────────────

def run_formula_verification():
    """
    Verify analytic properties:
      - Probabilities sum to 1.0.
      - As sigma→0, normal→1 and humped→0 for yield.
    """
    print("\n── Analytic formula verification ──")

    lam_sigma_pairs = [(0.3, 0.5), (0.5, 0.8), (1.0, 0.3), (2.0, 0.2)]

    for lam, sigma in lam_sigma_pairs:
        yp = yield_asymptotic_probs(lam, sigma)
        total = sum(yp.values())
        print(f"  yield (lam={lam}, sigma={sigma}): "
              f"inv={yp['inverse']:.4f} hump={yp['humped']:.4f} "
              f"norm={yp['normal']:.4f}  sum={total:.6f}")
        assert abs(total - 1.0) < 1e-10, "Yield probs do not sum to 1"

        fp = forward_asymptotic_probs_generic(lam, sigma)
        total_f = sum(fp.values())
        print(f"  fwd  (lam={lam}, sigma={sigma}): "
              f"inv={fp['inverse']:.4f} hump={fp['humped']:.4f} "
              f"norm={fp['normal']:.4f}  sum={total_f:.6f}")
        assert abs(total_f - 1.0) < 1e-10, "Forward probs do not sum to 1"

        for shape, prob in yp.items():
            _record('formula_yield', lam, sigma, np.inf, 'analytic', shape,
                    prob, prob, 0.0, 0.0, np.nan)
        for shape, prob in fp.items():
            _record('formula_fwd_generic', lam, sigma, np.inf, 'analytic', shape,
                    prob, prob, 0.0, 0.0, np.nan)

    # sigma→0 limit
    print("\n  sigma→0 check (yield):")
    for sigma_small in [0.1, 0.01, 0.001]:
        yp = yield_asymptotic_probs(1.0, sigma_small)
        print(f"    sigma={sigma_small}: normal={yp['normal']:.6f} humped={yp['humped']:.6f}")

    # terminal-decreasing fixed probs
    tdp = forward_asymptotic_probs_terminal_decreasing()
    total_td = sum(tdp.values())
    print(f"\n  terminal-decreasing: {tdp}  sum={total_td}")

    print("  All formula checks passed.\n")


# ── 2. Yield convergence ───────────────────────────────────────────────────────

def run_yield_convergence():
    """
    For yield curve with FlatCurve, compare MC probs to analytic asymptotic limits.
    """
    print("── Yield convergence study ──")
    curve_type = 'yield'
    r0_init = 0.0

    param_sets = [
        (0.5, 0.5),
        (1.0, 0.3),
        (0.25, 0.8),
    ]

    conv_data = {}   # (lam, sigma) -> {shape -> [prob_at_each_t]}

    for lam, sigma in param_sets:
        curve  = FlatCurve(0.0)
        hw     = HullWhiteCurveState(curve, lam, sigma)
        theory = yield_asymptotic_probs(lam, sigma)
        key    = (lam, sigma)
        conv_data[key] = {s: [] for s in theory}

        print(f"  (lam={lam}, sigma={sigma})")
        for t in T_SEQUENCE:
            mc_probs = monte_carlo_shape_probs(
                t, lam, sigma, r0_init, hw,
                curve_type=curve_type, n_samples=N_SAMPLES, seed=SEED
            )
            mc_se = monte_carlo_mc_se(mc_probs, N_SAMPLES)

            for shape, th_prob in theory.items():
                emp = mc_probs.get(shape, 0.0)
                se  = mc_se.get(shape, 0.0)
                conv_data[key][shape].append(emp)
                _record('yield_convergence', lam, sigma, t, 'mc', shape,
                        emp, th_prob, abs(emp - th_prob), se, r0_init)

            row_str = ' '.join(f"{s}={mc_probs.get(s,0):.3f}" for s in theory)
            print(f"    t={t:4d}: {row_str}")

    return conv_data


# ── 3. Forward generic convergence ────────────────────────────────────────────

def run_forward_generic_convergence():
    """
    Forward generic case: lambda < 1/tau.
    Uses MonotoneCurve with tau=4 → 1/tau=0.25, lambda=0.2 < 0.25.
    """
    print("── Forward generic convergence ──")
    lam, sigma, tau = 0.2, 0.5, 4.0
    curve  = MonotoneCurve(tau, 0.0, 1.0)   # f0(x) = exp(-x/4)
    hw     = HullWhiteCurveState(curve, lam, sigma)
    theory = forward_asymptotic_probs_generic(lam, sigma)
    r0_init = 0.0

    print(f"  lam={lam} sigma={sigma} tau={tau} (1/tau={1/tau:.2f} > lam={lam})")
    print(f"  Analytic: {theory}")

    for t in T_SEQUENCE:
        mc_probs = monte_carlo_shape_probs(
            t, lam, sigma, r0_init, hw,
            curve_type='forward', n_samples=N_SAMPLES, seed=SEED
        )
        mc_se = monte_carlo_mc_se(mc_probs, N_SAMPLES)
        for shape, th_prob in theory.items():
            emp = mc_probs.get(shape, 0.0)
            se  = mc_se.get(shape, 0.0)
            _record('fwd_generic', lam, sigma, t, 'mc', shape,
                    emp, th_prob, abs(emp - th_prob), se, r0_init)
        row_str = ' '.join(f"{s}={mc_probs.get(s,0):.3f}" for s in theory)
        print(f"  t={t:4d}: {row_str}")


# ── 4. Forward terminal-increasing convergence ─────────────────────────────────

def run_forward_terminal_increasing():
    """
    Terminal-increasing case (f0'(x)→0+ as x→∞).
    Uses Svensson with beta3 < 0, tau2 > tau1, so leading decay ~ -beta3/tau2^2 * x*e^(-x/tau2) > 0.
    Parameters: lam=0.5, tau1=0.5, tau2=3.0, beta3=-0.02 (<0 → terminal increasing)
    """
    print("── Forward terminal-increasing convergence ──")
    lam   = 0.5
    sigma = 0.3
    curve = SvenssonCurve(
        tau1=0.5, tau2=3.0,
        beta0=0.03, beta1=-0.01, beta2=0.01, beta3=-0.02,
    )
    # Verify terminal increasing
    f0p_large = float(curve.f0_prime(500.0))
    print(f"  f0'(500) = {f0p_large:.4e}  (should be > 0 for terminal-increasing)")

    hw     = HullWhiteCurveState(curve, lam, sigma)
    theory = forward_asymptotic_probs_terminal_increasing(lam, sigma)
    r0_init = 0.0

    print(f"  Analytic: {theory}")
    for t in T_SEQUENCE:
        mc_probs = monte_carlo_shape_probs(
            t, lam, sigma, r0_init, hw,
            curve_type='forward', n_samples=N_SAMPLES, seed=SEED
        )
        mc_se = monte_carlo_mc_se(mc_probs, N_SAMPLES)
        for shape, th_prob in theory.items():
            emp = mc_probs.get(shape, 0.0)
            se  = mc_se.get(shape, 0.0)
            _record('fwd_term_increasing', lam, sigma, t, 'mc', shape,
                    emp, th_prob, abs(emp - th_prob), se, r0_init)
        row_str = ' '.join(f"{s}={mc_probs.get(s,0):.3f}" for s in theory)
        print(f"  t={t:4d}: {row_str}")


# ── 5. Forward terminal-decreasing convergence ─────────────────────────────────

def run_forward_terminal_decreasing():
    """
    Terminal-decreasing case (f0'(x)→0- as x→∞).
    Uses MonotoneCurve with beta1 < 0, lambda > 1/tau so lambda dominates.
    f0'(x) = -(beta1/tau)*exp(-x/tau); with beta1<0 → f0' = (|beta1|/tau)*exp(-x/tau) > 0?
    Wait - for terminal-decreasing we need f0'(x) → 0-.
    MonotoneCurve: f0'(x) = -(beta1/tau)*exp(-x/tau).
    With beta1 > 0: f0'(x) < 0 → terminal-decreasing.
    Use lam=1.0, tau=2.0 (1/tau=0.5 < lam=1.0), beta1=1 → f0'(x) = -(1/2)*exp(-x/2) < 0.
    """
    print("── Forward terminal-decreasing convergence ──")
    lam   = 1.0
    sigma = 0.5
    tau   = 2.0
    curve = MonotoneCurve(tau, 0.0, 1.0)   # f0'(x) = -(1/2)*exp(-x/2) < 0
    f0p_large = float(curve.f0_prime(500.0))
    print(f"  f0'(500) = {f0p_large:.4e}  (should be < 0 for terminal-decreasing)")

    hw     = HullWhiteCurveState(curve, lam, sigma)
    theory = forward_asymptotic_probs_terminal_decreasing()
    r0_init = 0.0

    print(f"  Analytic: {theory}")
    for t in T_SEQUENCE:
        mc_probs = monte_carlo_shape_probs(
            t, lam, sigma, r0_init, hw,
            curve_type='forward', n_samples=N_SAMPLES, seed=SEED
        )
        mc_se = monte_carlo_mc_se(mc_probs, N_SAMPLES)
        for shape, th_prob in theory.items():
            emp = mc_probs.get(shape, 0.0)
            se  = mc_se.get(shape, 0.0)
            _record('fwd_term_decreasing', lam, sigma, t, 'mc', shape,
                    emp, th_prob, abs(emp - th_prob), se, r0_init)
        row_str = ' '.join(f"{s}={mc_probs.get(s,0):.3f}" for s in theory)
        print(f"  t={t:4d}: {row_str}")


# ── 6. r0 sensitivity ─────────────────────────────────────────────────────────

def run_r0_sensitivity():
    """
    For large t=50, verify asymptotic probs are insensitive to r0_init.
    """
    print("── r0 sensitivity study (t=50) ──")
    r0_vals  = [-3, 0, 3]
    t_large  = 50
    lam, sigma = 0.5, 0.5
    curve    = FlatCurve(0.0)
    hw       = HullWhiteCurveState(curve, lam, sigma)
    theory   = yield_asymptotic_probs(lam, sigma)

    for r0_init in r0_vals:
        mc_probs = monte_carlo_shape_probs(
            t_large, lam, sigma, r0_init, hw,
            curve_type='yield', n_samples=N_SAMPLES, seed=SEED
        )
        mc_se = monte_carlo_mc_se(mc_probs, N_SAMPLES)
        for shape, th_prob in theory.items():
            emp = mc_probs.get(shape, 0.0)
            se  = mc_se.get(shape, 0.0)
            _record('r0_sensitivity', lam, sigma, t_large, 'mc', shape,
                    emp, th_prob, abs(emp - th_prob), se, r0_init)
        row_str = ' '.join(f"{s}={mc_probs.get(s,0):.3f}" for s in theory)
        print(f"  r0={r0_init:+.0f}: {row_str}")


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_yield_convergence(conv_data, out_path):
    """Plot prob vs t for each yield shape, one panel per (lam, sigma)."""
    n = len(conv_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ((lam, sigma), shape_series) in zip(axes, conv_data.items()):
        theory = yield_asymptotic_probs(lam, sigma)
        for shape, series in shape_series.items():
            ax.plot(T_SEQUENCE[:len(series)], series,
                    marker='o', label=f'{shape} (MC)')
            ax.axhline(theory[shape], linestyle='--', alpha=0.6,
                       label=f'{shape} (theory)')
        ax.set_xlabel('t')
        ax.set_ylabel('probability')
        ax.set_title(f'Yield  lam={lam}  sigma={sigma}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Yield asymptotic probability convergence')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _gather_mc_series(case_tag, theory_probs):
    """Retrieve MC series from ALL_RECORDS for given case and theory keys."""
    series = {s: [] for s in theory_probs}
    for row in ALL_RECORDS:
        if row['case'] == case_tag and row['method'] == 'mc':
            s = row['shape']
            if s in series:
                series[s].append((row['t'], row['prob']))
    # Sort by t
    for s in series:
        series[s].sort(key=lambda x: x[0])
        series[s] = [p for _, p in series[s]]
    return series


def fig_forward_convergence(out_path):
    """Plot prob vs t for forward shapes (generic, term-inc, term-dec)."""
    cases = [
        ('fwd_generic',        forward_asymptotic_probs_generic(0.2, 0.5),     'Forward generic'),
        ('fwd_term_increasing', forward_asymptotic_probs_terminal_increasing(0.5, 0.3),
         'Terminal-increasing'),
        ('fwd_term_decreasing', forward_asymptotic_probs_terminal_decreasing(),
         'Terminal-decreasing'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (case_tag, theory, title) in zip(axes, cases):
        series = _gather_mc_series(case_tag, theory)
        for shape, probs in series.items():
            if probs:
                ax.plot(T_SEQUENCE[:len(probs)], probs,
                        marker='o', label=f'{shape} (MC)')
                ax.axhline(theory.get(shape, np.nan), linestyle='--', alpha=0.6,
                           label=f'{shape} (theory)')
        ax.set_xlabel('t')
        ax.set_ylabel('probability')
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Forward asymptotic probability convergence')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def fig_r0_sensitivity(out_path):
    """Plot prob vs r0_init for large t=50, yield curve."""
    r0_vals = [-3, 0, 3]
    theory  = yield_asymptotic_probs(0.5, 0.5)
    shapes  = list(theory.keys())

    # Collect empirical probs per r0
    emp_by_shape = {s: [] for s in shapes}
    for r0_init in r0_vals:
        for s in shapes:
            val = next(
                (row['prob'] for row in ALL_RECORDS
                 if row['case'] == 'r0_sensitivity' and row['r0_init'] == r0_init
                 and row['shape'] == s),
                np.nan
            )
            emp_by_shape[s].append(val)

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in shapes:
        ax.plot(r0_vals, emp_by_shape[s], marker='o', label=f'{s} (MC)')
        ax.axhline(theory[s], linestyle='--', alpha=0.6, label=f'{s} (theory)')
    ax.set_xlabel('r0_init')
    ax.set_ylabel('probability')
    ax.set_title('Yield shape probs vs r0_init  (t=50, lam=0.5, sigma=0.5)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ── Save results ───────────────────────────────────────────────────────────────

def save_csv(path):
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ALL_RECORDS)
    print(f"  Saved {len(ALL_RECORDS)} rows → {path}")


# ── Print summary table ────────────────────────────────────────────────────────

def print_summary_table():
    """Print table of MC probabilities vs theoretical limits."""
    header = f"{'Case':<25} {'lam':>5} {'sigma':>6} {'t':>5} {'Shape':<10} {'MC':>7} {'Theory':>7} {'|err|':>7} {'se':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for row in ALL_RECORDS:
        if row['method'] == 'mc' and row['t'] in [1, 10, 100]:
            print(
                f"{row['case']:<25} {row['lam']:>5.2f} {row['sigma']:>6.2f} "
                f"{row['t']:>5.0f} {row['shape']:<10} "
                f"{row['prob']:>7.4f} {row['theoretical_limit']:>7.4f} "
                f"{row['abs_error']:>7.4f} {row['mc_se']:>7.4f}"
            )
    print("=" * len(header) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run_formula_verification()
    conv_data = run_yield_convergence()
    run_forward_generic_convergence()
    run_forward_terminal_increasing()
    run_forward_terminal_decreasing()
    run_r0_sensitivity()

    save_csv(os.path.join(RESULTS_DIR, 'exp6_results.csv'))

    print("Creating figures ...")
    fig_yield_convergence(
        conv_data,
        out_path=os.path.join(RESULTS_DIR, 'exp6_yield_convergence.png'),
    )
    fig_forward_convergence(
        out_path=os.path.join(RESULTS_DIR, 'exp6_forward_convergence.png'),
    )
    fig_r0_sensitivity(
        out_path=os.path.join(RESULTS_DIR, 'exp6_r0_sensitivity.png'),
    )

    print_summary_table()
    print("Experiment 6 complete.")


if __name__ == '__main__':
    main()
