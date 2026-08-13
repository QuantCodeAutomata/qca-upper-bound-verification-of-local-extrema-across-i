"""
Experiment 4: General Nelson-Siegel classification via transformed-plane geometry.

Excludes the equality case lambda = 1/tau (requires |lambda - 1/tau| > 0.05).

Pipeline per state (lam, tau, sigma, beta0, beta1, beta2, t, r):
  1. Compute transformed parameters via compute_transformed_params
  2. Compute regime discriminant D via compute_D
  3. Classify forward regime via classify_ns_forward_regime
  4. Compute line-family coefficients a(x), b(x), c(x) on a dense x_grid
  5. Cross-check: partial_x f = a_f + b_f*tilde_beta1 + c_f*r (tol 1e-8)
  6. Compute envelope in (tilde_beta1, r) plane and plot
  7. Classify shapes by direct root counting
  8. Compare against regime prediction

Non-attainability check: 'dhd' shape absent across all NS parameter combinations.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from itertools import product

sys.path.insert(0, '/workspace/project')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.initial_curves import NelsonSiegelCurve
from src.hull_white import HullWhiteCurveState
from src.root_counting import count_extrema_and_classify
from src.transformed_plane import (
    compute_transformed_params,
    forward_line_coeffs,
    yield_line_coeffs,
    compute_envelope,
    compute_D,
    classify_ns_forward_regime,
    FORWARD_REGIME_SHAPES,
    YIELD_REGIME_SHAPES,
)

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Dense x-grid for line-coefficient computation and envelope
X_GRID = np.linspace(0.01, 40.0, 200)

# Regime colour map for plots
REGIME_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c',
                 4: '#d62728', 5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 0: '#7f7f7f'}


# ── Classification helpers ────────────────────────────────────────────────────

def classify_forward(hw: HullWhiteCurveState, t: float, r: float, n_grid: int = 200) -> str:
    d_func = lambda x: float(hw.d_forward_dx(x, t, r))
    _, shape, _ = count_extrema_and_classify(d_func, n_grid=n_grid)
    return shape


def classify_yield(hw: HullWhiteCurveState, t: float, r: float, n_grid: int = 200) -> str:
    d_func = lambda x: float(hw.d_yield_dx(np.array([x]), t, r))
    _, shape, _ = count_extrema_and_classify(d_func, n_grid=n_grid)
    return shape


# ── Transformed-plane pipeline helpers ───────────────────────────────────────

def compute_line_arrays(
    x_grid: np.ndarray,
    tau: float, lam: float,
    tb0: float, tb2: float, ts2: float,
) -> tuple:
    """Compute forward and yield line-family coefficient arrays over x_grid."""
    af = np.array([forward_line_coeffs(x, tau, tb0, tb2, ts2, lam)[0] for x in x_grid])
    bf = np.array([forward_line_coeffs(x, tau, tb0, tb2, ts2, lam)[1] for x in x_grid])
    cf = np.array([forward_line_coeffs(x, tau, tb0, tb2, ts2, lam)[2] for x in x_grid])
    ay = np.array([yield_line_coeffs(x, tau, tb0, tb2, ts2, lam)[0] for x in x_grid])
    by = np.array([yield_line_coeffs(x, tau, tb0, tb2, ts2, lam)[1] for x in x_grid])
    cy = np.array([yield_line_coeffs(x, tau, tb0, tb2, ts2, lam)[2] for x in x_grid])
    return af, bf, cf, ay, by, cy


def cross_check_derivatives(
    hw: HullWhiteCurveState,
    x_check: float, t: float, r: float,
    tau: float, lam: float,
    tb0: float, tb1: float, tb2: float, ts2: float,
    tol: float = 1e-8,
) -> tuple:
    """
    Verify partial_x f = a_f + b_f*tilde_beta1 + c_f*r
    and   partial_x y = a_y + b_y*tilde_beta1 + c_y*r
    at a single interior x_check.  Returns (fwd_ok, yld_ok, fwd_err, yld_err).
    """
    a_f, b_f, c_f = forward_line_coeffs(x_check, tau, tb0, tb2, ts2, lam)
    a_y, b_y, c_y = yield_line_coeffs(x_check, tau, tb0, tb2, ts2, lam)

    pred_f = a_f + b_f * tb1 + c_f * r
    pred_y = a_y + b_y * tb1 + c_y * r

    direct_f = float(hw.d_forward_dx(x_check, t, r))
    direct_y = float(hw.d_yield_dx(np.array([x_check]), t, r))

    err_f = abs(pred_f - direct_f)
    err_y = abs(pred_y - direct_y)
    return err_f < tol, err_y < tol, err_f, err_y


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_transformed_plane_plot(
    envelope_data: list,
    reference_points: list,
    results_dir: str,
) -> None:
    """
    Plot envelope curves and reference points in the (tilde_beta1, r) plane.

    envelope_data  : list of (label, tb1_env_arr, r_env_arr)
    reference_points: list of (label, tb1_val, r_val, regime)
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    used_labels = set()
    for label, tb1_env, r_env in envelope_data:
        mask = np.isfinite(tb1_env) & np.isfinite(r_env)
        if mask.sum() < 2:
            continue
        linestyle = '-' if '(fwd)' in label else '--'
        color = 'steelblue' if '(fwd)' in label else 'darkorange'
        lbl = label if label not in used_labels else '_'
        used_labels.add(label)
        ax.plot(tb1_env[mask], r_env[mask], linestyle, color=color,
                lw=1.5, alpha=0.8, label=lbl)

    for label, tb1_val, r_val, regime in reference_points:
        color = REGIME_COLORS.get(regime, 'k')
        ax.scatter([tb1_val], [r_val], c=color, s=80, zorder=6,
                   edgecolors='k', linewidths=0.5, label=f'{label} (reg {regime})')

    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r'$\tilde{\beta}_1$', fontsize=13)
    ax.set_ylabel('$r$', fontsize=13)
    ax.set_title('Transformed plane: envelope curves and reference states', fontsize=12)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.35)

    out = os.path.join(results_dir, 'exp4_transformed_plane.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results_dir = '/workspace/project/results'
    os.makedirs(results_dir, exist_ok=True)

    # ── Parameter sets: three (lam, tau) groups that avoid lambda = 1/tau
    #    Group A: 1/tau < lam  → regimes 1, 2, 3
    #    Group B: lam < 1/tau ≤ 2*lam → regimes 4, 5 (and 6/7 if D conditions met)
    #    Group C: 2*lam < 1/tau → regimes 5, 6
    lam_tau_groups = [
        (2.0,  1.0,  'A: 1/tau=1 < lam=2  [regimes 1,2,3]'),
        (0.5,  1.0,  'B: lam=0.5 < 1/tau=1 ≤ 2lam=1  [regimes 4,5]'),
        (0.25, 1.0,  'C: 2lam=0.5 < 1/tau=1  [regimes 5,6]'),
    ]

    sigma_vals = [0.2, 0.5, 1.0]
    beta2_vals = [-2, -1, 1, 2]
    beta1_vals = [-1, 0, 1]
    beta0_vals = [0, 1]
    t_vals     = [0.5, 1.0, 5.0]
    r_vals     = [-2.0, 0.0, 2.0]

    # Enforce exclusion of lambda = 1/tau case
    EXCLUSION_TOL = 0.05

    records           = []
    cross_check_fails = []
    envelope_data     = []   # for plot
    reference_points  = []   # for plot

    # ── Step-through for each (lam, tau) group
    for lam, tau, group_label in lam_tau_groups:
        assert abs(lam - 1.0 / tau) > EXCLUSION_TOL, \
            f"Forbidden: |lam - 1/tau| = {abs(lam - 1/tau):.4f} ≤ {EXCLUSION_TOL}"

        print(f'\nGroup {group_label}')

        # Pick one representative state for envelope plot and cross-check detail
        rep_beta2 = -1
        rep_sigma = 0.5
        rep_beta0, rep_beta1 = 0.0, 0.0
        rep_t, rep_r = 1.0, 0.0

        rep_curve = NelsonSiegelCurve(tau, rep_beta0, rep_beta1, rep_beta2)
        rep_hw    = HullWhiteCurveState(rep_curve, lam, rep_sigma)
        rep_params = compute_transformed_params(lam, tau, rep_sigma,
                                               rep_beta0, rep_beta1, rep_beta2, rep_t)
        rep_tb0, rep_tb1 = rep_params['tilde_beta0'], rep_params['tilde_beta1']
        rep_tb2, rep_ts2 = rep_params['tilde_beta2'], rep_params['tilde_sigma2']

        # Step 4: dense x_grid coefficients (forward)
        af, bf, cf, ay, by, cy = compute_line_arrays(
            X_GRID, tau, lam, rep_tb0, rep_tb2, rep_ts2)

        # Step 5: numerical cross-check at several interior x values
        x_checks = [0.5, 1.0, 2.0, 5.0, 10.0]
        print(f'  Cross-check (forward & yield derivatives) at x ∈ {x_checks}:')
        for xc in x_checks:
            fok, yok, fe, ye = cross_check_derivatives(
                rep_hw, xc, rep_t, rep_r,
                tau, lam, rep_tb0, rep_tb1, rep_tb2, rep_ts2)
            status = 'OK' if (fok and yok) else 'FAIL'
            print(f'    x={xc:5.1f}  fwd_err={fe:.2e}  yld_err={ye:.2e}  [{status}]')
            if not (fok and yok):
                cross_check_fails.append({'group': group_label, 'x': xc,
                                          'fwd_err': fe, 'yld_err': ye})

        # Step 6: compute and store envelope for the plot
        tb1_env_f, r_env_f = compute_envelope(X_GRID, af, bf, cf)
        tb1_env_y, r_env_y = compute_envelope(X_GRID, ay, by, cy)
        envelope_data.append((f'lam={lam},τ={tau} (fwd)', tb1_env_f, r_env_f))
        envelope_data.append((f'lam={lam},τ={tau} (yld)', tb1_env_y, r_env_y))

        # Store reference point
        rep_D      = compute_D(rep_tb2, tau, rep_ts2, lam)
        rep_regime = classify_ns_forward_regime(lam, tau, rep_tb2, rep_ts2, rep_D)
        reference_points.append((f'lam={lam},τ={tau}', rep_tb1, rep_r, rep_regime))

        # ── Main classification loop for this (lam, tau) group
        for beta0, beta1, beta2, sigma in product(
                beta0_vals, beta1_vals, beta2_vals, sigma_vals):

            curve = NelsonSiegelCurve(tau, beta0, beta1, beta2)
            hw    = HullWhiteCurveState(curve, lam, sigma)

            for t in t_vals:
                params  = compute_transformed_params(lam, tau, sigma,
                                                     beta0, beta1, beta2, t)
                tb0     = params['tilde_beta0']
                tb1     = params['tilde_beta1']
                tb2     = params['tilde_beta2']
                ts2     = params['tilde_sigma2']

                D       = compute_D(tb2, tau, ts2, lam)
                regime  = classify_ns_forward_regime(lam, tau, tb2, ts2, D)

                for r in r_vals:
                    fwd_shape = classify_forward(hw, t, r)
                    yld_shape = classify_yield(hw, t, r)

                    # Step 8: compare regime prediction against direct classification.
                    # Regimes 6 and 7 differ only by cond_5_6 (not computed here);
                    # a state predicted as regime 6 may truly belong to regime 7.
                    # We accept regime-7 shapes for regime-6 predictions.
                    regime_shapes = set(FORWARD_REGIME_SHAPES.get(regime, []))
                    if regime == 6:
                        regime_shapes |= set(FORWARD_REGIME_SHAPES.get(7, []))
                    regime_match = fwd_shape in regime_shapes if regime_shapes else True

                    records.append({
                        'lam':              lam,
                        'tau':              tau,
                        'sigma':            sigma,
                        'beta0':            beta0,
                        'beta1':            beta1,
                        'beta2':            beta2,
                        't':                t,
                        'tilde_beta1':      tb1,
                        'r':                r,
                        'D':                D,
                        'regime_predicted': regime,
                        'forward_shape':    fwd_shape,
                        'yield_shape':      yld_shape,
                        'regime_match':     regime_match,
                    })

    df = pd.DataFrame(records)

    # ── Save CSV
    csv_path = os.path.join(results_dir, 'exp4_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved: {csv_path}  ({len(df)} rows)')

    # ── Transformed-plane plot
    print('\nGenerating transformed-plane plot …')
    make_transformed_plane_plot(envelope_data, reference_points, results_dir)

    # ── Summary: regimes found and violations
    print('\n' + '=' * 60)
    print('SUMMARY: exp_4_ns_general')
    print('=' * 60)

    regimes_found = sorted(df['regime_predicted'].unique())
    print(f'\nForward regimes found: {regimes_found}')

    # Regime 7 is never predicted because classify_ns_forward_regime is called with
    # cond_5_6=False.  States that truly belong to regime 7 are predicted as regime 6,
    # but their shapes are accepted (regime_match uses the union of regimes 6 & 7).
    all_7 = set(range(1, 8)).issubset(set(regimes_found))
    print(f'All 7 regimes predicted: {"YES" if all_7 else "NO"}')
    if not all_7:
        missing = sorted(set(range(1, 8)) - set(regimes_found))
        print(f'  Regime(s) {missing} not predicted: regime 7 requires cond_5_6=True '
              f'(a secondary condition not computed here).')
        print('  States in regime 7 are classified as regime 6 with extended shape set.')
    all_6_covered = set(range(1, 7)).issubset(set(regimes_found))
    print(f'Regimes 1–6 all represented: {"YES" if all_6_covered else "NO"}')

    violations = df[~df['regime_match']]
    print(f'\nRegime-prediction violations (shape ∉ extended regime set): {len(violations)}')
    if len(violations) > 0:
        print(violations[['lam', 'tau', 'beta2', 'sigma', 't', 'r',
                           'regime_predicted', 'forward_shape']].head(10).to_string(index=False))

    # ── Cross-check summary
    print(f'\nDerivative cross-check failures (tol=1e-8): {len(cross_check_fails)}')
    if cross_check_fails:
        for f in cross_check_fails:
            print(f"  {f}")

    # ── Non-attainability check: 'dhd' shape must be absent for Nelson-Siegel
    print('\n--- Non-attainability of "dhd" shape ---')
    dhd_fwd = df[df['forward_shape'] == 'dhd']
    dhd_yld = df[df['yield_shape'] == 'dhd']
    print(f'  "dhd" in forward shapes: {len(dhd_fwd)} occurrences '
          f'({"CONFIRMED ABSENT" if len(dhd_fwd) == 0 else "UNEXPECTED - FOUND!"})')
    print(f'  "dhd" in yield shapes  : {len(dhd_yld)} occurrences '
          f'({"CONFIRMED ABSENT" if len(dhd_yld) == 0 else "UNEXPECTED - FOUND!"})')

    # ── Shape distribution per regime
    print('\n--- Forward shape distribution per regime ---')
    for reg in sorted(regimes_found):
        sub   = df[df['regime_predicted'] == reg]
        shapes = sorted(sub['forward_shape'].unique())
        expected = FORWARD_REGIME_SHAPES.get(reg, [])
        extra    = [s for s in shapes if s not in expected]
        print(f'  Regime {reg}: observed={shapes}')
        print(f'            expected={sorted(expected)}')
        if extra:
            print(f'            EXTRA (not in expected set): {extra}')

    print('=' * 60)
    print('Done.')


if __name__ == '__main__':
    main()
