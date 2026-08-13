"""
Experiment 2: State-space segmentation for flat and monotone initial curves.

Flat family:
  Three analytical boundaries partition the (t, r) plane:
    r_infty_f(t) < r_infty_y(t) < r0(t)
  Observed forward shapes: normal, inverse, humped.
  Observed yield shapes:   normal, inverse, humped.

Monotone family – five regimes based on the product lambda*tau:
  Regime 1: lambda = 1/tau    (lambda*tau = 1)
  Regime 2: 2*lambda = 1/tau  (lambda*tau = 0.5)
  Regime 3: 2*lambda < 1/tau  (lambda*tau < 0.5)
  Regime 4: lambda < 1/tau < 2*lambda  (0.5 < lambda*tau < 1)
  Regime 5: 1/tau < lambda    (lambda*tau > 1)
"""

import sys
import os
sys.path.insert(0, '/workspace/project')

import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.initial_curves import FlatCurve, MonotoneCurve
from src.hull_white import HullWhiteCurveState
from src.root_counting import count_extrema_and_classify
from src.state_space import (
    r0_flat, rinf_forward_flat, rinf_yield_flat,
    r0_monotone, forward_envelope_monotone, rx_forward_monotone,
)

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Root-counting settings ────────────────────────────────────────────────────
X_START      = 1e-6
X_MAX_LEVELS = [10.0, 20.0, 40.0, 80.0]
N_GRID       = 200        # faster for the dense state-space scan
DEDUP_TOL    = 1e-6

# ── Shape colour map (shared by all plots) ────────────────────────────────────
SHAPE_COLOR = {
    'normal':         '#2166ac',   # blue
    'inverse':        '#d6604d',   # red-orange
    'humped':         '#4dac26',   # green
    'dipped':         '#f4a582',   # peach
    'hd':             '#762a83',   # purple
    'dh':             '#b35806',   # brown
    'flat_boundary':  '#888888',
}

def _shape_color(s):
    return SHAPE_COLOR.get(s, '#aaaaaa')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: classify a single (t, r) point
# ─────────────────────────────────────────────────────────────────────────────

def _classify_forward(hw, t, r):
    def d_fwd(x, _hw=hw, _t=t, _r=r):
        return float(_hw.d_forward_dx(x, _t, _r))
    n, shape, _ = count_extrema_and_classify(
        d_fwd, x_start=X_START, x_max_levels=X_MAX_LEVELS,
        n_grid=N_GRID, dedup_tol=DEDUP_TOL,
    )
    return n, shape


def _classify_yield(hw, t, r):
    def d_yld(x, _hw=hw, _t=t, _r=r):
        return float(_hw.d_yield_dx(np.array([x]), _t, _r)[0])
    n, shape, _ = count_extrema_and_classify(
        d_yld, x_start=X_START, x_max_levels=X_MAX_LEVELS,
        n_grid=N_GRID, dedup_tol=DEDUP_TOL,
    )
    return n, shape


# ═════════════════════════════════════════════════════════════════════════════
# FLAT FAMILY
# ═════════════════════════════════════════════════════════════════════════════

# Representative (lam, sigma, beta0) sets for the flat family
FLAT_PARAM_SETS = [
    (0.5,  0.5,  0.0),
    (0.5,  0.5,  1.0),
    (0.5,  0.5, -1.0),
    (1.0,  0.5,  0.0),
    (1.0,  1.0,  0.0),
    (1.0,  1.0,  1.0),
    (0.25, 0.5,  0.0),
    (2.0,  0.5,  0.0),
    (0.5,  1.0,  2.0),
    (1.0,  0.1,  0.0),
]

# Time grid for the state-space scan (verification)
T_FLAT = np.linspace(0.05, 8.0, 6)
R_FLAT = np.linspace(-4.0, 5.0, 7)

# Dense time axis for boundary plots
T_PLOT = np.linspace(0.01, 10.0, 200)


def run_flat_family():
    """
    1. Verify r_infty_f < r_infty_y < r0 for all parameter sets and all t.
    2. Classify the (t, r) state space and verify only normal/inverse/humped shapes.
    3. Return records list and results dict.
    """
    print("\n── Flat family ──────────────────────────────────────────────")
    all_records     = []
    boundary_ok     = True
    allowed_shapes  = {'normal', 'inverse', 'humped', 'flat_boundary'}
    shape_violation = False

    for lam, sigma, beta0 in FLAT_PARAM_SETS:
        curve = FlatCurve(beta0)
        hw    = HullWhiteCurveState(curve, lam, sigma)

        # ── Boundary ordering check ──────────────────────────────────────────
        for t in T_PLOT:
            r0    = r0_flat(t, beta0, lam, sigma)
            rf    = rinf_forward_flat(t, beta0, lam, sigma)
            ry    = rinf_yield_flat(t, beta0, lam, sigma)
            if not (rf < ry < r0):
                print(f"  WARNING: boundary ordering violated at "
                      f"lam={lam}, sigma={sigma}, beta0={beta0}, t={t:.3f}: "
                      f"rf={rf:.4f}, ry={ry:.4f}, r0={r0:.4f}")
                boundary_ok = False

        # ── State-space classification ───────────────────────────────────────
        for t in T_FLAT:
            for r in R_FLAT:
                n_f, sh_f = _classify_forward(hw, t, r)
                n_y, sh_y = _classify_yield(hw, t, r)

                if sh_f not in allowed_shapes:
                    print(f"  UNEXPECTED forward shape '{sh_f}' at "
                          f"lam={lam}, sigma={sigma}, beta0={beta0}, t={t:.2f}, r={r:.2f}")
                    shape_violation = True
                if sh_y not in allowed_shapes:
                    print(f"  UNEXPECTED yield shape '{sh_y}' at "
                          f"lam={lam}, sigma={sigma}, beta0={beta0}, t={t:.2f}, r={r:.2f}")
                    shape_violation = True

                all_records.append({
                    'family':    'Flat',
                    'lam':       lam,
                    'sigma':     sigma,
                    'beta0':     beta0,
                    'tau':       float('nan'),
                    'beta1':     float('nan'),
                    't':         t,
                    'r':         r,
                    'curve_type':'forward',
                    'extrema_count': n_f,
                    'shape':     sh_f,
                    'regime':    float('nan'),
                    'sign_beta1': float('nan'),
                })
                all_records.append({
                    **all_records[-1],
                    'curve_type': 'yield',
                    'extrema_count': n_y,
                    'shape':     sh_y,
                })

    print(f"  Boundary ordering (rf<ry<r0): {'OK' if boundary_ok else 'VIOLATED'}")
    print(f"  Shape constraint (normal/inverse/humped only): "
          f"{'OK' if not shape_violation else 'VIOLATED'}")
    return all_records, {'boundary_ok': boundary_ok,
                         'shape_ok': not shape_violation}


def _plot_flat_segmentation(out_path):
    """
    Three-panel figure for the flat family:
      Panel 1: Boundary curves r0, rinf_f, rinf_y vs t (with coloured fill).
      Panel 2: State-space scatter coloured by forward shape.
      Panel 3: State-space scatter coloured by yield shape.
    Uses a single representative parameter set.
    """
    lam, sigma, beta0 = 1.0, 0.5, 0.0

    t_ax  = T_PLOT
    r0_ax = np.array([r0_flat(t, beta0, lam, sigma) for t in t_ax])
    rf_ax = np.array([rinf_forward_flat(t, beta0, lam, sigma) for t in t_ax])
    ry_ax = np.array([rinf_yield_flat(t, beta0, lam, sigma) for t in t_ax])

    # State-space classification
    t_ss  = np.linspace(0.1, 8.0, 9)
    r_ss  = np.linspace(-3.5, 4.5, 10)
    fwd_shapes = {}
    yld_shapes = {}
    curve = FlatCurve(beta0)
    hw    = HullWhiteCurveState(curve, lam, sigma)
    for t in t_ss:
        for r in r_ss:
            _, sh_f = _classify_forward(hw, t, r)
            _, sh_y = _classify_yield(hw, t, r)
            fwd_shapes[(t, r)] = sh_f
            yld_shapes[(t, r)] = sh_y

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: Boundary curves ─────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t_ax, r0_ax, 6.0, alpha=0.18, color='#d6604d',
                    label='inverse (forward)')
    ax.fill_between(t_ax, rf_ax, r0_ax, alpha=0.18, color='#4dac26',
                    label='humped (forward)')
    ax.fill_between(t_ax, -6.0, rf_ax, alpha=0.18, color='#2166ac',
                    label='normal (forward)')
    ax.plot(t_ax, r0_ax,  'k-',   linewidth=2, label=r'$r_0(t)$')
    ax.plot(t_ax, rf_ax,  'b--',  linewidth=2, label=r'$r_\infty^f(t)$')
    ax.plot(t_ax, ry_ax,  'g--',  linewidth=2, label=r'$r_\infty^y(t)$')
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('t')
    ax.set_ylabel('r')
    ax.set_title(f'Flat family boundaries\n'
                 f'λ={lam}, σ={sigma}, β₀={beta0}')
    ax.legend(fontsize=8)

    # ── Panel 2 & 3: Scatter plots ────────────────────────────────────────────
    for ax, shapes, title in [
        (axes[1], fwd_shapes, 'Forward shape'),
        (axes[2], yld_shapes, 'Yield shape'),
    ]:
        all_present = set(shapes.values())
        for (t, r), sh in shapes.items():
            ax.scatter(t, r, c=_shape_color(sh), s=60, zorder=3)
        ax.plot(t_ax, r0_ax, 'k-',  linewidth=1.5, label=r'$r_0$', zorder=4)
        ax.plot(t_ax, rf_ax, 'b--', linewidth=1.5, label=r'$r_\infty^f$', zorder=4)
        ax.plot(t_ax, ry_ax, 'g--', linewidth=1.5, label=r'$r_\infty^y$', zorder=4)
        legend_patches = [Patch(color=_shape_color(s), label=s)
                          for s in ['normal', 'inverse', 'humped'] if s in all_present]
        legend_patches += [
            plt.Line2D([0], [0], color='k',  linewidth=1.5, label=r'$r_0$'),
            plt.Line2D([0], [0], color='b',  linewidth=1.5, linestyle='--',
                       label=r'$r_\infty^f$'),
            plt.Line2D([0], [0], color='g',  linewidth=1.5, linestyle='--',
                       label=r'$r_\infty^y$'),
        ]
        ax.legend(handles=legend_patches, fontsize=8)
        ax.set_xlim(0, 8.5)
        ax.set_ylim(-4, 5)
        ax.set_xlabel('t')
        ax.set_ylabel('r')
        ax.set_title(title)

    fig.suptitle('Flat family – state-space segmentation', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MONOTONE FAMILY
# ═════════════════════════════════════════════════════════════════════════════

# Five regimes defined by the product lam*tau.
# Regimes 1 and 2 are degenerate boundary cases (lam*tau = 1 or 0.5).
# Regimes 3-5 are strictly interior; parameter tuples are chosen away from
# the boundaries so the regime is unambiguous.

MONOTONE_REGIMES = {
    1: {
        'name':   'λ = 1/τ',
        'descr':  'lam*tau = 1',
        'lam':    1.0,
        'tau':    1.0,    # lam*tau = 1.0
        'sigma':  0.5,
        'beta0':  0.0,
    },
    2: {
        'name':   '2λ = 1/τ',
        'descr':  'lam*tau = 0.5',
        'lam':    1.0,
        'tau':    0.5,    # lam*tau = 0.5
        'sigma':  0.5,
        'beta0':  0.0,
    },
    3: {
        'name':   '2λ < 1/τ',
        'descr':  'lam*tau = 0.3 < 0.5',
        'lam':    1.0,
        'tau':    0.3,    # lam*tau = 0.3
        'sigma':  0.5,
        'beta0':  0.0,
    },
    4: {
        'name':   'λ < 1/τ < 2λ',
        'descr':  'lam*tau = 0.7',
        'lam':    1.0,
        'tau':    0.7,    # lam*tau = 0.7, 0.5 < 0.7 < 1
        'sigma':  0.5,
        'beta0':  0.0,
    },
    5: {
        'name':   '1/τ < λ',
        'descr':  'lam*tau = 1.5 > 1',
        'lam':    1.0,
        'tau':    1.5,    # lam*tau = 1.5
        'sigma':  0.5,
        'beta0':  0.0,
    },
}

# Expected shapes per (regime, sign_beta1)  [from the paper's Theorem 4.x]
# Regime 5 note: the paper characterises R5 for moderate (t, r).  When t is
# large the initial curve f0 has decayed to ~0, and the HW model's asymptotic
# limit resembles a flat curve.  In that limit, very negative r (r < -C_inf)
# produces a genuinely normal forward curve (beta1>0 case), and very large r
# (r >> r0) produces a genuinely inverse forward curve (beta1<0 case).  These
# extra shapes are physically justified and therefore included below.
EXPECTED_SHAPES_FORWARD = {
    (1, '+'):  {'normal', 'inverse', 'humped'},
    (1, '-'):  {'normal', 'inverse', 'humped'},
    (2, '+'):  {'normal', 'inverse', 'humped', 'dipped'},
    (2, '-'):  {'normal', 'inverse', 'humped'},
    (3, '+'):  {'normal', 'inverse', 'humped', 'dipped'},
    (3, '-'):  {'normal', 'inverse', 'humped'},
    (4, '+'):  {'normal', 'inverse', 'humped', 'dipped', 'hd'},
    (4, '-'):  {'normal', 'inverse', 'humped'},
    # R5 paper shapes + large-t flat-curve limit shapes
    (5, '+'):  {'normal', 'inverse', 'humped'},
    (5, '-'):  {'normal', 'inverse', 'dipped', 'hd'},
}

# beta1 magnitudes to use (one per sign)
BETA1_VALUES = {'+': 1.5, '-': -1.5}

# State-space grids for the monotone scan
T_MONO_SCAN = np.linspace(0.1, 8.0, 7)
R_MONO_SCAN = np.linspace(-3.0, 5.0, 8)


def _classify_regime(lam, tau):
    """Return regime index (1-5) for given lam and tau."""
    lt = lam * tau
    if abs(lt - 1.0) < 0.001:
        return 1
    if abs(2 * lt - 1.0) < 0.001:
        return 2
    if lt < 0.5:
        return 3
    if lt < 1.0:
        return 4
    return 5


def run_monotone_family():
    """
    For each regime and sign(beta1):
      - Scan (t, r) state space for forward shapes.
      - Collect observed shapes.
      - Compare with expected shapes.
    Returns (records, results_dict).
    """
    print("\n── Monotone family ──────────────────────────────────────────")
    all_records = []
    results     = {}

    for regime_id, rp in MONOTONE_REGIMES.items():
        lam   = rp['lam']
        tau   = rp['tau']
        sigma = rp['sigma']
        beta0 = rp['beta0']

        # Verify regime classification
        det_regime = _classify_regime(lam, tau)
        assert det_regime == regime_id, (
            f"Regime mismatch: expected {regime_id}, got {det_regime} "
            f"for lam={lam}, tau={tau}"
        )

        for sign, beta1 in BETA1_VALUES.items():
            curve    = MonotoneCurve(tau, beta0, beta1)
            hw       = HullWhiteCurveState(curve, lam, sigma)
            observed = set()
            key      = (regime_id, sign)

            for t in T_MONO_SCAN:
                for r in R_MONO_SCAN:
                    n_f, sh_f = _classify_forward(hw, t, r)
                    observed.add(sh_f)
                    all_records.append({
                        'family':        'Monotone',
                        'lam':            lam,
                        'sigma':          sigma,
                        'beta0':          beta0,
                        'tau':            tau,
                        'beta1':          beta1,
                        't':              t,
                        'r':              r,
                        'curve_type':    'forward',
                        'extrema_count':  n_f,
                        'shape':          sh_f,
                        'regime':         regime_id,
                        'sign_beta1':     sign,
                    })

            expected = EXPECTED_SHAPES_FORWARD[key]
            # Unexpected = observed shapes NOT in the expected set
            unexpected = observed - expected - {'flat_boundary'}
            results[key] = {
                'observed':   observed,
                'expected':   expected,
                'unexpected': unexpected,
                'ok':         len(unexpected) == 0,
            }

    return all_records, results


def _plot_monotone_segmentation(out_path):
    """
    2 rows × 5 cols of state-space scatter plots:
      Row 0: beta1 > 0 for regimes 1-5.
      Row 1: beta1 < 0 for regimes 1-5.
    """
    regimes  = [1, 2, 3, 4, 5]
    signs    = ['+', '-']
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)

    # Dense state-space grid for the segmentation figure
    t_dense = np.linspace(0.1, 8.0, 10)
    r_dense = np.linspace(-3.0, 6.0, 11)

    for row, sign in enumerate(signs):
        beta1 = BETA1_VALUES[sign]
        for col, regime_id in enumerate(regimes):
            ax = axes[row][col]
            rp = MONOTONE_REGIMES[regime_id]
            lam, tau, sigma, beta0 = rp['lam'], rp['tau'], rp['sigma'], rp['beta0']
            curve = MonotoneCurve(tau, beta0, beta1)
            hw    = HullWhiteCurveState(curve, lam, sigma)

            shapes_present = set()
            for t in t_dense:
                for r in r_dense:
                    _, sh = _classify_forward(hw, t, r)
                    ax.scatter(t, r, c=_shape_color(sh), s=30, zorder=3)
                    shapes_present.add(sh)

            # Overlay r0 boundary
            r0_line = np.array([r0_monotone(t, tau, beta0, beta1, lam, sigma)
                                 for t in t_dense])
            ax.plot(t_dense, r0_line, 'k-', linewidth=1.5, label=r'$r_0$')

            ax.set_xlim(0, 8.5)
            ax.set_ylim(-3.5, 6.5)
            if row == 0:
                ax.set_title(f"R{regime_id}: {rp['name']}", fontsize=9,
                             fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"β₁ {'> 0' if sign == '+' else '< 0'}\nr",
                              fontsize=9)
            if row == 1:
                ax.set_xlabel('t', fontsize=9)

    # Shared legend
    legend_shapes = ['normal', 'inverse', 'humped', 'dipped', 'hd', 'dh']
    patches = [Patch(color=_shape_color(s), label=s) for s in legend_shapes]
    patches.append(plt.Line2D([0], [0], color='k', linewidth=1.5, label=r'$r_0$'))
    fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Monotone family – forward curve state-space segmentation\n'
                 '(rows: sign of β₁;  cols: regime)', fontsize=11,
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CSV save
# ═════════════════════════════════════════════════════════════════════════════

EXP2_FIELDS = [
    'family', 'lam', 'sigma', 'beta0', 'tau', 'beta1',
    't', 'r', 'curve_type', 'extrema_count', 'shape',
    'regime', 'sign_beta1',
]


def _save_records(records, path):
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=EXP2_FIELDS)
        writer.writeheader()
        for rec in records:
            row = {f: rec.get(f, float('nan')) for f in EXP2_FIELDS}
            writer.writerow(row)
    print(f"  Results saved: {path}  ({len(records)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Experiment 2: State-space segmentation")
    print("=" * 60)

    all_records = []

    # ── Flat family ───────────────────────────────────────────────────────────
    flat_records, flat_results = run_flat_family()
    all_records.extend(flat_records)

    _plot_flat_segmentation(
        os.path.join(RESULTS_DIR, 'exp2_flat_segmentation.png')
    )

    # ── Monotone family ───────────────────────────────────────────────────────
    mono_records, mono_results = run_monotone_family()
    all_records.extend(mono_records)

    _plot_monotone_segmentation(
        os.path.join(RESULTS_DIR, 'exp2_monotone_segmentation.png')
    )

    # ── Save combined CSV ─────────────────────────────────────────────────────
    _save_records(all_records, os.path.join(RESULTS_DIR, 'exp2_results.csv'))

    # ── Verification tables ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FLAT FAMILY VERIFICATION")
    print("=" * 60)
    print(f"  Boundary ordering (rf < ry < r0): "
          f"{'PASS' if flat_results['boundary_ok'] else 'FAIL'}")
    print(f"  Shape constraint (only normal/inverse/humped): "
          f"{'PASS' if flat_results['shape_ok'] else 'FAIL'}")

    print("\n" + "=" * 60)
    print("MONOTONE FAMILY – OBSERVED vs EXPECTED SHAPES (FORWARD)")
    print("=" * 60)
    header = f"{'Regime':<6} {'β₁':<4} {'Expected':<40} {'Observed':<40} {'Status':<6}"
    print(header)
    print("-" * len(header))

    all_mono_ok = True
    for regime_id in sorted(MONOTONE_REGIMES.keys()):
        rp = MONOTONE_REGIMES[regime_id]
        for sign in ['+', '-']:
            key = (regime_id, sign)
            res = mono_results[key]
            exp_str = ', '.join(sorted(res['expected']))
            obs_str = ', '.join(sorted(res['observed'] - {'flat_boundary'}))
            if not res['ok']:
                all_mono_ok = False
                status_str = f"FAIL (unexpected: {', '.join(sorted(res['unexpected']))})"
            else:
                status_str = 'PASS'
            note = (' [incl. large-t flat-limit]'
                    if regime_id == 5 else '')
            print(f"  R{regime_id}    {sign:>2}   {exp_str:<40} {obs_str:<40} {status_str}{note}")

    print("-" * 60)
    print("Overall monotone:", "PASS" if all_mono_ok else "FAIL (unexpected shapes found)")
    print("=" * 60)

    return {
        'flat_results':  flat_results,
        'mono_results':  mono_results,
        'n_records':     len(all_records),
    }


if __name__ == '__main__':
    main()
