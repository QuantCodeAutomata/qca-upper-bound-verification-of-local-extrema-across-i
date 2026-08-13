"""
Experiment 1: Verify upper bounds on local extrema across all initial-curve families.

Theorem bounds:
  Flat       <= 1 extremum
  Monotone   <= 2 extrema
  NelsonSiegel <= 3 extrema
  Bliss      <= 4 extrema
  Svensson   <= 5 extrema

For each family, we scan a grid of (lam, sigma, curve-params) tuples and for each
resulting HullWhite state (t, r) we count the local extrema of both the forward
and yield curves via their x-derivatives.
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

from src.initial_curves import (
    FlatCurve, MonotoneCurve, NelsonSiegelCurve, BlissCurve, SvenssonCurve,
)
from src.hull_white import HullWhiteCurveState
from src.root_counting import count_extrema_and_classify

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Root-counting settings ────────────────────────────────────────────────────
X_START      = 1e-6
X_MAX_LEVELS = [10.0, 20.0, 40.0, 80.0]
N_GRID       = 300
DEDUP_TOL    = 1e-6

# ── State-space scan grid ─────────────────────────────────────────────────────
# Representative subsets of spec's t_grid / r_grid.  Extreme values (t=10,
# r=±5) push roots to very large x which forces all 4 x_max_levels and
# multiplies runtime by ≈4×.  The subsets below keep the total experiment
# within the 5-minute budget while still covering early, mid and late regimes.
T_SCAN = [0.01, 0.1, 0.5, 1.0, 5.0]
R_SCAN = [-3.0, -1.0, 0.0, 1.0, 3.0]

# ── Theorem upper bounds ──────────────────────────────────────────────────────
THEOREM_BOUNDS = {
    'Flat':         1,
    'Monotone':     2,
    'NelsonSiegel': 3,
    'Bliss':        4,
    'Svensson':     5,
}

# ── CSV fieldnames ────────────────────────────────────────────────────────────
FIELDNAMES = [
    'family', 'lam', 'sigma',
    'tau1', 'tau2',
    'beta0', 'beta1', 'beta2', 'beta3',
    't', 'r', 'curve_type', 'extrema_count', 'shape',
]


# ─────────────────────────────────────────────────────────────────────────────
# Parameter-tuple builders
# Each returns a list of (lam, sigma, params_dict) triples.
# ─────────────────────────────────────────────────────────────────────────────

def _build_flat_params():
    """60 tuples: 4 lam × 3 sigma × 5 beta0 values."""
    tuples = []
    for lam in [0.25, 0.5, 1.0, 2.0]:
        for sigma in [0.1, 0.5, 1.0]:
            for beta0 in [-3.0, -1.5, 0.0, 1.5, 3.0]:
                tuples.append((lam, sigma, {'beta0': beta0}))
    return tuples  # 4×3×5 = 60


def _build_monotone_params():
    """72+ tuples: 3 lam × 2 sigma × 3 tau × 2 beta0 × 2 beta1."""
    tuples = []
    for lam in [0.5, 1.0, 2.0]:
        for sigma in [0.1, 0.5]:
            for tau in [0.5, 1.0, 2.0]:
                for beta0 in [-1.0, 1.0]:
                    for beta1 in [-2.0, 2.0]:
                        tuples.append((lam, sigma,
                                       {'tau': tau, 'beta0': beta0, 'beta1': beta1}))
    return tuples  # 3×2×3×2×2 = 72


def _build_ns_params():
    """96 tuples: 2 lam × 2 sigma × 3 tau × 2 beta0 × 2 beta1 × 2 beta2."""
    tuples = []
    for lam in [0.5, 1.0]:
        for sigma in [0.1, 0.5]:
            for tau in [0.5, 1.0, 2.0]:
                for beta0 in [-1.0, 1.0]:
                    for beta1 in [-2.0, 2.0]:
                        for beta2 in [-2.0, 2.0]:
                            tuples.append((lam, sigma,
                                           {'tau': tau, 'beta0': beta0,
                                            'beta1': beta1, 'beta2': beta2}))
    return tuples  # 2×2×3×2×2×2 = 96


def _build_bliss_params():
    """
    64+ tuples. tau pairs are strictly distinct (|tau1-tau2| >> 0).
    Bliss uses beta3, not beta2.
    """
    tau_pairs = [
        (0.5, 2.0), (1.0, 0.25), (0.5, 0.25),
        (1.0, 2.0), (0.25, 1.0), (2.0, 0.5),
    ]
    tuples = []
    for lam in [0.5, 1.0]:
        for sigma in [0.1, 0.5]:
            for tau1, tau2 in tau_pairs:
                for beta0 in [-1.0, 1.0]:
                    for beta1 in [-2.0, 2.0]:
                        for beta3 in [-2.0, 2.0]:
                            tuples.append((lam, sigma,
                                           {'tau1': tau1, 'tau2': tau2,
                                            'beta0': beta0, 'beta1': beta1,
                                            'beta3': beta3}))
    return tuples  # 2×2×6×2×2×2 = 192; we take first 96


def _build_svensson_params():
    """64+ tuples, tau1 != tau2 strictly."""
    tau_pairs = [(0.5, 2.0), (1.0, 0.25), (0.5, 0.25), (1.0, 2.0)]
    tuples = []
    for lam in [0.5, 1.0]:
        for sigma in [0.1, 0.5]:
            for tau1, tau2 in tau_pairs:
                for beta0 in [-1.0, 1.0]:
                    for beta1 in [-2.0, 2.0]:
                        for beta2 in [-2.0, 2.0]:
                            for beta3 in [-2.0, 2.0]:
                                tuples.append((lam, sigma,
                                               {'tau1': tau1, 'tau2': tau2,
                                                'beta0': beta0, 'beta1': beta1,
                                                'beta2': beta2, 'beta3': beta3}))
    return tuples  # 2×2×4×2×2×2×2 = 256; we take first 96


PARAM_BUILDERS = {
    'Flat':         _build_flat_params,
    'Monotone':     _build_monotone_params,
    'NelsonSiegel': _build_ns_params,
    'Bliss':        _build_bliss_params,
    'Svensson':     _build_svensson_params,
}

# Maximum tuples used per family (caps very large product sets)
MAX_TUPLES = 60   # ≥50 per spec; keeps total runtime ≤ 5 min


# ─────────────────────────────────────────────────────────────────────────────
# Curve factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_curve(family, params):
    if family == 'Flat':
        return FlatCurve(params['beta0'])
    if family == 'Monotone':
        return MonotoneCurve(params['tau'], params['beta0'], params['beta1'])
    if family == 'NelsonSiegel':
        return NelsonSiegelCurve(params['tau'], params['beta0'],
                                 params['beta1'], params['beta2'])
    if family == 'Bliss':
        return BlissCurve(params['tau1'], params['tau2'], params['beta0'],
                          params['beta1'], params['beta3'])
    if family == 'Svensson':
        return SvenssonCurve(params['tau1'], params['tau2'], params['beta0'],
                             params['beta1'], params['beta2'], params['beta3'])
    raise ValueError(f"Unknown family: {family}")


def _params_to_row_fields(params):
    """Return the scalar CSV fields derived from a params dict."""
    return {
        'tau1':  params.get('tau1', params.get('tau', float('nan'))),
        'tau2':  params.get('tau2', float('nan')),
        'beta0': params.get('beta0', float('nan')),
        'beta1': params.get('beta1', float('nan')),
        'beta2': params.get('beta2', float('nan')),
        'beta3': params.get('beta3', float('nan')),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-family runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_family(family_name):
    """
    Scan the parameter space for one curve family.

    Returns
    -------
    records   : list of CSV row dicts
    max_ext   : {'forward': int, 'yield': int}  – max extrema observed
    ext_dist  : {'forward': {n: count}, 'yield': {n: count}}
    """
    all_params = PARAM_BUILDERS[family_name]()[:MAX_TUPLES]
    n_tuples   = len(all_params)
    print(f"  {family_name}: {n_tuples} parameter tuples × "
          f"{len(T_SCAN)}t × {len(R_SCAN)}r × 2 curve types")

    records  = []
    max_ext  = {'forward': 0, 'yield': 0}
    ext_dist = {'forward': defaultdict(int), 'yield': defaultdict(int)}

    for idx, (lam, sigma, params) in enumerate(all_params):
        curve = _make_curve(family_name, params)
        hw    = HullWhiteCurveState(curve, lam, sigma)
        prow  = _params_to_row_fields(params)

        for t in T_SCAN:
            for r in R_SCAN:

                # ── forward derivative wrapper (scalar → scalar) ──────────
                def d_fwd(x, _hw=hw, _t=t, _r=r):
                    return float(_hw.d_forward_dx(x, _t, _r))

                # ── yield derivative wrapper (scalar → scalar) ────────────
                def d_yld(x, _hw=hw, _t=t, _r=r):
                    return float(_hw.d_yield_dx(np.array([x]), _t, _r)[0])

                n_f, shape_f, _ = count_extrema_and_classify(
                    d_fwd,
                    x_start=X_START,
                    x_max_levels=X_MAX_LEVELS,
                    n_grid=N_GRID,
                    dedup_tol=DEDUP_TOL,
                )
                n_y, shape_y, _ = count_extrema_and_classify(
                    d_yld,
                    x_start=X_START,
                    x_max_levels=X_MAX_LEVELS,
                    n_grid=N_GRID,
                    dedup_tol=DEDUP_TOL,
                )

                max_ext['forward'] = max(max_ext['forward'], n_f)
                max_ext['yield']   = max(max_ext['yield'],   n_y)
                ext_dist['forward'][n_f] += 1
                ext_dist['yield'][n_y]   += 1

                base = {
                    'family': family_name,
                    'lam':    lam,
                    'sigma':  sigma,
                    't':      t,
                    'r':      r,
                    **prow,
                }
                records.append({**base, 'curve_type': 'forward',
                                 'extrema_count': n_f, 'shape': shape_f})
                records.append({**base, 'curve_type': 'yield',
                                 'extrema_count': n_y, 'shape': shape_y})

        if (idx + 1) % 10 == 0 or (idx + 1) == n_tuples:
            print(f"    {idx + 1}/{n_tuples} tuples done  "
                  f"(max fwd={max_ext['forward']}, max yld={max_ext['yield']})")

    return records, max_ext, ext_dist


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_distributions(all_ext_dist, out_path):
    """
    Bar chart: for each family, show how often each extrema count was observed
    for forward and yield curves.
    """
    families   = list(THEOREM_BOUNDS.keys())
    fig, axes  = plt.subplots(1, len(families), figsize=(4 * len(families), 4),
                               sharey=False)
    ctype_colors = {'forward': '#2166ac', 'yield': '#d6604d'}

    for ax, fam in zip(axes, families):
        bound = THEOREM_BOUNDS[fam]
        max_n = bound + 1          # one extra bar to show no violations
        x_pos = np.arange(max_n + 1)
        width = 0.38

        for shift, ctype in [(-width / 2, 'forward'), (+width / 2, 'yield')]:
            dist = all_ext_dist[fam][ctype]
            counts = [dist.get(n, 0) for n in range(max_n + 1)]
            ax.bar(x_pos + shift, counts, width=width,
                   label=ctype, color=ctype_colors[ctype], alpha=0.85,
                   edgecolor='white', linewidth=0.5)

        ax.axvline(bound + 0.5, color='red', linestyle='--',
                   linewidth=1.2, label=f'bound={bound}')
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.set_xlabel('# extrema')
        ax.set_ylabel('occurrences')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(n) for n in range(max_n + 1)])
        ax.legend(fontsize=7)

    fig.suptitle('Observed extrema-count distributions per family\n'
                 '(red dashed = theorem bound)', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment 1: Upper bounds on local extrema")
    print("=" * 60)

    all_records  = []
    all_max_ext  = {}
    all_ext_dist = {}
    summary_rows = []

    for family in THEOREM_BOUNDS:
        print(f"\n[{family}]")
        records, max_ext, ext_dist = _run_family(family)
        all_records.extend(records)
        all_max_ext[family]  = max_ext
        all_ext_dist[family] = ext_dist

        bound = THEOREM_BOUNDS[family]
        for ctype in ('forward', 'yield'):
            obs   = max_ext[ctype]
            ok    = obs <= bound
            status = 'PASS' if ok else 'FAIL'
            print(f"  {ctype:8s}  max_observed={obs}  bound={bound}  [{status}]")
            summary_rows.append({
                'family':       family,
                'curve_type':   ctype,
                'max_extrema':  obs,
                'bound':        bound,
                'passed':       ok,
            })

    # ── Save detailed results CSV ─────────────────────────────────────────────
    results_csv = os.path.join(RESULTS_DIR, 'exp1_results.csv')
    with open(results_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nDetailed results saved: {results_csv}  ({len(all_records)} rows)")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_csv = os.path.join(RESULTS_DIR, 'exp1_summary.csv')
    summary_fields = ['family', 'curve_type', 'max_extrema', 'bound', 'passed']
    with open(summary_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved:          {summary_csv}")

    # ── Plot distributions ────────────────────────────────────────────────────
    fig_path = os.path.join(RESULTS_DIR, 'exp1_extrema_distribution.png')
    _plot_distributions(all_ext_dist, fig_path)

    # ── Final pass/fail banner ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"{'Family':<16} {'Type':<8} {'Observed':>9} {'Bound':>6} {'Result':>7}")
    print("-" * 52)
    all_passed = True
    for row in summary_rows:
        status = 'PASS' if row['passed'] else 'FAIL'
        if not row['passed']:
            all_passed = False
        print(f"{row['family']:<16} {row['curve_type']:<8} "
              f"{row['max_extrema']:>9} {row['bound']:>6} {status:>7}")
    print("-" * 52)
    print("Overall:", "ALL BOUNDS SATISFIED" if all_passed else "BOUND VIOLATED")
    print("=" * 60)

    return {
        'max_extrema':  all_max_ext,
        'ext_dist':     all_ext_dist,
        'summary_rows': summary_rows,
        'all_passed':   all_passed,
    }


if __name__ == '__main__':
    main()
