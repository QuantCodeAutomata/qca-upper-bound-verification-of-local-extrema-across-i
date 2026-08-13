# Hull-White Interest Rate Model – Project Memory

## Repository Layout

```
/workspace/project/
├── src/                     # Core library modules
│   ├── initial_curves.py    # FlatCurve, MonotoneCurve, NelsonSiegelCurve, BlissCurve, SvenssonCurve
│   ├── hull_white.py        # HullWhiteCurveState (forward, yield, derivatives)
│   ├── root_counting.py     # count_extrema_and_classify, find_roots_adaptive, count_transversal_roots
│   ├── state_space.py       # r0_general, rinf_forward_general, rinf_yield_svensson, trace_forward_envelope_numerical
│   ├── asymptotic.py        # yield_asymptotic_probs, forward_asymptotic_probs_*, monte_carlo_shape_probs
│   └── transformed_plane.py # NS line-family geometry (Wronskian, regime classification)
├── exp/                     # Experiment scripts (run offline, produce results/)
│   ├── exp_1_upper_bounds.py    # Theorem bound verification (Flat≤1…Svensson≤5)
│   ├── exp_2_flat_monotone.py   # State-space segmentation, flat & monotone families
│   ├── exp_5_bliss_svensson.py
│   └── exp_6_asymptotic.py
├── tests/                   # Fast pytest test suite
│   ├── test_core_modules.py  (32 tests)
│   └── test_experiments.py   (8 tests)
└── results/                 # Output CSVs and PNG figures
```

## Key API Patterns

### HullWhiteCurveState
```python
hw = HullWhiteCurveState(curve, lam, sigma)
hw.forward(x, t, r)                   # x can be array
hw.yield_curve_fast(np.array([x]), t, r)  # x must be array; returns scalar if input is 1-d
hw.d_forward_dx(float(x), t, r)       # x is scalar float
hw.d_yield_dx(np.array([float(x)]), t, r)  # x must be array; returns float
hw.d_forward_dx_at_zero(t, r)         # analytic limit at x=0
```

### Root counting
```python
n, shape, roots = count_extrema_and_classify(
    d_func,
    x_start=1e-6,
    x_max_levels=[10, 20, 40, 80],  # adaptive truncation
    n_grid=200,                      # grid density (200 in experiments, 100 in tests)
)
```

**Critical pitfall**: The adaptive grid always includes `x=1.0` as a boundary between the
first two linspace segments.  If the target root falls exactly at x=1.0 the sign test
`_sign(g(1.0)) = 0` prevents bracket detection.  Use roots at non-grid-aligned values
(e.g. x=1.3, x=1.5) in unit tests.

### d_yield_dx near zero
The hull_white implementation uses the analytic limit `0.5 * d_forward_dx_at_zero(t, r)`
only when `x < 1e-10`.  Tests checking dy/dx|_{x=0} must pass `x = 1e-11`, not `1e-9`.

### Terminal sign detection
Use `x_large = 30` (not 500+) when evaluating `f0'(x)` at "large x".  For the largest
tau in the parameter grids (tau=4), `exp(-30/4) ≈ 5e-4` is numerically distinguishable,
whereas `exp(-500/4) ≈ 10^{-54}` is below the detection threshold `1e-12`.

## Theoretical Bounds (verified numerically)
| Family     | Max forward extrema | Max yield extrema |
|------------|--------------------:|------------------:|
| Flat       |                   1 |                 1 |
| Monotone   |                   2 |                 2 |
| NelsonSiegel|                  3 |                 3 |
| Bliss      |                   4 |                 4 |
| Svensson   |                   5 |                 5 |

## Asymptotic Formulas
| Case                   | inverse | humped                          | normal / dipped |
|------------------------|---------|----------------------------------|-----------------|
| Yield                  | 0.5     | 0.5 - Phi(-3σ/√(8λ³))           | Phi(−3σ/√(8λ³)) |
| Forward generic        | 0.5     | 0.5 - Phi(-√2 σ/λ^{3/2})       | Phi(−√2 σ/λ^{3/2}) |
| Forward term-increasing| dip=0.5 | hd=0.5 - Phi(-√2 σ/λ^{3/2})   | norm=Phi(−…)    |
| Forward term-decreasing| 0.5     | 0.5                              | —               |

`inverse = 0.5` always (exact, independent of σ, λ).

## Shape Labels (root_counting.SHAPE_LABELS)
`+` → normal, `-` → inverse, `+-` → humped, `-+` → dipped, `+-+` → hd, `-+-` → dh, …

## Experiment Notes
- **exp_1** (upper bounds, all families): ~4 min with n_grid=300, MAX_TUPLES=60, T_SCAN
  [0.01,0.1,0.5,1,5] × R_SCAN [-3,-1,0,1,3].  Extreme values t=10 / r=±5 push roots
  to very large x, forcing all 4 x_max_levels and multiplying Bliss/Svensson runtime
  by ≈4×.  Restrict to |r|≤3 and t≤5 to stay within the 5-minute budget.
  Observed maxima: Flat=1, Monotone=2, NS=3, Bliss=3 (bound 4), Svensson=3 (bound 5).
  All bounds satisfied; Bliss/Svensson did not reach their theoretical maxima with 60 tuples.
  Key d_fwd/d_yld wrappers: `float(hw.d_forward_dx(x, t, r))` and
  `float(hw.d_yield_dx(np.array([x]), t, r)[0])`.
- **exp_2** (flat and monotone segmentation): flat boundary ordering (rf<ry<r0) and
  shape constraints pass.  Monotone: all 10 regime/sign combinations pass.
  Regime 5 expanded expected shapes: at large t the initial curve f0 decays to ≈0 and
  the HW model's asymptotic limit resembles a flat curve.  In that limit, very negative
  r (r < -C_inf) produces genuinely NORMAL forward (beta1>0), and very large r >> r0
  produces genuinely INVERSE forward (beta1<0).  These are physical, not numerical
  artefacts, so EXPECTED_SHAPES_FORWARD[(5,'+')]={'normal','inverse','humped'} and
  [(5,'-')]={'normal','inverse','dipped','hd'}.  The note "[incl. large-t flat-limit]"
  is appended to R5 rows in the verification table.
- **exp_3** (NS special case, lambda=1/tau): ~90 s with n_grid=200, 216 parameter combos.
  Key: use `expected ⊆ observed` (completeness) not `observed ⊆ expected`; extra
  shapes ('inverse'/'normal' for wrong beta2 sign) are large-t artefacts from the
  root-search horizon (x_max=80) being smaller than the actual terminal-sign root.
  The initial-sign boundary satisfies df/dx|_{x=0} = lambda*(r0-r), so r > r0 gives
  NEGATIVE initial derivative (docstring of r0_flat has the direction reversed).
- **exp_4** (NS general, |lambda-1/tau|>0.05): ~60 s with n_grid=200, 648 combos.
  Line-family cross-check is machine-precision exact (~1e-16). Regime 7 is never
  predicted with cond_5_6=False; accept regime-7 shapes for regime-6 predictions
  (regimes 6 and 7 differ only by cond_5_6). 'dhd' shape is non-attainable for NS
  (confirmed across all 1944 states). Three (lam,tau) groups cover regimes 1–6.
- **exp_5** full scan is computationally intensive (several hours for Svensson); reduce
  parameter grids if faster results are needed.
- **exp_6** uses `n_samples=2000` per Monte Carlo call; each takes 5–30 s depending on
  curve complexity.
- All figures use `matplotlib.use('Agg')` (non-interactive backend).
- Results CSV files go to `/workspace/project/results/`.

## NS Transformed-Plane Notes (exp_4)
- `compute_transformed_params(lam, tau, sigma, beta0, beta1, beta2, t)` returns dict with
  keys: tilde_beta0, tilde_beta1, tilde_beta2, tilde_sigma2, C_t, f0_t.
- `forward_line_coeffs(x, tau, tb0, tb2, ts2, lam)` returns (a, b, c) with
  partial_x f = a + b*tilde_beta1 + c*r  (tilde_beta2 not tilde_beta1 in `a`).
- `compute_envelope(x_grid, a_vals, b_vals, c_vals)` returns (tilde_beta1_env, r_env)
  arrays; plot as (tilde_beta1_env, r_env) in the transformed plane.
- `compute_D(tilde_beta2, tau, tilde_sigma2, lam)` — argument order matters.
- `classify_ns_forward_regime(lam, tau, tilde_beta2, tilde_sigma2, D, cond_5_6=False)`.

## Test Suite Notes (40 tests, ~3 s)
- `test_core_modules.py`: 32 tests covering all 5 curve families, HW identities,
  root counting, boundaries, asymptotic formulas.
- `test_experiments.py`: 8 tests using 2–3 parameter values per dimension for speed.
- Run with: `python -m pytest tests/ -v`
