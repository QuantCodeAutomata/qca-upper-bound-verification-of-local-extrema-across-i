# Upper-Bound Verification of Local Extrema in the Hull-White Model

Computational reproduction of theorems on the maximum number of local extrema of
forward and yield curves in the Hull-White one-factor model across five initial
forward-curve families: flat, monotone, Nelson-Siegel, Bliss, and Svensson.

## Background

The project implements and numerically validates the result that the maximum number
of local extrema depends on the richness of the initial-curve parameterisation:

| Family        | Upper bound |
|:--------------|:-----------:|
| Flat          | 1           |
| Monotone      | 2           |
| Nelson-Siegel | 3           |
| Bliss         | 4           |
| Svensson      | 5           |

All experiments are purely synthetic — no external market data are required.

## Repository Structure

```
├── data/                # placeholder (no external data)
├── src/                 # core modules
│   ├── initial_curves.py    # Flat, Monotone, Nelson-Siegel, Bliss, Svensson families
│   ├── hull_white.py        # HullWhiteCurveState: forward, yield, derivatives
│   ├── root_counting.py     # adaptive root counting and shape classification
│   ├── state_space.py       # boundary curves and state-space segmentation
│   ├── transformed_plane.py # transformed-plane geometry and Wronskian envelopes
│   └── asymptotic.py        # asymptotic probability formulas and Monte Carlo
├── exp/                 # experiment scripts
│   ├── exp_1_upper_bounds.py
│   ├── exp_2_flat_monotone.py
│   ├── exp_3_ns_special.py
│   ├── exp_4_ns_general.py
│   ├── exp_5_bliss_svensson.py
│   └── exp_6_asymptotic.py
├── tests/               # pytest suite (40 tests)
│   ├── test_core_modules.py
│   └── test_experiments.py
├── results/             # generated CSVs and PNG plots
│   └── RESULTS.md       # summary of findings
└── README.md
```

## Installation

```bash
pip install numpy scipy pandas matplotlib mpmath
```

All dependencies are standard scientific Python; no finance-specific packages required.

## Running the Experiments

```bash
# individual experiments
python exp/exp_1_upper_bounds.py
python exp/exp_2_flat_monotone.py
python exp/exp_3_ns_special.py
python exp/exp_4_ns_general.py
python exp/exp_5_bliss_svensson.py
python exp/exp_6_asymptotic.py
```

Each script writes CSV and PNG files to `results/`.

## Running Tests

```bash
pytest tests/ -q
```

Expected output: **40 passed**.

## Key Results

- **Zero upper-bound violations** across all five families and both forward and yield curves.
- **State-space segmentation** for flat and monotone families matches analytical boundary predictions.
- **Nelson-Siegel sign-split** (β₂ < 0 vs β₂ > 0): shape sets confirmed for both the special case λ = 1/τ and the general case.
- **Transformed-plane regime match** for general Nelson-Siegel: 100 % agreement (1 944 / 1 944 states).
- **Bliss and Svensson** small-time attainability: 38/38 test cases match initial-curve shape.
- **Asymptotic probabilities**: analytic formulas sum to exactly 1.000000; Monte Carlo confirms convergence at large t.

See [`results/RESULTS.md`](results/RESULTS.md) for detailed tables.
