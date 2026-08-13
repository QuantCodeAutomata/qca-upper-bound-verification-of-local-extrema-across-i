"""
Root counting and curve shape classification for Hull-White model experiments.

Core functionality:
  - count_transversal_roots: Count sign-change roots of a derivative function
  - classify_shape: Map derivative sign sequence to shape label
  - find_roots_adaptive: Adaptive truncation root finding on (0, infinity)
  - count_extrema: Count local extrema of a curve (= transversal roots of derivative)
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from typing import Callable, List, Tuple, Optional


# Shape label constants
# Sign sequences: + means d/dx > 0, - means < 0 at regions between roots
SHAPE_LABELS = {
    "+": "normal",          # monotone increasing
    "-": "inverse",         # monotone decreasing
    "+-": "humped",         # increases then decreases (hump)
    "-+": "dipped",         # decreases then increases (dip)
    "+-+": "hd",            # hump then dip
    "-+-": "dh",            # dip then hump
    "+-+-": "hdh",          # hump-dip-hump
    "-+-+": "dhd",          # dip-hump-dip (non-attainable for NS)
    "+-+-+": "hdhd",        # higher order
    "-+-+-": "dhdh",        # higher order
    "+-+-+-": "hdhdhd",     # higher order
    "-+-+-+": "dhdhd",      # higher order
}


def _sign(x: float) -> int:
    """Return +1, -1, or 0 for sign of x."""
    if x > 1e-12:
        return 1
    elif x < -1e-12:
        return -1
    return 0


def find_sign_changes(
    g: Callable[[float], float],
    x_grid: np.ndarray,
    tol: float = 1e-10,
) -> List[Tuple[float, float]]:
    """
    Find intervals [x_i, x_{i+1}] where g changes sign on the given grid.

    Parameters
    ----------
    g : callable, function to evaluate
    x_grid : sorted array of x values
    tol : tolerance for zero detection

    Returns
    -------
    List of (left, right) bracket pairs where g has a sign change.
    """
    # Try vectorized evaluation first (much faster); fall back to scalar loop.
    try:
        vals = np.asarray(g(x_grid), dtype=float)
        if vals.shape != x_grid.shape:
            raise ValueError
    except Exception:
        vals = np.array([g(xi) for xi in x_grid])
    brackets = []
    for i in range(len(vals) - 1):
        s1 = _sign(vals[i])
        s2 = _sign(vals[i + 1])
        if s1 != 0 and s2 != 0 and s1 != s2:
            brackets.append((x_grid[i], x_grid[i + 1]))
        elif s1 == 0 and s2 != 0:
            # near-zero at left endpoint: check if actually a root
            pass
        elif s2 == 0 and s1 != 0:
            pass
    return brackets


def find_roots_in_bracket(
    g: Callable[[float], float],
    bracket: Tuple[float, float],
    xtol: float = 1e-12,
) -> float:
    """Find a single root in [bracket[0], bracket[1]] using Brent's method."""
    try:
        root = brentq(g, bracket[0], bracket[1], xtol=xtol, maxiter=1000)
        return root
    except ValueError:
        return np.nan


def find_roots_adaptive(
    g: Callable[[float], float],
    x_start: float = 1e-6,
    x_max_levels: List[float] = None,
    n_grid: int = 500,
    xtol: float = 1e-12,
    dedup_tol: float = 1e-6,
    min_sign_change_tol: float = 1e-8,
) -> List[float]:
    """
    Find all transversal roots of g on (0, infinity) using adaptive truncation.

    Strategy:
    1. Evaluate g on a logarithmically spaced grid up to X_max.
    2. For each sign-change bracket, refine with brentq.
    3. Increase X_max until no new roots appear and terminal sign is stable.
    4. Deduplicate roots within dedup_tol.

    Parameters
    ----------
    g : callable
    x_start : float, start of search (small positive number)
    x_max_levels : list of X_max values to try (adaptive truncation)
    n_grid : number of grid points per level
    xtol : root-finding tolerance
    dedup_tol : tolerance for declaring two roots identical
    min_sign_change_tol : minimum |g| difference to count as sign change

    Returns
    -------
    Sorted list of transversal roots.
    """
    if x_max_levels is None:
        x_max_levels = [10.0, 20.0, 40.0, 80.0]

    all_roots = []
    prev_n_roots = -1

    for X_max in x_max_levels:
        # Use log-spaced grid to resolve early features + dense coverage
        x_grid = np.concatenate([
            np.linspace(x_start, 1.0, n_grid // 4),
            np.linspace(1.0, 5.0, n_grid // 4),
            np.linspace(5.0, X_max, n_grid // 2),
        ])
        x_grid = np.unique(x_grid)

        brackets = find_sign_changes(g, x_grid)
        level_roots = []
        for bracket in brackets:
            r = find_roots_in_bracket(g, bracket, xtol=xtol)
            if not np.isnan(r):
                level_roots.append(r)

        # Deduplicate
        level_roots = sorted(level_roots)
        deduped = []
        for r in level_roots:
            if not deduped or abs(r - deduped[-1]) > dedup_tol:
                deduped.append(r)

        all_roots = deduped

        # Check stability: if root count didn't change, we're done
        if len(all_roots) == prev_n_roots:
            break
        prev_n_roots = len(all_roots)

    # Final verification: check each root is a genuine sign change (transversal)
    transversal_roots = []
    for r in all_roots:
        eps = max(dedup_tol / 10, 1e-8)
        g_left = g(max(x_start, r - eps))
        g_right = g(r + eps)
        if _sign(g_left) != 0 and _sign(g_right) != 0 and _sign(g_left) != _sign(g_right):
            transversal_roots.append(r)

    return transversal_roots


def count_transversal_roots(
    g: Callable[[float], float],
    x_start: float = 1e-6,
    x_max_levels: List[float] = None,
    n_grid: int = 500,
    xtol: float = 1e-12,
    dedup_tol: float = 1e-6,
) -> int:
    """Count transversal roots of g on (0, infinity)."""
    roots = find_roots_adaptive(g, x_start, x_max_levels, n_grid, xtol, dedup_tol)
    return len(roots)


def classify_shape(
    d_func: Callable[[float], float],
    roots: List[float],
    x_start: float = 1e-6,
    x_large: float = 100.0,
) -> str:
    """
    Classify the shape of a curve from its derivative function d_func
    and a list of (transversal) roots.

    Parameters
    ----------
    d_func : derivative function g(x) = d(curve)/dx
    roots : sorted list of transversal roots in (0, infinity)
    x_start : small positive x for evaluating initial sign
    x_large : large x for evaluating terminal sign

    Returns
    -------
    shape label (e.g., "normal", "humped", etc.)
    """
    # Evaluate initial sign (just after x=0)
    g0 = d_func(x_start)
    g_inf = d_func(x_large)

    if len(roots) == 0:
        # Monotone: sign of d_func
        if g0 > 0:
            return "normal"
        elif g0 < 0:
            return "inverse"
        else:
            return "flat_boundary"

    # Build sign sequence from initial sign and number of roots
    init_sign = "+" if g0 > 0 else "-"
    sign_seq = init_sign
    current = init_sign
    for _ in roots:
        current = "-" if current == "+" else "+"
        sign_seq += current

    return SHAPE_LABELS.get(sign_seq, f"unknown:{sign_seq}")


def count_extrema_and_classify(
    d_func: Callable[[float], float],
    x_start: float = 1e-6,
    x_max_levels: List[float] = None,
    n_grid: int = 500,
    xtol: float = 1e-12,
    dedup_tol: float = 1e-6,
    x_large: float = 100.0,
) -> Tuple[int, str, List[float]]:
    """
    Count local extrema and classify curve shape.

    Returns
    -------
    (n_extrema, shape_label, roots_list)
    """
    roots = find_roots_adaptive(d_func, x_start, x_max_levels, n_grid, xtol, dedup_tol)
    n_extrema = len(roots)
    shape = classify_shape(d_func, roots, x_start, x_large)
    return n_extrema, shape, roots


def scan_parameter_grid(
    hw_state,
    curve_type: str,
    t_grid: np.ndarray,
    r_grid: np.ndarray,
    x_start: float = 1e-6,
    x_max_levels: List[float] = None,
    n_grid: int = 300,
    xtol: float = 1e-12,
    dedup_tol: float = 1e-6,
) -> dict:
    """
    Scan a grid of (t, r) states and record max extrema count and shapes.

    Parameters
    ----------
    hw_state : HullWhiteCurveState object
    curve_type : "forward" or "yield"
    t_grid, r_grid : arrays of t and r values to scan
    x_start : minimum maturity for root search
    x_max_levels : adaptive truncation levels
    n_grid : grid density for root search

    Returns
    -------
    dict with keys: max_extrema, records (list of dicts), shape_counts
    """
    if x_max_levels is None:
        x_max_levels = [10.0, 20.0, 40.0, 80.0]

    records = []
    max_extrema = 0
    shape_counts = {}

    for t in t_grid:
        for r in r_grid:
            if curve_type == "forward":
                def d_func(x, _t=t, _r=r):
                    return float(hw_state.d_forward_dx(x, _t, _r))
            else:  # yield
                def d_func(x, _t=t, _r=r):
                    return float(hw_state.d_yield_dx(np.array([x]), _t, _r))

            n_ext, shape, roots = count_extrema_and_classify(
                d_func, x_start, x_max_levels, n_grid, xtol, dedup_tol
            )
            if n_ext > max_extrema:
                max_extrema = n_ext

            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            records.append({
                "t": t,
                "r": r,
                "curve_type": curve_type,
                "extrema_count": n_ext,
                "shape": shape,
                "roots": roots,
            })

    return {
        "max_extrema": max_extrema,
        "records": records,
        "shape_counts": shape_counts,
    }


def check_near_tangential(
    g: Callable[[float], float],
    x_candidate: float,
    eps: float = 1e-5,
) -> bool:
    """
    Check if a root is near-tangential (g doesn't clearly change sign).
    Returns True if the root is likely tangential (non-transversal).
    """
    g_left = g(x_candidate - eps)
    g_right = g(x_candidate + eps)
    # If both sides have the same sign, the root is tangential
    if g_left * g_right > 0:
        return True
    return False
