"""
lc_utils.py
───────────
Shared light curve utilities used by both calibrate_with_castles.py and lens_hunter.py.

Implements all defenses Gemini identified:
  1. Overlap filter  — discard pairs with < 1 year common window
  2. Density filter  — discard pairs where one curve has < 30% points of other
  3. ZDCF            — Z-transformed Discrete Correlation Function,
                       designed for irregular time-series without interpolation
                       (replaces scipy.signal.correlate for lag detection)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY FILTERS  (Gemini's pre-flight checks)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityReport:
    pair_id:          str
    n_A:              int
    n_B:              int
    overlap_days:     float
    density_ratio:    float   # min(nA,nB) / max(nA,nB)
    passes_overlap:   bool
    passes_density:   bool
    passes_all:       bool
    reject_reason:    str


def quality_check(lc_A: pd.DataFrame,
                  lc_B: pd.DataFrame,
                  pair_id: str = "pair",
                  min_overlap_days: float = 365.0,
                  min_density_ratio: float = 0.30) -> QualityReport:
    """
    Pre-flight data quality check.
    Returns a QualityReport. Only proceed to scoring if passes_all is True.

    Gemini's two rules:
      - Overlap filter:  common MJD window must be >= min_overlap_days (default 1 yr)
      - Density filter:  shorter curve must have >= 30% points of longer curve
    """
    t_A = lc_A["mjd"].values
    t_B = lc_B["mjd"].values

    overlap_start = max(t_A.min(), t_B.min())
    overlap_end   = min(t_A.max(), t_B.max())
    overlap_days  = max(0.0, overlap_end - overlap_start)

    n_A = len(lc_A)
    n_B = len(lc_B)
    density_ratio = min(n_A, n_B) / max(n_A, n_B) if max(n_A, n_B) > 0 else 0.0

    passes_overlap = overlap_days  >= min_overlap_days
    passes_density = density_ratio >= min_density_ratio

    reasons = []
    if not passes_overlap:
        reasons.append(f"overlap {overlap_days:.0f}d < {min_overlap_days:.0f}d")
    if not passes_density:
        reasons.append(f"density ratio {density_ratio:.2f} < {min_density_ratio:.2f}")

    return QualityReport(
        pair_id        = pair_id,
        n_A            = n_A,
        n_B            = n_B,
        overlap_days   = overlap_days,
        density_ratio  = density_ratio,
        passes_overlap = passes_overlap,
        passes_density = passes_density,
        passes_all     = passes_overlap and passes_density,
        reject_reason  = "; ".join(reasons) if reasons else "OK",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ZDCF  (Z-transformed Discrete Correlation Function)
# ══════════════════════════════════════════════════════════════════════════════
#
# The ZDCF (Alexander 1997) calculates time lags directly from data point
# PAIRS, not from an interpolated grid. This eliminates the np.interp trap
# Gemini identified — no flat lines, no spurious 0-lag peaks.
#
# Algorithm:
#   1. For every pair of points (t_i from A, t_j from B), compute lag = t_i - t_j
#   2. Bin the pairs by lag
#   3. In each bin, compute the Fisher Z-transformed correlation coefficient
#   4. The lag with the highest |Z| is the best estimate
#
# Reference: Alexander, T. 1997, ASSL, 218, 163

def zdcf(t_A: np.ndarray, f_A: np.ndarray,
         t_B: np.ndarray, f_B: np.ndarray,
         lag_range: Tuple[float, float] = (-90, 90),
         n_bins: int = 60,
         min_pairs_per_bin: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-transformed Discrete Correlation Function.

    Parameters
    ----------
    t_A, f_A : arrays for light curve A (times and fluxes)
    t_B, f_B : arrays for light curve B
    lag_range : (min_lag, max_lag) in days to search
    n_bins    : number of lag bins
    min_pairs_per_bin : minimum data pairs per bin (bins with fewer are masked)

    Returns
    -------
    lag_centers : center of each lag bin (days)
    zdcf_vals   : Z-transformed correlation coefficient per bin
    n_pairs     : number of data pairs per bin (for uncertainty estimation)
    """
    # Normalise both series
    fA_n = (f_A - f_A.mean()) / (f_A.std() + 1e-9)
    fB_n = (f_B - f_B.mean()) / (f_B.std() + 1e-9)

    # All pairwise lags and products
    # Shape: (n_A, n_B)
    lag_matrix  = t_A[:, None] - t_B[None, :]        # lag = t_A - t_B
    prod_matrix = fA_n[:, None] * fB_n[None, :]

    # Flatten
    lags_all  = lag_matrix.ravel()
    prods_all = prod_matrix.ravel()

    # Bin edges
    lag_min, lag_max = lag_range
    edges = np.linspace(lag_min, lag_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    zdcf_vals = np.full(n_bins, np.nan)
    n_pairs   = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (lags_all >= edges[i]) & (lags_all < edges[i+1])
        n    = mask.sum()
        n_pairs[i] = n
        if n < min_pairs_per_bin:
            continue
        r = prods_all[mask].mean()
        r = np.clip(r, -0.9999, 0.9999)
        # Fisher Z-transform: stabilises variance
        z = 0.5 * np.log((1 + r) / (1 - r))
        zdcf_vals[i] = z

    return centers, zdcf_vals, n_pairs


def best_lag_from_zdcf(lag_centers: np.ndarray,
                        zdcf_vals: np.ndarray,
                        n_pairs: np.ndarray) -> Tuple[float, float]:
    """
    Extract best lag and its uncertainty from ZDCF output.

    Returns (best_lag_days, uncertainty_days)
    Uncertainty estimated from the half-width of the ZDCF peak above 50% max.
    """
    valid = ~np.isnan(zdcf_vals)
    if valid.sum() < 3:
        return 0.0, 999.0

    zv = zdcf_vals.copy()
    zv[~valid] = 0

    peak_idx  = np.argmax(np.abs(zv))
    best_lag  = abs(lag_centers[peak_idx])
    peak_val  = abs(zv[peak_idx])

    # Half-width at 50% of peak for uncertainty
    half_max  = peak_val * 0.5
    above     = np.abs(zv) >= half_max
    if above.sum() > 1:
        uncertainty = (lag_centers[above].max() - lag_centers[above].min()) / 2
    else:
        uncertainty = (lag_centers[1] - lag_centers[0])   # bin width

    return float(best_lag), float(uncertainty)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: prepare light curve for ZDCF
# ══════════════════════════════════════════════════════════════════════════════

def prep_lc(lc: pd.DataFrame,
            flux_col: str = "flux",
            time_col: str = "mjd",
            sigma_clip: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean and prepare a light curve for ZDCF.
    - Drops NaN values
    - Sigma-clips outliers
    - Returns (times, fluxes) as numpy arrays sorted by time
    """
    df = lc[[time_col, flux_col]].dropna().copy()
    df = df.sort_values(time_col)

    # Sigma clipping
    f    = df[flux_col].values
    med  = np.median(f)
    mad  = np.median(np.abs(f - med))
    good = np.abs(f - med) < sigma_clip * mad * 1.4826
    df   = df[good]

    return df[time_col].values, df[flux_col].values


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing lc_utils.py ...\n")

    rng = np.random.default_rng(42)

    # Simulate two light curves with a known 14-day lag and unequal sampling
    t_dense = np.sort(rng.uniform(58000, 58600, 400))   # 400 pts
    t_sparse = np.sort(rng.uniform(58100, 58500, 80))    # 80 pts (sparse)

    signal_fn = lambda t: np.sin(t / 60) + 0.3 * np.sin(t / 20) + rng.normal(0, 0.05, len(t))

    f_dense  = signal_fn(t_dense)
    f_sparse = signal_fn(t_sparse - 14.0) * 0.7 + rng.normal(0, 0.05, len(t_sparse))

    lc_A = pd.DataFrame({"mjd": t_dense,  "flux": f_dense})
    lc_B = pd.DataFrame({"mjd": t_sparse, "flux": f_sparse})

    # ── Quality check ────────────────────────────────────────────────────────
    qr = quality_check(lc_A, lc_B, pair_id="test_pair")
    print(f"Quality check: overlap={qr.overlap_days:.0f}d  "
          f"density={qr.density_ratio:.2f}  passes={qr.passes_all}")
    print(f"Reject reason: {qr.reject_reason}\n")

    # ── ZDCF ────────────────────────────────────────────────────────────────
    tA, fA = prep_lc(lc_A)
    tB, fB = prep_lc(lc_B)

    centers, zvals, npairs = zdcf(tA, fA, tB, fB, lag_range=(-60, 60), n_bins=50)
    best, unc = best_lag_from_zdcf(centers, zvals, npairs)

    print(f"ZDCF result:   best lag = {best:.1f} days  (true = 14.0 days)")
    print(f"               uncertainty = ± {unc:.1f} days")
    print(f"               recovered: {'✓' if abs(best - 14.0) <= 3 else '✗'}")