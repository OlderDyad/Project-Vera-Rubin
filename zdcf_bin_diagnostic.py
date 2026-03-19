"""
zdcf_bin_diagnostic.py
======================
Tests three hypotheses about the 887.5-day coincidence:

HYPOTHESIS 1: Bin sharpening
  What happens to the measured lag when we use 10-day bins instead of 25-day?
  Does J1004 and J1029 diverge to their true published values?

HYPOTHESIS 2: Rung ladder bias
  Are any rungs systematically favouring lags in the 700-900 day range?
  Test: run the scoring ladder on SYNTHETIC light curves with known delays
  spanning 50-1500 days and plot which delays score highest.

HYPOTHESIS 3: Survey/seasonal alias
  ZTF has a ~365-day observing cycle. Does the power spectrum of ZTF cadence
  create aliases that could produce false peaks near 875-900 days?
  (875 = 365*2 + 145, 900 = 365*2 + 170 — not obvious multiples, but test it)

Run from .venv or .venv64 with scipy, numpy, matplotlib installed.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('outputs/diagnostic', exist_ok=True)
np.random.seed(42)

# ── Shared utilities ──────────────────────────────────────────────────────────

def simulate_quasar_lc(mjd, tau=300, sigma_drw=0.3, noise=0.02):
    """
    Simulate a quasar light curve as a Damped Random Walk (DRW).
    tau = correlation timescale (days), sigma_drw = variability amplitude
    """
    n = len(mjd)
    flux = np.zeros(n)
    flux[0] = 1.0
    for i in range(1, n):
        dt = mjd[i] - mjd[i-1]
        e_factor = np.exp(-dt / tau)
        flux[i] = (flux[i-1] * e_factor +
                   sigma_drw * np.sqrt(1 - e_factor**2) * np.random.randn())
    flux = flux - flux.mean() + 1.0
    flux += np.random.normal(0, noise, n)
    return flux

def zdcf(t1, f1, t2, f2, lag_range=(-1000, 1000), bin_width=25.0):
    """
    Z-transformed Discrete Correlation Function.
    Returns (lags, r_values, n_pairs_per_bin)
    """
    lags_out = []
    r_out = []
    n_out = []

    lag_min, lag_max = lag_range
    bins = np.arange(lag_min, lag_max + bin_width, bin_width)

    for i in range(len(bins) - 1):
        bin_lo, bin_hi = bins[i], bins[i+1]
        bin_center = (bin_lo + bin_hi) / 2.0

        pairs_f1 = []
        pairs_f2 = []

        for i1, (ti1, fi1) in enumerate(zip(t1, f1)):
            for i2, (ti2, fi2) in enumerate(zip(t2, f2)):
                lag = ti2 - ti1
                if bin_lo <= lag < bin_hi:
                    pairs_f1.append(fi1)
                    pairs_f2.append(fi2)

        n = len(pairs_f1)
        if n < 4:
            continue

        r, _ = stats.pearsonr(pairs_f1, pairs_f2)
        if np.isnan(r):
            continue

        lags_out.append(bin_center)
        r_out.append(r)
        n_out.append(n)

    return np.array(lags_out), np.array(r_out), np.array(n_out)

def make_ztf_cadence(mjd_start=58000, mjd_end=61000, avg_gap=3.5, seasonal_gap=90):
    """
    Simulate realistic ZTF cadence with seasonal gaps.
    Returns array of MJD observation times.
    """
    obs = []
    mjd = mjd_start
    while mjd < mjd_end:
        # Check if we're in a seasonal gap (object behind sun ~90 days/year)
        day_of_year = mjd % 365.25
        if 150 < day_of_year < 240:  # ~90 day gap each year
            mjd += 1
            continue
        obs.append(mjd)
        mjd += avg_gap + np.random.exponential(1.0)
    return np.array(obs)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: BIN WIDTH SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

print("="*60)
print("TEST 1: BIN WIDTH SENSITIVITY")
print("="*60)
print("Testing whether 10-day bins vs 25-day bins resolve the")
print("difference between J1004 (~821d) and J1029 (~746d)...\n")

t_A = make_ztf_cadence(58000, 61000)
t_B = make_ztf_cadence(58000, 61000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TEST 1: Bin Width Sensitivity\nCan finer bins separate J1004 (821d) from J1029 (746d)?',
             fontsize=13, fontweight='bold')

for row, true_delay in enumerate([821, 746]):
    sys_name = "J1004 (true: 821d)" if true_delay == 821 else "J1029 (true: 746d)"

    # Simulate light curves with the known true delay
    f_A = simulate_quasar_lc(t_A, tau=200, sigma_drw=0.25)
    # Image B is a delayed, slightly noisy copy of A
    f_B_interp = np.interp(t_B - true_delay, t_A, f_A)
    f_B = f_B_interp + np.random.normal(0, 0.03, len(t_B))

    for col, bw in enumerate([25, 10]):
        ax = axes[row][col]
        lags, r_vals, n_pairs = zdcf(t_A, f_A, t_B, f_B,
                                      lag_range=(600, 1100), bin_width=bw)
        if len(lags) > 0:
            peak_idx = np.argmax(r_vals)
            peak_lag = lags[peak_idx]
            peak_r = r_vals[peak_idx]

            ax.bar(lags, r_vals, width=bw*0.85, color='steelblue', alpha=0.7)
            ax.axvline(true_delay, color='red', ls='--', lw=2,
                       label=f'True delay: {true_delay}d')
            ax.axvline(peak_lag, color='orange', ls='-', lw=2,
                       label=f'ZDCF peak: {peak_lag:.1f}d')
            ax.set_xlabel('Lag (days)')
            ax.set_ylabel('ZDCF r')
            ax.set_title(f'{sys_name}\nBin width = {bw}d | Peak = {peak_lag:.1f}d | r = {peak_r:.3f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            residual = abs(peak_lag - true_delay) / true_delay * 100
            print(f"  {sys_name}, bin={bw}d: peak={peak_lag:.1f}d, "
                  f"residual={residual:.1f}%, r={peak_r:.3f}")

plt.tight_layout()
plt.savefig('outputs/diagnostic/test1_bin_width.png', dpi=120, bbox_inches='tight')
print(f"\n  → Saved: /tmp/test1_bin_width.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: RUNG LADDER BIAS SCAN
# ═══════════════════════════════════════════════════════════════════════════════

print("="*60)
print("TEST 2: RUNG LADDER BIAS SCAN")
print("="*60)
print("Testing whether the scoring ladder systematically favours")
print("certain lag ranges (e.g. 700-900 days)...\n")

def score_at_lag(t_A, f_A, t_B, f_B, true_delay, bin_width=25):
    """Score a simulated pair at its true delay using simplified rung logic."""
    lags, r_vals, _ = zdcf(t_A, f_A, t_B, f_B,
                            lag_range=(true_delay-300, true_delay+300),
                            bin_width=bin_width)
    if len(r_vals) == 0:
        return 0.0, 0.0

    peak_idx = np.argmax(r_vals)
    r1 = max(0.0, float(r_vals[peak_idx]))
    measured_lag = lags[peak_idx]

    # R2: flux ratio stability
    ratio = f_A.mean() / (f_B.mean() + 1e-9)
    cv = f_A.std() / (f_A.mean() + 1e-9)
    r2 = max(0.0, 1.0 - cv / 0.35)

    # R4: fractional variability after shift
    f_B_shifted = np.interp(t_A, t_B - measured_lag, f_B,
                             left=np.nan, right=np.nan)
    mask = ~np.isnan(f_B_shifted)
    if mask.sum() > 10:
        try:
            corr, pval = stats.pearsonr(f_A[mask], f_B_shifted[mask])
            r4 = max(0.0, float(corr))
        except:
            r4 = 0.0
    else:
        r4 = 0.0

    composite = np.mean([r1, r2, r4])
    return composite, r1

# Scan across a wide range of true delays
test_delays = np.arange(50, 1500, 25)
composite_scores = []
r1_scores = []

n_trials = 5  # average over multiple simulations per delay
print(f"  Scanning {len(test_delays)} delay values ({test_delays[0]}d to {test_delays[-1]}d)...")

for true_delay in test_delays:
    trial_composites = []
    trial_r1s = []
    for _ in range(n_trials):
        t_A = make_ztf_cadence(58000, 61000)
        t_B = make_ztf_cadence(58000, 61000)
        f_A = simulate_quasar_lc(t_A, tau=200)
        f_B = np.interp(t_B - true_delay, t_A, f_A) + np.random.normal(0, 0.03, len(t_B))
        c, r1 = score_at_lag(t_A, f_A, t_B, f_B, true_delay)
        trial_composites.append(c)
        trial_r1s.append(r1)
    composite_scores.append(np.mean(trial_composites))
    r1_scores.append(np.mean(trial_r1s))

composite_scores = np.array(composite_scores)
r1_scores = np.array(r1_scores)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle('TEST 2: Rung Ladder Bias Scan\nDoes scoring favour certain delay ranges?',
             fontsize=13, fontweight='bold')

ax1.plot(test_delays, r1_scores, 'steelblue', lw=1.5, label='R1 (ZDCF peak r)')
ax1.axvspan(700, 950, alpha=0.15, color='red', label='887.5d ± 75d region')
ax1.axvline(821, color='red', ls='--', lw=1.5, label='J1004 published (821d)')
ax1.axvline(746, color='orange', ls='--', lw=1.5, label='J1029 published (746d)')
ax1.axvspan(0, 200, alpha=0.1, color='green', label='Short-delay regime')
# Mark ZTF seasonal alias regions
for alias in [365, 730, 1095]:
    ax1.axvline(alias, color='purple', ls=':', lw=1, alpha=0.5)
ax1.set_ylabel('Mean R1 score')
ax1.set_title('R1 (ZDCF correlation) vs True Delay\nPurple dotted = ZTF seasonal alias (365d multiples)')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1500)

ax2.plot(test_delays, composite_scores, 'darkorange', lw=1.5, label='Composite score (R1+R2+R4 mean)')
ax2.axvspan(700, 950, alpha=0.15, color='red', label='887.5d ± 75d region')
ax2.axvline(821, color='red', ls='--', lw=1.5, label='J1004 (821d)')
ax2.axvline(746, color='orange', ls='--', lw=1.5, label='J1029 (746d)')
ax2.axhline(0.25, color='gray', ls='--', lw=1, label='Score threshold (0.25)')
# Annotate the baseline compression zone
ax2.axvspan(1100, 1500, alpha=0.1, color='gray', label='Baseline compression zone')
for alias in [365, 730, 1095]:
    ax2.axvline(alias, color='purple', ls=':', lw=1, alpha=0.5)
ax2.set_xlabel('True Time Delay (days)')
ax2.set_ylabel('Mean Composite Score')
ax2.set_title('Composite Score vs True Delay\nGray shading = where ZTF baseline too short for full scoring')
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1500)

plt.tight_layout()
plt.savefig('outputs/diagnostic/test2_ladder_bias.png', dpi=120, bbox_inches='tight')

# Find peak of bias
peak_bias_delay = test_delays[np.argmax(composite_scores)]
print(f"\n  Composite score peak at: {peak_bias_delay}d")
print(f"  Score at 821d (J1004): {composite_scores[np.argmin(abs(test_delays-821))]:.3f}")
print(f"  Score at 746d (J1029): {composite_scores[np.argmin(abs(test_delays-746))]:.3f}")
print(f"  Score at 887d (our measurement): {composite_scores[np.argmin(abs(test_delays-887))]:.3f}")
print(f"  → Saved: /tmp/test2_ladder_bias.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: SEASONAL ALIAS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("="*60)
print("TEST 3: SEASONAL ALIAS ANALYSIS")
print("="*60)
print("Testing whether ZTF seasonal gaps create false ZDCF peaks")
print("near 887 days...\n")

# Simulate an UNCORRELATED pair — no real delay
# If 887d appears here, it's an alias
t_A = make_ztf_cadence(58000, 61000)
t_B = make_ztf_cadence(58000, 61000)

# Completely independent DRW light curves (no delay)
f_A_noise = simulate_quasar_lc(t_A, tau=200, sigma_drw=0.3)
f_B_noise = simulate_quasar_lc(t_B, tau=200, sigma_drw=0.3)  # independent

# Also test a correlated pair at a non-alias delay (say 500d)
f_B_500 = np.interp(t_B - 500, t_A, f_A_noise) + np.random.normal(0, 0.03, len(t_B))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('TEST 3: Seasonal Alias Analysis\nDo ZTF gaps create false ZDCF peaks near 887 days?',
             fontsize=13, fontweight='bold')

for ax, (f2, label, color) in zip(axes, [
    (f_B_noise, 'UNCORRELATED pair (no true delay) — aliases only', 'red'),
    (f_B_500,   'Correlated pair, TRUE delay = 500d', 'steelblue'),
    (np.interp(t_B - 821, t_A, f_A_noise) + np.random.normal(0, 0.03, len(t_B)),
     'Correlated pair, TRUE delay = 821d (J1004 published)', 'green'),
]):
    lags, r_vals, _ = zdcf(t_A, f_A_noise, t_B, f2,
                            lag_range=(0, 1500), bin_width=25)
    if len(lags) == 0:
        continue

    ax.bar(lags, r_vals, width=22, alpha=0.7, color=color)

    # Mark seasonal aliases
    for alias in [365, 730, 1095]:
        ax.axvline(alias, color='purple', ls=':', lw=1.5,
                   label=f'{alias}d (1-yr alias)' if alias == 365 else f'{alias}d')
    ax.axvline(887.5, color='black', ls='--', lw=2, label='887.5d (our measurement)')
    ax.axvline(821, color='red', ls='--', lw=1.5, alpha=0.7, label='821d (J1004)')
    ax.axvline(746, color='orange', ls='--', lw=1.5, alpha=0.7, label='746d (J1029)')

    if len(r_vals) > 0:
        peak = lags[np.argmax(r_vals)]
        ax.set_title(f'{label}\nPeak at {peak:.1f}d', fontsize=10)

    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('ZDCF r')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1500)

plt.tight_layout()
plt.savefig('outputs/diagnostic/test3_aliases.png', dpi=120, bbox_inches='tight')

# Quantify alias strength at 887d for uncorrelated pair
lags_nc, r_nc, _ = zdcf(t_A, f_A_noise, t_B, f_B_noise,
                          lag_range=(750, 1050), bin_width=25)
if len(r_nc) > 0:
    idx_887 = np.argmin(abs(lags_nc - 887.5))
    alias_r = r_nc[idx_887] if idx_887 < len(r_nc) else 0.0
    print(f"  ZDCF r at 887.5d for UNCORRELATED pair: {alias_r:.3f}")
    print(f"  (If close to 0, no systematic alias at 887d)")
    print(f"  → Saved: /tmp/test3_aliases.png\n")

# ── Summary ────────────────────────────────────────────────────────────────────
print("="*60)
print("SUMMARY")
print("="*60)
print("""
Test 1 (Bin sharpening):
  Do 10-day bins separate J1004 (821d) from J1029 (746d)?
  → See test1_bin_width.png

Test 2 (Ladder bias):
  Does the scoring ladder favour 700-900 day delays?
  → See test2_ladder_bias.png
  Key: if the composite score is elevated at 887d even for random
  pairs, the detection is less meaningful.

Test 3 (Seasonal aliases):
  Does the ZTF seasonal cadence create false ZDCF peaks near 887d?
  → See test3_aliases.png
  Key: if the uncorrelated pair shows a peak near 887d, that
  suggests our detections could be cadence artifacts.
""")