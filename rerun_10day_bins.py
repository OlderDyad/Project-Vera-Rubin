"""
rerun_10day_bins.py
===================
Reruns ZDCF on the actual ZTF light curves for J1004 and J1029
using 10-day bins instead of 25-day bins.

Expected: J1004 moves from 887.5d toward ~821d
          J1029 moves from 887.5d toward ~746d

If that happens: paper gets much stronger (residuals drop dramatically)
If it doesn't:   something else is going on in the real data

Also tests bin widths: 25d, 15d, 10d, 5d for both systems.
"""

import sys, time, math, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy import stats

sys.path.insert(0, '.')
os.makedirs('outputs/diagnostic', exist_ok=True)

# ── Fetch real ZTF data ────────────────────────────────────────────────────────

TARGETS = {
    "SDSS J1004+4112": {
        "oid_A": "ZTF21aaoleaj",
        "oid_B": "ZTF19aanfixr",
        "published": 821.0,
        "ra": 151.065, "dec": 41.209,
    },
    "SDSS J1029+2623": {
        "oid_A": "ZTF18aceypuf",
        "oid_B": "ZTF19aailazi",
        "published": 746.0,
        "ra": 157.306, "dec": 26.392,
    },
}

def fetch_lc(oid):
    url = f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    dets = r.json().get("detections", [])
    df = pd.DataFrame(dets)[["mjd","magpsf","sigmapsf","fid"]].rename(
        columns={"magpsf":"mag","sigmapsf":"mag_err","fid":"band"})
    df["flux"]     = 10**(-0.4*(df["mag"]-25.0))
    df["flux_err"] = df["flux"]*df["mag_err"]*0.4*np.log(10)
    return df.sort_values("mjd").reset_index(drop=True)

def zdcf(t1, f1, t2, f2, lag_range=(500, 1200), bin_width=25.0):
    """Full ZDCF implementation matching lc_utils.py logic."""
    bins = np.arange(lag_range[0], lag_range[1] + bin_width, bin_width)
    lags, r_vals, n_vals = [], [], []

    # Normalise fluxes
    f1n = (f1 - f1.mean()) / (f1.std() + 1e-12)
    f2n = (f2 - f2.mean()) / (f2.std() + 1e-12)

    for i in range(len(bins)-1):
        blo, bhi = bins[i], bins[i+1]
        bc = (blo + bhi) / 2

        pairs1, pairs2 = [], []
        for i1, (ti1, fi1) in enumerate(zip(t1, f1n)):
            for i2, (ti2, fi2) in enumerate(zip(t2, f2n)):
                lag = ti2 - ti1
                if blo <= lag < bhi:
                    pairs1.append(fi1)
                    pairs2.append(fi2)

        n = len(pairs1)
        if n < 4:
            continue
        r, _ = stats.pearsonr(pairs1, pairs2)
        if np.isnan(r):
            continue
        lags.append(bc)
        r_vals.append(r)
        n_vals.append(n)

    return np.array(lags), np.array(r_vals), np.array(n_vals)

# ── Fetch light curves ─────────────────────────────────────────────────────────

print("Fetching real ZTF light curves from ALeRCE...")
lcs = {}
for name, info in TARGETS.items():
    print(f"  {name}:")
    for key in ["oid_A", "oid_B"]:
        oid = info[key]
        lc = fetch_lc(oid)
        # Use combined bands (all detections)
        lcs[(name, key)] = lc
        print(f"    {oid}: {len(lc)} detections, "
              f"bands={dict(lc['band'].value_counts().to_dict())}")
        time.sleep(0.3)

# ── Run ZDCF at multiple bin widths ───────────────────────────────────────────

print("\nRunning ZDCF at multiple bin widths...")
bin_widths = [25, 15, 10, 5]
results = {}

fig, axes = plt.subplots(len(TARGETS), len(bin_widths),
                          figsize=(20, 10))
fig.suptitle('Real ZTF Data: ZDCF Peak vs Bin Width\n'
             'Do finer bins separate J1004 (821d) from J1029 (746d)?',
             fontsize=13, fontweight='bold')

for row, (name, info) in enumerate(TARGETS.items()):
    lc_A = lcs[(name, "oid_A")]
    lc_B = lcs[(name, "oid_B")]

    t_A = lc_A["mjd"].values
    f_A = lc_A["flux"].values
    t_B = lc_B["mjd"].values
    f_B = lc_B["flux"].values

    published = info["published"]
    print(f"\n  {name} (published: {published}d)")
    results[name] = []

    for col, bw in enumerate(bin_widths):
        ax = axes[row][col] if len(TARGETS) > 1 else axes[col]
        lags, r_vals, n_vals = zdcf(t_A, f_A, t_B, f_B,
                                     lag_range=(500, 1200), bin_width=bw)

        if len(lags) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        peak_idx = np.argmax(r_vals)
        peak_lag = lags[peak_idx]
        peak_r   = r_vals[peak_idx]
        residual = abs(peak_lag - published) / published * 100

        results[name].append({
            "bin_width": bw,
            "peak_lag": peak_lag,
            "peak_r": peak_r,
            "residual": residual,
        })

        print(f"    bin={bw:2d}d: peak={peak_lag:6.1f}d  "
              f"r={peak_r:.3f}  residual={residual:.1f}%")

        # Plot
        ax.bar(lags, r_vals, width=bw*0.85,
               color='steelblue' if row == 0 else 'darkorange', alpha=0.7)
        ax.axvline(published, color='red', ls='--', lw=2,
                   label=f'Published: {published:.0f}d')
        ax.axvline(887.5, color='gray', ls=':', lw=1.5,
                   label='Old peak: 887.5d')
        ax.axvline(peak_lag, color='black', ls='-', lw=2,
                   label=f'New peak: {peak_lag:.1f}d')
        ax.set_title(f'{name}\nbin={bw}d | peak={peak_lag:.1f}d | '
                     f'resid={residual:.1f}%', fontsize=9)
        ax.set_xlabel('Lag (days)', fontsize=8)
        ax.set_ylabel('ZDCF r', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(500, 1200)

plt.tight_layout()
plt.savefig('outputs/diagnostic/real_data_bin_sensitivity.png',
            dpi=130, bbox_inches='tight')
print(f"\n  → Saved: outputs/diagnostic/real_data_bin_sensitivity.png")

# ── Convergence summary ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CONVERGENCE SUMMARY")
print("="*60)
print(f"{'System':<22} {'bin':>5} {'Peak':>8} {'Published':>10} {'Residual':>10} {'r':>7}")
print("-"*65)
for name, res_list in results.items():
    pub = TARGETS[name]["published"]
    for r in res_list:
        print(f"{name:<22} {r['bin_width']:>5}d {r['peak_lag']:>8.1f}d "
              f"{pub:>10.0f}d {r['residual']:>9.1f}% {r['peak_r']:>7.3f}")
    print()

# Key question: do the two systems diverge as bins narrow?
print("KEY QUESTION: Do the peaks diverge as bins narrow?")
for name, res_list in results.items():
    if len(res_list) >= 2:
        wide = res_list[0]["peak_lag"]   # 25d bins
        narrow = res_list[-1]["peak_lag"] # 5d bins
        pub = TARGETS[name]["published"]
        direction = "→ converging ✓" if abs(narrow-pub) < abs(wide-pub) else "→ diverging ✗"
        print(f"  {name}: {wide:.1f}d (25d bins) → {narrow:.1f}d (5d bins) "
              f"[published: {pub:.0f}d] {direction}")

print("\nDone. Check outputs/diagnostic/real_data_bin_sensitivity.png")