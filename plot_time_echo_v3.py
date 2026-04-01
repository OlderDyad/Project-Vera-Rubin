"""
plot_time_echo_v3.py — Time-echo alignment plots.
Uses whichever band has the most overlap between the two objects.
"""
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from pathlib import Path

DB_PATH = Path("outputs/survey_v3/survey_results.db")

CANDIDATES = {
    "SDSS1029+2623": {"oid_A": "ZTF18aceypuf",  "oid_B": "ZTF19aailazi",  "lag": 887.5},
    "SDSS1004+4112": {"oid_A": "ZTF19aaplmpl",   "oid_B": "ZTF19aailogj",  "lag": 887.5},
}

if DB_PATH.exists():
    try:
        conn = sqlite3.connect(DB_PATH)
        # We only want the BEST match (highest score) for each anchor
        query = "SELECT anchor, lag_days, score FROM candidates ORDER BY score DESC"
        rows = conn.execute(query).fetchall()
        conn.close()
        
        processed_anchors = set()
        for anchor, lag_days, score in rows:
            if anchor in CANDIDATES and anchor not in processed_anchors:
                # Only overwrite if the score is high enough to be 'trusted'
                if score > 0.4: 
                    CANDIDATES[anchor]["lag"] = lag_days
                    print(f"  [DB TRUSTED] {anchor}: {lag_days:.1f}d (Score: {score:.2f})")
                else:
                    print(f"  [DB IGNORED] {anchor}: Low score ({score:.2f}), using hardcoded lag.")
                processed_anchors.add(anchor)
    except Exception as e:
        print(f"  DB check failed ({e}), sticking with hardcoded values.")

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAND_LABELS = {1: "g", 2: "r", 3: "i"}
BAND_COLORS = {1: ("royalblue", "crimson"),
               2: ("darkorange", "purple"),
               3: ("green",     "brown")}

def fetch_lc(oid):
    url = f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    df = pd.DataFrame(r.json()['detections'])
    df = df[['mjd', 'magpsf', 'sigmapsf', 'fid']].rename(
        columns={'magpsf': 'mag', 'sigmapsf': 'err'})
    return df.sort_values('mjd').reset_index(drop=True)

def best_band(lc_A, lc_B):
    """Pick the band with the most combined detections in both objects."""
    counts = {}
    for fid in [1, 2, 3]:
        nA = (lc_A['fid'] == fid).sum()
        nB = (lc_B['fid'] == fid).sum()
        counts[fid] = nA + nB
    return max(counts, key=counts.get)

for name, info in CANDIDATES.items():
    print(f"\nPlotting {name}  (lag={info['lag']:.1f}d)...")
    lc_A = fetch_lc(info['oid_A'])
    lc_B = fetch_lc(info['oid_B'])

    if lc_A is None or lc_B is None:
        print(f"  Error fetching LCs for {name}")
        continue

    fid = best_band(lc_A, lc_B)
    band_name = BAND_LABELS.get(fid, str(fid))
    col_A, col_B = BAND_COLORS.get(fid, ("blue", "red"))

    sub_A = lc_A[lc_A['fid'] == fid].copy()
    sub_B = lc_B[lc_B['fid'] == fid].copy()
    print(f"  Band: {band_name}  |  A={len(sub_A)} pts  B={len(sub_B)} pts")

    if len(sub_A) < 2 or len(sub_B) < 2:
        # Fall back to all bands
        sub_A = lc_A.copy()
        sub_B = lc_B.copy()
        band_name = "all"
        col_A, col_B = "royalblue", "crimson"
        print(f"  Falling back to all bands: A={len(sub_A)} B={len(sub_B)}")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.errorbar(sub_A['mjd'], sub_A['mag'], yerr=sub_A['err'],
                fmt='o', color=col_A, alpha=0.65, markersize=5,
                label=f'Image A ({info["oid_A"]})')

    shifted_mjd = sub_B['mjd'] - info['lag']
    ax.errorbar(shifted_mjd, sub_B['mag'], yerr=sub_B['err'],
                fmt='s', color=col_B, alpha=0.65, markersize=5,
                label=f'Image B −{info["lag"]:.1f}d ({info["oid_B"]})')

    # Shade overlap window
    t0 = max(sub_A['mjd'].min(), shifted_mjd.min())
    t1 = min(sub_A['mjd'].max(), shifted_mjd.max())
    if t1 > t0:
        ax.axvspan(t0, t1, alpha=0.07, color='green', zorder=0)
        overlap_yr = (t1 - t0) / 365.25
        ax.text((t0 + t1) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] < ax.get_ylim()[1]
                else ax.get_ylim()[1],
                f"overlap\n{overlap_yr:.1f} yr", ha='center', fontsize=8,
                color='green', alpha=0.7)

    ax.invert_yaxis()
    ax.set_title(f"Time-Echo Alignment: {name}  "
                 f"(Lag: {info['lag']:.1f}d, {band_name}-band)", fontsize=13)
    ax.set_xlabel("Modified Julian Date (MJD)")
    ax.set_ylabel(f"{band_name}-band Magnitude")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    save_path = OUT_DIR / f"{name}_alignment.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.close()

print("\nDone.")
