"""
plot_time_echo_v2.py — Time-echo alignment plots, lag sourced from survey DB.
Falls back to hardcoded values if DB not available.
"""
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from pathlib import Path

DB_PATH = Path("outputs/survey_v3/survey_results.db")

# Hardcoded fallback (correct values)
CANDIDATES = {
    "SDSS1029+2623": {"oid_A": "ZTF18aceypuf", "oid_B": "ZTF19aailazi", "lag": 887.5},
    "SDSS1004+4112": {"oid_A": "ZTF19aaplmpl", "oid_B": "ZTF19aailogj", "lag": 887.5},
}

# Override lags from DB if available
if DB_PATH.exists():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT anchor, lag_days FROM candidates ORDER BY score DESC"
        ).fetchall()
        conn.close()
        for anchor, lag_days in rows:
            if anchor in CANDIDATES:
                CANDIDATES[anchor]["lag"] = lag_days
                print(f"  DB lag for {anchor}: {lag_days:.1f}d")
    except Exception as e:
        print(f"  DB read failed ({e}), using hardcoded lags")
else:
    print("  DB not found, using hardcoded lags")

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_lc(oid):
    url = f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    df = pd.DataFrame(r.json()['detections'])
    df = df[df['fid'] == 1][['mjd', 'magpsf', 'sigmapsf']]
    df = df.rename(columns={'magpsf': 'mag', 'sigmapsf': 'err'})
    return df.sort_values('mjd').reset_index(drop=True)

for name, info in CANDIDATES.items():
    print(f"Plotting {name}  (lag={info['lag']:.1f}d)...")
    lc_A = fetch_lc(info['oid_A'])
    lc_B = fetch_lc(info['oid_B'])

    if lc_A is None or lc_B is None:
        print(f"  Error fetching LCs for {name}")
        continue

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.errorbar(lc_A['mjd'], lc_A['mag'], yerr=lc_A['err'],
                fmt='o', color='royalblue', alpha=0.6, markersize=5,
                label=f'Image A ({info["oid_A"]})')

    ax.errorbar(lc_B['mjd'] - info['lag'], lc_B['mag'], yerr=lc_B['err'],
                fmt='s', color='crimson', alpha=0.6, markersize=5,
                label=f'Image B Shifted −{info["lag"]:.1f}d ({info["oid_B"]})')

    ax.invert_yaxis()
    ax.set_title(f"Time-Echo Alignment: {name}  (Lag: {info['lag']:.1f} days)",
                 fontsize=13)
    ax.set_xlabel("Modified Julian Date (MJD)")
    ax.set_ylabel("g-band Magnitude")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Annotate overlap region
    t_overlap_start = max(lc_A['mjd'].min(), (lc_B['mjd'] - info['lag']).min())
    t_overlap_end   = min(lc_A['mjd'].max(), (lc_B['mjd'] - info['lag']).max())
    if t_overlap_end > t_overlap_start:
        ax.axvspan(t_overlap_start, t_overlap_end, alpha=0.06,
                   color='green', label='Overlap window')
        ax.legend(fontsize=10)

    save_path = OUT_DIR / f"{name}_alignment.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.close()

print("Done.")
