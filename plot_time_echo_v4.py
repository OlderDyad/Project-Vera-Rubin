# Save as plot_time_echo_v5.py

import matplotlib.pyplot as plt
import pandas as pd
import requests
from pathlib import Path

# Updated OIDs for SDSS1004 to use more common primary IDs
CANDIDATES = {
    "SDSS1029+2623": {"oid_A": "ZTF18aceypuf", "oid_B": "ZTF19aailazi", "lag": 887.5},
    "SDSS1004+4112": {"oid_A": "ZTF18abvmpiv", "oid_B": "ZTF19aaplmpl", "lag": 40.8}, 
}

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_lc(oid):
    url = f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return None
        dets = r.json().get('detections', [])
        if not dets: return None
        df = pd.DataFrame(dets)
        return df[['mjd', 'magpsf', 'sigmapsf', 'fid']].rename(columns={'magpsf': 'mag', 'sigmapsf': 'err'})
    except: return None

for name, info in CANDIDATES.items():
    print(f"\nChecking data for {name}...")
    lc_A = fetch_lc(info['oid_A'])
    lc_B = fetch_lc(info['oid_B'])
    
    if lc_A is None or lc_B is None:
        print(f"  [SKIP] Could not fetch data for {name}")
        continue

    # Find band with most data
    best_fid = 0
    max_pts = 0
    for fid in [1, 2]:
        pts = len(lc_A[lc_A['fid'] == fid]) + len(lc_B[lc_B['fid'] == fid])
        if pts > max_pts:
            max_pts = pts
            best_fid = fid

    sub_A = lc_A[lc_A['fid'] == best_fid]
    sub_B = lc_B[lc_B['fid'] == best_fid]

    if len(sub_A) < 10 or len(sub_B) < 10:
        print(f"  [SKIP] Not enough overlapping points (A:{len(sub_A)}, B:{len(sub_B)})")
        continue

    print(f"  [PLOTTING] Band {best_fid}: A={len(sub_A)}, B={len(sub_B)}")

    plt.figure(figsize=(12, 6))
    plt.errorbar(sub_A['mjd'], sub_A['mag'], yerr=sub_A['err'], fmt='o', label='Image A', alpha=0.5)
    plt.errorbar(sub_B['mjd'] - info['lag'], sub_B['mag'], yerr=sub_B['err'], fmt='s', label='Image B Shifted', alpha=0.5)

    plt.gca().invert_yaxis()
    plt.xlim(sub_A['mjd'].min() - 50, sub_A['mjd'].max() + 50)
    plt.title(f"Time-Echo: {name} (Lag: {info['lag']}d)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / f"{name}_V5.png")
    plt.close()

print("\nDone.")