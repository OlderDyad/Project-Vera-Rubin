"""
rank_goldilocks.py
──────────────────
Filters candidates.csv to find 'Goldilocks' candidates:
1. Separation between 1.5" and 10.0" (not a dupe, but physically close).
2. High physics confidence (rungs_passed >= 3).
3. Sorted by the highest total cross-correlation score.
"""

import pandas as pd
from pathlib import Path

RESULTS_FILE = Path("outputs/hunter/candidates.csv")
GOLDILOCKS_FILE = Path("outputs/hunter/goldilocks_candidates.csv")

def find_goldilocks(min_sep=1.5, max_sep=10.0, min_rungs=3):
    if not RESULTS_FILE.exists():
        print(f"⚠ Results file not found at {RESULTS_FILE}. Ensure lens_hunter.py is running.")
        return

    # Load current data from the Hunter
    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        print("No candidates evaluated yet.")
        return

    print(f"\n💎 Mining Goldilocks Candidates from {len(df):,} total results...")
    print("═" * 70)

    # Apply the Goldilocks filters
    # 1. Spatial filter: Above the jitter floor, below the cluster ceiling
    mask_sep = (df['sep_arcsec'] >= min_sep) & (df['sep_arcsec'] <= max_sep)
    
    # 2. Physics filter: At least 3 rungs passed (validating the echo)
    mask_phys = (df['rungs_passed'] >= min_rungs)
    
    goldilocks = df[mask_sep & mask_phys].copy()
    
    # Sort by total score descending
    goldilocks = goldilocks.sort_values("total_score", ascending=False)

    if not goldilocks.empty:
        print(f"✓ Found {len(goldilocks)} high-confidence candidates in the 1.5\"–10\" zone.")
        print("-" * 70)
        
        # Display the elite table
        cols = ["anchor", "sep_arcsec", "best_lag_days", "rungs_passed", "total_score"]
        print(goldilocks[cols].head(15).to_string(index=False))
        
        # Save to the file SIMBAD is waiting for
        GOLDILOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        goldilocks.to_csv(GOLDILOCKS_FILE, index=False)
        print("\n" + "═" * 70)
        print(f"Elite list saved to: {GOLDILOCKS_FILE}")
    else:
        print(f"No candidates found yet meeting the {min_rungs}-rung criteria in the Goldilocks zone.")
        print(f"Check again after the Hunter hits another 5,000 anchors.")

if __name__ == "__main__":
    find_goldilocks()