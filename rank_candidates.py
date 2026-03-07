"""
rank_candidates.py
──────────────────
Scans the massive candidates.csv output from lens_hunter.py,
filters for high-confidence passes, and isolates the top 1% 
highest-scoring pairs for manual follow-up.
"""

import pandas as pd
from pathlib import Path

RESULTS_FILE = Path("outputs/hunter/candidates.csv")
TOP_RESULTS_FILE = Path("outputs/hunter/top_1_percent_candidates.csv")

def analyze_results(min_rungs_passed=4):
    if not RESULTS_FILE.exists():
        print(f"⚠ Cannot find {RESULTS_FILE}. Is the Hunter still running?")
        return

    print(f"\n📊 Analyzing Lens Hunter Results")
    print("═" * 60)
    
    # Load the massive dataset
    df = pd.read_csv(RESULTS_FILE)
    total_evaluated = len(df)
    print(f"Total pairs evaluated: {total_evaluated}")

    if total_evaluated == 0:
        print("No candidates to analyze yet.")
        return

    # Filter 1: Must pass a strict number of rungs (physics check)
    df_filtered = df[df["rungs_passed"] >= min_rungs_passed]
    print(f"Pairs passing >={min_rungs_passed} rungs: {len(df_filtered)}")

    # Filter 2: Calculate the Top 1% threshold
    top_1_percent_count = max(1, int(total_evaluated * 0.01))
    
    # Sort by total score descending, and grab the top 1%
    top_candidates = df_filtered.sort_values("total_score", ascending=False).head(top_1_percent_count)
    
    print("\n🏆 Top 1% Candidates:")
    print("─" * 60)
    
    # Print a clean, readable table to the terminal
    cols_to_show = ["pair_id", "anchor", "sep_arcsec", "best_lag_days", "rungs_passed", "total_score"]
    
    # Check if the dataframe has these columns to avoid KeyError
    available_cols = [c for c in cols_to_show if c in top_candidates.columns]
    
    if not top_candidates.empty:
        print(top_candidates[available_cols].to_string(index=False))
        
        # Save this elite list to its own file for easy access
        top_candidates.to_csv(TOP_RESULTS_FILE, index=False)
        print("\n" + "═" * 60)
        print(f"✓ Saved elite candidates to {TOP_RESULTS_FILE}")
    else:
        print("No candidates met the minimum rung criteria.")

if __name__ == "__main__":
    # You can change the strictness here. 4 is a good baseline for physics confidence.
    analyze_results(min_rungs_passed=4)