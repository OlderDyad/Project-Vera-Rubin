"""
status_check.py v2.0
────────────────────
Calculates live ETA and now reports the number of high-priority 
'Goldilocks' candidates identified so far.
"""

import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROGRESS_FILE = Path("outputs/hunter/progress.txt")
GOLDILOCKS_FILE = Path("outputs/hunter/goldilocks_candidates.csv")
TOTAL_TARGETS = 100000

def check_eta():
    print("\n⏱️ LENS HUNTER LIVE DASHBOARD")
    print("═" * 45)

    if not PROGRESS_FILE.exists():
        print("⚠ Cannot find progress.txt. Has the Hunter started?")
        return

    # Calculate Speed
    with open(PROGRESS_FILE, "r") as f:
        start_idx = int(f.read().strip())
    
    print("Calculating live velocity (sampling 10s of data)...")
    time.sleep(10)
    
    with open(PROGRESS_FILE, "r") as f:
        end_idx = int(f.read().strip())

    # Goldilocks count
    g_count = 0
    if GOLDILOCKS_FILE.exists():
        try:
            g_df = pd.read_csv(GOLDILOCKS_FILE)
            g_count = len(g_df)
        except: pass

    # Velocity Math
    anchors_processed = end_idx - start_idx
    speed_per_sec = anchors_processed / 10.0
    sec_per_anchor = 1.0 / speed_per_sec if speed_per_sec > 0 else 0
    
    remaining_targets = TOTAL_TARGETS - end_idx
    remaining_seconds = remaining_targets * sec_per_anchor
    
    eta_delta = timedelta(seconds=int(remaining_seconds))
    completion_time = datetime.now() + eta_delta

    # Progress Bar (Visual)
    pct = (end_idx / TOTAL_TARGETS)
    bar = "█" * int(pct * 20) + "-" * (20 - int(pct * 20))

    # Print Report
    print(f"\nProgress: |{bar}| {pct*100:.2f}%")
    print(f"Position: {end_idx:,} / {TOTAL_TARGETS:,}")
    print(f"Speed:    {speed_per_sec:.2f} anchors/sec ({sec_per_anchor:.2f}s per)")
    
    print("\n🎯 DISCOVERY STATS")
    print(f"Goldilocks Candidates (1.5\"-10\"): {g_count}")
    
    print("\n🏁 PROJECTION")
    print(f"Time Remaining: {eta_delta}")
    print(f"Estimated Finish: {completion_time.strftime('%A at %I:%M %p')}")
    print("═" * 45)

if __name__ == "__main__":
    check_eta()