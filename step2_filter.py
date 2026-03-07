"""
step2_filter.py
───────────────
Applies science filters to narrow ~500 raw alerts down to
your best candidates. Think of each filter as one stage
in a DP (dynamic programming) pipeline — each stage
"memoizes" only the events worth carrying forward.

Run:
    python step2_filter.py
"""

import pandas as pd
from pathlib import Path

INPUT_FILE  = Path("data/alerts_raw.csv")
OUTPUT_FILE = Path("data/alerts_filtered.csv")

# ── Tune these thresholds for your science goal ────────────────────────────────
FILTERS = {
    # Signal-to-noise: removes noise spikes
    "min_detections":  3,       # at least N observations

    # Brightness range (magnitudes — lower = brighter)
    "mag_min":        14.0,     # don't saturate the detector
    "mag_max":        22.5,     # don't go too faint

    # Variability: how much has the source changed?
    "min_delta_mag":   0.10,    # ignore static / barely-varying sources

    # Recency: only consider alerts updated in the last N days
    # Set to None to disable
    "max_age_days":   None,
}

# Column name mapping from ALeRCE API
# (adjust if your data has different column names)
COL_NDET     = "ndet"
COL_MAGMEAN  = "magmean"
COL_MAGMIN   = "magmin"
COL_MAGMAX   = "magmax"
COL_LASTMJD  = "lastmjd"


def filter_alerts(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)
    steps = []

    # Stage 1 — Minimum detections
    if COL_NDET in df.columns:
        df = df[df[COL_NDET] >= FILTERS["min_detections"]]
        steps.append(f"  After ≥{FILTERS['min_detections']} detections:  {len(df):>5} alerts")

    # Stage 2 — Magnitude range
    mag_col = next((c for c in [COL_MAGMEAN, COL_MAGMIN, "magpsf"] if c in df.columns), None)
    if mag_col:
        df = df[
            (df[mag_col] >= FILTERS["mag_min"]) &
            (df[mag_col] <= FILTERS["mag_max"])
        ]
        steps.append(f"  After magnitude cut:          {len(df):>5} alerts")

    # Stage 3 — Variability (delta magnitude)
    if COL_MAGMIN in df.columns and COL_MAGMAX in df.columns:
        df = df.copy()
        df["delta_mag"] = df[COL_MAGMAX] - df[COL_MAGMIN]
        df = df[df["delta_mag"] >= FILTERS["min_delta_mag"]]
        steps.append(f"  After variability cut:        {len(df):>5} alerts")
    else:
        df["delta_mag"] = 0.0

    # Stage 4 — Recency filter
    if FILTERS["max_age_days"] and COL_LASTMJD in df.columns:
        # MJD: Modified Julian Date — roughly 1 day per unit
        recent_mjd = df[COL_LASTMJD].max() - FILTERS["max_age_days"]
        df = df[df[COL_LASTMJD] >= recent_mjd]
        steps.append(f"  After recency cut:            {len(df):>5} alerts")

    return df, original, steps


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple scoring heuristic to rank candidates.
    Higher score = more interesting / anomalous.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    # Reward large variability
    if "delta_mag" in df.columns:
        score += (df["delta_mag"] / df["delta_mag"].max()) * 40

    # Reward high detection count (well-observed)
    if COL_NDET in df.columns:
        score += (df[COL_NDET] / df[COL_NDET].max()) * 30

    # Reward recent activity
    if COL_LASTMJD in df.columns:
        score += ((df[COL_LASTMJD] - df[COL_LASTMJD].min()) /
                  (df[COL_LASTMJD].max() - df[COL_LASTMJD].min() + 1e-9)) * 30

    df["discovery_score"] = score.round(1)
    return df.sort_values("discovery_score", ascending=False)


if __name__ == "__main__":
    print("\n🔭 Rubin Discovery Toolkit — Step 2: Filter Alerts")
    print("─" * 55)

    if not INPUT_FILE.exists():
        print(f"✗ Input file not found: {INPUT_FILE}")
        print("  Run step1_fetch_alerts.py first.")
        exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} raw alerts from {INPUT_FILE}\n")

    df_filtered, original, steps = filter_alerts(df)

    print("Filter pipeline:")
    print(f"  Starting alerts:              {original:>5}")
    for s in steps:
        print(s)

    reduction = (1 - len(df_filtered) / original) * 100 if original else 0
    print(f"\n  Reduction: {reduction:.1f}%  →  {len(df_filtered)} candidates remain\n")

    if df_filtered.empty:
        print("⚠  No alerts passed filters. Try relaxing the thresholds in FILTERS dict.")
        exit(0)

    # Score and rank
    df_scored = score_candidates(df_filtered)

    df_scored.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved {len(df_scored)} scored candidates to {OUTPUT_FILE}")

    print("\nTop 10 candidates by discovery score:")
    display_cols = [c for c in ["oid", "classxf", "discovery_score", "delta_mag", COL_NDET, COL_LASTMJD] if c in df_scored.columns]
    print(df_scored[display_cols].head(10).to_string(index=False))

    print("\n→ Next: run  python step3_crossmatch.py")
