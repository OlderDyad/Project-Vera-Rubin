"""
step1_fetch_alerts.py
─────────────────────
Fetches real astronomical alerts from the ALeRCE broker.
ALeRCE ingests the ZTF stream (same format as Rubin/LSST).
No account or data rights needed — it's fully public.

Run:
    python step1_fetch_alerts.py
"""

import requests
import pandas as pd
import json
from pathlib import Path

OUTPUT_FILE = Path("data/alerts_raw.csv")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
ALERCE_API = "https://api.alerce.online/alerts/v1"

QUERY_PARAMS = {
    "page":      1,
    "page_size": 500,       # fetch 500 alerts per call (max 1000)
    "order_by":  "lastmjd", # most recently updated first
    "order_mode":"DESC",
}

# Optional: filter to a specific classifier class
# Options: "SN", "AGN", "VS" (variable star), "asteroid", etc.
# Remove the key entirely to get all classes
CLASSIFIER_FILTER = None   # e.g. set to "SN" to hunt supernovae only


def fetch_alerts(class_filter=None, max_pages=3):
    """Fetch alerts from ALeRCE REST API, returns a DataFrame."""
    all_alerts = []
    params = QUERY_PARAMS.copy()

    if class_filter:
        params["classxf"] = class_filter
        print(f"  Filtering to class: {class_filter}")

    for page in range(1, max_pages + 1):
        params["page"] = page
        print(f"  Fetching page {page}/{max_pages}...")

        try:
            resp = requests.get(
                f"{ALERCE_API}/query",
                params=params,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if not items:
                print("  No more alerts returned.")
                break
            all_alerts.extend(items)
            print(f"  Got {len(items)} alerts (total so far: {len(all_alerts)})")

        except requests.exceptions.ConnectionError:
            print("\n  ⚠  Could not reach ALeRCE API.")
            print("     Make sure you have internet access, then re-run.")
            return None
        except Exception as e:
            print(f"  ⚠  Error on page {page}: {e}")
            break

    if not all_alerts:
        return None

    df = pd.DataFrame(all_alerts)
    print(f"\n✓ Fetched {len(df)} alerts total")
    print(f"  Columns: {list(df.columns)}\n")
    return df


def summarize(df):
    """Print a quick summary of what we got."""
    print("═" * 55)
    print("ALERT SUMMARY")
    print("═" * 55)

    if "classxf" in df.columns:
        print("\nTop classifications:")
        print(df["classxf"].value_counts().head(10).to_string())

    if "meanra" in df.columns:
        print(f"\nRA range:   {df['meanra'].min():.2f}° — {df['meanra'].max():.2f}°")
    if "meandec" in df.columns:
        print(f"Dec range:  {df['meandec'].min():.2f}° — {df['meandec'].max():.2f}°")

    if "lastmjd" in df.columns:
        print(f"Most recent MJD: {df['lastmjd'].max():.2f}")

    print("\nFirst 5 alert IDs:")
    id_col = "oid" if "oid" in df.columns else df.columns[0]
    print(df[id_col].head().to_string())
    print("═" * 55)


if __name__ == "__main__":
    print("\n🔭 Rubin Discovery Toolkit — Step 1: Fetch Alerts")
    print("─" * 55)

    df = fetch_alerts(class_filter=CLASSIFIER_FILTER, max_pages=3)

    if df is not None:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Saved to {OUTPUT_FILE}")
        summarize(df)
        print("\n→ Next: run  python step2_filter.py")
    else:
        print("\n✗ No data retrieved. Check your internet connection.")
