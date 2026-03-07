"""
fetch_quasar_seeds.py
─────────────────────
Downloads a large catalog of known quasars from the Million Quasars (Milliquas)
catalog via the CDS VizieR database. Filters out southern targets that ZTF 
cannot see, formatting the output perfectly for lens_hunter.py.
"""

import requests
import pandas as pd
from pathlib import Path
import time

SEEDS_DIR = Path("seeds")
SEEDS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = SEEDS_DIR / "quasar_seeds.csv"

def fetch_milliquas(max_records=100000):
    print(f"\n📡 Querying CDS VizieR for up to {max_records} known quasars...")
    print(f"   Catalog: Million Quasars (Milliquas) v7.2+ [VII/290]")
    print(f"   Filter: Declination > -28° (ZTF Northern Hemisphere visibility)")
    
    # VizieR Simple Cone Search (ASU) API
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    params = {
        "-source": "VII/290/catalog",
        "-out.max": str(max_records),
        "-out": "Name,RAJ2000,DEJ2000,z",
        "DEJ2000": ">-28",      # ZTF footprint
        "-sort": "RAJ2000"     # Sequential sweep across the sky
    }

    start_time = time.time()
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠ API Error: {e}")
        return

    print(f"   Download complete in {time.time() - start_time:.1f} seconds. Parsing data...")

    # VizieR TSV output includes metadata comments (#) and a separator row (---)
    # We strip those out to create a clean DataFrame.
    lines = [line for line in response.text.splitlines() if not line.startswith("#") and line.strip()]
    
    if len(lines) < 3:
        print("⚠ Not enough data returned. VizieR might be rate-limiting you.")
        return

    # Extract headers and split by tabs
    header = lines[0].split('\t')
    data = [line.split('\t') for line in lines[2:]]  # Skip the '---' line
    
    df = pd.DataFrame(data, columns=header)

    # Rename columns to match what lens_hunter.py is expecting
    df = df.rename(columns={
        "Name": "name", 
        "RAJ2000": "ra", 
        "DEJ2000": "dec", 
        "z": "redshift"
    })
    
    # Clean up column types and drop any corrupted rows
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    df = df.dropna(subset=["ra", "dec"])

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "═"*60)
    print(f"✓ Success: Saved {len(df)} quasar anchors to {OUTPUT_FILE}")
    print("═"*60)
    print("\nYou can now run the hunter:")
    print("  python lens_hunter.py --strategy anchored --policy uncertain --radius 60")

if __name__ == "__main__":
    fetch_milliquas(max_records=100000)