"""
step3_crossmatch.py
───────────────────
Cross-matches your filtered candidates against SIMBAD
(the master catalog of known astronomical objects).

Objects with NO match are your most interesting candidates —
they may be genuinely new or poorly studied sources.

Run:
    python step3_crossmatch.py
"""

import pandas as pd
import time
from pathlib import Path
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

INPUT_FILE  = Path("data/alerts_filtered.csv")
OUTPUT_FILE = Path("data/alerts_crossmatched.csv")

# Search radius in arcseconds (5" is standard for point sources)
SEARCH_RADIUS_ARCSEC = 5.0

# Only crossmatch top N candidates (saves time/API calls)
TOP_N = 50

# ALeRCE column names for coordinates
COL_RA  = "meanra"
COL_DEC = "meandec"
COL_ID  = "oid"


def crossmatch_simbad(ra: float, dec: float, radius_arcsec: float = 5.0):
    """
    Query SIMBAD for known objects within radius of (ra, dec).
    Returns (matched: bool, object_type: str, object_id: str)
    """
    try:
        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        result = Simbad.query_region(coord, radius=radius_arcsec * u.arcsec)

        if result is None or len(result) == 0:
            return False, "—", "NO KNOWN COUNTERPART ⭐"

        # Return the closest/first match
        obj_id   = str(result["MAIN_ID"][0])
        obj_type = str(result["OTYPE"][0]) if "OTYPE" in result.colnames else "unknown"
        return True, obj_type, obj_id

    except Exception as e:
        return None, "error", str(e)


def run_crossmatch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().head(TOP_N)
    n = len(df)

    matched_list   = []
    otype_list     = []
    simbad_id_list = []

    ra_col  = next((c for c in [COL_RA,  "ra",  "meanra"]  if c in df.columns), None)
    dec_col = next((c for c in [COL_DEC, "dec", "meandec"] if c in df.columns), None)

    if not ra_col or not dec_col:
        print(f"  ✗ Could not find RA/Dec columns. Available: {list(df.columns)}")
        return df

    print(f"  Cross-matching {n} candidates against SIMBAD")
    print(f"  Search radius: {SEARCH_RADIUS_ARCSEC}\"")
    print(f"  (This may take ~{n * 1.2:.0f} seconds due to API rate limits)\n")

    for i, (_, row) in enumerate(df.iterrows()):
        ra  = row[ra_col]
        dec = row[dec_col]
        obj_id = row.get(COL_ID, f"row_{i}")

        matched, otype, simbad_id = crossmatch_simbad(ra, dec, SEARCH_RADIUS_ARCSEC)

        matched_list.append(matched)
        otype_list.append(otype)
        simbad_id_list.append(simbad_id)

        status = "✓ KNOWN" if matched else "★ NEW  "
        print(f"  [{i+1:>3}/{n}] {str(obj_id):<20}  {status}  {simbad_id[:40]}")

        # Be polite to the SIMBAD server
        time.sleep(0.5)

    df["simbad_matched"] = matched_list
    df["simbad_otype"]   = otype_list
    df["simbad_id"]      = simbad_id_list

    return df


if __name__ == "__main__":
    print("\n🔭 Rubin Discovery Toolkit — Step 3: Cross-match")
    print("─" * 55)

    if not INPUT_FILE.exists():
        print(f"✗ Input file not found: {INPUT_FILE}")
        print("  Run step2_filter.py first.")
        exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} filtered candidates\n")

    df_xm = run_crossmatch(df)

    df_xm.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved cross-matched results to {OUTPUT_FILE}")

    # Highlight unknowns
    unknowns = df_xm[df_xm["simbad_matched"] == False]
    print(f"\n{'═'*55}")
    print(f"  ★  {len(unknowns)} candidates have NO known SIMBAD counterpart")
    print(f"     These are your highest-priority discovery targets!")
    print(f"{'═'*55}")

    if not unknowns.empty:
        id_col = COL_ID if COL_ID in unknowns.columns else unknowns.columns[0]
        print("\nUnknown candidates:")
        show_cols = [c for c in [id_col, "classxf", "discovery_score", "delta_mag"] if c in unknowns.columns]
        print(unknowns[show_cols].head(20).to_string(index=False))

    print("\n→ Next: run  python step4_ml_anomaly.py")
