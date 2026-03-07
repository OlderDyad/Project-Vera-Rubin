"""
analyze_colors.py
─────────────────
Queries DECaLS DR9/SDSS for g, r, and i magnitudes to calculate 
color indices (g-r, r-i) for your novel candidates.
"""
import pandas as pd
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import astropy.units as u
from pathlib import Path

INPUT_FILE = Path("outputs/hunter/physics_vetted_candidates.csv")
OUTPUT_FILE = Path("outputs/hunter/color_vetted_candidates.csv")

def get_colors():
    if not INPUT_FILE.exists(): return
    df = pd.read_csv(INPUT_FILE)
    results = []

    print(f"🌈 Quantifying colors for {len(df)} candidates...")

    for _, row in df.iterrows():
        pos = coords.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
        row_dict = row.to_dict()
        
        try:
            # Query SDSS PhotoObj for magnitudes
            result = SDSS.query_region(pos, radius=2*u.arcsec, photoobj_fields=['g','r','i'])
            
            if result:
                g, r, i = result[0]['g'], result[0]['r'], result[0]['i']
                # Calculate Indices
                row_dict['g_minus_r'] = round(g - r, 3)
                row_dict['r_minus_i'] = round(r - i, 3)
                print(f"  ✓ {row['anchor']}: g-r={row_dict['g_minus_r']}, r-i={row_dict['r_minus_i']}")
            else:
                row_dict['g_minus_r'], row_dict['r_minus_i'] = "N/A", "N/A"
        except:
            row_dict['g_minus_r'], row_dict['r_minus_i'] = "Error", "Error"

        results.append(row_dict)

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Color data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    get_colors()