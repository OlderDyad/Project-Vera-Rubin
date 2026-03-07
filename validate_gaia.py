import pandas as pd
from astroquery.utils.tap.core import TapPlus
from pathlib import Path
import time

INPUT_FILE = Path("outputs/hunter/candidates.csv")
VALIDATED_FILE = Path("outputs/hunter/validated_candidates.csv")

def check_gaia_motion():
    if not INPUT_FILE.exists():
        print(f"❌ Could not find {INPUT_FILE}")
        return
        
    df = pd.read_csv(INPUT_FILE)
    results = []

    # Initialize the TAP service directly to avoid 'GaiaClass' attribute errors
    gaia_tap = TapPlus(url="https://gea.esac.esa.int/tap-server/tap")

    print(f"🛰️ Gaia Validation on {len(df)} candidates...")

    for _, row in df.iterrows():
        ra, dec = row['ra'], row['dec']
        
        # SQL query for Gaia DR3
        query = f"SELECT TOP 5 pmra, pmdec, parallax FROM gaiadr3.gaia_source WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.00027)) = 1"
        
        try:
            # Launch the job via the TAP service
            job = gaia_tap.launch_query_async(query)
            r = job.get_results()
            
            row_dict = row.to_dict()
            if len(r) > 0:
                # If values are masked (null), default to 0.0
                p = r['parallax'][0]
                row_dict['gaia_parallax'] = float(p) if p is not None else 0.0
                row_dict['gaia_status'] = "Match Found"
            else:
                row_dict['gaia_parallax'] = 0.0
                row_dict['gaia_status'] = "No Gaia Source (Likely Deep Space)"

            results.append(row_dict)
            print(f"  ✓ {row['anchor']}: Parallax={row_dict['gaia_parallax']:.3f}")
            
            # This 5-second delay is mandatory to avoid the 'Heavy Query Shower' ban
            time.sleep(5) 
            
        except Exception as e:
            print(f"  ⚠ Gaia Error for {row['anchor']}: {e}")

    pd.DataFrame(results).to_csv(VALIDATED_FILE, index=False)
    print(f"\n✅ Validation complete. Saved to {VALIDATED_FILE}")

if __name__ == "__main__":
    check_gaia_motion()