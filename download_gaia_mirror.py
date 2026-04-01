import pandas as pd
from astroquery.utils.tap.core import TapPlus
from pathlib import Path
import time

# Config
# Update this if your new 250,000 targets have a different filename
SEED_FILES = ["seeds/quasar_seeds.csv", "seeds/agn_seeds.csv"]
MIRROR_DIR = Path("data/gaia_mirror/")
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

def fetch_local_gaia_blocks():
    # 🩹 FIX 1: Rerouted to the Heidelberg University mirror to bypass ESA 503 outages
    gaia_tap = TapPlus(url="https://gaia.ari.uni-heidelberg.de/tap")
    
    for seed_file in SEED_FILES:
        seed_path = Path(seed_file)
        if not seed_path.exists(): continue
        df = pd.read_csv(seed_path)
        
        # 🩹 FIX 2: Sort by RA so the 1000 seeds are physically next to each other in the sky
        df = df.sort_values('ra').reset_index(drop=True)
        
        print(f"🌍 Downloading Gaia blocks for {len(df)} seeds in {seed_path.name}...")
        
        # Process in batches of 1000
        for i in range(0, len(df), 1000):
            batch = df.iloc[i:i+1000]
            
            # Now the mean RA/Dec accurately points to the center of the sorted cluster
            query = f"""
            SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), 
                               CIRCLE('ICRS', {batch['ra'].mean()}, {batch['dec'].mean()}, 2.0))
            """
            
            try:
                job = gaia_tap.launch_job_async(query)
                results = job.get_results().to_pandas()
                
                # Save locally on your E: drive with a clean filename
                out_file = MIRROR_DIR / f"gaia_block_{seed_path.stem}_{i}.parquet"
                results.to_parquet(out_file)
                print(f"  ✓ Saved block {i} ({len(results)} stars)")
                
            except Exception as e:
                print(f"  ❌ Error on block {i}: {e}")
                time.sleep(5) # Give the German server a 5-second breather if it hiccups

if __name__ == "__main__":
    fetch_local_gaia_blocks()