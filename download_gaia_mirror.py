import pandas as pd
from astroquery.utils.tap.core import TapPlus
from pathlib import Path

# Config
SEED_FILES = ["seeds/quasar_seeds.csv", "seeds/agn_seeds.csv"]
MIRROR_DIR = Path("data/gaia_mirror/")
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

def fetch_local_gaia_blocks():
    gaia_tap = TapPlus(url="https://gea.esac.esa.int/tap-server/tap")
    
    for seed_file in SEED_FILES:
        if not Path(seed_file).exists(): continue
        df = pd.read_csv(seed_file)
        
        print(f"🌍 Downloading Gaia blocks for {len(df)} seeds in {seed_file}...")
        
        # We process in batches of 1000 to keep the ESA server happy
        for i in range(0, len(df), 1000):
            batch = df.iloc[i:i+1000]
            # Custom ADQL to get only essential motion data for your specific sky-patch
            # This is the "Rubin-Ready" way to pre-vet your entire seed list
            query = f"""
            SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), 
                               CIRCLE('ICRS', {batch['ra'].mean()}, {batch['dec'].mean()}, 2.0))
            """
            job = gaia_tap.launch_query_async(query)
            results = job.get_results().to_pandas()
            
            # Save locally on your E: drive
            results.to_parquet(MIRROR_DIR / f"gaia_block_{i}.parquet")
            print(f"  ✓ Saved block {i} ({len(results)} stars)")

if __name__ == "__main__":
    fetch_local_gaia_blocks()