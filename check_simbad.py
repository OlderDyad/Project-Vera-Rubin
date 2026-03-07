"""
check_simbad.py v2.1
────────────────────
Defensive version: handles missing Z_VALUE columns and prevents 
KeyErrors if queries fail.
"""
import pandas as pd
from pathlib import Path
import time
from astroquery.simbad import Simbad
import warnings
warnings.filterwarnings("ignore")

GOLDILOCKS_FILE = Path("outputs/hunter/goldilocks_candidates.csv")
FINAL_DISCOVERIES_FILE = Path("outputs/hunter/novel_candidates.csv")

def check_simbad_goldilocks():
    if not GOLDILOCKS_FILE.exists():
        print(f"⚠ Cannot find {GOLDILOCKS_FILE}. Run rank_goldilocks.py first.")
        return

    df = pd.read_csv(GOLDILOCKS_FILE)
    if df.empty:
        print("No candidates to check.")
        return

    print(f"\n🌍 Querying SIMBAD for {len(df)} Goldilocks Candidates...")
    print("═" * 70)

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('otype', 'otypes', 'z_value')

    results_data = []

    for _, row in df.iterrows():
        anchor = row['anchor']
        row_dict = row.to_dict()
        
        # Default values in case of failure
        row_dict['simbad_z'] = "N/A"
        row_dict['simbad_otype'] = "Unknown"
        row_dict['is_known_lens'] = False
        
        try:
            result = custom_simbad.query_object(anchor)
            
            if result is not None:
                # Safely get Redshift
                if 'Z_VALUE' in result.colnames and not result['Z_VALUE'].mask[0]:
                    row_dict['simbad_z'] = f"{result['Z_VALUE'][0]:.4f}"
                
                # Safely get Types
                if 'OTYPE' in result.colnames:
                    otype = result['OTYPE'][0]
                    otypes = result['OTYPES'][0]
                    if isinstance(otype, bytes): otype = otype.decode('utf-8')
                    if isinstance(otypes, bytes): otypes = otypes.decode('utf-8')
                    
                    row_dict['simbad_otype'] = otype
                    # Search for lens keywords
                    row_dict['is_known_lens'] = any(word in otypes for word in ['Lens', 'Grav', 'gLs'])

            status = "[KNOWN]" if row_dict['is_known_lens'] else "[NOVEL]"
            print(f"  {status} {anchor:25} | z={row_dict['simbad_z']:7} | {row_dict['simbad_otype']}")
                
        except Exception as e:
            print(f"  [⚠] {anchor}: Query failed ({e})")
        
        results_data.append(row_dict)
        time.sleep(1.2)

    # Save findings
    full_results_df = pd.DataFrame(results_data)
    if not full_results_df.empty:
        novel_df = full_results_df[full_results_df['is_known_lens'] == False]
        novel_df.to_csv(FINAL_DISCOVERIES_FILE, index=False)
        print("\n" + "═" * 70)
        print(f"✓ Processed {len(full_results_df)} candidates.")
        print(f"✓ Saved {len(novel_df)} potential discoveries to {FINAL_DISCOVERIES_FILE}")
    else:
        print("No data retrieved.")

if __name__ == "__main__":
    check_simbad_goldilocks()