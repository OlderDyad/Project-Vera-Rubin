"""
mass_modeler.py
───────────────
Calculates the estimated mass of the lensing galaxy in Solar Masses (M_sun)
using the observed separation and time delay.
"""
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("outputs/hunter/validated_candidates.csv")
OUTPUT_FILE = Path("outputs/hunter/physics_vetted_candidates.csv")

# Physical Constants
C = 299792458.0  # Speed of light (m/s)
G = 6.67430e-11  # Gravitational constant
M_SUN = 1.989e30 # Mass of the Sun (kg)

def estimate_lens_mass(sep_arcsec, delta_t_days):
    """
    A simplified mass estimate based on the observed separation and lag.
    """
    # Convert arcsec to radians
    theta_e = (sep_arcsec / 2.0) * (np.pi / (180.0 * 3600.0))
    
    # Convert days to seconds
    delta_t_sec = delta_t_days * 86400.0
    
    # Geometric approximation for mass within Einstein Radius
    # M = (c^3 * delta_t) / (4 * G * (1 + z_l)) 
    # We assume a standard lens redshift (z_l) of 0.5 for the sanity check
    z_l = 0.5
    mass_kg = (C**3 * delta_t_sec) / (4 * G * (1 + z_l))
    
    return mass_kg / M_SUN

def run_physics_check():
    if not INPUT_FILE.exists():
        print("⚠ Run validate_gaia.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"⚖️ Calculating Lens Masses for {len(df)} candidates...")

    df['est_mass_msun'] = df.apply(lambda x: estimate_lens_mass(x['sep_arcsec'], x['best_lag_days']), axis=1)

    # Label the result
    def classify_mass(m):
        if m < 1e10: return "Sub-Galactic / Dwarf"
        if 1e10 <= m <= 2e12: return "Galaxy-Scale (Likely)"
        return "Cluster-Scale"

    df['mass_classification'] = df['est_mass_msun'].apply(classify_mass)
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    for _, row in df.iterrows():
        print(f"  Target: {row['anchor']}")
        print(f"    - Est. Mass: {row['est_mass_msun']:.2e} M_sun")
        print(f"    - Category:  {row['mass_classification']}")

if __name__ == "__main__":
    run_physics_check()