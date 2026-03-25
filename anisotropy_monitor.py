"""
anisotropy_monitor.py  v2.0
============================
Cosmological anisotropy monitor for the Rubin Gravitational Lens Pipeline.

ARCHITECTURE
============
Foundation layer (shared by all modules):
  - Coordinate transforms: (RA,Dec) → galactic (l,b), ecliptic, CMB frame
  - Sky tessellation: HEALPix-like equal-area sky bins
  - Detection tagging: every lens gets full sky metadata on discovery
  - Database schema: anisotropy_detections table in survey_results.db
  - Doppler boost calculator: expected signal from our CMB motion
  - Statistical utilities: Poisson, Fisher, dipole estimators

Science modules (each imports foundation, adds own analysis):
  Module A: H0 directional map    — H0(l,b) heatmap, anisotropy test
  Module B: Cosmic dipole monitor — N+/N- asymmetry, variability dipole
  Module C: S8 tension probe      — strong lens count vs ΛCDM prediction

Design principle: every new lens detected by rubin_survey_v3.py should
be tagged and stored with one call to tag_and_store_detection(). All
three science modules then query the same database — no double-processing.

References:
  CMB dipole:    Planck 2020, A&A 641 A1 (Aghanim et al.)
  Kinematic dipole: Ellis & Baldwin 1984, MNRAS 206 377
  Strong lens rates: Oguri & Marshall 2010, MNRAS 405 2579
  S8 tension:    DES Y3 (Amon et al. 2022), KiDS-1000 (Heymans et al. 2021)
  H0 anisotropy: Migkas et al. 2021, A&A 649 A24
"""

import sqlite3
import numpy as np
from scipy import integrate, stats
from pathlib import Path
from datetime import datetime, timezone
import json

Path("outputs/diagnostic").mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# ── FOUNDATION: CONSTANTS ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

C_KM_S      = 2.998e5    # km/s
H0_FIDUCIAL = 70.0
OM = 0.3
OL = 0.7

# CMB dipole apex — direction of our motion (Planck 2020, Table 3)
CMB_APEX = {
    "ra":       167.99,    # degrees (J2000)
    "dec":       -6.99,    # degrees
    "l":        264.021,   # galactic longitude
    "b":         48.253,   # galactic latitude
    "v_kms":    369.82,    # our velocity toward apex (km/s)
    "beta":     369.82 / 2.998e5,   # v/c
}

# Typical AGN spectral index (S_nu ∝ nu^-alpha)
AGN_SPECTRAL_INDEX = 0.5

# Strong lens rate parameters (Oguri & Marshall 2010, Rubin depth)
QUAD_RATE_PER_SQ_DEG = 0.004   # expected quads/deg² at Rubin depth

# S8 reference values
S8_PLANCK    = 0.832   # CMB early-universe prediction (Planck 2020)
S8_WEAK_LENS = 0.770   # DES/KiDS weak lensing (lower = less clumpy)

# Sky areas
ZTF_SKY_DEG2   = 15000.0
RUBIN_SKY_DEG2 = 18000.0

# Number of sky bins for the H0 heatmap
N_SKY_BINS_L = 12   # bins in galactic longitude (30° each)
N_SKY_BINS_B =  6   # bins in galactic latitude  (30° each)

# ══════════════════════════════════════════════════════════════════════
# ── FOUNDATION: COORDINATE TRANSFORMS ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def eq_to_galactic(ra_deg, dec_deg):
    """
    Equatorial (RA, Dec) J2000 → Galactic (l, b).
    Uses IAU standard NGP at RA=192.8595°, Dec=+27.1284° (J2000).
    Returns (l_deg, b_deg).
    """
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ra_ngp  = np.radians(192.8595)
    dec_ngp = np.radians(27.1284)
    l_cp    = np.radians(122.9320)

    sin_b = (np.sin(dec_ngp)*np.sin(dec) +
             np.cos(dec_ngp)*np.cos(dec)*np.cos(ra - ra_ngp))
    b = np.arcsin(np.clip(sin_b, -1, 1))

    cos_b = np.cos(b)
    if abs(cos_b) < 1e-10:
        l = 0.0
    else:
        sl = np.cos(dec) * np.sin(ra - ra_ngp) / cos_b
        cl = (np.sin(dec) - np.sin(dec_ngp)*np.sin(b)) / (np.cos(dec_ngp)*cos_b)
        l  = l_cp - np.arctan2(sl, cl)

    return round(np.degrees(l) % 360, 4), round(np.degrees(b), 4)

def angular_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle angular separation in degrees."""
    r1,d1,r2,d2 = map(np.radians, [ra1,dec1,ra2,dec2])
    cos_s = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
    return np.degrees(np.arccos(np.clip(cos_s, -1, 1)))

def sky_bin(l_deg, b_deg, n_bands=12):
    """
    Equal-area sky tessellation (sinusoidal projection).
    Each band has equal solid angle. Bands near the poles have fewer
    longitude pixels; bands near the equator have more.
    Total: ~360 pixels at n_bands=12, each ~115 deg².

    Returns (pixel_id, i_band, i_lon, n_lon_in_band, b_center_deg)
    pixel_id is unique and stable across runs.
    """
    # Equal-area latitude bands: divide sin(b) uniformly
    sin_b  = np.sin(np.radians(np.clip(b_deg, -89.99, 89.99)))
    i_band = int((sin_b + 1) / 2 * n_bands)
    i_band = max(0, min(n_bands - 1, i_band))

    # Band center latitude
    b_ctr  = np.degrees(np.arcsin(-1 + (i_band + 0.5) * 2 / n_bands))

    # Number of longitude bins in this band (proportional to cos(b_ctr))
    n_lon  = max(1, int(round(n_bands * np.cos(np.radians(b_ctr)) * np.pi)))

    # Longitude bin within band
    i_lon  = int(l_deg / 360 * n_lon) % n_lon

    # Stable unique pixel ID
    pixel_id = i_band * 1000 + i_lon

    return pixel_id, i_band, i_lon, n_lon, round(b_ctr, 2)

# ══════════════════════════════════════════════════════════════════════
# ── FOUNDATION: DOPPLER BOOST ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def doppler_boost(ra_deg, dec_deg, alpha=AGN_SPECTRAL_INDEX):
    """
    Compute kinematic Doppler boost factor for a sky direction.

    For a source at angle theta from the CMB dipole apex:
      S_obs = S_rest × delta^(3+alpha)
    where delta = 1/gamma/(1 - beta*cos(theta)) ≈ 1 + beta*cos(theta)
    for beta << 1.

    Returns:
      flux_boost:      multiplicative flux enhancement (>1 toward apex)
      count_boost:     number count enhancement (Ellis & Baldwin 1984)
      variability_boost: fractional increase in variability amplitude
    """
    sep_deg    = angular_sep_deg(ra_deg, dec_deg,
                                  CMB_APEX["ra"], CMB_APEX["dec"])
    cos_theta  = np.cos(np.radians(sep_deg))
    beta       = CMB_APEX["beta"]

    # Flux boost: S_obs/S_rest ≈ 1 + (3+alpha)*beta*cos(theta)
    flux_boost = 1 + (3 + alpha) * beta * cos_theta

    # Number count boost: N+/N- (Ellis & Baldwin 1984)
    # For source count slope x: dN/dS ∝ S^{-x-1}
    # Expected kinematic dipole amplitude = (2 + alpha - x) * beta
    x_slope    = 1.0
    count_boost = 1 + (2 + alpha - x_slope) * beta * cos_theta

    # Variability boost: time dilation causes faster apparent variability
    # fractional change ≈ (3+alpha)*beta*cos(theta)
    variability_boost = (3 + alpha) * beta * cos_theta

    return {
        "cos_theta":         round(cos_theta, 5),
        "flux_boost":        round(flux_boost, 6),
        "count_boost":       round(count_boost, 6),
        "variability_boost": round(variability_boost, 6),
    }

# ══════════════════════════════════════════════════════════════════════
# ── FOUNDATION: COSMOLOGICAL DISTANCES ───────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def comoving_Mpc(z, H0=H0_FIDUCIAL):
    f = lambda zp: 1.0 / np.sqrt(OM*(1+zp)**3 + OL)
    r, _ = integrate.quad(f, 0, z)
    return (C_KM_S/H0) * r

def D_dt_Mpc(z_L, z_S, H0=H0_FIDUCIAL):
    """Time-delay distance D_Δt = (1+z_L) D_L D_S / D_LS."""
    DC_L = comoving_Mpc(z_L, H0)
    DC_S = comoving_Mpc(z_S, H0)
    DA_L  = DC_L/(1+z_L)
    DA_S  = DC_S/(1+z_S)
    DA_LS = (DC_S-DC_L)/(1+z_S)
    return (1+z_L)*DA_L*DA_S/DA_LS

def h0_from_delay(measured_d, published_d, H0_ref=H0_FIDUCIAL,
                   plateau_err_d=100.0, sys_pct=15.0):
    """H0 = H0_ref × (Δt_pub / Δt_meas), scaling method."""
    H0_est = H0_ref * (published_d / measured_d)
    frac   = np.sqrt((plateau_err_d/measured_d)**2 + (sys_pct/100)**2)
    return round(H0_est, 2), round(H0_est*frac, 2)

# ══════════════════════════════════════════════════════════════════════
# ── FOUNDATION: DATABASE ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

DB_PATH = "outputs/survey_v3/survey_results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS anisotropy_detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lens            TEXT UNIQUE NOT NULL,
    -- Equatorial
    ra_deg          REAL,
    dec_deg         REAL,
    -- Galactic
    l_deg           REAL,
    b_deg           REAL,
    sky_bin_l       INTEGER,
    sky_bin_b       INTEGER,
    -- Physical
    z_lens          REAL,
    z_source        REAL,
    sep_arcsec      REAL,
    measured_delay_d REAL,
    -- H0
    H0_estimate     REAL,
    H0_uncertainty  REAL,
    -- Kinematic Doppler
    dipole_cos_theta    REAL,
    flux_boost          REAL,
    count_boost         REAL,
    variability_boost   REAL,
    -- Hemisphere flags
    galactic_north  INTEGER,
    toward_dipole   INTEGER,
    high_gal_lat    INTEGER,    -- |b| > 30 deg
    -- Classification
    lens_type       TEXT,       -- galaxy_quad, cluster_quad, double, etc.
    survey          TEXT,       -- ZTF, Rubin_DP02, LSST_Y1, etc.
    -- Metadata
    logged_at       TEXT
)
"""

def get_conn(db_path=DB_PATH):
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn

def tag_and_store_detection(conn, lens_name, ra, dec, z_L, z_S,
                             sep_arcsec, measured_delay_d,
                             published_delay_d, H0_ref=H0_FIDUCIAL,
                             plateau_err_d=100.0, sys_pct=15.0,
                             lens_type="unknown", survey="ZTF"):
    """
    MAIN ENTRY POINT.
    Tag a new lens detection with full sky metadata and store in DB.
    Call this once per new detection from rubin_survey_v3.py output.

    Returns: dict with all computed metadata.
    """
    # Coordinate transforms
    l_deg, b_deg = eq_to_galactic(ra, dec)
    pixel_id, i_band, i_l, n_lon_band, b_ctr = sky_bin(l_deg, b_deg)
    i_b = i_band   # alias for backward compatibility

    # Doppler boost
    db = doppler_boost(ra, dec)

    # H0 estimate
    H0_est, H0_err = h0_from_delay(measured_delay_d, published_delay_d,
                                    H0_ref, plateau_err_d, sys_pct)

    # Hemisphere flags
    gal_north    = 1 if b_deg > 0 else 0
    toward_dipole = 1 if db["cos_theta"] > 0 else 0
    high_lat     = 1 if abs(b_deg) > 30 else 0

    meta = {
        "lens": lens_name, "ra": ra, "dec": dec,
        "l": l_deg, "b": b_deg, "i_l": i_l, "i_b": i_b,
        "z_L": z_L, "z_S": z_S, "sep": sep_arcsec,
        "delay_d": measured_delay_d, "H0": H0_est, "H0_err": H0_err,
        "dipole_cos": db["cos_theta"],
        "flux_boost": db["flux_boost"],
        "count_boost": db["count_boost"],
        "var_boost": db["variability_boost"],
        "gal_north": gal_north, "toward_dipole": toward_dipole,
        "high_lat": high_lat, "lens_type": lens_type, "survey": survey,
    }

    try:
        conn.execute("""
            INSERT OR REPLACE INTO anisotropy_detections
            (lens, ra_deg, dec_deg, l_deg, b_deg, sky_bin_l, sky_bin_b,
             z_lens, z_source, sep_arcsec, measured_delay_d,
             H0_estimate, H0_uncertainty,
             dipole_cos_theta, flux_boost, count_boost, variability_boost,
             galactic_north, toward_dipole, high_gal_lat,
             lens_type, survey, logged_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            lens_name, ra, dec, l_deg, b_deg, i_l, i_b,
            z_L, z_S, sep_arcsec, measured_delay_d,
            H0_est, H0_err,
            db["cos_theta"], db["flux_boost"],
            db["count_boost"], db["variability_boost"],
            gal_north, toward_dipole, high_lat,
            lens_type, survey,
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
    except Exception as e:
        print(f"  DB error for {lens_name}: {e}")

    return meta

# ══════════════════════════════════════════════════════════════════════
# ── FOUNDATION: STATISTICAL UTILITIES ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def dipole_significance(n_toward, n_away):
    """
    Test whether N(toward)/N(away) is consistent with isotropy.
    Uses binomial test (exact) for small N, normal approx for large N.
    Returns (ratio, p_value, sigma_equivalent).
    """
    n_total = n_toward + n_away
    if n_total == 0:
        return None, None, None
    ratio = n_toward / max(n_away, 1)
    # Under isotropy: P(toward) = 0.5, binomial test
    p_val = stats.binom_test(n_toward, n_total, 0.5, alternative='two-sided') \
            if hasattr(stats, 'binom_test') else \
            2 * min(stats.binom.cdf(n_toward, n_total, 0.5),
                    1 - stats.binom.cdf(n_toward-1, n_total, 0.5))
    sigma = abs(stats.norm.ppf(p_val/2)) if p_val > 0 else 10.0
    return round(ratio, 3), round(p_val, 5), round(sigma, 2)

def h0_anisotropy_significance(h0_list_A, h0_list_B):
    """
    Test whether H0 differs between two sky regions (A vs B).
    Uses Welch's t-test (unequal variance).
    Returns (delta_H0, p_value, sigma_equivalent).
    """
    if len(h0_list_A) < 2 or len(h0_list_B) < 2:
        return None, None, None
    delta = np.mean(h0_list_A) - np.mean(h0_list_B)
    t_stat, p_val = stats.ttest_ind(h0_list_A, h0_list_B, equal_var=False)
    sigma = abs(stats.norm.ppf(p_val/2)) if p_val > 0 else 10.0
    return round(delta, 2), round(p_val, 5), round(sigma, 2)

def s8_from_lens_count(n_obs, sky_area_deg2, S8_ref=S8_PLANCK,
                        rate_ref=QUAD_RATE_PER_SQ_DEG):
    """
    Infer S8 from observed strong lens count.
    N_quad ∝ S8² (from lens cross-section scaling with sigma_8).
    Returns (S8_inferred, uncertainty_1sigma).
    """
    if n_obs == 0 or sky_area_deg2 == 0:
        return None, None
    rate_obs = n_obs / sky_area_deg2
    S8_inf   = S8_ref * np.sqrt(rate_obs / rate_ref)
    # Poisson uncertainty on count → uncertainty on S8
    S8_err   = S8_inf / (2 * np.sqrt(n_obs))
    return round(S8_inf, 4), round(S8_err, 4)

# ══════════════════════════════════════════════════════════════════════
# ── MODULE A: H0 DIRECTIONAL MAP ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def module_a_h0_map(conn):
    print()
    print("  ┌─ MODULE A: H0 DIRECTIONAL MAP ─────────────────────────┐")
    print("  │  Tests: Does H0 vary with sky direction?               │")
    print("  │  Signal: H0(toward dipole) ≠ H0(away from dipole)     │")
    print("  │  Ref: Migkas et al. 2021 (X-ray cluster anisotropy)   │")
    print("  └─────────────────────────────────────────────────────────┘")

    rows = conn.execute("""
        SELECT lens, l_deg, b_deg, sky_bin_l, sky_bin_b,
               H0_estimate, H0_uncertainty,
               dipole_cos_theta, toward_dipole, high_gal_lat,
               z_lens, survey
        FROM anisotropy_detections ORDER BY logged_at
    """).fetchall()

    print(f"\n  Detections in database: {len(rows)}")
    print()

    if not rows:
        print("  No detections yet. Waiting for survey detections.")
        return

    # Per-detection summary
    print(f"  {'Lens':<20} {'l':>7} {'b':>6} {'H0':>8} {'±':>5} "
          f"{'cosθ':>7} {'Dir':>8} {'bin':>8}")
    print(f"  {'─'*20} {'─'*7} {'─'*6} {'─'*8} {'─'*5} "
          f"{'─'*7} {'─'*8} {'─'*8}")
    for r in rows:
        direction = "→apex" if r[8] else "←anti"
        binstr    = f"({r[3]},{r[4]})"
        print(f"  {r[0]:<20} {r[1]:>7.1f} {r[2]:>+6.1f} "
              f"{r[5]:>8.1f} {r[6]:>5.1f} {r[7]:>7.3f} "
              f"{direction:>8} {binstr:>8}")

    # Directional split
    toward = [(r[5], r[6]) for r in rows if r[8] == 1]
    away   = [(r[5], r[6]) for r in rows if r[8] == 0]
    north  = [(r[5], r[6]) for r in rows if r[2] > 0]
    south  = [(r[5], r[6]) for r in rows if r[2] < 0]

    print()
    print("  DIRECTIONAL H0 SPLIT:")
    for label, group in [("Toward CMB apex", toward), ("Away from apex", away),
                          ("Galactic North",  north),  ("Galactic South", south)]:
        if group:
            h0_vals = [g[0] for g in group]
            h0_errs = [g[1] for g in group]
            wmean   = np.average(h0_vals, weights=[1/e**2 for e in h0_errs])
            print(f"    {label:<20}: {wmean:.1f} km/s/Mpc  (N={len(group)})")
        else:
            print(f"    {label:<20}: — (no detections)")

    # Significance test
    if toward and away:
        delta, pval, sig = h0_anisotropy_significance(
            [t[0] for t in toward], [a[0] for a in away])
        print()
        print(f"  Anisotropy test (toward vs away):")
        print(f"    ΔH0 = {delta:+.1f} km/s/Mpc")
        print(f"    p-value = {pval:.4f}  ({sig:.1f}σ)")
        if sig and sig > 2:
            print(f"    ⚠ POTENTIAL SIGNAL — requires more detections to confirm")
        else:
            print(f"    Consistent with isotropy at current N")

    # Sky map using equal-area bins
    print()
    print("  H0 SKY MAP (equal-area sinusoidal bins, ~115 deg² each):")
    # Build pixel_id → H0 list from DB
    bin_data = {}
    for r in rows:
        # Recompute pixel from stored l,b
        pid, ib, il, nl, bc = sky_bin(r[1], r[2])
        if pid not in bin_data:
            bin_data[pid] = {"h0_vals": [], "l": r[1], "b": r[2], "ib": ib}
        bin_data[pid]["h0_vals"].append(r[5])

    if bin_data:
        print()
        print(f"  {'Pixel':>7} {'l_deg':>7} {'b_deg':>6} {'N':>3} {'H0':>8}  Symbol")
        print(f"  {'─'*7} {'─'*7} {'─'*6} {'─'*3} {'─'*8}  {'─'*6}")
        for pid, info in sorted(bin_data.items()):
            h0_mean = np.mean(info["h0_vals"])
            sym = "▼ LOW" if h0_mean < 67 else ("▲ HIGH" if h0_mean > 72 else "● MID")
            print(f"  {pid:>7} {info['l']:>7.1f}° {info['b']:>+6.1f}° "
                  f"{len(info['h0_vals']):>3} {h0_mean:>8.1f}  {sym}")
    else:
        print("  (No detections yet — sky map will populate as lenses are found)")
    print()
    print("  Legend: ▼=H0<67 (low, toward CMB)  ●=H0 67-72  ▲=H0>72 (high, SH0ES)")
    print("  Database grows: each Rubin detection adds a pixel to this map")
    print()
    # H0 vs redshift table (tests DESI running H0)
    print("  H0 vs REDSHIFT (tests DESI running H0 theory):")
    print(f"  {'Lens':<20} {'z_L':>6} {'H0':>8} {'±':>6} {'vs CMB':>8} {'vs SH0ES':>9}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*6} {'─'*8} {'─'*9}")
    sorted_rows = sorted(rows, key=lambda r: r[10])  # sort by z_lens
    for r in sorted_rows:
        d_cmb   = r[5] - 67.4
        d_shoes = r[5] - 73.0
        print(f"  {r[0]:<20} {r[10]:>6.3f} {r[5]:>8.1f} {r[6]:>6.1f} "
              f"{d_cmb:>+8.1f} {d_shoes:>+9.1f}")
    if len(rows) > 1:
        z_vals = [r[10] for r in sorted_rows]
        h_vals = [r[5]  for r in sorted_rows]
        print(f"  H0 trend: Δ{h_vals[-1]-h_vals[0]:+.1f} km/s/Mpc over "
              f"Δz={z_vals[-1]-z_vals[0]:.2f}")
        print(f"  (DESI w0waCDM predicts ~-10 km/s/Mpc shift from z=0.3 to z=0.7)")
    print()
    print("  N lenses needed for directional discrimination:")
    # To detect ~5 km/s/Mpc anisotropy at 3σ
    sigma_h0_per_lens = 12.0  # typical per-lens H0 uncertainty
    signal            = 5.0   # km/s/Mpc anisotropy signal to detect
    n_half_needed     = int((3 * sigma_h0_per_lens / signal)**2) + 1
    print(f"    Target signal: ±{signal:.0f} km/s/Mpc dipole in H0")
    print(f"    Per-lens σ(H0): ~{sigma_h0_per_lens:.0f} km/s/Mpc (cluster lenses)")
    print(f"    N per hemisphere for 3σ: ~{n_half_needed} lenses")
    print(f"    Total needed: ~{2*n_half_needed} lenses")
    print(f"    Rubin galaxy lenses (5% sys): ~{int(n_half_needed/9)} per hemisphere")

# ══════════════════════════════════════════════════════════════════════
# ── MODULE B: COSMIC DIPOLE MONITOR ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def module_b_dipole(conn):
    print()
    print("  ┌─ MODULE B: COSMIC DIPOLE MONITOR ──────────────────────┐")
    print("  │  Tests: Cosmological Principle — is universe isotropic?│")
    print("  │  Signal: N(toward)/N(away) > kinematic prediction      │")
    print("  │  Ref: Ellis & Baldwin 1984, Secrest et al. 2022        │")
    print("  └─────────────────────────────────────────────────────────┘")

    rows = conn.execute("""
        SELECT lens, l_deg, b_deg, toward_dipole, high_gal_lat,
               dipole_cos_theta, count_boost, variability_boost, z_lens
        FROM anisotropy_detections
    """).fetchall()

    n_total   = len(rows)
    n_toward  = sum(1 for r in rows if r[3] == 1)
    n_away    = n_total - n_toward
    n_clean   = sum(1 for r in rows if r[4] == 1)   # |b| > 30°

    print(f"\n  Total lensed AGN pairs: {n_total}")
    print(f"  Clean sky (|b|>30°):   {n_clean}")
    print(f"  Toward CMB apex:       {n_toward}")
    print(f"  Away from CMB apex:    {n_away}")

    # Kinematic dipole prediction (Ellis & Baldwin 1984)
    beta      = CMB_APEX["beta"]
    alpha     = AGN_SPECTRAL_INDEX
    x_slope   = 1.0
    kin_amplitude = (2 + alpha - x_slope) * beta
    pred_ratio    = 1 + 2 * kin_amplitude

    print()
    print("  KINEMATIC DIPOLE PREDICTION (Ellis & Baldwin 1984):")
    print(f"    Our velocity: {CMB_APEX['v_kms']:.0f} km/s  β = {beta:.6f}")
    print(f"    Predicted N+/N- (kinematic only):   {pred_ratio:.5f}")
    print(f"    Predicted dipole amplitude:          {kin_amplitude:.5f}  ({kin_amplitude*100:.3f}%)")
    print(f"    Observed in radio/quasar surveys:    ~0.010–0.016 (2–3× larger)")
    print(f"    This excess is the 'dipole anomaly'")

    # Observed ratio + significance
    ratio, pval, sig = dipole_significance(n_toward, n_away)
    if ratio is not None:
        print()
        print(f"  OBSERVED DIPOLE (current {n_total} lenses):")
        print(f"    N+/N- = {ratio:.3f}  (predicted kinematic: {pred_ratio:.4f})")
        print(f"    Binomial p-value: {pval:.4f}  ({sig:.1f}σ)")

    # Variability dipole
    if rows:
        boost_toward = np.mean([r[6] for r in rows if r[3]==1]) if n_toward > 0 else 0
        boost_away   = np.mean([r[6] for r in rows if r[3]==0]) if n_away   > 0 else 0
        print()
        print("  VARIABILITY DIPOLE (Doppler boost):")
        print(f"    Mean count boost toward apex: {boost_toward:.6f}")
        print(f"    Mean count boost away:        {boost_away:.6f}")
        print(f"    Expected variability boost toward: "
              f"{(3+alpha)*beta*100:.4f}%")
        print(f"    → Detectable with ~{int(1/((3+alpha)*beta)**2)} pairs")

    # N needed
    print()
    print("  DETECTION REQUIREMENTS:")
    for sigma_target in [3, 5]:
        # To detect dipole amplitude A at sigma_target sigma:
        # A*sqrt(N/4) > sigma_target  → N > (sigma_target/A)^2 * 4
        for label, amplitude in [
            ("Kinematic only", kin_amplitude),
            ("Observed anomaly (2×kin)", 2*kin_amplitude),
        ]:
            n_needed = int((sigma_target / amplitude)**2 * 4) + 1
            print(f"    {sigma_target}σ, {label:<30}: N ≈ {n_needed:,} lensed pairs")
    print()
    print("  Rubin LSST projected lensed AGN catalog: ~10,000–50,000")
    print("  → Dipole anomaly test feasible in Rubin year 3–5")
    print("  → Key advantage: variability filter removes stellar contamination")
    print("    that plagues radio/optical dipole studies (Ellis & Baldwin flaw)")

# ══════════════════════════════════════════════════════════════════════
# ── MODULE C: S8 TENSION PROBE ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def module_c_s8(conn):
    print()
    print("  ┌─ MODULE C: S8 TENSION PROBE ───────────────────────────┐")
    print("  │  Tests: Is matter less clumpy than ΛCDM predicts?      │")
    print("  │  Method: Strong lens count rate as S8 proxy            │")
    print("  │  Ref: Oguri & Marshall 2010; DES Y3, KiDS-1000        │")
    print("  └─────────────────────────────────────────────────────────┘")

    rows = conn.execute("""
        SELECT lens, z_lens, z_source, sep_arcsec, lens_type, survey
        FROM anisotropy_detections
    """).fetchall()

    n_obs   = len(rows)
    n_quads = sum(1 for r in rows if "quad" in (r[4] or ""))

    print(f"\n  Observed lenses (ZTF baseline): {n_obs}")
    print(f"  Confirmed quads:               {n_quads}")

    # ΛCDM prediction
    # ZTF resolution correction:
    # ZTF seeing ~2" resolves only wide-separation lenses (sep > 3")
    # This is ~3% of all quads (Oguri & Marshall 2010, cluster-scale lenses)
    ZTF_RESOLUTION_FRAC = 0.03   # fraction of quads resolvable by ZTF
    ztf_rate = QUAD_RATE_PER_SQ_DEG * ZTF_RESOLUTION_FRAC

    n_pred_planck = ztf_rate * ZTF_SKY_DEG2
    n_pred_s8low  = n_pred_planck * (S8_WEAK_LENS / S8_PLANCK)**2

    print()
    print(f"  ΛCDM PREDICTIONS (ZTF, {ZTF_SKY_DEG2:.0f} deg², wide-sep only):")
    print(f"    Resolution cut: sep > 3 arcsec (~{ZTF_RESOLUTION_FRAC*100:.0f}% of all quads)")
    print(f"    Planck S8={S8_PLANCK}:       {n_pred_planck:.1f} wide-sep quads expected")
    print(f"    Weak lensing S8={S8_WEAK_LENS}: {n_pred_s8low:.1f} wide-sep quads expected")
    print(f"    Sensitivity ratio: {n_pred_s8low/n_pred_planck:.3f}")

    # S8 inference from observations
    if n_quads > 0:
        S8_inf, S8_err = s8_from_lens_count(n_quads, ZTF_SKY_DEG2,
                                              rate_ref=ztf_rate)
        print()
        print(f"  INFERRED S8 FROM {n_quads} QUAD DETECTIONS:")
        print(f"    S8 = {S8_inf:.3f} ± {S8_err:.3f}")
        print(f"    Planck:   {S8_PLANCK:.3f}  (CMB early universe)")
        print(f"    WL surveys: {S8_WEAK_LENS:.3f}  (DES/KiDS weak lensing)")
        if S8_inf < S8_PLANCK - S8_err:
            print(f"    → Consistent with LOW S8 (less clumpy than CMB predicts)")
        else:
            print(f"    → Consistent with Planck at current N")

    # Rubin projection
    n_rubin_pl  = QUAD_RATE_PER_SQ_DEG * RUBIN_SKY_DEG2
    n_rubin_low = n_rubin_pl * (S8_WEAK_LENS / S8_PLANCK)**2
    delta_n     = n_rubin_pl - n_rubin_low

    print()
    print(f"  RUBIN LSST PROJECTION ({RUBIN_SKY_DEG2:.0f} deg²):")
    print(f"    Planck S8:        {n_rubin_pl:.0f} quads")
    print(f"    Low S8:           {n_rubin_low:.0f} quads")
    print(f"    Separation ΔN:    {delta_n:.0f} quads")
    sigma_disc = delta_n / np.sqrt(n_rubin_low)
    print(f"    Statistical sig:  {sigma_disc:.1f}σ with full Rubin survey")
    print(f"    → S8 discrimination: {'✓ YES' if sigma_disc > 3 else '✗ marginal'}")

    # Spatial S8 map
    print()
    print("  SPATIAL S8 PROBE (as catalog grows):")
    print("  If S8 varies across the sky → dark matter clumping is anisotropic")
    print("  Method: compute S8(l,b) from local quad density in sky bins")
    print("  This connects Module C to Module A (anisotropic dark energy/matter)")
    print()
    print("  KEY ADVANTAGE over weak lensing:")
    print("  Strong lens count = discrete events (quads/doubles/clusters)")
    print("  No PSF modeling, no shear calibration, no intrinsic alignment")
    print("  Systematic floor is fundamentally different → independent probe")

# ══════════════════════════════════════════════════════════════════════
# ── FOUNDATION TEST: POPULATE FROM EXISTING PIPELINE ─────────────────
# ══════════════════════════════════════════════════════════════════════

KNOWN_LENSES = {
    "SDSS1004+4112": {
        "ra": 151.065, "dec": 41.209,
        "z_L": 0.68,  "z_S": 1.734,
        "sep": 14.52, "pub_delay": 821.0,
        "H0_ref": 70.0, "sys_pct": 15.0,
        "lens_type": "cluster_quad",
    },
    "SDSS1029+2623": {
        "ra": 157.306, "dec": 26.392,
        "z_L": 0.584, "z_S": 2.197,
        "sep": 22.37, "pub_delay": 744.0,
        "H0_ref": 70.0, "sys_pct": 20.0,
        "lens_type": "cluster_quad",
    },
}

def load_and_tag_pipeline_detections(conn):
    """Load ZDCF detections from candidates table and tag them."""
    tagged = 0
    try:
        rows = conn.execute(
            "SELECT anchor, lag_days FROM candidates"
        ).fetchall()
    except:
        rows = []

    for anchor, lag in rows:
        key = anchor.replace(" ", "")
        if key in KNOWN_LENSES:
            info = KNOWN_LENSES[key]
            tag_and_store_detection(
                conn, key,
                info["ra"], info["dec"],
                info["z_L"], info["z_S"],
                info["sep"], abs(lag),
                info["pub_delay"],
                H0_ref=info["H0_ref"],
                sys_pct=info["sys_pct"],
                lens_type=info["lens_type"],
                survey="ZTF"
            )
            tagged += 1
    return tagged


# ══════════════════════════════════════════════════════════════════════
# ── INTEGRATION HOOK FOR rubin_survey_v3.py ───────────────────────────
# ══════════════════════════════════════════════════════════════════════
# To enable real-time anisotropy tagging, add this import and call
# to rubin_survey_v3.py inside process_seed() after a match is stored:
#
#   from anisotropy_monitor import tag_and_store_detection, get_conn
#
#   # After storing candidate to survey DB:
#   aniso_conn = get_conn()
#   if aniso_conn:
#       tag_and_store_detection(
#           aniso_conn,
#           lens_name    = f"{anchor}_{oid_A}",
#           ra           = ra,          # anchor RA
#           dec          = dec,         # anchor Dec
#           z_L          = 0.5,         # from catalog if known, else 0.5
#           z_S          = 2.0,         # from catalog if known, else 2.0
#           sep_arcsec   = sep,
#           measured_delay_d = abs(result["best_lag"]),
#           published_delay_d = abs(result["best_lag"]),  # self-ref if unknown
#           lens_type    = "galaxy_double" if sep < 5 else "cluster_quad",
#           survey       = "LSST_Y1"    # or "ZTF"
#       )
#       aniso_conn.close()
# ─────────────────────────────────────────────────────────────────────

def get_hook_code():
    """Return the integration hook code as a string for embedding."""
    return """
    # ── ANISOTROPY MONITOR HOOK ──────────────────────────────────────
    try:
        from anisotropy_monitor import tag_and_store_detection, get_conn as _ac
        _aconn = _ac()
        if _aconn:
            tag_and_store_detection(
                _aconn, lens_name=f"{anchor}_{oid_A}",
                ra=ra, dec=dec, z_L=0.5, z_S=2.0,
                sep_arcsec=sep,
                measured_delay_d=abs(result["best_lag"]),
                published_delay_d=abs(result["best_lag"]),
                lens_type="galaxy_double" if sep < 5 else "cluster_quad",
                survey="ZTF"
            )
            _aconn.close()
    except Exception as _e:
        pass  # anisotropy tagging is non-critical
    # ─────────────────────────────────────────────────────────────────
    """

# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   ANISOTROPY MONITOR v2.0                                   ║")
    print("║   Testing the Cosmological Principle with lensed AGN        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Foundation: coordinate transforms, Doppler boost, sky bins,")
    print("  statistical utilities, and a unified detection database.")
    print()
    print("  Science modules:")
    print("    A. H0 directional map  → anisotropic dark energy?")
    print("    B. Cosmic dipole       → is the universe isotropic?")
    print("    C. S8 tension probe    → is matter clumping correct?")

    conn = get_conn()
    if conn is None:
        print(f"\n  ERROR: Database not found at {DB_PATH}")
        print("  Run rubin_survey_v3.py first.")
        return

    # Tag existing pipeline detections
    n_tagged = load_and_tag_pipeline_detections(conn)
    n_total  = conn.execute(
        "SELECT COUNT(*) FROM anisotropy_detections"
    ).fetchone()[0]
    print(f"\n  Tagged {n_tagged} pipeline detections.")
    print(f"  Total in anisotropy_detections: {n_total}")

    # Run modules
    module_a_h0_map(conn)
    module_b_dipole(conn)
    module_c_s8(conn)

    conn.close()

    # Summary
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SUMMARY                                                     ║")
    print(f"║  Detections tagged: {n_total:<39}║")
    print("║                                                              ║")
    print("║  Foundation layer ready. To add a new Rubin detection:      ║")
    print("║    tag_and_store_detection(conn, name, ra, dec, ...)        ║")
    print("║  All three modules update automatically.                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()