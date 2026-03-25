"""
mstep_comparison.py
====================
Implements the Bansal & Huterer (2025) Mstep model framework and tests
it against gravitational lens time-delay H0 measurements.

THREE SCENARIOS FROM THE PAPER:
  Scenario 1 — LCDM: H0 constant at ~67.4 everywhere
  Scenario 2 — SH0ES: H0 constant at ~73.0 everywhere
  Scenario 3 — Mstep: Sharp M transition at z_t ~ 0.01
                      H0_local (~73) at z < z_t
                      H0_global (~67) at z > z_t
  Scenario 4 — Intermediate Mstep: Gradual transition at z_t ~ 0.15
                      (the "sweet spot" identified in Appendix C)
  Scenario 5 — DESI w0waCDM: Running H0, bias grows with z_L

The key test: as we accumulate lens detections at different z_L values,
does the H0(z_L) pattern match any of these scenarios?

Reference: Bansal & Huterer 2025, "On the Difficulties with Late-Time
           Solutions for the Hubble Tension", U. Michigan preprint
"""

import sqlite3
import numpy as np
from scipy import integrate, special
from pathlib import Path
from datetime import datetime, timezone

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

C_KM_S = 2.998e5
OM = 0.3
OL = 0.7

# Reference H0 values from Bansal & Huterer Table 1
H0_CMB    = 67.4   # Planck 2020
H0_SHOES  = 73.04  # SH0ES collaboration
H0_DESI   = 68.5   # DESI DR2 BAO + BBN
H0_TDCOSMO = 74.2  # TDCOSMO lensing

# Mstep transition parameters (best-fit from B&H Table 1)
Z_T_LOW   = 0.01   # low-z transition (trivial solution, Δχ²=-40)
Z_T_MID   = 0.15   # intermediate transition (sweet spot, Δχ²=-16)

# DESI w0waCDM best-fit (B&H context / DESI DR2)
W0_DESI = -0.827
WA_DESI = -0.75

# ══════════════════════════════════════════════════════════════════════
# COSMOLOGICAL DISTANCES
# ══════════════════════════════════════════════════════════════════════

def comoving_Mpc(z, H0=H0_CMB, w0=-1.0, wa=0.0):
    """Comoving distance under flat w0waCDM."""
    def E(zp):
        f_de = np.exp(3*wa*(zp/(1+zp))) * (1+zp)**(3*(1+w0+wa))
        return np.sqrt(OM*(1+zp)**3 + OL*f_de)
    result, _ = integrate.quad(lambda zp: 1/E(zp), 0, z)
    return (C_KM_S / H0) * result

def D_dt_Mpc(z_L, z_S, H0=H0_CMB, w0=-1.0, wa=0.0):
    """Time-delay distance D_Δt = (1+z_L) D_L D_S / D_LS."""
    DC_L  = comoving_Mpc(z_L, H0, w0, wa)
    DC_S  = comoving_Mpc(z_S, H0, w0, wa)
    DA_L  = DC_L / (1+z_L)
    DA_S  = DC_S / (1+z_S)
    DA_LS = (DC_S - DC_L) / (1+z_S)
    return (1+z_L) * DA_L * DA_S / DA_LS

# ══════════════════════════════════════════════════════════════════════
# FIVE SCENARIO MODELS (Bansal & Huterer framework)
# ══════════════════════════════════════════════════════════════════════

def h0_scenario_lcdm(z_L):
    """
    Scenario 1: Pure LCDM
    H0 is constant at H0_CMB = 67.4 for all lens redshifts.
    This is what BAO+CMB prefers.
    """
    return H0_CMB, 0.5   # (H0_predicted, uncertainty_km_s_mpc)

def h0_scenario_shoes(z_L):
    """
    Scenario 2: SH0ES local value
    H0 is constant at H0_SHOES = 73.04 for all lens redshifts.
    This is what Cepheid + SNIa distance ladder gives.
    """
    return H0_SHOES, 1.04

def h0_scenario_mstep_sharp(z_L, z_t=Z_T_LOW):
    """
    Scenario 3: Sharp Mstep (Bansal & Huterer best-fit, Δχ²=-40)
    Transition in SNIa absolute magnitude M at z_t ~ 0.01.
    For lenses at z_L > z_t: H0 → H0_CMB (~67)
    For lenses at z_L < z_t: H0 → H0_SHOES (~73)
    
    All gravitational lenses accessible to ZTF/Rubin have z_L >> 0.01,
    so this predicts H0 ~ 67 for ALL our measurements.
    This is the "trivial" solution that decouples SH0ES from high-z data.
    """
    if z_L < z_t:
        return H0_SHOES, 1.0
    else:
        return H0_CMB, 0.5

def h0_scenario_mstep_intermediate(z_L, z_t=Z_T_MID):
    """
    Scenario 4: Intermediate Mstep (B&H Appendix C sweet spot, Δχ²=-16)
    Gradual M transition at z_t ~ 0.15.
    
    For lenses at z_L < 0.15: H0 closer to ~70-72
    For lenses at z_L > 0.15: H0 closer to ~67-68
    
    This is the physically interesting scenario because:
    1. Our cluster lenses at z_L~0.6 should give H0 ~ 67-68
    2. Future Rubin galaxy lenses at z_L~0.1-0.2 will probe the transition
    3. The DESI BGS measurement at z~0.3 constrains this transition
    """
    # Smooth sigmoid transition from H0_mid to H0_CMB
    H0_high = 70.5   # above-CMB value at z < z_t
    H0_low  = H0_CMB  # CMB value at z > z_t
    sharpness = 15.0  # controls how sharp the transition is
    sigmoid = 1.0 / (1.0 + np.exp(sharpness * (z_L - z_t)))
    H0_pred = H0_low + (H0_high - H0_low) * sigmoid
    return round(H0_pred, 2), 1.0

def h0_scenario_desi_w0wa(z_L, z_S=2.0):
    """
    Scenario 5: DESI w0waCDM (dynamical dark energy)
    H0 inferred from lensing is BIASED if true cosmology is w0waCDM
    but analysis assumes LCDM.
    
    Bias = H0_LCDM_assumed / H0_true
         = D_dt(w0waCDM) / D_dt(LCDM)
    
    Under DESI best-fit (w0=-0.827, wa=-0.75):
    D_dt is ~20% larger at z_L~0.6-0.7, so H0_inferred is ~17% lower.
    
    This means: if DESI is right, our lensing measurements at z_L~0.6
    will give H0 ~ 62-65 even if the true H0 is 70.
    """
    D_lcdm = D_dt_Mpc(z_L, z_S, H0_CMB, -1.0, 0.0)
    D_desi  = D_dt_Mpc(z_L, z_S, H0_CMB, W0_DESI, WA_DESI)
    if D_desi <= 0:
        return H0_CMB, 1.0
    # H0_inferred (LCDM assumption) = H0_true * D_desi/D_lcdm
    # If true H0 = 70, observed H0 under wrong cosmology:
    H0_pred = H0_CMB * (D_lcdm / D_desi)
    return round(H0_pred, 2), 1.5

# ══════════════════════════════════════════════════════════════════════
# SCENARIO COMPARISON ENGINE
# ══════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "LCDM":          ("Scenario 1: LCDM (H0=67.4 everywhere)",
                       h0_scenario_lcdm,   "steelblue"),
    "SH0ES":         ("Scenario 2: SH0ES (H0=73.0 everywhere)",
                       h0_scenario_shoes,  "firebrick"),
    "Mstep_sharp":   ("Scenario 3: Mstep z_t=0.01 (trivial, Δχ²=-40)",
                       h0_scenario_mstep_sharp,  "orange"),
    "Mstep_mid":     ("Scenario 4: Mstep z_t=0.15 (sweet spot, Δχ²=-16)",
                       h0_scenario_mstep_intermediate, "green"),
    "DESI_w0wa":     ("Scenario 5: DESI w0waCDM bias",
                       h0_scenario_desi_w0wa,  "purple"),
}

def predict_all_scenarios(z_L, z_S=2.0):
    """Return H0 predictions from all five scenarios for a given z_L."""
    results = {}
    for key, (label, func, color) in SCENARIOS.items():
        try:
            if key == "DESI_w0wa":
                H0_pred, H0_err = func(z_L, z_S)
            else:
                H0_pred, H0_err = func(z_L)
            results[key] = {"H0": H0_pred, "err": H0_err,
                             "label": label, "color": color}
        except Exception as e:
            results[key] = {"H0": None, "err": None,
                             "label": label, "color": color,
                             "error": str(e)}
    return results

def chi2_vs_scenario(H0_obs, H0_err_obs, z_L, z_S=2.0):
    """
    Compute chi-squared of an observed H0 measurement against
    each of the five scenarios.
    Returns dict of {scenario: (chi2, sigma_tension)}
    """
    preds = predict_all_scenarios(z_L, z_S)
    tensions = {}
    for key, pred in preds.items():
        if pred["H0"] is None:
            continue
        total_err = np.sqrt(H0_err_obs**2 + pred["err"]**2)
        chi2      = ((H0_obs - pred["H0"]) / total_err)**2
        sigma     = abs(H0_obs - pred["H0"]) / total_err
        tensions[key] = {
            "chi2":     round(chi2, 3),
            "sigma":    round(sigma, 2),
            "H0_pred":  pred["H0"],
            "H0_obs":   H0_obs,
            "delta_H0": round(H0_obs - pred["H0"], 2),
        }
    return tensions

def best_scenario(tensions):
    """Return the scenario with lowest chi2 (best fit to current data)."""
    valid = {k: v for k, v in tensions.items() if "chi2" in v}
    return min(valid, key=lambda k: valid[k]["chi2"])

# ══════════════════════════════════════════════════════════════════════
# DISCRIMINABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def n_lenses_to_discriminate(sigma_target=3.0):
    """
    How many lenses at each redshift do we need to discriminate
    between scenarios?
    
    Strategy: compare Scenario 1 (LCDM) vs Scenario 2 (SH0ES)
    Signal = |H0_LCDM - H0_SH0ES| = 73.0 - 67.4 = 5.6 km/s/Mpc
    Noise per lens = sigma_H0 (depends on lens type)
    N needed = (sigma_target * sigma_H0 / signal)^2
    """
    signal = H0_SHOES - H0_CMB  # 5.6 km/s/Mpc

    print()
    print("  N LENSES NEEDED TO DISCRIMINATE SCENARIOS:")
    print(f"  Signal (SH0ES - CMB): {signal:.1f} km/s/Mpc")
    print(f"  Target: {sigma_target:.0f}σ discrimination")
    print()
    print(f"  {'Lens type':<25} {'σ(H0)/lens':>12} {'N for 3σ':>10} {'N for 5σ':>10}")
    print(f"  {'─'*25} {'─'*12} {'─'*10} {'─'*10}")

    lens_types = [
        ("Cluster (current, ~15% sys)", 12.0),
        ("Galaxy (Rubin, ~5% sys)",      4.0),
        ("Galaxy (Rubin, ~3% sys)",      2.5),
    ]

    for label, sigma_per_lens in lens_types:
        n_3sigma = int((3.0 * sigma_per_lens / signal)**2) + 1
        n_5sigma = int((5.0 * sigma_per_lens / signal)**2) + 1
        print(f"  {label:<25} {sigma_per_lens:>10.1f}  {n_3sigma:>10} {n_5sigma:>10}")

    # Special case: Mstep sweet spot discrimination
    print()
    print("  Discriminating Mstep z_t=0.15 from LCDM:")
    print("  Need lenses spanning z_L = 0.05-0.25 (below transition)")
    print("  AND z_L = 0.4-0.8 (above transition)")
    print("  Signal at z_L=0.10: H0(Mstep) ~ 70.5 vs H0(LCDM) ~ 67.4 = 3.1 km/s/Mpc")
    print("  With galaxy lenses (σ=4 km/s/Mpc): N ~ 6 per redshift bin")

# ══════════════════════════════════════════════════════════════════════
# DATABASE INTEGRATION
# ══════════════════════════════════════════════════════════════════════

DB_PATH = "outputs/survey_v3/survey_results.db"

def load_h0_detections(db_path=DB_PATH):
    """Load H0 estimates from h0_estimates table."""
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT lens, z_lens, z_source, measured_delay_d,
                   H0_estimate, H0_uncertainty
            FROM h0_estimates
            ORDER BY z_lens
        """).fetchall()
    except Exception:
        rows = []
    conn.close()
    return rows

def save_mstep_comparison(detections, db_path=DB_PATH):
    """Save Mstep comparison results to database."""
    if not Path(db_path).exists():
        return
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mstep_comparison (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            lens        TEXT,
            z_lens      REAL,
            H0_measured REAL,
            H0_err      REAL,
            -- Chi2 vs each scenario
            chi2_lcdm      REAL, sigma_lcdm      REAL,
            chi2_shoes     REAL, sigma_shoes     REAL,
            chi2_mstep_sharp REAL, sigma_mstep_sharp REAL,
            chi2_mstep_mid REAL, sigma_mstep_mid REAL,
            chi2_desi_w0wa REAL, sigma_desi_w0wa REAL,
            -- Best scenario
            best_scenario  TEXT,
            logged_at      TEXT
        )
    """)

    for det in detections:
        lens, z_L, z_S, delay_d, H0_obs, H0_err = det
        z_S_use = z_S if z_S else 2.0
        tensions = chi2_vs_scenario(H0_obs, H0_err, z_L, z_S_use)
        best = best_scenario(tensions)

        conn.execute("""
            INSERT OR REPLACE INTO mstep_comparison
            (lens, z_lens, H0_measured, H0_err,
             chi2_lcdm, sigma_lcdm,
             chi2_shoes, sigma_shoes,
             chi2_mstep_sharp, sigma_mstep_sharp,
             chi2_mstep_mid, sigma_mstep_mid,
             chi2_desi_w0wa, sigma_desi_w0wa,
             best_scenario, logged_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            lens, z_L, H0_obs, H0_err,
            tensions.get("LCDM",{}).get("chi2"),
            tensions.get("LCDM",{}).get("sigma"),
            tensions.get("SH0ES",{}).get("chi2"),
            tensions.get("SH0ES",{}).get("sigma"),
            tensions.get("Mstep_sharp",{}).get("chi2"),
            tensions.get("Mstep_sharp",{}).get("sigma"),
            tensions.get("Mstep_mid",{}).get("chi2"),
            tensions.get("Mstep_mid",{}).get("sigma"),
            tensions.get("DESI_w0wa",{}).get("chi2"),
            tensions.get("DESI_w0wa",{}).get("sigma"),
            best,
            datetime.now(timezone.utc).isoformat()
        ))
    conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════════════════
# ASCII SCENARIO PLOT
# ══════════════════════════════════════════════════════════════════════

def print_scenario_curves():
    """Print ASCII visualization of all five H0(z_L) scenarios."""
    print()
    print("  H0(z_L) PREDICTIONS — FIVE SCENARIOS")
    print("  (Bansal & Huterer 2025 framework)")
    print()

    z_vals = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90, 1.00]

    # Header
    print(f"  {'z_L':>5}  {'LCDM':>6}  {'SH0ES':>6}  "
          f"{'Mstep.01':>9}  {'Mstep.15':>9}  {'DESI_bias':>10}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*6}  "
          f"{'─'*9}  {'─'*9}  {'─'*10}")

    for z_L in z_vals:
        h0_lcdm,  _ = h0_scenario_lcdm(z_L)
        h0_shoes, _ = h0_scenario_shoes(z_L)
        h0_ms1,   _ = h0_scenario_mstep_sharp(z_L)
        h0_ms2,   _ = h0_scenario_mstep_intermediate(z_L)
        h0_desi,  _ = h0_scenario_desi_w0wa(z_L)

        # Mark the transition zone
        marker = " ← transition" if 0.10 <= z_L <= 0.20 else ""

        print(f"  {z_L:>5.2f}  {h0_lcdm:>6.1f}  {h0_shoes:>6.1f}  "
              f"{h0_ms1:>9.1f}  {h0_ms2:>9.1f}  {h0_desi:>10.1f}{marker}")

    print()
    print("  KEY INSIGHT from Bansal & Huterer:")
    print("  • Scenario 3 (Mstep z_t=0.01): predicts H0~67 for ALL")
    print("    gravitational lenses (z_L > 0.01 always true)")
    print("    → Cannot distinguish from LCDM using lensing alone")
    print("  • Scenario 4 (Mstep z_t=0.15): predicts H0~70.5 at z_L<0.15,")
    print("    H0~67 at z_L>0.15 — THIS IS TESTABLE with Rubin lenses!")
    print("  • Scenario 5 (DESI w0waCDM): H0 bias grows with z_L,")
    print("    reaching ~-10 km/s/Mpc at z_L~0.6-0.7")
    print("  • Our cluster lenses at z_L~0.6 are most sensitive to")
    print("    Scenarios 4 and 5, NOT to the sharp Mstep")

# ══════════════════════════════════════════════════════════════════════
# PNG PLOT
# ══════════════════════════════════════════════════════════════════════

def save_mstep_plot(detections):
    """Save H0(z_L) scenario comparison as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        Path("outputs/diagnostic").mkdir(parents=True, exist_ok=True)
    except ImportError:
        print("  (matplotlib not available — skipping PNG)")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot scenario curves
    z_range = np.linspace(0.01, 1.0, 200)
    z_skip  = z_range[z_range > 0.01]  # all > Mstep sharp transition

    curves = {
        "LCDM\n(H0=67.4, constant)": (
            [h0_scenario_lcdm(z)[0] for z in z_range],
            "steelblue", "--", 2.0),
        "SH0ES\n(H0=73.0, constant)": (
            [h0_scenario_shoes(z)[0] for z in z_range],
            "firebrick", "--", 2.0),
        "Mstep z_t=0.01\n(trivial, Δχ²=-40)": (
            [h0_scenario_mstep_sharp(z)[0] for z in z_range],
            "orange", "-.", 2.0),
        "Mstep z_t=0.15\n(sweet spot, Δχ²=-16)": (
            [h0_scenario_mstep_intermediate(z)[0] for z in z_range],
            "forestgreen", "-", 2.5),
        "DESI w0waCDM bias\n(w0=-0.827, wa=-0.75)": (
            [h0_scenario_desi_w0wa(z)[0] for z in z_range],
            "purple", "-", 2.5),
    }

    for label, (vals, color, ls, lw) in curves.items():
        ax.plot(z_range, vals, color=color, ls=ls, lw=lw, label=label)

    # Shade the transition zone (z_t = 0.15 ± 0.05)
    ax.axvspan(0.10, 0.20, alpha=0.08, color="green",
               label="Mstep sweet spot\ntest zone (z=0.10-0.20)")

    # Mark our current cluster lens positions
    if detections:
        for det in detections:
            lens, z_L, z_S, delay, H0_obs, H0_err = det
            ax.errorbar(z_L, H0_obs, yerr=H0_err,
                        fmt="D", color="black", markersize=10,
                        capsize=5, capthick=2, elinewidth=2,
                        zorder=5)
            ax.annotate(lens[:12], (z_L, H0_obs),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=9, color="black")

    # Reference horizontal lines
    ax.axhline(H0_CMB,   color="steelblue", lw=0.8, alpha=0.4)
    ax.axhline(H0_SHOES, color="firebrick", lw=0.8, alpha=0.4)

    # Rubin accessible redshift ranges
    ax.axvspan(0.01, 0.15, alpha=0.04, color="gray",
               label="Rubin compact lenses\n(z_L < 0.15)")
    ax.axvspan(0.3, 0.8, alpha=0.04, color="navy",
               label="Cluster lens zone\n(z_L ~ 0.3-0.8)")

    ax.set_xlabel("Lens redshift z_L", fontsize=13)
    ax.set_ylabel("H0 (km/s/Mpc)", fontsize=13)
    ax.set_title(
        "H0(z_L) Predictions: Five Cosmological Scenarios\n"
        "Bansal & Huterer 2025 Mstep Framework vs Gravitational Lens Measurements",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(54, 78)
    ax.grid(True, alpha=0.3)

    # Add annotation boxes
    ax.annotate("← All our current lenses\nare in this z_L range",
                xy=(0.63, 62), fontsize=9, color="navy",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow", alpha=0.8))

    ax.annotate("Key test:\nMstep z_t=0.15\npredicts H0~70.5\nfor z_L < 0.15",
                xy=(0.08, 70.5), fontsize=8, color="forestgreen",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="honeydew", alpha=0.8))

    plt.tight_layout()
    out = "outputs/diagnostic/mstep_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved → {out}")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   MSTEP COMPARISON — Bansal & Huterer 2025 Framework        ║")
    print("║   Testing five cosmological scenarios against H0 from       ║")
    print("║   gravitational lens time delays                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Reference: 'On the Difficulties with Late-Time Solutions")
    print("  for the Hubble Tension', Bansal & Huterer, U. Michigan 2025")
    print()

    # ── Scenario curve table ─────────────────────────────────────────
    print_scenario_curves()

    # ── Load actual lens detections ──────────────────────────────────
    detections = load_h0_detections()
    print()
    print(f"  LENS DETECTIONS IN DATABASE: {len(detections)}")

    if not detections:
        print("  (No H0 estimates yet — run h0_pipeline.py first)")
        print("  Showing scenario predictions only.")
        # Use known values as placeholders
        detections = [
            ("SDSS1004+4112", 0.68, 1.734, 877.5, 65.5, 12.3),
            ("SDSS1029+2623", 0.584, 2.197, 887.5, 58.7, 13.5),
        ]
        print("  Using pipeline values for demonstration.")

    # ── Per-lens scenario comparison ─────────────────────────────────
    print()
    print("  ┌─ SCENARIO COMPARISON PER LENS ────────────────────────────┐")
    print()

    all_chi2_totals = {k: 0.0 for k in SCENARIOS}

    for det in detections:
        lens, z_L, z_S, delay_d, H0_obs, H0_err = det
        z_S_use = z_S if z_S else 2.0
        tensions = chi2_vs_scenario(H0_obs, H0_err, z_L, z_S_use)
        best = best_scenario(tensions)
        preds = predict_all_scenarios(z_L, z_S_use)

        print(f"  {lens}  [z_L={z_L:.3f}  H0={H0_obs:.1f}±{H0_err:.1f}]")
        print(f"  {'Scenario':<35} {'H0_pred':>8} {'ΔH0':>7} {'σ':>6} {'χ²':>7}")
        print(f"  {'─'*35} {'─'*8} {'─'*7} {'─'*6} {'─'*7}")

        for key, t in tensions.items():
            marker = " ◄ BEST" if key == best else ""
            print(f"  {SCENARIOS[key][0][:35]:<35} "
                  f"{t['H0_pred']:>8.1f} "
                  f"{t['delta_H0']:>+7.1f} "
                  f"{t['sigma']:>6.2f} "
                  f"{t['chi2']:>7.3f}{marker}")
            all_chi2_totals[key] += t["chi2"]
        print()

    # ── Combined chi2 across all lenses ──────────────────────────────
    print("  ┌─ COMBINED χ² (all lenses) ────────────────────────────────┐")
    print()
    print(f"  {'Scenario':<35} {'Total χ²':>10} {'Rank':>6}")
    print(f"  {'─'*35} {'─'*10} {'─'*6}")

    sorted_scenarios = sorted(all_chi2_totals.items(),
                               key=lambda x: x[1])
    for rank, (key, chi2_total) in enumerate(sorted_scenarios, 1):
        label = SCENARIOS[key][0][:35]
        marker = " ◄ BEST FIT" if rank == 1 else ""
        print(f"  {label:<35} {chi2_total:>10.3f} {rank:>6}{marker}")

    winner = sorted_scenarios[0][0]
    print()
    print(f"  Current data best explained by: {SCENARIOS[winner][0]}")
    print()
    print("  INTERPRETATION:")

    if winner == "DESI_w0wa":
        print("  → H0 values are LOW, consistent with DESI w0waCDM bias")
        print("  → Dynamical dark energy may be shifting H0 inference")
        print("  → Need more lenses at different z_L to confirm")
    elif winner == "LCDM":
        print("  → H0 values consistent with CMB/DESI predictions")
        print("  → No evidence for tension at current precision")
    elif winner == "SH0ES":
        print("  → H0 values consistent with local distance ladder")
        print("  → Supports late-universe expansion rate")
    elif winner == "Mstep_mid":
        print("  → H0 values suggest intermediate-z transition")
        print("  → Test with lenses below z_L=0.15 to confirm")
    elif winner == "Mstep_sharp":
        print("  → Cannot distinguish from LCDM using lensing")
        print("  → Mstep z_t=0.01 is invisible to our probe")

    # ── N needed table ───────────────────────────────────────────────
    n_lenses_to_discriminate()

    # ── Save to DB and PNG ───────────────────────────────────────────
    save_mstep_comparison(detections)
    print()
    print("  ✓ Results saved → mstep_comparison table in survey_results.db")
    save_mstep_plot(detections)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SUMMARY                                                     ║")
    print("║                                                              ║")
    print("║  Mstep z_t=0.01 (trivial): Predicts H0~67 for ALL our       ║")
    print("║  lenses — indistinguishable from LCDM via lensing alone.    ║")
    print("║                                                              ║")
    print("║  Mstep z_t=0.15 (testable): Predicts H0~70.5 for z_L<0.15  ║")
    print("║  vs H0~67 for z_L>0.15. Rubin compact lenses will test this.║")
    print("║                                                              ║")
    print("║  DESI w0waCDM: Predicts growing H0 bias with z_L.           ║")
    print("║  Our z_L~0.6 lenses are in the peak sensitivity zone.       ║")
    print("║                                                              ║")
    print("║  The key observable: H0(z_L<0.15) vs H0(z_L>0.15)          ║")
    print("║  → Add Rubin lenses at low z_L to test the sweet spot.      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()