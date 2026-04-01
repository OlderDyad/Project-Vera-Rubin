"""
generate_discovery_report.py
────────────────────────────
Takes the surviving novel candidates from the SIMBAD check and generates
a publication-ready dossier for each one. Combines ZTF light curves, 
physics scores, and a live optical sky photograph from DECaLS.
"""

import pandas as pd
import requests
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from io import BytesIO
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

NOVEL_FILE = Path("outputs/hunter/novel_candidates.csv")
REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic constants (matching your existing toolkit)
DARK="#04060f"; PANEL="#080d1a"; BORDER="#0d1f3a"; TEXT="#c8ddf5"
MUTED="#4a6080"; ACCENT="#00c8ff"; GREEN="#00e5a0"; GOLD="#ffd166"; RED="#ff4757"

def fetch_lightcurve(oid):
    """Pulls ZTF detections to plot the visual light curves."""
    try:
        r = requests.get(f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve", timeout=20)
        r.raise_for_status()
        dets = r.json().get("detections", [])
        if not dets: return None
        df = pd.DataFrame(dets)[["mjd", "magpsf", "sigmapsf", "fid"]].rename(
            columns={"magpsf": "mag", "sigmapsf": "mag_err", "fid": "band"})
        df["flux"] = 10**(-0.4*(df["mag"]-25.0))
        return df.sort_values("mjd").reset_index(drop=True)
    except: return None

def fetch_object_coords(oid):
    """Gets the exact RA/Dec of the object to center the telescope photo."""
    try:
        r = requests.get(f"https://api.alerce.online/ztf/v1/objects/{oid}", timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("meanra"), data.get("meandec")
    except: return None, None

def fetch_decals_cutout(ra, dec, size=256, pixscale=0.262):
    """Downloads a live optical JPEG cutout from the DECaLS sky survey."""
    print(f"  📸 Fetching DECaLS image at RA={ra:.4f}, Dec={dec:.4f}...")
    url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&layer=ls-dr10&pixscale={pixscale}&size={size}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception as e:
        print(f"  ⚠ Failed to fetch DECaLS image: {e}")
        return None

def build_dossier(row, lc_A, lc_B, img_decals, ra, dec):
    """Constructs the multi-panel visual report."""
    pair_id = row.get("pair_id", f"{row['oid_A']}x{row['oid_B']}")
    anchor = row.get("anchor", "Unknown")
    
    fig = plt.figure(figsize=(14, 8), facecolor=DARK)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.2)
    
    def format_ax(ax, title=""):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values(): s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        if title: ax.set_title(title, color="white", fontsize=10, pad=8)

    # --- TOP PANEL: Light Curves ---
    ax_lc = fig.add_subplot(gs[0, :])
    format_ax(ax_lc, f"Light Curves: {pair_id}")
    if lc_A is not None:
        ax_lc.plot(lc_A["mjd"], lc_A["flux"], "o", c=ACCENT, ms=3, alpha=0.7, label=f"Image A ({row['oid_A']})")
    if lc_B is not None:
        ax_lc.plot(lc_B["mjd"], lc_B["flux"], "s", c=GOLD, ms=3, alpha=0.7, label=f"Image B ({row['oid_B']})")
    ax_lc.set_xlabel("MJD"); ax_lc.set_ylabel("Flux")
    ax_lc.legend(facecolor=PANEL, labelcolor="white", fontsize=8)

    # --- BOTTOM LEFT: DECaLS Image ---
    ax_img = fig.add_subplot(gs[1, 0])
    format_ax(ax_img, "DECaLS DR10 Optical Imaging")
    if img_decals:
        ax_img.imshow(img_decals)
        ax_img.axis("off")
        # Add a crosshair to the center
        size = img_decals.size[0]
        ax_img.plot([size/2], [size/2], '+', color=RED, markersize=15, markeredgewidth=1.5, alpha=0.8)
    else:
        ax_img.text(0.5, 0.5, "Image Unavailable", color=RED, ha="center", va="center", transform=ax_img.transAxes)

    # --- BOTTOM RIGHT: Physics & Metadata ---
    ax_stats = fig.add_subplot(gs[1, 1])
    format_ax(ax_stats, "Candidate Dossier")
    ax_stats.axis("off")
    
    stats_text = [
        f"TARGET INFO",
        f"Anchor:  {anchor}",
        f"RA:      {ra:.5f}" if ra else "RA:      Unknown",
        f"Dec:     {dec:.5f}" if dec else "Dec:     Unknown",
        f"Sep:     {row.get('sep_arcsec', 0):.3f} arcsec",
        f"",
        f"LENSING PHYSICS",
        f"Lag:     {row.get('best_lag_days', 0):.1f} days",
        f"Rungs:   {row.get('rungs_passed', 0)} / 6 Passed",
        f"Score:   {row.get('total_score', 0):.3f}",
        f"",
        f"STATUS",
        f"Novelty: Not in SIMBAD as Lens",
        f"Action:  Needs human visual review"
    ]
    
    y_pos = 0.95
    for line in stats_text:
        color = ACCENT if "LENSING PHYSICS" in line or "TARGET INFO" in line or "STATUS" in line else TEXT
        weight = "bold" if color == ACCENT else "normal"
        ax_stats.text(0.05, y_pos, line, color=color, fontsize=10, fontfamily="monospace", fontweight=weight, transform=ax_stats.transAxes)
        y_pos -= 0.065

    fig.suptitle(f"DISCOVERY CANDIDATE: {anchor}", color="white", fontsize=14, fontweight="bold", y=0.96)
    
    out_path = REPORT_DIR / f"report_{pair_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved dossier -> {out_path.name}")

def generate_reports():
    if not NOVEL_FILE.exists():
        print(f"⚠ Cannot find {NOVEL_FILE}. Run check_simbad.py first.")
        return

    df = pd.read_csv(NOVEL_FILE)
    if df.empty:
        print("No novel candidates to generate reports for.")
        return

    print(f"\n📑 Generating Dossiers for {len(df)} Novel Candidates...")
    print("═" * 60)

    for _, row in df.iterrows():
        pair_id = row.get("pair_id", f"{row['oid_A']}x{row['oid_B']}")
        print(f"\nProcessing {pair_id}...")
        
        lc_A = fetch_lightcurve(row['oid_A'])
        lc_B = fetch_lightcurve(row['oid_B'])
        
        ra, dec = fetch_object_coords(row['oid_A'])
        img = fetch_decals_cutout(ra, dec) if ra and dec else None
        
        build_dossier(row, lc_A, lc_B, img, ra, dec)
        time.sleep(1) # Be polite to APIs

    print("\n" + "═" * 60)
    print(f"All reports generated in {REPORT_DIR}/")

if __name__ == "__main__":
    generate_reports()