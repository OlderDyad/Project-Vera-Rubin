"""
step5_visualize.py
──────────────────
Generates publication-quality visualizations:
  1. Sky map of your anomalous candidates
  2. Top candidate summary table (saved as PNG)
  3. Fetches and plots a light curve for your #1 candidate

Run:
    python step5_visualize.py
"""

import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

INPUT_FILE = Path("data/alerts_anomalies.csv")
FALLBACK   = Path("data/alerts_filtered.csv")
OUT_SKYMAP = Path("outputs/skymap.png")
OUT_LC     = Path("outputs/lightcurve_top_candidate.png")
OUT_SUMMARY = Path("outputs/candidate_summary.png")

for p in [OUT_SKYMAP, OUT_LC, OUT_SUMMARY]:
    p.parent.mkdir(exist_ok=True)

DARK_BG   = "#04060f"
PANEL_BG  = "#080d1a"
BORDER    = "#0d1f3a"
TEXT      = "#c8ddf5"
MUTED     = "#4a6080"
ACCENT    = "#00c8ff"
DANGER    = "#ff4757"
GOLD      = "#ffd166"
GREEN     = "#00e5a0"
PURPLE    = "#7b4fff"


def style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color="white", fontsize=11, pad=8)


def plot_skymap(df):
    ra_col  = next((c for c in ["meanra",  "ra"]  if c in df.columns), None)
    dec_col = next((c for c in ["meandec", "dec"] if c in df.columns), None)
    if not ra_col or not dec_col:
        print("  ⚠  No RA/Dec columns found for sky map.")
        return

    fig = plt.figure(figsize=(12, 5), facecolor=DARK_BG)
    ax  = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)

    # Convert RA to radians, centered on 180°
    ra_rad  = np.deg2rad(df[ra_col]  - 180)
    dec_rad = np.deg2rad(df[dec_col])

    is_anomaly = df.get("anomaly", pd.Series(False, index=df.index))

    normal    = ~is_anomaly
    anomalous =  is_anomaly

    ax.scatter(ra_rad[normal],    dec_rad[normal],
               s=8,  c=ACCENT,  alpha=0.4, label="Candidate")
    ax.scatter(ra_rad[anomalous], dec_rad[anomalous],
               s=40, c=DANGER,  alpha=0.9, label="Anomaly ★",
               edgecolors="#ff8a93", linewidths=0.6, zorder=5)

    ax.set_title("Sky Distribution of Transient Candidates",
                 color="white", fontsize=13, pad=12)
    ax.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9,
              loc="lower right", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(OUT_SKYMAP, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Sky map → {OUT_SKYMAP}")


def fetch_lightcurve_alerce(oid: str):
    """Fetch photometric light curve from ALeRCE for a given object ID."""
    url = f"https://api.alerce.online/alerts/v1/lightcurve/{oid}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        detections = data.get("detections", [])
        if not detections:
            return None
        df = pd.DataFrame(detections)
        return df
    except Exception as e:
        print(f"    Light curve fetch failed: {e}")
        return None


def plot_lightcurve(oid: str, lc_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    style_ax(ax, f"Light Curve: {oid}")

    band_colors = {"g": GREEN, "r": DANGER, "i": GOLD, "z": PURPLE}

    fid_map = {1: "g", 2: "r", 3: "i"}   # ZTF band IDs
    band_col = next((c for c in ["fid", "band", "filter"] if c in lc_df.columns), None)
    mjd_col  = next((c for c in ["mjd", "jd"] if c in lc_df.columns), None)
    mag_col  = next((c for c in ["magpsf", "mag", "magnitude"] if c in lc_df.columns), None)
    err_col  = next((c for c in ["sigmapsf", "mag_err", "magerr"] if c in lc_df.columns), None)

    if not mjd_col or not mag_col:
        print(f"    ⚠  Unexpected light curve columns: {list(lc_df.columns)}")
        return

    plotted = False
    for band_id, band_name in fid_map.items():
        mask = lc_df[band_col] == band_id if band_col else pd.Series(True, index=lc_df.index)
        subset = lc_df[mask]
        if subset.empty:
            continue

        color = band_colors.get(band_name, ACCENT)
        kwargs = dict(fmt="o", color=color, label=f"{band_name}-band",
                      markersize=5, linewidth=0.8, capsize=2, alpha=0.85)
        if err_col and err_col in subset.columns:
            ax.errorbar(subset[mjd_col], subset[mag_col], yerr=subset[err_col], **kwargs)
        else:
            ax.scatter(subset[mjd_col], subset[mag_col], c=color, s=30,
                       label=f"{band_name}-band", alpha=0.85)
        plotted = True

    if not plotted:
        ax.scatter(lc_df[mjd_col], lc_df[mag_col], c=ACCENT, s=30, alpha=0.8)

    ax.invert_yaxis()   # astronomer convention: brighter = lower magnitude
    ax.set_xlabel("MJD (Modified Julian Date)")
    ax.set_ylabel("Magnitude (brighter ↑)")
    ax.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_LC, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Light curve → {OUT_LC}")


def plot_summary_table(df):
    top = df.sort_values("anomaly_score", ascending=False).head(10) if "anomaly_score" in df.columns else df.head(10)
    show_cols = [c for c in ["oid", "classxf", "anomaly_score", "delta_mag", "ndet", "simbad_id"] if c in top.columns]
    top_show = top[show_cols].copy()

    if "anomaly_score" in top_show.columns:
        top_show["anomaly_score"] = top_show["anomaly_score"].round(3)
    if "delta_mag" in top_show.columns:
        top_show["delta_mag"] = top_show["delta_mag"].round(3)

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(top_show) + 1.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    col_labels = [c.replace("_", " ").title() for c in top_show.columns]
    tbl = ax.table(
        cellText=top_show.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL_BG if r % 2 == 0 else "#0a1020")
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=ACCENT if r == 0 else TEXT)

    ax.set_title("Top Anomalous Transient Candidates",
                 color="white", fontsize=13, pad=10, loc="left")
    plt.tight_layout()
    plt.savefig(OUT_SUMMARY, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Summary table → {OUT_SUMMARY}")


if __name__ == "__main__":
    print("\n🔭 Rubin Discovery Toolkit — Step 5: Visualize")
    print("─" * 55)

    for path in [INPUT_FILE, FALLBACK]:
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} records from {path}\n")
            break
    else:
        print("✗ No data found. Run step2_filter.py first.")
        exit(1)

    plot_skymap(df)
    plot_summary_table(df)

    # Fetch live light curve for the top candidate
    id_col = next((c for c in ["oid", "alertId"] if c in df.columns), None)
    if id_col:
        sort_col = "anomaly_score" if "anomaly_score" in df.columns else df.columns[0]
        top_id = df.sort_values(sort_col, ascending=False).iloc[0][id_col]
        print(f"\n  Fetching light curve for top candidate: {top_id}")
        lc = fetch_lightcurve_alerce(str(top_id))
        if lc is not None and not lc.empty:
            plot_lightcurve(str(top_id), lc)
        else:
            print("  (Light curve not available offline — run with internet access)")

    print(f"\n✓ All visualizations saved to outputs/")
    print("  Open the PNG files in VS Code (just click them in the file explorer)")
    print("\n🎉 Pipeline complete! Review outputs/ for your discovery candidates.")
