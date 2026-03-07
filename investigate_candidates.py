"""
investigate_candidates.py
──────────────────────────
Deep-dive report on top candidates from lens_hunter.py output.
For each top candidate, fetches fresh light curves, re-runs scoring,
and produces a detailed multi-panel investigation plot.

Usage:
    python investigate_candidates.py
    python investigate_candidates.py --top 5
    python investigate_candidates.py --pair ZTF17aacvwexxZTF20aahbcrr
"""

import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.ndimage import uniform_filter1d

from lc_utils import quality_check, zdcf, best_lag_from_zdcf, prep_lc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_FILE = Path("outputs/hunter/candidates.csv")
OUT_DIR      = Path("outputs/investigation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK="#04060f"; PANEL="#080d1a"; BORDER="#0d1f3a"
TEXT="#c8ddf5"; MUTED="#4a6080"; ACCENT="#00c8ff"
GREEN="#00e5a0"; GOLD="#ffd166"; RED="#ff4757"; PURPLE="#7b4fff"


def fetch_lightcurve(oid):
    try:
        r = requests.get(
            f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve",
            timeout=25)
        r.raise_for_status()
        dets = r.json().get("detections", [])
        if not dets: return None
        df = pd.DataFrame(dets)
        if not all(c in df.columns for c in ["mjd","magpsf","sigmapsf"]): return None
        df = df[["mjd","magpsf","sigmapsf","fid"]].rename(
            columns={"magpsf":"mag","sigmapsf":"mag_err","fid":"band"})
        df["flux"]     = 10**(-0.4*(df["mag"]-25.0))
        df["flux_err"] = df["flux"]*df["mag_err"]*0.4*np.log(10)
        return df.sort_values("mjd").reset_index(drop=True)
    except: return None


def fetch_object_info(oid):
    try:
        r = requests.get(
            f"https://api.alerce.online/ztf/v1/objects/{oid}",
            timeout=15)
        r.raise_for_status()
        return r.json()
    except: return {}


def zdcf_full(lc_A, lc_B, lag_range=(-90,90), n_bins=90):
    tA,fA = prep_lc(lc_A); tB,fB = prep_lc(lc_B)
    return zdcf(tA, fA, tB, fB, lag_range=lag_range, n_bins=n_bins)


def structure_function(t, f, dt_bins=30):
    """
    First-order structure function: SF(tau) = <(f(t+tau) - f(t))^2>
    Rising SF = stochastic variability (DRW).
    Flat or oscillating SF = periodic or noise-dominated.
    """
    taus = []; sfs = []
    edges = np.linspace(0, (t.max()-t.min())/2, dt_bins+1)
    for i in range(dt_bins):
        tau_lo, tau_hi = edges[i], edges[i+1]
        pairs = [(f[j]-f[k])**2
                 for j in range(len(t))
                 for k in range(j+1, len(t))
                 if tau_lo <= abs(t[j]-t[k]) < tau_hi]
        if pairs:
            taus.append((tau_lo+tau_hi)/2)
            sfs.append(np.mean(pairs))
    return np.array(taus), np.array(sfs)


def investigate(pair_id, lc_A, lc_B, oid_A, oid_B,
                anchor, sep, known_lag=None):

    print(f"\n  Investigating {pair_id}")
    print(f"  Anchor: {anchor}  Sep: {sep:.3f}\"")
    print(f"  A: {oid_A}  ({len(lc_A)} pts)")
    print(f"  B: {oid_B}  ({len(lc_B)} pts)")

    # Gemini's duplicate-ID check: ZTF cannot resolve objects < 1.5"
    ZTF_MIN_SEP = 1.5
    if sep < ZTF_MIN_SEP:
        print(f"  WARNING: sep={sep:.3f} arcsec < {ZTF_MIN_SEP} arcsec -- LIKELY ZTF DUPLICATE ID, not a real pair")
        print(f"     Verify at: https://www.legacysurvey.org/viewer?ra={lc_A['flux'].mean():.4f}&dec=0&zoom=16&layer=ls-dr10")

    qr = quality_check(lc_A, lc_B, pair_id)
    print(f"  Quality: overlap={qr.overlap_days:.0f}d  density={qr.density_ratio:.2f}  "
          f"{'✓' if qr.passes_all else '⚠ ' + qr.reject_reason}")

    # ZDCF
    centers, zvals, npairs = zdcf_full(lc_A, lc_B)
    best_lag, lag_unc = best_lag_from_zdcf(centers, zvals, npairs)
    valid = ~np.isnan(zvals)
    peak_z = np.nanmax(np.abs(zvals)) if valid.any() else 0
    peak_r = float(np.tanh(peak_z))

    print(f"  ZDCF lag: {best_lag:.1f} ± {lag_unc:.1f} days  (peak r={peak_r:.3f})")
    if known_lag:
        print(f"  Known lag: {known_lag} days  Error: {abs(best_lag-known_lag):.1f}d")

    # Structure functions
    tA, fA = prep_lc(lc_A); tB, fB = prep_lc(lc_B)
    sfA_t, sfA = structure_function(tA, fA)
    sfB_t, sfB = structure_function(tB, fB)

    # Fractional variability at best lag
    lB2 = lc_B.copy(); lB2["mjd"] += best_lag
    tmin = max(lc_A["mjd"].min(), lB2["mjd"].min())
    tmax = min(lc_A["mjd"].max(), lB2["mjd"].max())
    tg   = np.arange(tmin, tmax, 2.0)
    fA_g = np.interp(tg, lc_A["mjd"].values, lc_A["flux"].values)
    fB_g = np.interp(tg, lB2["mjd"].values,  lB2["flux"].values)
    fA_n = (fA_g - fA_g.mean())/(fA_g.mean()+1e-9)
    fB_n = (fB_g - fB_g.mean())/(fB_g.mean()+1e-9)
    frac_r, frac_p = stats.pearsonr(fA_n, fB_n) if len(tg)>10 else (0,1)

    # Microlensing decomposition
    win  = max(3, int(30/3))
    fA_s = uniform_filter1d(fA_g, win); fB_s = uniform_filter1d(fB_g, win)
    long_r,  _ = stats.pearsonr(fA_s, fB_s)
    fA_r = fA_g - fA_s; fB_r = fB_g - fB_s
    short_r = 0.0
    if fA_r.std()>1e-9 and fB_r.std()>1e-9:
        short_r, _ = stats.pearsonr(fA_r, fB_r)

    print(f"  Frac var r={frac_r:.3f}  Micro: long={long_r:.3f} short={short_r:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.38,
                             left=0.07, right=0.97, top=0.93, bottom=0.08)

    def sa(ax, title=""):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values(): s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        if title: ax.set_title(title, color="white", fontsize=9, pad=5)

    # Row 0: Full light curves
    ax0 = fig.add_subplot(gs[0, :2])
    sa(ax0, f"Light Curves — {oid_A[:16]} × {oid_B[:16]}")
    ax0.plot(lc_A["mjd"], lc_A["flux"], "o", c=ACCENT, ms=2.5, alpha=0.8, label="Object A")
    ax0.plot(lc_B["mjd"], lc_B["flux"], "s", c=GOLD,  ms=2.5, alpha=0.8, label="Object B")
    ax0.set_xlabel("MJD"); ax0.set_ylabel("Flux")
    ax0.legend(facecolor=PANEL, labelcolor="white", fontsize=8)

    # Row 0 col 2: Summary card
    ax_s = fig.add_subplot(gs[0, 2])
    sa(ax_s, "Investigation Summary")
    ax_s.axis("off")
    summary = [
        f"Anchor: {anchor}",
        f"Separation: {sep:.3f}\"",
        f"",
        f"ZDCF lag: {best_lag:.1f} ± {lag_unc:.1f} d",
        f"Peak r:   {peak_r:.3f}",
        f"Overlap:  {qr.overlap_days:.0f} d",
        f"Density:  {qr.density_ratio:.2f}",
        f"",
        f"Frac var r: {frac_r:.3f} (p={frac_p:.2e})",
        f"Micro long: {long_r:.3f}",
        f"Micro short:{short_r:.3f}",
        f"",
        f"Pts A: {len(lc_A)}  B: {len(lc_B)}",
    ]
    if known_lag:
        summary.append(f"Known lag: {known_lag} d")
        summary.append(f"Lag error: {abs(best_lag-known_lag):.1f} d")

    # Verdict
    score_checks = [
        peak_r > 0.3,
        frac_r > 0.5,
        long_r > 0.5 and short_r < 0.45,
    ]
    n_pass = sum(score_checks)
    if n_pass == 3:   verdict = "★★★ HIGH INTEREST"
    elif n_pass == 2: verdict = "★★  MODERATE"
    elif n_pass == 1: verdict = "★   WEAK SIGNAL"
    else:             verdict = "    LIKELY NOISE"
    summary += ["", f"Verdict: {verdict}"]

    ax_s.text(0.05, 0.97, "\n".join(summary),
              transform=ax_s.transAxes, color=TEXT,
              fontsize=8, va="top", fontfamily="monospace")

    # Row 1: ZDCF
    ax1 = fig.add_subplot(gs[1, 0])
    sa(ax1, "ZDCF (Z-transformed correlation)")
    if valid.any():
        ax1.plot(centers[valid], zvals[valid], c=ACCENT, lw=1.5)
        ax1.fill_between(centers[valid], 0, zvals[valid],
                          alpha=0.15, color=ACCENT)
    ax1.axvline(best_lag,  c=RED,   lw=1.5, ls="--", label=f"Lag={best_lag:.1f}d")
    ax1.axvline(-best_lag, c=RED,   lw=0.8, ls="--", alpha=0.4)
    ax1.axhline(0,         c=MUTED, lw=0.5)
    if known_lag:
        ax1.axvline(known_lag, c=GREEN, lw=1.2, ls=":", label=f"Known={known_lag}d")
    ax1.set_xlabel("Lag (days)"); ax1.set_ylabel("Z-score")
    ax1.legend(facecolor=PANEL, labelcolor="white", fontsize=7)

    # Row 1: Structure functions
    ax2 = fig.add_subplot(gs[1, 1])
    sa(ax2, "Structure Functions")
    if len(sfA_t)>1: ax2.plot(sfA_t, sfA, c=ACCENT, lw=1.2, label="SF — A")
    if len(sfB_t)>1: ax2.plot(sfB_t, sfB, c=GOLD,  lw=1.2, label="SF — B")
    ax2.set_xlabel("Time lag (days)"); ax2.set_ylabel("SF(τ)")
    ax2.set_yscale("log")
    ax2.legend(facecolor=PANEL, labelcolor="white", fontsize=7)

    # Row 1: Fractional variability scatter
    ax3 = fig.add_subplot(gs[1, 2])
    sa(ax3, f"Fractional Variability (lag={best_lag:.1f}d)")
    if len(tg)>10:
        ax3.scatter(fA_n, fB_n, c=tg, cmap="plasma", s=10, alpha=0.6)
        lim = max(abs(fA_n).max(), abs(fB_n).max()) * 1.1
        ax3.plot([-lim,lim],[-lim,lim], c=GREEN, lw=1, ls="--", alpha=0.7)
        ax3.text(0.05, 0.93, f"r={frac_r:.3f}", transform=ax3.transAxes,
                 color=GREEN if frac_r>0.6 else RED, fontsize=10, fontweight="bold")
    ax3.set_xlabel("Frac var A"); ax3.set_ylabel("Frac var B")

    # Row 2: Microlensing decomposition
    ax4 = fig.add_subplot(gs[2, :2])
    sa(ax4, "Microlensing Decomposition (30-day smoothing)")
    if len(tg)>10:
        ax4.plot(tg, fA_g, c=ACCENT, lw=0.6, alpha=0.4, label="A raw")
        ax4.plot(tg, fB_g, c=GOLD,   lw=0.6, alpha=0.4, label="B raw")
        ax4.plot(tg, fA_s, c=ACCENT, lw=1.5, label="A smooth (macro)")
        ax4.plot(tg, fB_s, c=GOLD,   lw=1.5, label="B smooth (macro)")
        # Residuals offset below
        offset = min(fA_g.min(), fB_g.min()) - 0.3
        ax4.plot(tg, fA_r + offset, c=ACCENT, lw=0.8, alpha=0.6, ls="--")
        ax4.plot(tg, fB_r + offset, c=GOLD,   lw=0.8, alpha=0.6, ls="--")
        ax4.axhline(offset, color=MUTED, lw=0.5, ls=":")
        ax4.text(tg[-1], offset+0.02, "residuals", color=MUTED, fontsize=7, ha="right")
    ax4.set_xlabel("MJD (shifted by lag)"); ax4.set_ylabel("Flux")
    ax4.legend(facecolor=PANEL, labelcolor="white", fontsize=7, ncol=4)

    # Row 2 col 2: Pair stats table
    ax5 = fig.add_subplot(gs[2, 2])
    sa(ax5, "Data Quality")
    ax5.axis("off")
    n_overlap_A = ((lc_A["mjd"]>=tmin) & (lc_A["mjd"]<=tmax)).sum()
    n_overlap_B = ((lc_B["mjd"]>=tmin) & (lc_B["mjd"]<=tmax)).sum()
    table_data = [
        ["", "Object A", "Object B"],
        ["OID", oid_A[-8:], oid_B[-8:]],
        ["Total pts", str(len(lc_A)), str(len(lc_B))],
        ["In overlap", str(n_overlap_A), str(n_overlap_B)],
        ["MJD start", f"{lc_A['mjd'].min():.0f}", f"{lc_B['mjd'].min():.0f}"],
        ["MJD end",   f"{lc_A['mjd'].max():.0f}", f"{lc_B['mjd'].max():.0f}"],
        ["Baseline",  f"{lc_A['mjd'].max()-lc_A['mjd'].min():.0f}d",
                      f"{lc_B['mjd'].max()-lc_B['mjd'].min():.0f}d"],
        ["Density ratio", f"{qr.density_ratio:.2f}", ""],
        ["Overlap days",  f"{qr.overlap_days:.0f}", ""],
    ]
    y = 0.97
    for row in table_data:
        is_header = row[0]==""
        color = "#ffffff" if is_header else TEXT
        ax5.text(0.02, y, row[0], transform=ax5.transAxes,
                 color=MUTED, fontsize=7.5, fontfamily="monospace")
        ax5.text(0.40, y, row[1], transform=ax5.transAxes,
                 color=ACCENT, fontsize=7.5, fontfamily="monospace")
        ax5.text(0.70, y, row[2], transform=ax5.transAxes,
                 color=GOLD,   fontsize=7.5, fontfamily="monospace")
        y -= 0.09

    fig.suptitle(f"Candidate Investigation — {pair_id}  |  Anchor: {anchor}",
                 color="white", fontsize=11, y=0.97)

    out_path = OUT_DIR / f"investigation_{pair_id[:50]}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  Plot -> {out_path}")
    return best_lag, peak_r, frac_r, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",  type=int, default=5,
                        help="Investigate top N candidates (default 5)")
    parser.add_argument("--pair", type=str, default=None,
                        help="Investigate a specific pair_id")
    parser.add_argument("--min-rungs", type=int, default=3)
    args = parser.parse_args()

    if not RESULTS_FILE.exists():
        print(f"No results file found at {RESULTS_FILE}")
        print("Run lens_hunter.py first.")
        return

    df = pd.read_csv(RESULTS_FILE)

    if args.pair:
        df = df[df["pair_id"]==args.pair]
        if df.empty:
            print(f"Pair '{args.pair}' not found in results")
            return
    else:
        df = df[df["rungs_passed"]>=args.min_rungs]
        df = df.sort_values("total_score", ascending=False).head(args.top)

    print(f"\n🔬 Investigating {len(df)} candidate(s)")
    print("─"*60)

    summaries = []
    for _, row in df.iterrows():
        oid_A = row["oid_A"]; oid_B = row["oid_B"]
        pair_id = row["pair_id"]
        anchor  = row.get("anchor","?")
        sep     = row.get("sep_arcsec", 0)

        print(f"\nFetching light curves for {pair_id}...")
        lc_A = fetch_lightcurve(oid_A)
        lc_B = fetch_lightcurve(oid_B)

        if lc_A is None or lc_B is None:
            print(f"  Could not fetch light curves — skipping")
            continue

        lag, r, frac_r, verdict = investigate(
            pair_id, lc_A, lc_B, oid_A, oid_B, anchor, sep)

        summaries.append({
            "pair_id": pair_id, "anchor": anchor,
            "sep\"": f"{sep:.3f}", "lag_d": f"{lag:.1f}",
            "zdcf_r": f"{r:.3f}", "frac_r": f"{frac_r:.3f}",
            "verdict": verdict,
        })

    print(f"\n{'═'*60}")
    print("  INVESTIGATION SUMMARY")
    print(f"{'═'*60}")
    if summaries:
        sdf = pd.DataFrame(summaries)
        print(sdf.to_string(index=False))
        sdf.to_csv(OUT_DIR/"investigation_summary.csv", index=False)
        print(f"\nSummary -> {OUT_DIR/'investigation_summary.csv'}")

    # Per-candidate lookup links
    ANCHOR_COORDS = {
        "PG 0844+349": (131.9113, 34.7506),
        "PG 1613+658": (243.5096, 65.7147),
        "HE 0435-1223": (69.5621, -12.2873),
        "3C 273": (187.2779, 2.0524),
        "Mrk 509": (311.0404, -10.7233),
        "NGC 5548": (214.4979, 25.1369),
        "PG 1543+489": (236.3938, 48.7761),
        "PG 0804+761": (122.7229, 76.0406),
    }
    print(f"\nPer-candidate lookup URLs:")
    for s in summaries:
        pid = s["pair_id"]; anchor = s["anchor"]
        sep_val = float(s.get("sep\"", s.get("sep_arcsec", 99)))
        dupe_warn = " LIKELY DUPLICATE (sep < 1.5 arcsec)" if sep_val < 1.5 else ""
        print(f"\n  {pid}{dupe_warn}")
        coords = ANCHOR_COORDS.get(anchor)
        if coords:
            ra, dec = coords
            print(f"    DECaLS: https://www.legacysurvey.org/viewer?ra={ra}&dec={dec}&zoom=16&layer=ls-dr10")
            print(f"    SIMBAD: https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra}+{dec}&Radius=0.1&Radius.unit=arcmin")
        else:
            print(f"    SIMBAD: https://simbad.u-strasbg.fr/simbad/sim-id?Ident={anchor.replace(' ','+')} ")

    print(f"\nNext steps for HIGH INTEREST candidates:")
    print(f"  1. Discard any pair with sep < 1.5 arcsec -- ZTF duplicate IDs, not real pairs")
    print(f"  2. For remaining candidates, open DECaLS viewer link above")
    print(f"     Look for two compact sources or an arc near the anchor")
    print(f"  3. Check SIMBAD link — is this a known lens already?")
    print(f"  4. If unknown + DECaLS shows two sources — candidate for a discovery paper")


if __name__=="__main__":
    main()