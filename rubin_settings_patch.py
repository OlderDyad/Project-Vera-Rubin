"""
rubin_settings_patch.py
========================
Shows the two SETTINGS blocks for ZTF vs Rubin modes.

The survey engine auto-detects which mode to use based on
the --survey flag, or you can manually set SURVEY_MODE.

Usage:
    python rubin_survey_v3.py --survey ztf    # current ZTF mode
    python rubin_survey_v3.py --survey rubin  # Rubin mode (when RSP approved)
"""

# ── ZTF mode (current) ────────────────────────────────────────────────────────
SETTINGS_ZTF = {
    "min_score":   0.28,
    "min_rungs":   2,
    "radius":      60.0,   # arcsec search radius
    "min_sep":     0.5,    # ZTF astrometric jitter floor
    "pair_sep":    50.0,   # arcsec ceiling
    "min_det":     30,     # minimum detections per object
    "lag_range":   (-900, 900),     # days — covers all ZTF-accessible delays
    "zdcf_bin":    5.0,             # day bins — validated on real ZTF data
    "timeout":     20,
    "max_obj":     8,
}

# ── Rubin LSST mode (when RSP account approved) ───────────────────────────────
SETTINGS_RUBIN = {
    "min_score":   0.25,   # slightly lower — better data quality compensates
    "min_rungs":   2,
    "radius":      30.0,   # tighter — Rubin catalog is denser, fewer false pairs
    "min_sep":     0.1,    # Rubin resolves down to ~0.2" — opens compact lenses
    "pair_sep":    50.0,   # keep wide ceiling — still want cluster lenses too
    "min_det":     20,     # lower — Rubin cadence is nightly, builds up fast
    "lag_range":   (-1500, 1500),   # wider — some cluster delays exceed 900d
    "zdcf_bin":    2.0,             # finer — needed for short delays like HE0435 (14d)
    "timeout":     30,
    "max_obj":     12,     # more — Rubin catalog far denser per field
}

# ── Compact lens catalog (inaccessible to ZTF, primary Rubin targets) ─────────
COMPACT_LENS_CATALOG = {
    # System: (ra, dec, z_lens, z_source, sep_arcsec, published_delay_d, type)
    "HE0435-1223":   (69.5621,  -12.2873, 0.46, 1.693, 1.46,  14.0,  "galaxy_quad"),
    "PG1115+080":    (169.5717,   7.7658, 0.31, 1.722, 1.79,  23.7,  "galaxy_quad"),
    "RXJ1131-1231":  (173.1025, -12.5319, 0.29, 0.657, 1.83,  91.4,  "galaxy_quad"),
    "B1608+656":     (242.7071,  65.5686, 0.63, 1.394, 1.36,  77.0,  "galaxy_quad"),
    "WFI2033-4723":  (308.8229, -47.3972, 0.66, 1.662, 1.66, 125.0,  "galaxy_quad"),
    "J1206+4332":    (181.4967,  43.5386, 0.75, 1.789, 1.37, 111.3,  "galaxy_double"),
    "HE1104-1805":   (166.6383, -18.3556, 0.73, 2.316, 3.19, 162.2,  "galaxy_double"),
    "SDSS0246-0825": ( 41.6958,  -8.4306, 0.72, 1.683, 1.04,  None,  "galaxy_double"),
    "HS2209+1914":   (332.8975,  19.4769, 0.51, 1.070, 1.20,  None,  "galaxy_double"),
    # Wide-separation lenses (already in ZTF catalog, keeping for continuity)
    "SDSS1004+4112": (151.065,   41.209,  0.68, 1.734, 14.52, 821.0, "cluster_quad"),
    "SDSS1029+2623": (157.306,   26.392,  0.584,2.197, 22.37, 744.0, "cluster_quad"),
}

print("Compact lens catalog for Rubin mode:")
print(f"{'System':<18} {'Sep\"':>5} {'z_L':>5} {'Delay(d)':>10} {'Type':<15}")
print("─"*60)
for name, info in COMPACT_LENS_CATALOG.items():
    ra,dec,zl,zs,sep,delay,ltype = info
    delay_str = f"{delay:.0f}" if delay else "unknown"
    print(f"{name:<18} {sep:>5.2f} {zl:>5.2f} {delay_str:>10} {ltype:<15}")

print()
print("To use Rubin mode: change SETTINGS to SETTINGS_RUBIN")
print("and add COMPACT_LENS_CATALOG seeds when RSP account is active")