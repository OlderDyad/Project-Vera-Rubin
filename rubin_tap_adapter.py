"""
rubin_tap_adapter.py — Drop-in replacement for ALeRCE fetch functions.
Targets the Rubin Science Platform TAP service (DP0.2 / LSST Operations).

Authentication:
  Store your RSP token in ~/.rsp-tap.token (one line, no newline)
  Generate at: https://data.lsst.cloud/auth/tokens

Usage:
  from rubin_tap_adapter import fetch_objects_tap, fetch_lc_tap

  # Find objects near a sky position
  objects = fetch_objects_tap(ra=157.306, dec=26.392, radius_arcsec=60,
                              min_nobs=30, max_results=8)

  # Fetch light curve for one object
  lc = fetch_lc_tap(dia_object_id=12345678901234567)
"""

import os
import math
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pyvo
    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

RSP_TAP_URL     = "https://data.lsst.cloud/api/tap"
TOKEN_FILE      = Path.home() / ".rsp-tap.token"

DIAOBJECT_TABLE = "dp02_dc2_catalogs.DiaObject"
FORCEDSOURCE_TABLE = "dp02_dc2_catalogs.ForcedSourceOnDiaObject"
CCDVISIT_TABLE     = "dp02_dc2_catalogs.CcdVisit"

BAND_TO_FID = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}
FID_TO_BAND = {v: k for k, v in BAND_TO_FID.items()}

LSST_ZP_NJY = 31.4   # AB mag zero-point for nanoJansky fluxes

# ── Authentication ─────────────────────────────────────────────────────────────

def _load_token() -> Optional[str]:
    """Load RSP bearer token from ~/.rsp-tap.token"""
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    env_token = os.environ.get("RSP_TAP_TOKEN")
    if env_token:
        return env_token
    return None


def get_tap_service() -> "pyvo.dal.TAPService":
    """Return an authenticated pyvo TAPService for the RSP."""
    if not HAS_PYVO:
        raise ImportError(
            "pyvo is required.\n"
            "Install with: pip install pyvo"
        )
    token = _load_token()
    if token is None:
        raise FileNotFoundError(
            f"RSP token not found.\n"
            f"  1. Generate a token at: https://data.lsst.cloud/auth/tokens\n"
            f"  2. Save it to: {TOKEN_FILE}\n"
            f"  Or set env var RSP_TAP_TOKEN=<your-token>"
        )
    # pyvo 1.5+: pass Authorization header via session
    import requests
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService(RSP_TAP_URL, session=session)


# ── Object Search ──────────────────────────────────────────────────────────────

def fetch_objects_tap(
    ra: float,
    dec: float,
    radius_arcsec: float = 60.0,
    min_nobs: int = 30,
    max_results: int = 8,
    timeout: int = 30,
) -> list:
    """
    Find DiaObjects near a sky position, sorted by observation count descending.
    Drop-in replacement for ALeRCE /objects endpoint.

    Returns list of dicts with keys: oid, meanra, meandec, ndet
    (same keys as ALeRCE response for compatibility with rubin_survey_v3.py)

    NOTE: DP0.2 DiaObject table uses 'ra' and 'decl' (not 'dec').
    """
    service = get_tap_service()
    radius_deg = radius_arcsec / 3600.0

    adql = f"""
        SELECT
            diaObjectId   AS oid,
            ra            AS meanra,
            decl          AS meandec,
            nDiaSources   AS ndet
        FROM {DIAOBJECT_TABLE}
        WHERE CONTAINS(
            POINT('ICRS', ra, decl),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
        AND nDiaSources >= {min_nobs}
        ORDER BY nDiaSources DESC
    """.strip()

    logger.debug("TAP object query: %s", adql)
    t0 = time.time()
    results = service.search(adql, maxrec=max_results)
    elapsed = time.time() - t0
    logger.debug("TAP object query returned %d rows in %.1fs", len(results), elapsed)

    rows = []
    for row in results:
        rows.append({
            "oid":     str(row["oid"]),
            "meanra":  float(row["meanra"]),
            "meandec": float(row["meandec"]),
            "ndet":    int(row["ndet"]),
        })
    return rows


# ── Light Curve Fetch ──────────────────────────────────────────────────────────

def fetch_lc_tap(
    dia_object_id,
    bands: Optional[list] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch the difference-image light curve for a DiaObject.
    Drop-in replacement for ALeRCE /objects/{oid}/lightcurve endpoint.

    Returns DataFrame with columns:
        mjd, mag, mag_err, flux, flux_err, band

    NOTE: DP0.2 DiaSource uses 'midPointMjdTai' for time,
          'psfFlux' / 'psfFluxErr' in nanoJanskys.
    """
    service = get_tap_service()

    band_filter = ""
    if bands:
        band_list = ", ".join(f"'{b}'" for b in bands)
        band_filter = f"AND band IN ({band_list})"

    adql = f"""
        SELECT
            cv.expMidptMJD       AS mjd,
            fsodo.psfFlux        AS flux_njy,
            fsodo.psfFluxErr     AS flux_njy_err,
            fsodo.band
        FROM dp02_dc2_catalogs.ForcedSourceOnDiaObject AS fsodo
        JOIN dp02_dc2_catalogs.CcdVisit AS cv
            ON cv.ccdVisitId = fsodo.ccdVisitId
        WHERE fsodo.diaObjectId = {dia_object_id}
        {band_filter}
        ORDER BY cv.expMidptMJD
    """.strip()

    logger.debug("TAP LC query for %s", dia_object_id)
    t0 = time.time()
    results = service.search(adql, maxrec=50000)
    elapsed = time.time() - t0
    logger.debug("TAP LC query returned %d rows in %.1fs", len(results), elapsed)

    if len(results) == 0:
        return pd.DataFrame(
            columns=["mjd", "mag", "mag_err", "flux", "flux_err", "band"])

    df = results.to_table().to_pandas()

    # Rename time column if needed
    if "midPointMjdTai" in df.columns:
        df = df.rename(columns={"midPointMjdTai": "mjd"})

    # Convert nJy fluxes → AB magnitudes
    flux_njy     = df["flux_njy"].values.astype(float)
    flux_njy_err = df["flux_njy_err"].values.astype(float)

    pos_mask     = flux_njy > 0
    mag          = np.full(len(df), np.nan)
    mag_err      = np.full(len(df), np.nan)
    flux_out     = np.full(len(df), np.nan)
    flux_err_out = np.full(len(df), np.nan)

    mag[pos_mask]     = -2.5 * np.log10(flux_njy[pos_mask]) + LSST_ZP_NJY
    mag_err[pos_mask] = (2.5 / np.log(10)) * (
        flux_njy_err[pos_mask] / flux_njy[pos_mask])

    # Normalise to ZP=25 for compatibility with rubin_survey_v3.py
    flux_out[pos_mask]     = 10 ** (-0.4 * (mag[pos_mask] - 25.0))
    flux_err_out[pos_mask] = (flux_out[pos_mask] * mag_err[pos_mask]
                               * 0.4 * np.log(10))

    df["mag"]      = mag
    df["mag_err"]  = mag_err
    df["flux"]     = flux_out
    df["flux_err"] = flux_err_out

    # Map LSST band string → int fid (ZTF-compatible downstream)
    df["band"] = df["band"].map(BAND_TO_FID).fillna(-1).astype(int)

    # Drop rows with negative/zero flux (difference imaging artefacts)
    df = df[pos_mask].copy()
    df = df[["mjd", "mag", "mag_err", "flux", "flux_err",
             "band"]].reset_index(drop=True)

    return df


# ── Compatibility shims ────────────────────────────────────────────────────────

def fetch_objects_rubin(ra, dec, radius, page_size=8, timeout=30):
    """Signature-compatible with ALeRCE fetch_objects()."""
    return fetch_objects_tap(
        ra=ra, dec=dec,
        radius_arcsec=radius,
        min_nobs=30,
        max_results=page_size,
        timeout=timeout,
    )


def fetch_lc_rubin(oid, timeout=30):
    """Signature-compatible with ALeRCE fetch_lc()."""
    return fetch_lc_tap(int(oid), timeout=timeout)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def test_connection():
    """Quick connectivity test — run once after saving token."""
    print(f"Token file: {TOKEN_FILE}")
    token_found = TOKEN_FILE.exists() and bool(TOKEN_FILE.read_text().strip())
    print(f"Token found: {token_found}")
    print(f"TAP URL: {RSP_TAP_URL}")
    print()

    service = get_tap_service()

    # List available schemas
    print("Querying available schemas...")
    r = service.search("SELECT schema_name FROM tap_schema.schemas", maxrec=20)
    schemas = [row["schema_name"] for row in r]
    print(f"Schemas: {schemas}")
    print()

    # Cone search near DC2 centre (RA=61.86, Dec=-35.79)
    print("Cone search: DC2 centre, 5-arcmin radius, nDiaSources >= 50 ...")
    objs = fetch_objects_tap(ra=61.86, dec=-35.79,
                              radius_arcsec=300, min_nobs=50, max_results=5)
    print(f"Found {len(objs)} DiaObjects:")
    for o in objs:
        print(f"  oid={o['oid']}  ra={o['meanra']:.4f}"
              f"  dec={o['meandec']:.4f}  ndet={o['ndet']}")

    if objs:
        oid = int(objs[0]["oid"])
        print(f"\nFetching light curve for oid={oid} ...")
        lc = fetch_lc_tap(oid)
        print(f"  {len(lc)} positive-flux detections")
        if len(lc):
            bands = dict(lc["band"].value_counts().items())
            print(f"  band counts (int fid): {bands}")
            print(f"  MJD range: {lc['mjd'].min():.1f} – {lc['mjd'].max():.1f}")

    print("\n✓ Connection test complete — Rubin DP0.2 is live.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_connection()
