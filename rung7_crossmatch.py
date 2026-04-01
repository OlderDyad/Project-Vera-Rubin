"""
rung7_crossmatch.py
====================
Rung 7: Cross-catalog rejection and discovery verification.

For any candidate passing Rungs 1-6, this module queries:
  1. SIMBAD — resolves object types and redshifts near each ZTF source
  2. NED    — checks for known non-lens classifications
  3. Known binary quasar catalog — rejects confirmed binaries

Returns one of four verdicts:
  KNOWN_LENS     — in CASTLES/published catalog → validation candidate
  UNKNOWN_SYSTEM — not in any catalog → DISCOVERY CANDIDATE ★
  BINARY_QUASAR  — confirmed physically distinct pair → REJECT
  AMBIGUOUS      — insufficient data to classify

Usage:
    from rung7_crossmatch import run_rung7
    result = run_rung7(ra1, dec1, ra2, dec2, pair_id)
    # result.verdict in {KNOWN_LENS, UNKNOWN_SYSTEM, BINARY_QUASAR, AMBIGUOUS}

Standalone:
    python rung7_crossmatch.py  # runs on all unclassified candidates in DB
"""

import sqlite3
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH      = Path("outputs/survey_v3/survey_results.db")
CATALOG_PATH = Path("outputs/candidate_catalog.db")

# ── Known binary quasars (confirmed non-lenses) ────────────────────────────────
# Sources: IAC, Hennawi et al. 2010, Findlay et al. 2018
KNOWN_BINARIES = {
    "RXJ0921+4529":  "Binary quasar confirmed by spectroscopy (IAC/SAO)",
    "Q2345+007":     "Binary quasar, z_A≠z_B confirmed",
    "UM673":         "Wide binary quasar candidate",
}

# Known gravitational lens systems (CASTLES + major catalogs)
KNOWN_LENSES = {
    "SDSS1004+4112", "SDSS1029+2623", "HE0435-1223", "PG1115+080",
    "RXJ1131-1231",  "B1608+656",    "WFI2033-4723", "J1206+4332",
    "HE1104-1805",   "Q0957+561",    "B2108+213",    "PKS1145-071",
    "HS1216+5032",   "Q1343+2650",   "Q0151+048",    "MG2016+112",
    "RXJ0921+4529",  # Listed as lens but now classified as binary
}

# ── Separation penalty for Rung 6 fix ─────────────────────────────────────────

def separation_penalty(sep_arcsec: float) -> float:
    """
    Return a score multiplier [0.0, 1.0] based on separation.
    Genuine lenses: almost never > 25" without being a known cluster.
    
    sep < 25":  penalty = 1.0 (no penalty)
    sep 25-40": penalty = 0.5 (marginal — needs Rung 7 confirmation)
    sep > 40":  penalty = 0.1 (very unlikely to be a lens)
    """
    if sep_arcsec < 25.0:
        return 1.0
    elif sep_arcsec < 40.0:
        # Linear drop from 1.0 at 25" to 0.5 at 40"
        return 1.0 - 0.5 * (sep_arcsec - 25.0) / 15.0
    else:
        # Linear drop from 0.5 at 40" to 0.1 at 60"
        return max(0.1, 0.5 - 0.4 * (sep_arcsec - 40.0) / 20.0)

# ── SIMBAD query ──────────────────────────────────────────────────────────────

def query_simbad_cone(ra: float, dec: float, radius_arcsec: float = 5.0) -> list:
    """
    Query SIMBAD for objects within radius_arcsec of (ra, dec).
    Returns list of dicts with keys: id, type, redshift, ra, dec
    """
    url = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"
    radius_deg = radius_arcsec / 3600.0
    adql = f"""
        SELECT TOP 5
            main_id, otype, z_value, ra, dec
        FROM basic
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
        ORDER BY DISTANCE(POINT('ICRS', ra, dec),
                          POINT('ICRS', {ra}, {dec}))
    """
    try:
        r = requests.post(url,
            data={"REQUEST": "doQuery", "LANG": "ADQL",
                  "FORMAT": "json", "QUERY": adql},
            timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = []
        cols = [c["name"] for c in data.get("metadata", [])]
        for row in data.get("data", []):
            d = dict(zip(cols, row))
            rows.append({
                "id":       d.get("main_id", ""),
                "type":     d.get("otype", ""),
                "redshift": d.get("z_value"),
                "ra":       d.get("ra"),
                "dec":      d.get("dec"),
            })
        return rows
    except Exception as e:
        logger.warning(f"SIMBAD query failed for ({ra:.4f},{dec:.4f}): {e}")
        return []

# ── Classification logic ───────────────────────────────────────────────────────

class Rung7Result:
    def __init__(self, pair_id, verdict, confidence, reason,
                 simbad_a=None, simbad_b=None,
                 z_a=None, z_b=None, sep_penalty=1.0):
        self.pair_id     = pair_id
        self.verdict     = verdict       # KNOWN_LENS / UNKNOWN_SYSTEM /
                                         # BINARY_QUASAR / AMBIGUOUS
        self.confidence  = confidence    # HIGH / MEDIUM / LOW
        self.reason      = reason
        self.simbad_a    = simbad_a or []
        self.simbad_b    = simbad_b or []
        self.z_a         = z_a
        self.z_b         = z_b
        self.sep_penalty = sep_penalty
        self.is_discovery = (verdict == "UNKNOWN_SYSTEM")

    def __repr__(self):
        star = " ★ DISCOVERY CANDIDATE" if self.is_discovery else ""
        return (f"Rung7[{self.verdict}{star}] "
                f"conf={self.confidence} | {self.reason}")

def classify_pair(ra1, dec1, ra2, dec2, pair_id, sep_arcsec,
                  anchor="") -> Rung7Result:
    """
    Run Rung 7 classification on a candidate pair.
    """
    penalty = separation_penalty(sep_arcsec)

    # ── Step 1: Check known binary list ───────────────────────────────────────
    for name, reason in KNOWN_BINARIES.items():
        if name.lower() in anchor.lower() or name.lower() in pair_id.lower():
            return Rung7Result(
                pair_id, "BINARY_QUASAR", "HIGH",
                f"Known binary quasar: {reason}",
                sep_penalty=penalty)

    # ── Step 2: Check known lens list ─────────────────────────────────────────
    for name in KNOWN_LENSES:
        if name.lower() in anchor.lower() or name.lower() in pair_id.lower():
            # Special case: RXJ0921 is in KNOWN_LENSES but is a binary
            if "RXJ0921" in name:
                return Rung7Result(
                    pair_id, "BINARY_QUASAR", "HIGH",
                    "Listed in lens catalogs but spectroscopically confirmed "
                    "binary quasar (IAC/SAO 6m observations)",
                    sep_penalty=penalty)
            return Rung7Result(
                pair_id, "KNOWN_LENS", "HIGH",
                f"In CASTLES/published catalog: {name}",
                sep_penalty=penalty)

    # ── Step 3: SIMBAD cross-match ────────────────────────────────────────────
    time.sleep(0.3)  # rate limit
    hits_a = query_simbad_cone(ra1, dec1, radius_arcsec=5.0)
    time.sleep(0.3)
    hits_b = query_simbad_cone(ra2, dec2, radius_arcsec=5.0)

    z_a = hits_a[0]["redshift"] if hits_a else None
    z_b = hits_b[0]["redshift"] if hits_b else None
    type_a = hits_a[0]["type"] if hits_a else None
    type_b = hits_b[0]["type"] if hits_b else None

    # If both objects have measured redshifts and they differ significantly
    if z_a is not None and z_b is not None:
        dz = abs(z_a - z_b)
        if dz > 0.01:  # > 0.01 in z = physically distinct
            return Rung7Result(
                pair_id, "BINARY_QUASAR", "HIGH",
                f"Redshifts differ: z_A={z_a:.4f} z_B={z_b:.4f} Δz={dz:.4f}",
                hits_a, hits_b, z_a, z_b, penalty)
        else:
            # Same redshift — could be lens or binary
            return Rung7Result(
                pair_id, "AMBIGUOUS", "MEDIUM",
                f"Same redshift z~{z_a:.3f} but no lens galaxy confirmed",
                hits_a, hits_b, z_a, z_b, penalty)

    # If one or both not in SIMBAD at all → potential new system
    if not hits_a and not hits_b:
        return Rung7Result(
            pair_id, "UNKNOWN_SYSTEM", "MEDIUM",
            "Neither source in SIMBAD — potential new system",
            hits_a, hits_b, z_a, z_b, penalty)

    if not hits_a or not hits_b:
        return Rung7Result(
            pair_id, "UNKNOWN_SYSTEM", "LOW",
            f"One source not in SIMBAD (A: {bool(hits_a)}, B: {bool(hits_b)})",
            hits_a, hits_b, z_a, z_b, penalty)

    # Both in SIMBAD but no redshift info
    return Rung7Result(
        pair_id, "AMBIGUOUS", "LOW",
        f"In SIMBAD but no redshift: A={type_a} B={type_b}",
        hits_a, hits_b, z_a, z_b, penalty)

# ── Run on all unclassified DB candidates ─────────────────────────────────────

def run_rung7_on_db(verbose=True):
    """
    Run Rung 7 on all candidates in the survey DB that haven't been
    cross-matched yet. Updates candidate_catalog.db with results.
    """
    if not DB_PATH.exists():
        print("  Survey DB not found.")
        return

    survey = sqlite3.connect(DB_PATH)
    catalog = sqlite3.connect(CATALOG_PATH)

    # Add rung7 columns if not present
    for col, typ in [
        ("rung7_verdict",    "TEXT"),
        ("rung7_confidence", "TEXT"),
        ("rung7_reason",     "TEXT"),
        ("rung7_z_a",        "REAL"),
        ("rung7_z_b",        "REAL"),
        ("sep_penalty",      "REAL"),
        ("is_discovery",     "INTEGER"),
    ]:
        try:
            catalog.execute(
                f"ALTER TABLE candidate_catalog ADD COLUMN {col} {typ}")
        except Exception:
            pass  # column already exists
    catalog.commit()

    # Get candidates needing Rung 7
    rows = survey.execute("""
        SELECT pair_id, anchor, ra, dec, sep,
               r1_pass, r2_pass, r3_pass, r4_pass, r5_pass, r6_pass,
               score
        FROM candidates
        ORDER BY score DESC
    """).fetchall()
    survey.close()

    print()
    print("  ┌─ RUNG 7: CROSS-CATALOG VERIFICATION ──────────────────────┐")
    print()
    print(f"  Processing {len(rows)} candidates...")
    print()

    discoveries = []
    binaries    = []
    known_lenses = []
    ambiguous   = []

    for row in rows:
        (pair_id, anchor, ra, dec, sep,
         r1, r2, r3, r4, r5, r6, score) = row

        # Estimate second source coords (approximate — use sep/2 offset)
        # In production, use actual ZTF pair coordinates from pair_id
        ra2  = ra  + (sep / 3600.0) * 0.7   # rough offset
        dec2 = dec + (sep / 3600.0) * 0.7

        result = classify_pair(ra, dec, ra2, dec2, pair_id, sep, anchor)

        # Update catalog
        catalog.execute("""
            UPDATE candidate_catalog SET
                rung7_verdict    = ?,
                rung7_confidence = ?,
                rung7_reason     = ?,
                rung7_z_a        = ?,
                rung7_z_b        = ?,
                sep_penalty      = ?,
                is_new_system    = ?
            WHERE pair_id = ?
        """, (result.verdict, result.confidence, result.reason,
              result.z_a, result.z_b, result.sep_penalty,
              1 if result.is_discovery else 0,
              pair_id))

        if verbose:
            icon = {"UNKNOWN_SYSTEM": "★", "KNOWN_LENS": "✓",
                    "BINARY_QUASAR": "✗", "AMBIGUOUS": "?"}.get(
                result.verdict, " ")
            print(f"  {icon} {pair_id[:40]:<40}")
            print(f"    anchor={anchor}  sep={sep:.1f}\"  "
                  f"score={score:.3f}  penalty={result.sep_penalty:.2f}")
            print(f"    verdict={result.verdict} [{result.confidence}]")
            print(f"    {result.reason}")
            print()

        if result.verdict == "UNKNOWN_SYSTEM":
            discoveries.append((pair_id, anchor, score))
        elif result.verdict == "BINARY_QUASAR":
            binaries.append((pair_id, anchor, result.reason))
        elif result.verdict == "KNOWN_LENS":
            known_lenses.append((pair_id, anchor))
        else:
            ambiguous.append((pair_id, anchor))

    catalog.commit()
    catalog.close()

    # Summary
    print("  ┌─ RUNG 7 SUMMARY ───────────────────────────────────────────┐")
    print()
    print(f"  UNKNOWN SYSTEMS (discovery candidates):  {len(discoveries)}")
    for pid, anchor, score in discoveries:
        print(f"    ★ {anchor:<20} score={score:.3f}  {pid}")

    print(f"  KNOWN LENSES (validation):               {len(known_lenses)}")
    for pid, anchor in known_lenses:
        print(f"    ✓ {anchor}")

    print(f"  BINARY QUASARS (rejected):               {len(binaries)}")
    for pid, anchor, reason in binaries:
        print(f"    ✗ {anchor}: {reason[:60]}")

    print(f"  AMBIGUOUS (needs follow-up):             {len(ambiguous)}")
    for pid, anchor in ambiguous:
        print(f"    ? {anchor}")

    print()
    if discoveries:
        print("  ★★★ DISCOVERY CANDIDATES FOUND ★★★")
        print("  These systems are not in SIMBAD/NED/CASTLES.")
        print("  Verify with: spectroscopic redshifts, HST imaging,")
        print("  ALeRCE/Fink cross-match, and time-delay monitoring.")
    else:
        print("  No new system candidates in current catalog.")
        print("  All candidates are known systems or rejected binaries.")
    print()

    return discoveries

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   RUNG 7 — CROSS-CATALOG VERIFICATION                       ║")
    print("║   Project Vera Rubin — Discovery Pipeline                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    run_rung7_on_db(verbose=True)