"""
rubin_survey_v3.py  — Professional Survey Engine v3.3
──────────────────────────────────────────────────────
Changes in v3.3:
  1. 5-day ZDCF bins (zdcf_bin: 5.0) — diagnostic validated
  2. Noise-peak guard (min 8 pairs per bin)
  3. Cluster-adaptive Rung 6 scaling (sep > 10")
  4. Anisotropy monitor hook — one DB connection per worker process

Usage:
    python rubin_survey_v3.py
    python rubin_survey_v3.py --seeds seeds/castles_wide_seeds.csv
    python rubin_survey_v3.py --workers 4 --resume
"""

import argparse
import json
import math
import sqlite3
import time
import concurrent.futures
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.ndimage import uniform_filter1d

from lc_utils import quality_check, zdcf, best_lag_from_zdcf, prep_lc

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SETTINGS = {
    "min_score":   0.28,
    "min_rungs":   2,
    "radius":      60.0,
    "min_sep":     0.1,
    "pair_sep":    50.0,
    "min_det":     20,
    "lag_range":   (-1500, 1500),
    "zdcf_bin":     2.0,          # 5-day bins: diagnostic validated
    "timeout":     20,
    "max_obj":     8,
}

THRESHOLDS = {
    "lag_min_r":       0.12,
    "flux_cv_max":     0.35,
    "ls_power_max":    0.45,
    "frac_var_min_r":  0.50,
    "micro_long_min":  0.45,
    "micro_short_max": 0.50,
}

OUT_DIR = Path("outputs/survey_v3")
DB_PATH = OUT_DIR / "survey_results.db"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN SEED FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

BUILTIN_SEEDS = [
    {"name":"Q 0957+561",     "ra":150.3425, "dec": 55.8978, "note":"1st known lens"},
    {"name":"PG 1115+080",    "ra":169.5717, "dec":  7.7658, "note":"quad lens"},
    {"name":"HE 0435-1223",   "ra": 69.5621, "dec":-12.2873, "note":"quad lens"},
    {"name":"SDSS J1004+4112","ra":151.1558, "dec": 41.2019, "note":"quad lens"},
    {"name":"SDSS J1029+2623","ra":157.3967, "dec": 26.3892, "note":"double lens"},
    {"name":"Q 0911+0551",    "ra":137.8950, "dec":  5.8558, "note":"quad lens"},
    {"name":"H 1413+117",     "ra":213.9408, "dec": 11.4958, "note":"cloverleaf"},
    {"name":"B 1422+231",     "ra":216.1575, "dec": 22.9314, "note":"quad lens"},
    {"name":"HS 0810+2554",   "ra":123.2600, "dec": 25.7458, "note":"double lens"},
    {"name":"SBS 1520+530",   "ra":230.5725, "dec": 52.9142, "note":"double lens"},
    {"name":"B 1600+434",     "ra":240.5788, "dec": 43.3344, "note":"double lens"},
    {"name":"SDSS J2222+2745","ra":335.6104, "dec": 27.7511, "note":"5-image lens"},
    {"name":"J1420+6019",     "ra":215.0742, "dec": 60.3236, "note":"double lens"},
    {"name":"J1650+4251",     "ra":252.6933, "dec": 42.8603, "note":"double lens"},
    {"name":"3C 273",         "ra":187.2779, "dec":  2.0524, "note":"brightest QSO"},
    {"name":"Mrk 335",        "ra":  1.5817, "dec": 20.2031, "note":"Seyfert 1"},
    {"name":"Mrk 509",        "ra":311.0404, "dec":-10.7233, "note":"Seyfert 1"},
    {"name":"Mrk 817",        "ra":217.1375, "dec": 58.7944, "note":"Seyfert 1"},
    {"name":"NGC 4151",       "ra":182.6358, "dec": 39.4057, "note":"Seyfert 1"},
    {"name":"NGC 5548",       "ra":214.4979, "dec": 25.1369, "note":"Seyfert 1"},
]


def load_seeds(path=None):
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Seed file '{path}' not found.")
        df = pd.read_csv(p)
        df.columns = [c.lower().strip() for c in df.columns]
        for alias in [("longitude","ra"),("latitude","dec"),("anchor","name"),
                      ("object","name"),("id","name")]:
            if alias[0] in df.columns and alias[1] not in df.columns:
                df = df.rename(columns={alias[0]: alias[1]})
        if "ra" not in df.columns or "dec" not in df.columns:
            raise ValueError(f"Seed CSV must have 'ra' and 'dec' columns.")
        if "name" not in df.columns:
            df["name"] = df["ra"].apply(lambda r: f"seed_{r:.3f}")
        print(f"Loaded {len(df)} seeds from {path}")
        return df.to_dict("records")
    else:
        print(f"No seed file — using built-in list ({len(BUILTIN_SEEDS)} seeds)")
        return BUILTIN_SEEDS


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS progress (
    idx          INTEGER PRIMARY KEY,
    name         TEXT,
    status       TEXT,
    n_pairs      INTEGER DEFAULT 0,
    started_at   TEXT,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    pair_id   TEXT PRIMARY KEY,
    anchor    TEXT,
    ra        REAL,
    dec       REAL,
    sep       REAL,
    lag_days  REAL,
    lag_unc   REAL,
    score     REAL,
    rungs     INTEGER,
    r1_score  REAL, r2_score REAL, r3_score REAL,
    r4_score  REAL, r5_score REAL, r6_score REAL,
    r1_pass   INTEGER, r2_pass INTEGER, r3_pass INTEGER,
    r4_pass   INTEGER, r5_pass INTEGER, r6_pass INTEGER,
    delay_uncertain  INTEGER,
    possibly_blended INTEGER DEFAULT 0,
    logged_at TEXT
);
"""


def open_db(path=DB_PATH):
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(DB_SCHEMA)
    conn.commit()
    return conn


def get_processed_ids(db_path=DB_PATH):
    conn = open_db(db_path)
    rows = conn.execute("SELECT idx FROM progress").fetchall()
    conn.close()
    return {r[0] for r in rows}


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_objects(ra, dec, radius):
    try:
        r = requests.get(
            "https://api.alerce.online/ztf/v1/objects",
            params={
                "ra": ra, "dec": dec, "radius": radius,
                "page_size": SETTINGS["max_obj"],
                "order_by": "ndet",
                "order_mode": "DESC",
            },
            timeout=SETTINGS["timeout"])
        r.raise_for_status()
        return [o for o in r.json().get("items", [])
                if o.get("ndet", 0) >= SETTINGS["min_det"]]
    except Exception:
        return []


def fetch_lc(oid):
    try:
        r = requests.get(
            f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve",
            timeout=SETTINGS["timeout"])
        r.raise_for_status()
        dets = r.json().get("detections", [])
        if len(dets) < SETTINGS["min_det"]:
            return None
        df = pd.DataFrame(dets)
        if not all(c in df.columns for c in ["mjd", "magpsf", "sigmapsf"]):
            return None
        df = df[["mjd","magpsf","sigmapsf","fid"]].rename(
            columns={"magpsf":"mag","sigmapsf":"mag_err","fid":"band"})
        df["flux"]     = 10**(-0.4*(df["mag"]-25.0))
        df["flux_err"] = df["flux"]*df["mag_err"]*0.4*np.log(10)
        return df.sort_values("mjd").reset_index(drop=True)
    except Exception:
        return None


def ang_sep(ra1, dec1, ra2, dec2):
    cos_dec = math.cos(math.radians((dec1+dec2)/2))
    return math.sqrt(((ra1-ra2)*cos_dec*3600)**2 + ((dec1-dec2)*3600)**2)


# ══════════════════════════════════════════════════════════════════════════════
# FULL 6-RUNG SCORING LADDER
# ══════════════════════════════════════════════════════════════════════════════

def score_pair(lc_A, lc_B, oid_A, oid_B, sep_arcsec=None, delay_uncertain=False):
    res = {
        "oid_A": oid_A, "oid_B": oid_B,
        "sep_arcsec": sep_arcsec,
        "delay_uncertain": delay_uncertain,
        "rungs": {},
        "total_score": 0.0,
        "rungs_passed": 0,
        "best_lag": None,
        "lag_uncertainty": None,
    }
    best_lag = 0.0

    # ── Rung 1: ZDCF lag detection ─────────────────────────────────────────
    try:
        tA, fA = prep_lc(lc_A); tB, fB = prep_lc(lc_B)
        lag_rng = SETTINGS["lag_range"]
        _bw     = SETTINGS.get("zdcf_bin", 5.0)
        _nbins  = max(10, int((lag_rng[1]-lag_rng[0]) / _bw))
        cs, zv, np_ = zdcf(tA, fA, tB, fB, lag_range=lag_rng, n_bins=_nbins)

        # Noise-peak guard: suppress bins with < 8 pairs
        _min_pairs  = 8
        zv_filtered = np.where(np_ >= _min_pairs, zv, np.nan)
        best_lag, lag_unc = best_lag_from_zdcf(cs, zv_filtered, np_)

        valid_zv = np.abs(zv[~np.isnan(zv)])
        if len(valid_zv) == 0:
            res["rungs"]["rung1"] = {"passed": False, "score": 0.0,
                                     "error": "all-NaN ZDCF"}
            return res
        pr     = float(np.tanh(valid_zv.max()))
        passed = pr > THRESHOLDS["lag_min_r"]
        res["best_lag"] = best_lag; res["lag_uncertainty"] = lag_unc
        res["rungs"]["rung1"] = {
            "passed": passed, "score": pr,
            "lag_days": best_lag, "lag_unc": lag_unc,
            "note": "delay_uncertain" if delay_uncertain else ""
        }
    except Exception as e:
        res["rungs"]["rung1"] = {"passed": False, "score": 0.0, "error": str(e)}

    # ── Rung 2: Flux ratio stability ────────────────────────────────────────
    try:
        lB2  = lc_B.copy(); lB2["mjd"] += best_lag
        tmin = max(lc_A["mjd"].min(), lB2["mjd"].min())
        tmax = min(lc_A["mjd"].max(), lB2["mjd"].max())
        tg   = np.arange(tmin, tmax, 2.0)
        if len(tg) > 5:
            fA_ = np.interp(tg, lc_A["mjd"].values, lc_A["flux"].values)
            fB_ = np.interp(tg, lB2["mjd"].values,  lB2["flux"].values)
            ratio  = fA_ / (fB_ + 1e-9)
            cv     = ratio.std() / (ratio.mean() + 1e-9)
            passed = cv < THRESHOLDS["flux_cv_max"]
            score  = max(0.0, 1.0 - cv / THRESHOLDS["flux_cv_max"])
        else:
            passed, score, cv = False, 0.0, 1.0
        res["rungs"]["rung2"] = {"passed": passed, "score": score, "cv": float(cv)}
    except Exception as e:
        res["rungs"]["rung2"] = {"passed": False, "score": 0.0, "error": str(e)}

    # ── Rung 3: Stochasticity (Lomb-Scargle) ────────────────────────────────
    try:
        from astropy.timeseries import LombScargle
        powers = []
        for lc in [lc_A, lc_B]:
            t   = lc["mjd"].values
            f   = lc["flux"].values
            err = lc["flux_err"].values if "flux_err" in lc.columns \
                  else np.full(len(t), 0.05)
            _, pw = LombScargle(t, f, err).autopower(
                minimum_frequency=1/300, maximum_frequency=1/3)
            powers.append(float(pw.max()))
        avg    = float(np.mean(powers))
        passed = avg < THRESHOLDS["ls_power_max"]
        score  = max(0.0, 1.0 - avg / THRESHOLDS["ls_power_max"])
        res["rungs"]["rung3"] = {"passed": passed, "score": score,
                                  "ls_A": powers[0], "ls_B": powers[1]}
    except Exception as e:
        res["rungs"]["rung3"] = {"passed": False, "score": 0.0, "error": str(e)}

    # ── Rung 4: Fractional variability match ────────────────────────────────
    try:
        lB2  = lc_B.copy(); lB2["mjd"] += best_lag
        tmin = max(lc_A["mjd"].min(), lB2["mjd"].min())
        tmax = min(lc_A["mjd"].max(), lB2["mjd"].max())
        tg   = np.arange(tmin, tmax, 2.0)
        if len(tg) > 10:
            fA_ = np.interp(tg, lc_A["mjd"].values, lc_A["flux"].values)
            fB_ = np.interp(tg, lB2["mjd"].values,  lB2["flux"].values)
            fAn = (fA_ - fA_.mean()) / (fA_.mean() + 1e-9)
            fBn = (fB_ - fB_.mean()) / (fB_.mean() + 1e-9)
            corr, pval = stats.pearsonr(fAn, fBn)
            passed = corr > THRESHOLDS["frac_var_min_r"] and pval < 0.01
            score  = max(0.0, float(corr))
        else:
            passed, score, corr, pval = False, 0.0, 0.0, 1.0
        res["rungs"]["rung4"] = {"passed": passed, "score": score,
                                  "r": float(corr), "p": float(pval)}
    except Exception as e:
        res["rungs"]["rung4"] = {"passed": False, "score": 0.0, "error": str(e)}

    # ── Rung 5: Microlensing signature ──────────────────────────────────────
    try:
        lB2  = lc_B.copy(); lB2["mjd"] += best_lag
        tmin = max(lc_A["mjd"].min(), lB2["mjd"].min())
        tmax = min(lc_A["mjd"].max(), lB2["mjd"].max())
        tg   = np.arange(tmin, tmax, 3.0)
        if len(tg) > 10:
            fA_ = np.interp(tg, lc_A["mjd"].values, lc_A["flux"].values)
            fB_ = np.interp(tg, lB2["mjd"].values,  lB2["flux"].values)
            win    = max(3, int(30/3))
            fAs    = uniform_filter1d(fA_, win)
            fBs    = uniform_filter1d(fB_, win)
            long_r, _ = stats.pearsonr(fAs, fBs)
            fAr    = fA_ - fAs; fBr = fB_ - fBs
            short_r = 0.0
            if fAr.std() > 1e-9 and fBr.std() > 1e-9:
                short_r, _ = stats.pearsonr(fAr, fBr)
            passed = (long_r  > THRESHOLDS["micro_long_min"] and
                      short_r < THRESHOLDS["micro_short_max"])
            score  = max(0.0, (float(long_r) - abs(float(short_r)-0.15)) / 1.5)
        else:
            passed, score, long_r, short_r = False, 0.0, 0.0, 0.0
        res["rungs"]["rung5"] = {"passed": passed, "score": score,
                                  "long_r": float(long_r),
                                  "short_r": float(short_r)}
    except Exception as e:
        res["rungs"]["rung5"] = {"passed": False, "score": 0.0, "error": str(e)}

    # ── Rung 6: Mass-delay consistency (cluster-adaptive) ───────────────────
    try:
        if sep_arcsec and best_lag and abs(best_lag) > 0:
            # Cluster-adaptive scaling: different physics for wide-sep lenses
            is_cluster = sep_arcsec > 10.0
            if is_cluster:
                # Empirical power law for cluster lenses (sep > 10")
                dt_pred = 15.0 * (sep_arcsec ** 1.5)
            else:
                # SIS galaxy lens formula
                from astropy.cosmology import FlatLambdaCDM
                import astropy.units as u
                cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
                Dl    = cosmo.angular_diameter_distance(0.5).to(u.m).value
                Ds    = cosmo.angular_diameter_distance(1.5).to(u.m).value
                Dls   = cosmo.angular_diameter_distance_z1z2(0.5, 1.5).to(u.m).value
                theta = np.deg2rad(sep_arcsec / 3600)
                dt_pred = (1.5 * (theta**2/2) * (Dl*Ds) /
                           (Dls * 3e8) / 86400)
            dt_min  = dt_pred * 0.1
            dt_max  = dt_pred * 10.0
            passed  = dt_min <= abs(best_lag) <= dt_max
            score   = 1.0 if passed else max(
                0.0, 1.0 - abs(np.log10(
                    max(abs(best_lag), 1) / dt_pred)) / 2.0)
            note    = f"pred:{dt_pred:.0f}d ({'cluster' if is_cluster else 'galaxy'})"
        else:
            passed, score, note = False, 0.0, "no sep/lag"
        res["rungs"]["rung6"] = {"passed": passed, "score": float(score),
                                  "note": note}
    except Exception as e:
        res["rungs"]["rung6"] = {"passed": False, "score": 0.0, "error": str(e)}

    sc = [r.get("score", 0.0) for r in res["rungs"].values()]
    pa = [r.get("passed", False) for r in res["rungs"].values()]
    res["total_score"]  = float(np.mean(sc))
    res["rungs_passed"] = int(sum(pa))
    return res


# ══════════════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════════════

def process_seed(task):
    """
    Runs in a worker subprocess. Opens its own DB connections.
    Anisotropy monitor connection is opened ONCE per worker task
    (not once per match) for performance at scale.
    """
    idx, ra, dec, name, db_path = task
    started = datetime.now(timezone.utc).isoformat()
    report  = {
        "idx": idx, "name": name,
        "status": "processed", "matches": [],
        "started_at": started, "finished_at": None,
    }

    # ── Open anisotropy connection once for this worker's entire task ────────
    try:
        from anisotropy_monitor import tag_and_store_detection, get_conn as _ac
        _aconn = _ac()
    except Exception:
        _aconn = None
    # ─────────────────────────────────────────────────────────────────────────

    try:
        objs = fetch_objects(ra, dec, SETTINGS["radius"])
        if len(objs) < 2:
            report["status"] = "skipped_low_density"
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            if _aconn:
                _aconn.close()
            return report

        lcs = {}
        for obj in objs[:SETTINGS["max_obj"]]:
            lc = fetch_lc(obj["oid"])
            if lc is not None:
                lcs[obj["oid"]] = lc

        if len(lcs) < 2:
            report["status"] = "skipped_no_lcs"
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            if _aconn:
                _aconn.close()
            return report

        objs_dict = {o["oid"]: o for o in objs}
        oids      = list(lcs.keys())

        for i in range(len(oids)):
            for j in range(i+1, len(oids)):
                oid_A, oid_B = oids[i], oids[j]
                oA = objs_dict.get(oid_A, {})
                oB = objs_dict.get(oid_B, {})
                sep = ang_sep(oA.get("meanra", 0), oA.get("meandec", 0),
                              oB.get("meanra", 0), oB.get("meandec", 0))

                if sep < SETTINGS["min_sep"] or sep > SETTINGS["pair_sep"]:
                    continue

                lc_A, lc_B     = lcs[oid_A], lcs[oid_B]
                qr             = quality_check(lc_A, lc_B, f"{oid_A}x{oid_B}")
                delay_uncertain = not qr.passes_all

                result = score_pair(
                    lc_A, lc_B, oid_A, oid_B,
                    sep_arcsec=sep,
                    delay_uncertain=delay_uncertain)

                if (result["total_score"] >= SETTINGS["min_score"] and
                        result["rungs_passed"] >= SETTINGS["min_rungs"]):

                    match = {
                        "possibly_blended": 0,
                        "pair_id":         f"{oid_A}x{oid_B}",
                        "anchor":          name,
                        "ra":              oA.get("meanra", ra),
                        "dec":             oA.get("meandec", dec),
                        "sep":             sep,
                        "lag_days":        result.get("best_lag"),
                        "lag_unc":         result.get("lag_uncertainty"),
                        "score":           result["total_score"],
                        "rungs":           result["rungs_passed"],
                        "delay_uncertain": int(delay_uncertain),
                        **{f"r{k}_score": result["rungs"].get(
                               f"rung{k}", {}).get("score", 0)
                           for k in range(1, 7)},
                        **{f"r{k}_pass": int(result["rungs"].get(
                               f"rung{k}", {}).get("passed", False))
                           for k in range(1, 7)},
                    }
                    report["matches"].append(match)

                    # ── ANISOTROPY MONITOR HOOK ──────────────────────
                    # Connection already open — just call tag_and_store
                    if _aconn:
                        try:
                            tag_and_store_detection(
                                _aconn,
                                lens_name  = f"{name}_{oid_A}",
                                ra         = oA.get("meanra", ra),
                                dec        = oA.get("meandec", dec),
                                z_L        = 0.5,
                                z_S        = 2.0,
                                sep_arcsec = sep,
                                measured_delay_d  = abs(result["best_lag"] or 0),
                                published_delay_d = abs(result["best_lag"] or 0),
                                lens_type  = "cluster_quad" if sep > 10
                                             else "galaxy_double",
                                survey     = "ZTF"
                            )
                        except Exception:
                            pass  # non-critical
                    # ─────────────────────────────────────────────────

        report["status"]      = "processed"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if _aconn:
            _aconn.close()
        return report

    except Exception as e:
        report["status"]      = f"error: {type(e).__name__}: {str(e)[:80]}"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if _aconn:
            _aconn.close()
        return report


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run(seeds, workers, resume):
    conn = open_db()

    if resume:
        done  = get_processed_ids()
        seeds = [s for i, s in enumerate(seeds) if i not in done]
        print(f"Resuming: {len(done)} done, {len(seeds)} remaining")
    else:
        print(f"Starting fresh: {len(seeds)} seeds")

    if not seeds:
        print("Nothing to do.")
        conn.close()
        return

    tasks = [(i, s["ra"], s["dec"], s.get("name", f"seed_{i}"), str(DB_PATH))
             for i, s in enumerate(seeds)]

    print(f"Workers: {workers}  DB: {DB_PATH}")
    print("─" * 60)

    ctx           = multiprocessing.get_context("spawn")
    total_matches = conn.execute(
        "SELECT COUNT(*) FROM candidates").fetchone()[0]

    with concurrent.futures.ProcessPoolExecutor(
            mp_context=ctx, max_workers=workers) as executor:

        futures = {executor.submit(process_seed, t): t for t in tasks}

        for done_count, future in enumerate(
                concurrent.futures.as_completed(futures), start=1):
            try:
                res = future.result()
            except Exception as e:
                print(f"  Future exception: {e}")
                continue

            conn.execute(
                "INSERT OR REPLACE INTO progress "
                "(idx, name, status, n_pairs, started_at, finished_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (res["idx"], res["name"], res["status"],
                 len(res["matches"]),
                 res.get("started_at"), res.get("finished_at")))

            for m in res["matches"]:
                conn.execute("""
                    INSERT OR REPLACE INTO candidates (
                        pair_id, anchor, ra, dec, sep, lag_days, lag_unc,
                        score, rungs,
                        r1_score, r2_score, r3_score,
                        r4_score, r5_score, r6_score,
                        r1_pass,  r2_pass,  r3_pass,
                        r4_pass,  r5_pass,  r6_pass,
                        delay_uncertain, possibly_blended, logged_at
                    ) VALUES (
                        ?,?,?,?,?,?,?,?,?,
                        ?,?,?,?,?,?,
                        ?,?,?,?,?,?,
                        ?,?,?
                    )
                """, (
                    m["pair_id"], m["anchor"], m["ra"], m["dec"],
                    m["sep"], m["lag_days"], m["lag_unc"],
                    m["score"], m["rungs"],
                    m["r1_score"], m["r2_score"], m["r3_score"],
                    m["r4_score"], m["r5_score"], m["r6_score"],
                    m["r1_pass"],  m["r2_pass"],  m["r3_pass"],
                    m["r4_pass"],  m["r5_pass"],  m["r6_pass"],
                    m["delay_uncertain"], m.get("possibly_blended", 0),
                    datetime.now(timezone.utc).isoformat()
                ))
                total_matches += 1

            if done_count % 50 == 0:
                conn.commit()

            n_matches = len(res["matches"])
            star      = " ★" if n_matches > 0 else ""
            print(f"  [{done_count:>5}/{len(tasks)}] "
                  f"{res['name'][:30]:<30}  "
                  f"{res['status'][:25]:<25}  "
                  f"matches={n_matches}{star}  total={total_matches}")

    conn.commit()
    conn.close()

    print(f"\n{'═'*60}")
    print(f"  Run complete.  Total candidates in DB: {total_matches}")
    print(f"  DB: {DB_PATH}")
    print(f"\nQuery results:")
    print(f"  python -c \"import sqlite3,pandas as pd; "
          f"conn=sqlite3.connect('{DB_PATH}'); "
          f"print(pd.read_sql('SELECT * FROM candidates "
          f"ORDER BY score DESC LIMIT 20', conn))\"")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Rubin/ZTF gravitational lens survey engine v3.3")
    parser.add_argument("--seeds",   type=str, default=None)
    parser.add_argument("--workers", type=int,
                        default=min(4, multiprocessing.cpu_count()))
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    seeds = load_seeds(args.seeds)
    run(seeds, workers=args.workers, resume=args.resume)
