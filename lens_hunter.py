"""
lens_hunter.py v2.6 - Professional Birthday Edition (Multi-Core)
─────────────────────────────────────────────────────────────
Optimized for high-volume discovery and 12-core Windows environments.
"""
import argparse, json, math, time, requests, concurrent.futures, multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.ndimage import uniform_filter1d
from lc_utils import quality_check, zdcf, best_lag_from_zdcf, prep_lc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Global Configuration ---
OUT_DIR = Path("outputs/hunter")
RESULTS_FILE = OUT_DIR / "candidates.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_DETECTIONS = 30
LAG_RANGE = (-90, 90)
THRESHOLDS = {
    "lag_min_r": 0.20, "flux_cv_max": 0.25, "ls_power_max": 0.40,
    "frac_var_min_r": 0.60, "micro_long_min": 0.50, "micro_short_max": 0.45
}

# --- Utility Functions ---
def load_seeds(seed_path):
    if not Path(seed_path).exists():
        raise FileNotFoundError(f"❌ Seed file not found: {seed_path}")
    df = pd.read_csv(seed_path)
    df.columns = [c.lower() for c in df.columns]
    if 'dec' in df.columns:
        df = df[df["dec"] > -28].reset_index(drop=True)
    return df

def ang_sep(ra1, dec1, ra2, dec2):
    c = math.cos(math.radians((dec1 + dec2) / 2))
    return math.sqrt(((ra1 - ra2) * c * 3600)**2 + ((dec1 - dec2) * 3600)**2)

def fetch_objects_near(ra, dec, radius_arcsec):
    try:
        r = requests.get("https://api.alerce.online/ztf/v1/objects",
                         params={"ra": ra, "dec": dec, "radius": radius_arcsec,
                                 "page_size": 50, "order_by": "ndet", "order_mode": "DESC"}, timeout=15)
        r.raise_for_status()
        return [o for o in r.json().get("items", []) if o.get("ndet", 0) >= MIN_DETECTIONS]
    except: return []

def fetch_lightcurve(oid):
    try:
        r = requests.get(f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve", timeout=15)
        r.raise_for_status()
        dets = r.json().get("detections", [])
        if not dets: return None
        df = pd.DataFrame(dets)
        df = df[["mjd", "magpsf", "sigmapsf", "fid"]].rename(
            columns={"magpsf": "mag", "sigmapsf": "mag_err", "fid": "band"})
        df["flux"] = 10**(-0.4*(df["mag"]-25.0))
        df["flux_err"] = df["flux"] * df["mag_err"] * 0.4 * np.log(10)
        return df.sort_values("mjd").reset_index(drop=True)
    except: return None

# --- Physics Rungs ---
def _r2(lA, lB, lag):
    lB2 = lB.copy(); lB2["mjd"] += lag
    tg = np.arange(max(lA["mjd"].min(), lB2["mjd"].min()), min(lA["mjd"].max(), lB2["mjd"].max()), 2.)
    if len(tg) < 5: return {"passed": False, "score": 0.}
    fA = np.interp(tg, lA["mjd"].values, lA["flux"].values)
    fB = np.interp(tg, lB2["mjd"].values, lB2["flux"].values)
    ratio = fA/(fB+1e-9); cv = ratio.std()/(ratio.mean()+1e-9)
    return {"passed": cv < THRESHOLDS["flux_cv_max"], "score": max(0, 1 - cv/THRESHOLDS["flux_cv_max"])}

def _r3(lA, lB):
    from astropy.timeseries import LombScargle
    powers = []
    for lc in [lA, lB]:
        _, pw = LombScargle(lc["mjd"], lc["flux"]).autopower(minimum_frequency=1/300, maximum_frequency=1/3)
        powers.append(float(pw.max()))
    avg = np.mean(powers)
    return {"passed": avg < THRESHOLDS["ls_power_max"], "score": max(0, 1 - avg/THRESHOLDS["ls_power_max"])}

def _r4(lA, lB, lag):
    lB2 = lB.copy(); lB2["mjd"] += lag
    tg = np.arange(max(lA["mjd"].min(), lB2["mjd"].min()), min(lA["mjd"].max(), lB2["mjd"].max()), 2.)
    if len(tg) < 10: return {"passed": False, "score": 0.}
    fA = np.interp(tg, lA["mjd"].values, lA["flux"].values)
    fB = np.interp(tg, lB2["mjd"].values, lB2["flux"].values)
    corr, _ = stats.pearsonr(fA, fB)
    return {"passed": corr > THRESHOLDS["frac_var_min_r"], "score": max(0., float(corr))}

def _r5(lA, lB, lag):
    lB2 = lB.copy(); lB2["mjd"] += lag
    tg = np.arange(max(lA["mjd"].min(), lB2["mjd"].min()), min(lA["mjd"].max(), lB2["mjd"].max()), 3.)
    if len(tg) < 10: return {"passed": False, "score": 0.}
    fA = np.interp(tg, lA["mjd"].values, lA["flux"].values)
    fB = np.interp(tg, lB2["mjd"].values, lB2["flux"].values)
    fAs = uniform_filter1d(fA, 10); fBs = uniform_filter1d(fB, 10)
    lr, _ = stats.pearsonr(fAs, fBs)
    return {"passed": lr > THRESHOLDS["micro_long_min"], "score": max(0, float(lr))}

def _r6(lag, sep):
    if not sep or lag <= 0: return {"passed": False, "score": 0.}
    dt_scale = 1.5 * ((np.deg2rad(sep/3600))**2 / 2) * (1e26 / 3e8) / 86400
    passed = (dt_scale * 0.05) <= lag <= (dt_scale * 5.0)
    return {"passed": passed, "score": 1.0 if passed else 0.0}

def score_pair(lc_A, lc_B, oid_A, oid_B, sep_arcsec):
    res = {"oid_A": oid_A, "oid_B": oid_B, "sep_arcsec": sep_arcsec, "rungs": {}}
    try:
        tA, fA = prep_lc(lc_A); tB, fB = prep_lc(lc_B)
        cs, zv, np_ = zdcf(tA, fA, tB, fB, lag_range=LAG_RANGE, n_bins=72)
        best_lag, lu = best_lag_from_zdcf(cs, zv, np_)
        valid_zv = np.abs(zv[~np.isnan(zv)])
        pr = float(np.tanh(valid_zv.max())) if len(valid_zv) > 0 else 0.0
        res["best_lag"] = best_lag
        res["lag_uncertainty"] = lu
        res["rungs"]["rung1"] = {"passed": pr > THRESHOLDS["lag_min_r"], "score": pr}
        res["rungs"]["rung2"] = _r2(lc_A, lc_B, best_lag)
        res["rungs"]["rung3"] = _r3(lc_A, lc_B)
        res["rungs"]["rung4"] = _r4(lc_A, lc_B, best_lag)
        res["rungs"]["rung5"] = _r5(lc_A, lc_B, best_lag)
        res["rungs"]["rung6"] = _r6(best_lag, sep_arcsec)
        sc = [r.get("score", 0.) for r in res["rungs"].values()]
        pa = [r.get("passed", False) for r in res["rungs"].values()]
        res["total_score"] = float(np.mean(sc))
        res["rungs_passed"] = int(sum(pa))
    except:
        res["total_score"] = 0.0; res["rungs_passed"] = 0
    return res

# --- The Worker Function ---
def process_single_seed(task_data):
    idx, ra, dec, name, radius, pair_sep = task_data
    results = [] 
    try:
        objs = fetch_objects_near(ra, dec, radius)
        if len(objs) < 2: return results
        
        lcs = {o['oid']: fetch_lightcurve(o['oid']) for o in objs[:5]}
        lcs = {k: v for k, v in lcs.items() if v is not None and len(v) >= MIN_DETECTIONS}
        
        oids = list(lcs.keys())
        for i in range(len(oids)):
            for j in range(i+1, len(oids)):
                oA = next(obj for obj in objs if obj['oid']==oids[i])
                oB = next(obj for obj in objs if obj['oid']==oids[j])
                sep = ang_sep(oA['meanra'], oA['meandec'], oB['meanra'], oB['meandec'])
                
                if 1.5 <= sep <= pair_sep:
                    res = score_pair(lcs[oids[i]], lcs[oids[j]], oids[i], oids[j], sep)
                    res.update({"anchor": name, "ra": oA['meanra'], "dec": oA['meandec']})
                    results.append(res)
        return results
    except Exception as e:
        return []

def save_results(all_results):
    if not all_results: return
    rows = []
    for r in all_results:
        row = {
            "anchor": r.get("anchor", ""), "ra": r.get("ra", ""), "dec": r.get("dec", ""),
            "oid_A": r["oid_A"], "oid_B": r["oid_B"], "sep_arcsec": r["sep_arcsec"],
            "best_lag_days": r.get("best_lag", 0), "total_score": r["total_score"],
            "rungs_passed": r["rungs_passed"]
        }
        for i in range(1, 7):
            row[f"rung{i}_pass"] = r["rungs"].get(f"rung{i}", {}).get("passed", False)
        rows.append(row)
    pd.DataFrame(rows).sort_values("total_score", ascending=False).to_csv(RESULTS_FILE, index=False)

# --- Execution Entry Point ---
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="seeds/all_candidates_250k.csv")
    p.add_argument("--radius", type=float, default=60.)
    p.add_argument("--pair-sep", type=float, default=15.)
    p.add_argument("--cores", type=int, default=8)
    args = p.parse_args()

    seeds_df = load_seeds(args.seeds)
    tasks = [(i, row['ra'], row['dec'], row.get('anchor', row.get('name', 'Unknown')), 
              args.radius, args.pair_sep) for i, row in seeds_df.iterrows()]

    print(f"🚀 Launching Big Run on {args.cores} cores for {len(seeds_df)} seeds...")

    all_results = []
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx, max_workers=args.cores) as executor:
        futures = [executor.submit(process_single_seed, t) for t in tasks]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res_list = future.result()
            if res_list:
                all_results.extend(res_list)
            if i % 100 == 0:
                print(f"✅ {i}/{len(seeds_df)} processed... Matches: {len(all_results)}")
                save_results(all_results)
    save_results(all_results)
    print("🏁 Run Complete.")