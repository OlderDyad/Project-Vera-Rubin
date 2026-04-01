"""
run_pipeline.py
================
Master run script for Project Vera Rubin — Gravitational Lens Discovery Pipeline.
Runs all four analysis modules in sequence and prints a consolidated summary.

Usage:
    python run_pipeline.py                    # full pipeline run
    python run_pipeline.py --survey-only      # discovery only, skip analysis
    python run_pipeline.py --analysis-only    # analysis only, skip discovery
    python run_pipeline.py --seeds seeds/castles_wide_seeds.csv

Modules executed in order:
    1. rubin_survey_v3.py   — ZDCF gravitational lens discovery
    2. h0_pipeline.py       — H0 cosmography (5 modules)
    3. anisotropy_monitor.py — Cosmological tensions (3 modules)
    4. mstep_comparison.py  — Bansal & Huterer 2025 scenario comparison
"""

import argparse
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH      = Path("outputs/survey_v3/survey_results.db")
SURVEY_SEEDS = "seeds/castles_wide_seeds.csv"
SURVEY_WORKERS = 4

MODULES = {
    "survey":     "rubin_survey_v3.py",
    "h0":         "h0_pipeline.py",
    "anisotropy": "anisotropy_monitor.py",
    "mstep":      "mstep_comparison.py",
    "candidates": "candidate_manager.py",
}

# ── Utilities ──────────────────────────────────────────────────────────────────

def header(title):
    width = 62
    print()
    print("╔" + "═"*width + "╗")
    print(f"║  {title:<{width-2}}║")
    print("╚" + "═"*width + "╝")

def section(title):
    print()
    print(f"  ┌─ {title} {'─'*(55-len(title))}┐")

def ok(msg):   print(f"  ✓ {msg}")
def warn(msg): print(f"  ⚠ {msg}")
def fail(msg): print(f"  ✗ {msg}")

def check_prerequisites():
    """Verify all required files exist before running."""
    section("PREREQUISITE CHECK")
    all_ok = True
    checks = [
        (MODULES["survey"],     "Survey engine"),
        (MODULES["h0"],         "H0 pipeline"),
        (MODULES["anisotropy"], "Anisotropy monitor"),
        (MODULES["mstep"],      "Mstep comparison"),
        ("lc_utils.py",         "ZDCF utilities"),
        (SURVEY_SEEDS,          "CASTLES seed catalog"),
    ]
    for path, label in checks:
        if Path(path).exists():
            ok(f"{label} ({path})")
        else:
            warn(f"{label} not found: {path}")
            if path in MODULES.values():
                all_ok = False
    print()
    return all_ok

def db_snapshot():
    """Read current database state."""
    if not DB_PATH.exists():
        return {"candidates": 0, "h0_estimates": 0,
                "anisotropy": 0, "mstep": 0}
    conn = sqlite3.connect(DB_PATH)
    def count(table):
        try:
            return conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            return 0
    snap = {
        "candidates":  count("candidates"),
        "h0_estimates": count("h0_estimates"),
        "anisotropy":  count("anisotropy_detections"),
        "mstep":       count("mstep_comparison"),
    }
    conn.close()
    return snap

def run_module(script, extra_args=None, label=None):
    """Run a Python module as a subprocess, stream output, return success."""
    label = label or script
    section(f"RUNNING: {label.upper()}")
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"  $ {' '.join(cmd)}")
    print()

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,   # stream directly to terminal
            text=True,
            timeout=600             # 10 minute timeout per module
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            ok(f"{label} completed in {elapsed:.1f}s")
            return True
        else:
            fail(f"{label} exited with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        fail(f"{label} timed out after 600s")
        return False
    except FileNotFoundError:
        fail(f"{label} not found — check working directory")
        return False
    except Exception as e:
        fail(f"{label} error: {e}")
        return False

def print_consolidated_summary(before, after, results):
    """Print final summary across all modules."""
    header("CONSOLIDATED PIPELINE SUMMARY")

    # Database delta
    print()
    print("  DATABASE CHANGES THIS RUN:")
    print(f"  {'Table':<25} {'Before':>8} {'After':>8} {'New':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    tables = [
        ("candidates",        "Lens candidates"),
        ("h0_estimates",      "H0 estimates"),
        ("anisotropy_detections", "Anisotropy tags"),
        ("mstep_comparison",  "Mstep comparisons"),
    ]
    for key, label in tables:
        b = before.get(key, 0)
        a = after.get(key, 0)
        new = a - b
        marker = " ◄ NEW" if new > 0 else ""
        print(f"  {label:<25} {b:>8} {a:>8} {new:>8}{marker}")

    # Module results
    print()
    print("  MODULE STATUS:")
    icons = {True: "✓", False: "✗"}
    for module, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"    {icons[success]} {module:<20} {status}")

    # Current science state
    conn = None
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)

    print()
    print("  CURRENT SCIENCE STATE:")
    if conn:
        try:
            rows = conn.execute("""
                SELECT anchor, lag_days, score, rungs
                FROM candidates ORDER BY score DESC LIMIT 5
            """).fetchall()
            if rows:
                print(f"  Top candidates (by score):")
                print(f"  {'Anchor':<20} {'Lag(d)':>8} {'Score':>7} {'Rungs':>6}")
                print(f"  {'─'*20} {'─'*8} {'─'*7} {'─'*6}")
                for anchor, lag, score, rungs in rows:
                    print(f"  {anchor:<20} {abs(lag or 0):>8.1f} "
                          f"{score:>7.3f} {rungs:>6}")
        except Exception:
            pass

        try:
            h0_rows = conn.execute("""
                SELECT lens, H0_estimate, H0_uncertainty, z_lens
                FROM h0_estimates ORDER BY z_lens
            """).fetchall()
            if h0_rows:
                print()
                print(f"  H0 estimates:")
                print(f"  {'Lens':<20} {'z_L':>6} {'H0':>8} {'±':>6}")
                print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*6}")
                for lens, h0, err, zl in h0_rows:
                    print(f"  {lens:<20} {zl:>6.3f} {h0:>8.1f} {err:>6.1f}")

                # Combined
                import numpy as np
                weights = [1/r[2]**2 for r in h0_rows]
                W = sum(weights)
                H0c = sum(w*r[1] for w, r in zip(weights, h0_rows)) / W
                sc  = 1.0 / np.sqrt(W)
                print(f"  {'COMBINED':<20} {'—':>6} {H0c:>8.1f} {sc:>6.1f}")
        except Exception:
            pass

        try:
            mstep = conn.execute("""
                SELECT best_scenario FROM mstep_comparison
                ORDER BY id DESC LIMIT 1
            """).fetchone()
            if mstep:
                print()
                print(f"  Best-fit cosmological scenario: {mstep[0]}")
                print(f"  (N=2 lenses — insufficient to discriminate; "
                      f"awaiting Rubin detections)")
        except Exception:
            pass

        conn.close()

    # Pending
    print()
    print("  PENDING EXTERNAL ITEMS:")
    pending = [
        ("ANTARES MR !1",  "Broker filter — awaiting team review"),
        ("Rubin alerts",   "Week 21 engineering tests — weeks away"),
        ("New discovery",  "First unknown system from pipeline"),
    ]
    for item, note in pending:
        print(f"    ⧖ {item:<18} {note}")

    # Next milestone
    print()
    print("  NEXT SCIENCE MILESTONE:")
    print("    First new lens discovery — unknown system, not in")
    print("    CASTLES/SIMBAD/NED, found by ZDCF variability alone.")
    print("    That is the publishable result (AAS rejected recovery")
    print("    of known delays; new discovery clears the bar).")
    print()
    print(f"  Run completed: "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project Vera Rubin — Master Pipeline Runner")
    parser.add_argument("--seeds",         default=SURVEY_SEEDS)
    parser.add_argument("--workers",       type=int, default=SURVEY_WORKERS)
    parser.add_argument("--survey-only",   action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    header("PROJECT VERA RUBIN — GRAVITATIONAL LENS DISCOVERY PIPELINE")
    print()
    print("  Published: McKnight 2026, RNAAS AAS75139")
    print("  Repo:      https://github.com/OlderDyad/Project-Vera-Rubin")
    print(f"  Time:      "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Prerequisites
    if not check_prerequisites():
        fail("Missing required files — aborting.")
        sys.exit(1)

    # DB snapshot before
    before = db_snapshot()

    results = {}
    run_survey   = not args.analysis_only
    run_analysis = not args.survey_only

    # ── Module 1: Discovery Survey ───────────────────────────────────────────
    if run_survey:
        survey_args = [
            "--seeds",   args.seeds,
            "--workers", str(args.workers),
        ]
        if args.resume:
            survey_args.append("--resume")
        results["Survey (rubin_survey_v3)"] = run_module(
            MODULES["survey"], survey_args, "ZDCF Discovery Survey")
    else:
        ok("Survey skipped (--analysis-only)")

    # ── Module 2: H0 Pipeline ────────────────────────────────────────────────
    if run_analysis:
        results["H0 pipeline"] = run_module(
            MODULES["h0"], label="H0 Cosmography Pipeline")
    else:
        ok("H0 pipeline skipped (--survey-only)")

    # ── Module 3: Anisotropy Monitor ─────────────────────────────────────────
    if run_analysis:
        results["Anisotropy monitor"] = run_module(
            MODULES["anisotropy"], label="Anisotropy Monitor")
    else:
        ok("Anisotropy monitor skipped (--survey-only)")

    # ── Module 4: Mstep Comparison ───────────────────────────────────────────
    if run_analysis:
        results["Mstep comparison"] = run_module(
            MODULES["mstep"], label="Mstep Scenario Comparison")
    else:
        ok("Mstep comparison skipped (--survey-only)")

    # ── Module 5: Candidate Manager ──────────────────────────────────────────
    results["Candidate manager"] = run_module(
        MODULES["candidates"], [], label="Candidate Manager")

    # DB snapshot after
    after = db_snapshot()

    # Consolidated summary
    print_consolidated_summary(before, after, results)

    # Exit code: 0 if all succeeded, 1 if any failed
    if all(results.values()):
        ok("All modules completed successfully.")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if not v]
        fail(f"Failed modules: {', '.join(failed)}")
        sys.exit(1)

if __name__ == "__main__":
    main()