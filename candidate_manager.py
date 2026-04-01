"""
candidate_manager.py
====================
Candidate tracking and deduplication system for Project Vera Rubin.

Solves two problems:
  1. PERSISTENCE — stores interesting candidates with full metadata,
     investigation notes, and follow-up status locally
  2. DEDUPLICATION — new pipeline runs never re-report known candidates

Three tiers of candidate status:
  KNOWN    — previously seen, scored, stored. Never re-reported.
  WATCH    — flagged for follow-up (interesting but needs more data)
  CONFIRMED — validated detection (published or publication-ready)

Usage:
    python candidate_manager.py              # show full catalog
    python candidate_manager.py --new        # show only new candidates
    python candidate_manager.py --watch      # show watch list
    python candidate_manager.py --flag HE0435-1223_ZTF18abwbjev watch
    python candidate_manager.py --note HE0435-1223_ZTF18abwbjev "sub-arcsec, lag uncertain"
    python candidate_manager.py --export     # export to CSV for review
"""

import sqlite3
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH       = Path("outputs/survey_v3/survey_results.db")
CATALOG_PATH  = Path("outputs/candidate_catalog.db")

# ── Catalog Schema ─────────────────────────────────────────────────────────────

CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_catalog (
    pair_id          TEXT PRIMARY KEY,
    anchor           TEXT,
    ra               REAL,
    dec              REAL,
    sep_arcsec       REAL,
    lag_days         REAL,
    lag_unc          REAL,
    score            REAL,
    rungs            INTEGER,
    -- Rung scores
    r1_score REAL, r2_score REAL, r3_score REAL,
    r4_score REAL, r5_score REAL, r6_score REAL,
    r1_pass  INTEGER, r2_pass  INTEGER, r3_pass  INTEGER,
    r4_pass  INTEGER, r5_pass  INTEGER, r6_pass  INTEGER,
    -- Status tracking
    status           TEXT DEFAULT 'known',
    -- known / watch / confirmed / rejected
    note             TEXT DEFAULT '',
    survey           TEXT DEFAULT 'ZTF',
    first_seen       TEXT,
    last_seen        TEXT,
    n_detections     INTEGER DEFAULT 1,
    -- Cross-match flags
    in_castles       INTEGER DEFAULT 0,
    in_simbad        INTEGER DEFAULT 0,
    in_ned           INTEGER DEFAULT 0,
    is_new_system    INTEGER DEFAULT 0,   -- 1 = not in any catalog = DISCOVERY
    -- Investigation
    investigated     INTEGER DEFAULT 0,
    investigation_notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS run_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_time     TEXT,
    survey       TEXT,
    seeds_file   TEXT,
    n_seeds      INTEGER,
    n_new        INTEGER,
    n_known      INTEGER,
    n_watch      INTEGER,
    n_confirmed  INTEGER,
    top_new_score REAL
);
"""

# ── Database helpers ───────────────────────────────────────────────────────────

def get_catalog():
    conn = sqlite3.connect(CATALOG_PATH)
    conn.executescript(CATALOG_SCHEMA)
    conn.commit()
    return conn

def get_survey_db():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH)

# ── Core: sync survey DB → catalog ────────────────────────────────────────────

def sync_from_survey(verbose=True):
    """
    Pull all candidates from survey_results.db into candidate_catalog.db.
    New candidates get status='known'.
    Already-known candidates get their last_seen updated.
    Returns (n_new, n_updated).
    """
    survey = get_survey_db()
    if survey is None:
        print("  Survey DB not found — run rubin_survey_v3.py first")
        return 0, 0

    catalog = get_catalog()
    now = datetime.now(timezone.utc).isoformat()

    rows = survey.execute("""
        SELECT pair_id, anchor, ra, dec, sep, lag_days, lag_unc,
               score, rungs,
               r1_score, r2_score, r3_score, r4_score, r5_score, r6_score,
               r1_pass,  r2_pass,  r3_pass,  r4_pass,  r5_pass,  r6_pass,
               logged_at
        FROM candidates
    """).fetchall()
    survey.close()

    n_new = n_updated = 0
    for row in rows:
        (pair_id, anchor, ra, dec, sep, lag, lag_unc,
         score, rungs,
         r1s, r2s, r3s, r4s, r5s, r6s,
         r1p, r2p, r3p, r4p, r5p, r6p,
         logged_at) = row

        existing = catalog.execute(
            "SELECT pair_id, n_detections FROM candidate_catalog WHERE pair_id=?",
            (pair_id,)).fetchone()

        if existing is None:
            # New candidate — insert
            catalog.execute("""
                INSERT INTO candidate_catalog
                (pair_id, anchor, ra, dec, sep_arcsec, lag_days, lag_unc,
                 score, rungs,
                 r1_score, r2_score, r3_score, r4_score, r5_score, r6_score,
                 r1_pass,  r2_pass,  r3_pass,  r4_pass,  r5_pass,  r6_pass,
                 status, note, first_seen, last_seen, n_detections)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (pair_id, anchor, ra, dec, sep, lag, lag_unc,
                  score, rungs,
                  r1s, r2s, r3s, r4s, r5s, r6s,
                  r1p, r2p, r3p, r4p, r5p, r6p,
                  'known', '', logged_at, now, 1))
            n_new += 1
        else:
            # Known — update last_seen and increment detection count
            catalog.execute("""
                UPDATE candidate_catalog
                SET last_seen=?, n_detections=n_detections+1,
                    score=MAX(score, ?), lag_days=?
                WHERE pair_id=?
            """, (now, score, lag, pair_id))
            n_updated += 1

    catalog.commit()
    catalog.close()

    if verbose:
        print(f"  Sync: {n_new} new candidates, {n_updated} updated")

    return n_new, n_updated

# ── Reporting ──────────────────────────────────────────────────────────────────

def print_catalog(status_filter=None, new_only=False):
    catalog = get_catalog()
    now = datetime.now(timezone.utc)

    query = "SELECT * FROM candidate_catalog"
    params = []
    if status_filter:
        query += " WHERE status=?"
        params.append(status_filter)
    elif new_only:
        query += " WHERE status='known' AND investigated=0"
    query += " ORDER BY score DESC"

    rows = catalog.execute(query, params).fetchall()
    cols = [d[0] for d in catalog.execute(query, params).description] \
        if rows else []
    catalog.close()

    if not rows:
        print("  No candidates match filter.")
        return

    print()
    print(f"  {'pair_id':<35} {'anchor':<20} {'sep\"':>5} "
          f"{'lag_d':>7} {'score':>6} {'status':>10} {'N':>3} {'note'}")
    print(f"  {'─'*35} {'─'*20} {'─'*5} "
          f"{'─'*7} {'─'*6} {'─'*10} {'─'*3} {'─'*30}")

    for row in rows:
        d = dict(zip(
            ['pair_id','anchor','ra','dec','sep','lag','lag_unc',
             'score','rungs',
             'r1s','r2s','r3s','r4s','r5s','r6s',
             'r1p','r2p','r3p','r4p','r5p','r6p',
             'status','note','survey','first_seen','last_seen',
             'n_det','in_castles','in_simbad','in_ned','is_new',
             'investigated','inv_notes'], row))

        new_marker  = " ★" if not d['investigated'] and d['status']=='known' else ""
        watch_marker = " 👁" if d['status'] == 'watch' else ""
        conf_marker  = " ✓" if d['status'] == 'confirmed' else ""
        rej_marker   = " ✗" if d['status'] == 'rejected' else ""

        print(f"  {d['pair_id'][:35]:<35} {d['anchor'][:20]:<20} "
              f"{d['sep']:>5.2f} {d['lag'] or 0:>7.0f} {d['score']:>6.3f} "
              f"{d['status']:>10} {d['n_det']:>3} "
              f"{d['note'][:30]}{new_marker}{watch_marker}{conf_marker}{rej_marker}")

def print_summary():
    catalog = get_catalog()
    rows = catalog.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='known' THEN 1 ELSE 0 END) as known,
            SUM(CASE WHEN status='watch' THEN 1 ELSE 0 END) as watch,
            SUM(CASE WHEN status='confirmed' THEN 1 ELSE 0 END) as confirmed,
            SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) as rejected,
            SUM(CASE WHEN investigated=0 AND status='known' THEN 1 ELSE 0 END) as uninvestigated,
            SUM(CASE WHEN is_new_system=1 THEN 1 ELSE 0 END) as new_systems,
            MAX(score) as top_score
        FROM candidate_catalog
    """).fetchone()
    catalog.close()

    if rows[0] == 0:
        print("  No candidates in catalog yet.")
        return

    total, known, watch, confirmed, rejected, uninvest, new_sys, top = rows
    print()
    print("  CANDIDATE CATALOG SUMMARY")
    print(f"  Total:        {total}")
    print(f"  Known:        {known}  (seen, not yet investigated)")
    print(f"  Watch list:   {watch}  (flagged for follow-up)")
    print(f"  Confirmed:    {confirmed}  (validated detections)")
    print(f"  Rejected:     {rejected}  (ruled out)")
    print(f"  Uninvestigated: {uninvest}")
    print(f"  Potential new systems: {new_sys}")
    print(f"  Top score:    {top:.3f}" if top else "")

# ── Actions ────────────────────────────────────────────────────────────────────

def flag_candidate(pair_id, status):
    """Change a candidate's status."""
    valid = {'known', 'watch', 'confirmed', 'rejected'}
    if status not in valid:
        print(f"  Invalid status '{status}'. Use: {valid}")
        return
    catalog = get_catalog()
    now = datetime.now(timezone.utc).isoformat()
    n = catalog.execute("""
        UPDATE candidate_catalog
        SET status=?, investigated=1, last_seen=?
        WHERE pair_id LIKE ?
    """, (status, now, f"%{pair_id}%")).rowcount
    catalog.commit()
    catalog.close()
    print(f"  Updated {n} candidate(s) → status='{status}'")

def add_note(pair_id, note):
    """Add or append a note to a candidate."""
    catalog = get_catalog()
    existing = catalog.execute(
        "SELECT note FROM candidate_catalog WHERE pair_id LIKE ?",
        (f"%{pair_id}%",)).fetchone()
    if existing:
        current = existing[0] or ""
        new_note = f"{current}; {note}".strip("; ")
        catalog.execute(
            "UPDATE candidate_catalog SET note=?, investigated=1 WHERE pair_id LIKE ?",
            (new_note, f"%{pair_id}%"))
        catalog.commit()
        print(f"  Note updated: {new_note}")
    else:
        print(f"  Candidate not found: {pair_id}")
    catalog.close()

def export_csv():
    """Export full catalog to CSV for external review."""
    import pandas as pd
    catalog = get_catalog()
    df = pd.read_sql("SELECT * FROM candidate_catalog ORDER BY score DESC", catalog)
    catalog.close()
    out = Path("outputs/candidate_catalog_export.csv")
    df.to_csv(out, index=False)
    print(f"  Exported {len(df)} candidates → {out}")

def new_candidates_report():
    """
    Show only candidates not previously reported.
    Called by run_pipeline.py after each survey run.
    """
    catalog = get_catalog()
    rows = catalog.execute("""
        SELECT pair_id, anchor, sep_arcsec, lag_days, score, rungs,
               r1_score, r2_score, r3_score, r4_score, r5_score, r6_score,
               first_seen
        FROM candidate_catalog
        WHERE investigated = 0
        ORDER BY score DESC
    """).fetchall()
    catalog.close()

    if not rows:
        print("  No new uninvestigated candidates.")
        return 0

    print()
    print(f"  NEW CANDIDATES THIS RUN: {len(rows)}")
    print(f"  {'pair_id':<35} {'anchor':<20} {'sep\"':>5} "
          f"{'lag_d':>7} {'score':>6} {'rungs':>6}")
    print(f"  {'─'*35} {'─'*20} {'─'*5} {'─'*7} {'─'*6} {'─'*6}")
    for r in rows:
        pid, anchor, sep, lag, score, rungs = r[:6]
        rung_scores = r[6:12]
        print(f"  {pid[:35]:<35} {anchor[:20]:<20} "
              f"{sep:>5.2f} {lag or 0:>7.0f} {score:>6.3f} {rungs:>6}/6")
        rung_str = "  " + " ".join(
            f"R{i+1}={'✓' if s > 0.12 else '·'}" for i, s in enumerate(rung_scores))
        print(rung_str)

    print()
    print("  To flag for follow-up:")
    print("    python candidate_manager.py --flag <pair_id_prefix> watch")
    print("  To add a note:")
    print("    python candidate_manager.py --note <pair_id_prefix> 'your note'")
    print("  To mark as investigated (suppress from future new-candidate reports):")
    print("    python candidate_manager.py --flag <pair_id_prefix> known")

    return len(rows)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Candidate tracking for Project Vera Rubin")
    parser.add_argument("--new",    action="store_true",
                        help="Show only uninvestigated new candidates")
    parser.add_argument("--watch",  action="store_true",
                        help="Show watch list")
    parser.add_argument("--confirmed", action="store_true",
                        help="Show confirmed detections")
    parser.add_argument("--all",    action="store_true",
                        help="Show full catalog")
    parser.add_argument("--flag",   nargs=2, metavar=("PAIR_ID", "STATUS"),
                        help="Flag a candidate: --flag <id> <status>")
    parser.add_argument("--note",   nargs=2, metavar=("PAIR_ID", "NOTE"),
                        help="Add a note: --note <id> 'your note'")
    parser.add_argument("--export", action="store_true",
                        help="Export catalog to CSV")
    parser.add_argument("--sync",   action="store_true",
                        help="Sync from survey DB only")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   CANDIDATE MANAGER — Project Vera Rubin                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Always sync first
    n_new, n_updated = sync_from_survey(verbose=True)

    if args.flag:
        flag_candidate(args.flag[0], args.flag[1])
    elif args.note:
        add_note(args.note[0], args.note[1])
    elif args.export:
        export_csv()
    elif args.watch:
        print_catalog(status_filter='watch')
    elif args.confirmed:
        print_catalog(status_filter='confirmed')
    elif args.new:
        new_candidates_report()
    elif args.sync:
        pass  # already synced above
    else:
        # Default: summary + new candidates
        print_summary()
        if n_new > 0:
            print()
            new_candidates_report()
        else:
            print()
            print("  No new candidates since last run.")
            print("  Use --all to see full catalog, --watch for watch list.")

    print()

if __name__ == "__main__":
    main()