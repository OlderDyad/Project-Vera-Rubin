import sqlite3
import pandas as pd

conn = sqlite3.connect("outputs/survey_v3/survey_results.db")

print("=== PROCESSED SEEDS ===")
df = pd.read_sql("SELECT name, n_pairs FROM progress WHERE status='processed'", conn)
print(df.to_string(index=False))

print("\n=== STATUS SUMMARY ===")
print(pd.read_sql("SELECT status, COUNT(*) as n FROM progress GROUP BY status", conn).to_string(index=False))

print("\n=== TOP CANDIDATES ===")
cands = pd.read_sql(
    "SELECT anchor, sep, lag_days, score, rungs, delay_uncertain "
    "FROM candidates ORDER BY score DESC LIMIT 15", conn)
if len(cands):
    print(cands.to_string(index=False))
else:
    print("(none yet)")

conn.close()