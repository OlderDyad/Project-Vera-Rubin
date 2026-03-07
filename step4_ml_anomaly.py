"""
step4_ml_anomaly.py
───────────────────
Uses Isolation Forest (unsupervised ML) to find statistically
unusual transients in your candidate set.

No training labels needed — the algorithm identifies objects
that behave differently from the majority population.
These outliers are your discovery candidates.

Run:
    python step4_ml_anomaly.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")   # save to file, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

INPUT_FILE   = Path("data/alerts_crossmatched.csv")
FALLBACK_FILE = Path("data/alerts_filtered.csv")   # if crossmatch wasn't run
OUTPUT_FILE  = Path("data/alerts_anomalies.csv")
PLOT_FILE    = Path("outputs/anomaly_plot.png")
PLOT_FILE.parent.mkdir(exist_ok=True)

# Isolation Forest settings
CONTAMINATION = 0.05   # expect ~5% anomalies (tune up/down)
RANDOM_STATE  = 42

# Features to use for anomaly detection
# These will be selected if present in the data
CANDIDATE_FEATURES = [
    "delta_mag",    # variability amplitude
    "ndet",         # number of detections
    "lastmjd",      # recency of activity
    "magmean",      # mean brightness
    "magmin",       # brightest observed
    "magmax",       # faintest observed
    "discovery_score",
]


def load_data():
    for path in [INPUT_FILE, FALLBACK_FILE]:
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} records from {path}")
            return df
    return None


def select_features(df):
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    print(f"  Using features: {available}")
    X = df[available].copy()

    # Fill missing values with column median
    X = X.fillna(X.median())
    return X, available


def run_isolation_forest(df, X):
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
    )
    predictions = model.fit_predict(X_scaled)       # -1 = anomaly, 1 = normal
    scores      = model.score_samples(X_scaled)     # lower = more anomalous

    df = df.copy()
    df["anomaly"]       = predictions == -1
    df["anomaly_score"] = -scores    # flip sign: higher = more anomalous

    return df, X_scaled


def plot_results(df, X_scaled, feature_names):
    """PCA projection showing anomalies vs normal objects."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    normal   = ~df["anomaly"]
    anomalous = df["anomaly"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#04060f")

    # ── Left: PCA scatter ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#080d1a")
    ax.scatter(coords[normal, 0],   coords[normal, 1],
               c="#1e3a5f", s=20, alpha=0.6, label="Normal")
    ax.scatter(coords[anomalous, 0], coords[anomalous, 1],
               c="#ff4757", s=60, alpha=0.9, label="Anomaly", zorder=5,
               edgecolors="#ff8a93", linewidths=0.5)

    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)", color="#c8ddf5")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)", color="#c8ddf5")
    ax.set_title("PCA: Normal vs Anomalous Alerts", color="white", pad=10)
    ax.tick_params(colors="#4a6080")
    for spine in ax.spines.values():
        spine.set_color("#0d1f3a")
    ax.legend(facecolor="#080d1a", labelcolor="white", framealpha=0.8)

    # ── Right: Anomaly score histogram ────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#080d1a")
    ax2.hist(df.loc[normal,   "anomaly_score"], bins=30, color="#1e3a5f",
             alpha=0.8, label="Normal", density=True)
    ax2.hist(df.loc[anomalous, "anomaly_score"], bins=15, color="#ff4757",
             alpha=0.9, label="Anomaly", density=True)
    ax2.set_xlabel("Anomaly Score (higher = more unusual)", color="#c8ddf5")
    ax2.set_ylabel("Density", color="#c8ddf5")
    ax2.set_title("Anomaly Score Distribution", color="white", pad=10)
    ax2.tick_params(colors="#4a6080")
    for spine in ax2.spines.values():
        spine.set_color("#0d1f3a")
    ax2.legend(facecolor="#080d1a", labelcolor="white", framealpha=0.8)

    fig.suptitle("Vera Rubin / LSST — ML Anomaly Detection Results",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved → {PLOT_FILE}")


if __name__ == "__main__":
    print("\n🔭 Rubin Discovery Toolkit — Step 4: ML Anomaly Detection")
    print("─" * 55)

    df = load_data()
    if df is None:
        print("✗ No input data found. Run step2_filter.py first.")
        exit(1)

    X, feature_names = select_features(df)

    if len(X) < 10:
        print(f"  ⚠  Only {len(X)} records — need at least 10 for meaningful anomaly detection.")
        exit(0)

    print(f"\n  Running Isolation Forest on {len(X)} candidates...")
    df, X_scaled = run_isolation_forest(df, X)

    n_anomalies = df["anomaly"].sum()
    print(f"\n  ★  Found {n_anomalies} anomalous transients ({n_anomalies/len(df)*100:.1f}%)")

    # Save results
    df_sorted = df.sort_values("anomaly_score", ascending=False)
    df_sorted.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved results → {OUTPUT_FILE}")

    # Plot
    try:
        plot_results(df, X_scaled, feature_names)
    except Exception as e:
        print(f"  ⚠  Plot failed: {e}")

    # Print top anomalies
    anomalies = df_sorted[df_sorted["anomaly"]].copy()
    print(f"\n{'═'*55}")
    print("TOP ANOMALOUS CANDIDATES (most unusual first):")
    print(f"{'═'*55}")
    id_col = next((c for c in ["oid", "alertId", "id"] if c in anomalies.columns), anomalies.columns[0])
    show = [c for c in [id_col, "classxf", "anomaly_score", "delta_mag", "ndet", "simbad_id"] if c in anomalies.columns]
    print(anomalies[show].head(15).to_string(index=False))

    print("\n→ Next: run  python step5_visualize.py")
    print("     Or open  outputs/anomaly_plot.png  to see the ML results")
