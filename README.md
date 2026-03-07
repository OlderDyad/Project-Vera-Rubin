# 🔭 Vera Rubin Observatory — Alert Discovery Toolkit

A beginner-friendly Python pipeline for mining LSST transient alerts
and finding potentially new astronomical objects.

---

## ⚡ Quick Setup (VS Code + Virtual Environment)

### 1. Open the project in VS Code
```
File → Open Folder → select this folder (rubin_discovery)
```

### 2. Open the integrated terminal
```
Terminal → New Terminal   (or Ctrl+` )
```

### 3. Create a virtual environment
```bash
python -m venv venv
```

### 4. Activate it
- **Windows:**   `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You'll see `(venv)` appear in your terminal prompt.

### 5. Install dependencies
```bash
pip install -r requirements.txt
```
*(Takes 1–2 minutes the first time)*

---

## 🚀 Run the Pipeline

Each step runs independently. Run them in order:

```bash
python step1_fetch_alerts.py     # Download real alerts from ALeRCE broker
python step2_filter.py           # Apply science filters, score candidates
python step3_crossmatch.py       # Check against SIMBAD catalog (~2 min)
python step4_ml_anomaly.py       # Isolation Forest anomaly detection
python step5_visualize.py        # Generate plots and light curves
```

Results appear in:
- `data/`     — CSV files at each pipeline stage
- `outputs/`  — PNG plots (click to view in VS Code)

---

## 📁 Project Structure

```
rubin_discovery/
├── requirements.txt              # Python dependencies
├── step1_fetch_alerts.py         # Fetch from ALeRCE broker API
├── step2_filter.py               # Filter + score candidates
├── step3_crossmatch.py           # SIMBAD catalog cross-match
├── step4_ml_anomaly.py           # Isolation Forest ML
├── step5_visualize.py            # Sky map + light curves
├── data/
│   ├── alerts_raw.csv            # Step 1 output
│   ├── alerts_filtered.csv       # Step 2 output
│   ├── alerts_crossmatched.csv   # Step 3 output
│   └── alerts_anomalies.csv      # Step 4 output
└── outputs/
    ├── anomaly_plot.png          # ML results visualization
    ├── skymap.png                # Sky distribution map
    ├── lightcurve_top_candidate.png
    └── candidate_summary.png
```

---

## 🎛 Customizing Your Search

### Hunt for a specific object type
In `step1_fetch_alerts.py`, change:
```python
CLASSIFIER_FILTER = None      # all types
CLASSIFIER_FILTER = "SN"      # supernovae only
CLASSIFIER_FILTER = "AGN"     # active galactic nuclei
CLASSIFIER_FILTER = "VS"      # variable stars
```

### Relax or tighten filters
In `step2_filter.py`, edit the `FILTERS` dictionary:
```python
FILTERS = {
    "min_detections": 3,      # lower = catch more, more noise
    "mag_min":        14.0,   # brightest allowed
    "mag_max":        22.5,   # faintest allowed
    "min_delta_mag":  0.10,   # minimum variability
    "max_age_days":   None,   # None = no recency cut
}
```

### Change anomaly sensitivity
In `step4_ml_anomaly.py`:
```python
CONTAMINATION = 0.05   # 0.01 = 1% flagged, 0.10 = 10% flagged
```

---

## 🌐 Data Sources

| Source | What it provides | URL |
|--------|-----------------|-----|
| **ALeRCE** | ZTF alerts + ML classifications | api.alerce.online |
| **SIMBAD** | Master catalog of known objects | simbad.u-strasbg.fr |
| **Lasair** | SQL-queryable ZTF/LSST stream | lasair-ztf.lsst.ac.uk |
| **Fink** | Anomaly scores pre-computed | fink-broker.org |

---

## 💡 Tips for Beginners

- **Start with ZTF data** — it's been live for years, same format as Rubin
- **Objects with no SIMBAD match** are your most interesting candidates
- **Isolation Forest** needs no training labels — perfect for exploration
- **Join the community**: community.lsst.org — post your candidates!
- **VS Code tip**: Click any PNG in the file explorer to view it instantly

---

*Data from ALeRCE broker (ZTF stream). Rubin/LSST full stream launching 2025.*
