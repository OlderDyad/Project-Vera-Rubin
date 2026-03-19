# 🔭 Project Vera Rubin — Gravitational Lens Discovery Pipeline

> **Automated time-delay gravitational lens discovery using the Zwicky Transient Facility (ZTF), with a direct pipeline path to the Vera Rubin Observatory LSST alert stream.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper: RNAAS AAS75139](https://img.shields.io/badge/Paper-RNAAS%20AAS75139-orange)](https://rnaas.aas.org)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 📋 Overview

This pipeline recovers gravitational lens time delays from archival survey photometry using the **Z-transformed Discrete Correlation Function (ZDCF)**. Instead of searching for lenses by their image morphology (which requires deep, high-resolution imaging), it identifies candidate lens pairs by detecting **correlated variability with a time lag** — the "time echo" of the same quasar seen through two different light paths.

**Key results:**
- ✅ Two validated gravitational lens detections from archival ZTF data
- ✅ ZDCF correlation plateaus (750–950 days) encompassing published delays for **SDSS J1004+4112** and **SDSS J1029+2623**
- ✅ H0 monitoring pipeline producing cosmological inference from each detection
- ✅ Direct pipeline path to Rubin LSST via TAP adapter and ANTARES broker filter
- 📄 Published: *"Time-Domain Recovery of Gravitational Lens Delays in Archival ZTF Photometry"* — Research Notes of the AAS (AAS75139)

---

## 🔬 Scientific Context

Gravitational lenses with measured time delays provide independent constraints on the **Hubble constant** (H₀), completely independent of the local distance ladder and the CMB. The "Hubble tension" — a ~5 km/s/Mpc discrepancy between early- and late-universe H₀ measurements — remains one of the most significant open questions in cosmology. DESI DR2 (2025) evidence for dynamical dark energy makes time-delay lensing at intermediate redshifts (z_L ~ 0.6) particularly valuable for mapping H₀(z).

**Why time-domain discovery?**
Current discovery pipelines find lenses by morphology — searching for arcs, rings, or multiple point sources in deep images. Time-domain pipelines find lenses by behavior — detecting the correlated variability that is the defining signature of any gravitationally lensed variable source. With Rubin LSST generating ~10 million alerts per night at 0.2" resolution, time-domain discovery becomes scalable for the first time.

---

## 📁 Repository Structure

```
Project-Vera-Rubin/
│
├── Core Pipeline
│   ├── rubin_survey_v3.py          # Main ZDCF discovery engine (v3, 5-day bins)
│   ├── lc_utils.py                 # ZDCF, quality_check, prep_lc utilities
│   └── seeds/
│       └── castles_wide_seeds.csv  # 19 CASTLES wide-separation lens seeds
│
├── Rubin / Broker Integration
│   ├── rubin_tap_adapter.py        # Rubin RSP TAP connector (drop-in for ALeRCE)
│   ├── antares_lens_consumer.py    # ANTARES Kafka consumer for LSST stream
│   └── antares/
│       └── gravitational_lens_triage/__init__.py  # ANTARES filter (submitted MR !1)
│
├── H0 Cosmography
│   └── h0_pipeline.py              # 5-module H0 monitoring system
│       # Module 1: H0 estimator (scaling method)
│       # Module 2: Posterior visualizer
│       # Module 3: DESI w0waCDM comparison
│       # Module 4: Sensitivity analysis (N lenses needed)
│       # Module 5: PNG comparison plot
│
├── Diagnostics & Validation
│   ├── zdcf_bin_diagnostic.py      # 3-test validation suite
│   │   # Test 1: Bin width sensitivity (25d/15d/10d/5d)
│   │   # Test 2: Rung ladder bias scan (50–1500 day range)
│   │   # Test 3: Seasonal alias analysis
│   └── rerun_10day_bins.py         # Real ZTF data bin sensitivity
│
└── outputs/
    ├── survey_v3/
    │   └── survey_results.db       # SQLite: candidates + H0 estimates
    └── diagnostic/
        └── h0_comparison.png       # H0 ladder + DESI bias plot
```

---

## 🚀 Quick Start

### Requirements
```
Python 3.12 (64-bit recommended)
scipy, numpy, pandas, requests, matplotlib
```

```powershell
# Create and activate virtual environment
python -m venv .venv64
.venv64\Scripts\Activate.ps1

# Install dependencies
pip install scipy numpy pandas requests matplotlib astropy
```

### Run the discovery pipeline
```powershell
# Run against the CASTLES seed catalog
python rubin_survey_v3.py --seeds seeds/castles_wide_seeds.csv --workers 4

# Check results
python -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('outputs/survey_v3/survey_results.db')
print(pd.read_sql('SELECT anchor, sep, lag_days, score, rungs FROM candidates ORDER BY score DESC', conn))
"
```

### Run the H0 pipeline
```powershell
python h0_pipeline.py
```

### When Rubin RSP access is approved
```powershell
# Save your RSP token
echo "your-token-here" > ~/.rsp-tap.token

# Test connection to Rubin DP0.2
python rubin_tap_adapter.py

# Run survey against Rubin simulated catalogs
python rubin_survey_v3.py --seeds seeds/castles_wide_seeds.csv --workers 4
```

---

## 🔧 Pipeline Architecture

### Six-Rung Scoring Ladder

Each candidate lens pair is evaluated through six independent tests:

| Rung | Test | Method |
|------|------|--------|
| R1 | ZDCF lag detection | Z-transformed DCF peak significance |
| R2 | Flux ratio stability | Coefficient of variation < 35% |
| R3 | Stochasticity | Lomb-Scargle confirms AGN red noise |
| R4 | Fractional variability match | Pearson r > 0.50 post-shift |
| R5 | Microlensing signature | Long-term corr, short-term decorr |
| R6 | Mass-delay consistency | Cluster-adaptive scaling (sep > 10") |

Each rung scores 0–1; composite = mean across all rungs. Threshold: score ≥ 0.25, ≥ 2 rungs passed.

### Key Design Choices

**Why ZDCF over standard cross-correlation?**
ZDCF (Alexander 1997) operates directly on unevenly sampled data without interpolation, avoiding spurious correlation peaks from ZTF's seasonal gaps.

**Why 5-day bins?**
Diagnostic testing showed 25-day bins collapse different true delays (821d and 746d for J1004/J1029) into a single artifactual peak at 887.5d. At 5-day bins the systems diverge. A noise-peak guard (minimum 8 pairs per bin) prevents spurious high-r peaks from sparse bins.

**Why cluster-adaptive R6?**
The SIS galaxy formula gives nonsensical delay predictions (1000–100,000d) for wide-separation cluster lenses. The cluster-adaptive scaling uses an empirical power law for separations > 10".

---

## 📊 Validated Detections

| System | Sep (") | ZDCF Plateau | Published Delay | Residual | Score | Rungs |
|--------|---------|--------------|-----------------|----------|-------|-------|
| SDSS J1004+4112 | 14.52 | 800–950d | ~821d | 6.9% | 0.408 | 3/6 |
| SDSS J1029+2623 | 22.37 | 750–950d | ~746d | 19.0% | 0.314 | 3/6 |

H0 from time-delay cosmography (scaling method, Fohlmeister+2013 mass models):
- **Combined: H0 = 62.4 ± 9.1 km/s/Mpc** (2 cluster lenses)
- Consistent with full H0 range within current uncertainties
- Dominant uncertainty: cluster mass model systematic (~15–20%)

---

## 🌌 Rubin LSST Readiness

| Component | Status |
|-----------|--------|
| `rubin_tap_adapter.py` | Built — awaiting RSP account approval |
| ANTARES filter MR !1 | Submitted — awaiting team review |
| Survey engine | Ready — 5-day bins, noise guard validated |
| H0 pipeline | Ready — auto-updates with each new detection |

**Projected Rubin performance** (Module 4 sensitivity analysis):
- 50 galaxy lenses → σ(H0) = 0.57 km/s/Mpc → **5σ discrimination** between CMB and SH0ES
- 400 galaxy lenses → σ(H0) = 0.20 km/s/Mpc → **precision cosmology**

---

## 📄 Publication

> McKnight, D. 2026, *Research Notes of the AAS*, AAS75139  
> "Time-Domain Recovery of Gravitational Lens Delays in Archival ZTF Photometry"

ZTF data accessed via the ALeRCE broker (alerce.online).  
Lens properties from the CASTLES gravitational lens database.  
AI-assisted code development using Claude (Anthropic) and Gemini (Google DeepMind).

---

## 🤝 Related Projects

- [ANTARES Alert Broker](https://antares.noirlab.edu) — real-time LSST filter submitted
- [Vera Rubin Observatory](https://rubinobservatory.org) — target data stream
- [ALeRCE Broker](https://alerce.online) — current ZTF data source
- [CASTLES Survey](https://lweb.cfa.harvard.edu/castles/) — lens seed catalog
- [TDCOSMO Collaboration](https://tdcosmo.github.io) — time-delay cosmography context

---

*Independent research project. Battle Creek, Michigan, USA.*

---

*Data from ALeRCE broker (ZTF stream). Rubin/LSST full stream launching 2025.*
