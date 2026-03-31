# 🔭 Project Vera Rubin — Gravitational Lens Discovery Pipeline

> **Automated time-delay gravitational lens discovery using the Zwicky Transient Facility (ZTF), with a direct pipeline path to the Vera Rubin Observatory LSST alert stream.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper: RNAAS AAS75139](https://img.shields.io/badge/Paper-RNAAS%20AAS75139-orange)](https://rnaas.aas.org)
[![Pages](https://img.shields.io/badge/Site-GitHub%20Pages-blue)](https://olderdyad.github.io/Project-Vera-Rubin/)

---

## 📋 Overview

This pipeline recovers gravitational lens time delays from archival survey photometry using the **Z-transformed Discrete Correlation Function (ZDCF)**. It identifies candidate lens pairs by detecting **correlated variability with a time lag** — the time echo of the same quasar seen through two different light paths — without relying on image morphology or prior lens catalogs.

**Key results to date:**
- ✅ Two validated detections from archival ZTF data — SDSS J1004+4112 and SDSS J1029+2623
- ✅ H0 = 62.4 ± 9.1 km/s/Mpc from time-delay cosmography (2 cluster lenses)
- ✅ Five-scenario cosmological framework testing Bansal & Huterer (2025) Mstep models
- ✅ Cosmological anisotropy monitor: H0 directional map, cosmic dipole, S8 tension probe
- ✅ Full Rubin LSST integration staged: TAP adapter, ANTARES broker filter, compact lens catalog
- 📄 **Published:** *"Time-Domain Recovery of Gravitational Lens Delays in Archival ZTF Photometry"* — Research Notes of the AAS (AAS75139, 2026)

---

## 🔬 Scientific Context

Gravitational lenses with measured time delays provide independent H0 constraints, completely free of the local distance ladder and the CMB sound horizon. The project targets three major open questions in cosmology:

| Tension | Method | Status |
|---|---|---|
| **Hubble tension** | H0 from time-delay cosmography | ✓ Pipeline operational |
| **Cosmic dipole anomaly** | AGN count asymmetry (variability-confirmed catalog) | ✓ Framework built |
| **S8 clumpiness tension** | Strong lens quad count rate vs ΛCDM | ✓ Framework built |

**Why time-domain discovery?** Rubin LSST will generate ~10 million alerts per night. This pipeline identifies lenses by their *behavior* — correlated variability with a lag — rather than by morphology, making it scalable to the full alert stream without human inspection per object.

**Why variability-confirmed AGN for the dipole?** Stellar contamination is the dominant systematic in radio/optical dipole studies. The ZDCF time-echo criterion is a physical filter: only gravitationally lensed extragalactic sources produce a delayed variability echo. Stars and dust cannot.

---

## 📁 Repository Structure

```
Project-Vera-Rubin/
│
├── run_pipeline.py             # ← START HERE: master run script
│
├── Core Discovery
│   ├── rubin_survey_v3.py      # ZDCF discovery engine (v3.3, 5-day bins)
│   ├── lc_utils.py             # ZDCF, quality_check, prep_lc utilities
│   └── seeds/
│       ├── castles_wide_seeds.csv   # 19 CASTLES wide-separation targets
│       └── compact_lenses.csv       # 9 compact targets for Rubin mode
│
├── Cosmological Analysis
│   ├── h0_pipeline.py          # H0 cosmography — 5 modules
│   │   # Module 1: H0 estimator (scaling method, H0 ∝ 1/Δt)
│   │   # Module 2: Posterior visualizer
│   │   # Module 3: DESI w0waCDM comparison
│   │   # Module 4: Sensitivity analysis (N lenses needed)
│   │   # Module 5: PNG comparison plot
│   │
│   ├── anisotropy_monitor.py   # Cosmological tensions — 3 modules
│   │   # Foundation: eq_to_galactic(), sky_bin(), doppler_boost(),
│   │   #             tag_and_store_detection(), statistical utilities
│   │   # Module A: H0 directional map (H0 vs sky position)
│   │   # Module B: Cosmic dipole monitor (N+/N- asymmetry)
│   │   # Module C: S8 tension probe (strong lens count rate)
│   │
│   └── mstep_comparison.py     # Bansal & Huterer 2025 framework
│       # Five scenarios: ΛCDM, SH0ES, Mstep z_t=0.01,
│       #                 Mstep z_t=0.15, DESI w0waCDM
│       # χ² comparison per lens, combined posterior, N needed
│
├── Rubin / Broker Integration
│   ├── rubin_tap_adapter.py         # Rubin RSP TAP connector
│   ├── antares_lens_consumer.py     # ANTARES Kafka consumer
│   └── rubin_settings_patch.py      # ZTF vs Rubin SETTINGS + compact catalog
│
├── Diagnostics
│   ├── zdcf_bin_diagnostic.py       # 3-test validation suite
│   └── rerun_10day_bins.py          # Bin sensitivity on real data
│
└── outputs/
    ├── survey_v3/survey_results.db  # SQLite: candidates + all analysis tables
    └── diagnostic/
        ├── h0_comparison.png        # H0 ladder + DESI bias plot
        └── mstep_comparison.png     # Five-scenario H0(z_L) curves
```

---

## 🚀 Quick Start

### Requirements
```
Python 3.12 (64-bit)
scipy numpy pandas requests matplotlib astropy
```

```powershell
# Activate 64-bit environment
.venv64\Scripts\Activate.ps1

# Run the full pipeline
python run_pipeline.py --seeds seeds/castles_wide_seeds.csv --workers 4

# Analysis only (skip discovery, re-run cosmology modules)
python run_pipeline.py --analysis-only

# Discovery only
python run_pipeline.py --survey-only --seeds seeds/castles_wide_seeds.csv
```

### Individual modules
```powershell
python h0_pipeline.py          # H0 cosmography
python anisotropy_monitor.py   # Cosmological tensions
python mstep_comparison.py     # Bansal & Huterer scenario comparison
```

---

## 🔧 Pipeline Architecture

### Six-Rung Scoring Ladder

| Rung | Test | Method |
|------|------|--------|
| R1 | ZDCF lag detection | Z-transformed DCF peak significance |
| R2 | Flux ratio stability | Coefficient of variation < 35% |
| R3 | Stochasticity | Lomb-Scargle confirms AGN red noise |
| R4 | Fractional variability match | Pearson r > 0.50 post-shift |
| R5 | Microlensing signature | Long-term corr, short-term decorr |
| R6 | Mass-delay consistency | Cluster-adaptive scaling (sep > 10") |

### Key Design Choices

**5-day ZDCF bins:** Diagnostic testing showed 25-day bins collapse different true delays (821d and 746d for J1004/J1029) into a single artifactual peak. At 5-day bins the systems separate correctly.

**Noise-peak guard:** Bins with fewer than 8 pairs are masked to NaN, preventing spurious high-r peaks from sparsely populated lag bins.

**Cluster-adaptive Rung 6:** The SIS galaxy formula gives nonsensical predictions for wide-separation cluster lenses (sep > 10"). An empirical power-law scaling is used instead.

**Anisotropy hook:** Every detection that passes scoring automatically calls `tag_and_store_detection()` in `anisotropy_monitor.py`, tagging galactic coordinates, CMB dipole alignment, Doppler boost factor, and H0 estimate. All three science modules update automatically — no manual steps.

---

## 📊 Validated Detections

| System | Sep (") | ZDCF Plateau | Published Delay | Score | Rungs |
|--------|---------|--------------|-----------------|-------|-------|
| SDSS J1004+4112 | 14.52 | 800–950d | ~821d | 0.408 | 3/6 |
| SDSS J1029+2623 | 22.37 | 750–950d | ~746d | 0.314 | 3/6 |

**Combined H0:** 62.4 ± 9.1 km/s/Mpc (2 cluster lenses, scaling method)
Dominant uncertainty: cluster mass model systematic (~15–20%, Wagner 2018 MSD)

**Cosmological scenario comparison** (Bansal & Huterer 2025):
- Mstep z_t=0.01 (trivial): predicts H0~67 for ALL our lenses — indistinguishable from ΛCDM via lensing
- Mstep z_t=0.15 (testable): predicts H0~70.5 for z_L < 0.15 vs H0~67 for z_L > 0.15 — **testable with Rubin compact lenses**
- DESI w0waCDM: H0 bias grows with z_L, ~-10 km/s/Mpc at z_L~0.6-0.7

---

## 🌌 Rubin LSST Readiness

### Settings Switch (when RSP account arrives)

```python
# In rubin_survey_v3.py, change:
SETTINGS = SETTINGS_ZTF    # current
# to:
SETTINGS = SETTINGS_RUBIN  # from rubin_settings_patch.py
```

Key differences:
| Parameter | ZTF | Rubin |
|---|---|---|
| min_sep | 0.5" | 0.1" — opens compact lenses |
| zdcf_bin | 5.0d | 2.0d — detects HE0435 (14d delay) |
| lag_range | ±900d | ±1500d |
| min_det | 30 | 20 |

### Compact Lenses Now Accessible with Rubin

| System | Sep (") | Delay (d) | ZTF? |
|---|---|---|---|
| HE0435-1223 | 1.46 | 14 | ✗ |
| PG1115+080 | 1.79 | 24 | ✗ |
| RXJ1131-1231 | 1.83 | 91 | ✗ |
| B1608+656 | 1.36 | 77 | ✗ |
| WFI2033-4723 | 1.66 | 125 | ✗ |

### Broker Integration

| Component | Status |
|---|---|
| `rubin_tap_adapter.py` | Built — awaiting RSP account |
| ANTARES filter MR !1 | Submitted — awaiting review |

### Science Milestones

| N lenses | What becomes possible |
|---|---|
| 5–10 | H0 directional comparison (toward/away CMB dipole) |
| 13 | 5σ H0 discrimination (galaxy lenses, 5% systematic) |
| 50+ | DESI H0(z) test, Mstep z_t=0.15 discrimination |
| 72 | 3σ S8 discrimination from quad count rate |
| 400+ | ±0.2 km/s/Mpc H0 precision |

---

## 📄 Publication

> McKnight, D. 2026, *Research Notes of the AAS*, AAS75139
> *"Time-Domain Recovery of Gravitational Lens Delays in Archival ZTF Photometry"*

---

## 📚 Key References

- Bansal & Huterer 2025 — Mstep framework, late-time solutions to Hubble tension
- Wagner 2018, A&A — Mass-sheet degeneracy in strong lensing (systematic floor explanation)
- Wagner 2019, MNRAS — Data-based distances for cosmology-free H0
- Fohlmeister et al. 2013, ApJ — Published delays for J1004 and J1029
- Ellis & Baldwin 1984, MNRAS — Kinematic dipole prediction
- Oguri & Marshall 2010, MNRAS — Strong lens rate predictions (S8 probe baseline)
- DESI Collaboration 2025 — DR2 BAO, dynamical dark energy evidence

---

## 🤝 Related Projects

- [ANTARES Alert Broker](https://antares.noirlab.edu)
- [Vera Rubin Observatory](https://rubinobservatory.org)
- [ALeRCE Broker](https://alerce.online)
- [CASTLES Survey](https://lweb.cfa.harvard.edu/castles/)
- [TDCOSMO Collaboration](https://tdcosmo.github.io)

---

*Independent research. Battle Creek, Michigan, USA.*
