# Project Vera Rubin — Phase 1 Technical Note v3
## Gravitational Lens Time-Delay Discovery Pipeline: ZTF Validation and Rubin LSST Readiness

**Author:** David E. McKnight, Independent Researcher, Battle Creek, Michigan  
**Date:** March 2026  
**Status:** Phase 1 Complete — Awaiting Rubin LSST Alert Stream  
**Publication:** McKnight 2026, RNAAS AAS75139 *(approved)*

---

## 1. Executive Summary

This document records the complete scientific and technical state of the Project Vera Rubin gravitational lens discovery pipeline as of March 2026. The pipeline has been validated on archival ZTF photometry, a proof-of-concept paper has been published in the Research Notes of the AAS, and all components are staged and ready for Rubin LSST data.

**Core result:** Two validated gravitational lens time-delay detections from ZTF archival data using the ZDCF method. H0 = 62.4 ± 9.1 km/s/Mpc from time-delay cosmography. Five-scenario cosmological framework operational. Three cosmological tension monitors running.

**Next target:** First new gravitational lens discovery — an unknown system not in any existing catalog, found from time-domain variability correlation alone. This is the publishable result that clears the AAS bar.

---

## 2. Publication Record

**AAS75139** — *"Time-Domain Recovery of Gravitational Lens Delays in Archival ZTF Photometry"*
- Submitted to RNAAS, approved by Editor-in-Chief Ethan T. Vishniac
- Status: **Published, DOI pending from IOP**
- Key claim: ZDCF pipeline recovers known time delays for J1004+4112 and J1029+2623 from archival ZTF photometry without prior knowledge of the target delays during pipeline execution

**Rejection note (AAS, March 2026):** The Editor-in-Chief replied: *"This is not a judgment about whether or not it is correct. Assuming it is correct, this result is not sufficiently important."* The result needed to be a NEW discovery, not a recovery of known delays. This redirects the scientific target clearly: the next paper must report a previously unknown gravitational lens system.

---

## 3. Validated Detections

### 3.1 SDSS J1004+4112
- Separation: 14.52"
- ZTF pair: ZTF21aaoleaj × ZTF19aanfixr
- ZDCF plateau: 800–950d, peak at 877.5d
- Published delay: ~821d (Fohlmeister et al. 2013)
- Residual: +6.9%
- Score: 0.408 | Rungs: 3/6
- H0: 65.5 ± 12.3 km/s/Mpc

### 3.2 SDSS J1029+2623
- Separation: 22.37"
- ZTF pair: ZTF18aceypuf × ZTF19aailazi
- ZDCF plateau: 750–950d, peak at 887.5d
- Published delay: ~746d (Fohlmeister et al. 2013)
- Residual: +19.3%
- Score: 0.314 | Rungs: 3/6
- H0: 58.7 ± 13.5 km/s/Mpc

### 3.3 Combined H0
- **H0 = 62.4 ± 9.1 km/s/Mpc** (inverse-variance weighted, 2 lenses)
- Method: Scaling approach — H0 = H0_ref × (Δt_published / Δt_measured)
- Dominant uncertainty: cluster mass model systematic (15–20%)
- Note: The 15–20% systematic floor is the local mass-sheet degeneracy (Wagner 2018, A&A) — irreducible from ZDCF alone. Breakable with resolved image shapes + velocity dispersions (available from Rubin resolved imaging).
- Consistent with all H0 measurements within uncertainties (0.55σ from CMB, 1.16σ from SH0ES)

---

## 4. Key Diagnostic Results

### 4.1 The 887.5-Day Coincidence
Both systems initially returned the same lag (887.5d), which was alarming. Three diagnostic tests resolved this:

1. **Bin sensitivity test:** At 25-day bins, both true delays (821d and 746d) fall in the same bin centered at 887.5d. At 5-day bins they separate correctly. This is a measurement resolution artifact, not a physical signal.

2. **Lag bias scan:** Pipeline sensitivity peaks at 600–1000d given ZTF's ~2500-day baseline. Documented as selection effect, not algorithm bias.

3. **Seasonal alias test:** Correlation at 887.5d for an uncorrelated pair = -0.24. No cadence-induced alias at this lag.

**Resolution:** Production pipeline uses 5-day bins with a noise-peak guard (minimum 8 pairs per bin). Both systems now show distinct ZDCF plateaus.

### 4.2 H0 Unit Conversion
Initial `h0_estimator.py` used Fermat potentials from Fohlmeister et al. (0.2270 arcsec²) with a direct arcsec² → rad² conversion, producing H0 = 4.3 km/s/Mpc (wrong by factor ~16). The Fermat potential in Fohlmeister is defined within a specific cluster mass model and cannot be converted by simple unit arithmetic. Corrected to the scaling method (Section 3.3), which is physically transparent and correctly propagates the mass model uncertainty.

---

## 5. Cosmological Analysis Framework

### 5.1 H0 Pipeline (h0_pipeline.py)
Five modules:
- **Module 1:** H0 estimator (scaling method)
- **Module 2:** ASCII posterior visualizer
- **Module 3:** DESI w0waCDM comparison — H0 bias of ~-10 km/s/Mpc at z_L~0.6-0.7 under DESI best-fit (w0=-0.827, wa=-0.75) is physically correct, not a bug. This is the signal we will measure with Rubin.
- **Module 4:** Sensitivity analysis — N=13 galaxy lenses gives 5σ discrimination
- **Module 5:** PNG comparison plot

### 5.2 Anisotropy Monitor (anisotropy_monitor.py)
Foundation layer provides coordinate transforms (IAU eq→galactic), equal-area sky tessellation (~360 pixels, ~115 deg² each), Doppler boost calculations (CMB apex at RA=167.99°, Dec=-6.99°, v=369.82 km/s), and the single entry point `tag_and_store_detection()` called automatically from the survey engine.

Three science modules:
- **Module A:** H0 directional map — tests H0 variation with sky position
- **Module B:** Cosmic dipole — tests Cosmological Principle via N+/N- asymmetry
- **Module C:** S8 probe — tests matter clumping via strong lens count rate

Current state (2 lenses): Both lenses at l~179-206°, b~+53-58° (galactic north, high latitude, clean sky). Both toward CMB dipole apex (cos θ = +0.63, +0.82). S8 inferred = 0.877 ± 0.310, consistent with Planck.

### 5.3 Mstep Comparison (mstep_comparison.py)
Implements Bansal & Huterer 2025 framework. Five scenarios:
1. ΛCDM: H0=67.4 everywhere
2. SH0ES: H0=73.0 everywhere
3. Mstep z_t=0.01: H0~67 for ALL lenses (z_L always > 0.01) — indistinguishable from ΛCDM via lensing alone
4. Mstep z_t=0.15: H0~70.5 for z_L<0.15, H0~67 for z_L>0.15 — **testable** with Rubin compact lenses
5. DESI w0waCDM: H0 bias grows with z_L

Key insight: The trivial Mstep solution (z_t~0.01) cannot be tested with gravitational lensing — all lenses have z_L >> 0.01. The physically interesting sweet spot (z_t~0.15) IS testable once Rubin finds compact lenses at z_L < 0.15. The discriminating observation is H0(z_L<0.15) vs H0(z_L>0.15).

Current χ² ranking (2 lenses, illustrative only):
1. Mstep z_t=0.15: χ²=0.441
2. ΛCDM: χ²=0.443
3. Mstep z_t=0.01: χ²=0.443
4. DESI w0waCDM: χ²=0.627
5. SH0ES: χ²=1.502

---

## 6. Rubin LSST Readiness

### 6.1 Settings Switch
`rubin_settings_patch.py` contains `SETTINGS_ZTF` (current) and `SETTINGS_RUBIN` (awaiting RSP account):

| Parameter | ZTF | Rubin | Reason |
|---|---|---|---|
| min_sep | 0.5" | 0.1" | Opens compact galaxy lenses |
| zdcf_bin | 5.0d | 2.0d | Detects HE0435 (14-day delay) |
| lag_range | ±900d | ±1500d | Covers all known delays |
| min_det | 30 | 20 | Rubin nightly cadence |
| radius | 60" | 30" | Rubin catalog is denser |

### 6.2 Compact Lens Catalog
Nine compact systems added to `COMPACT_LENS_CATALOG` in `rubin_settings_patch.py`:
HE0435-1223 (1.46", 14d), PG1115+080 (1.79", 24d), RXJ1131-1231 (1.83", 91d), B1608+656 (1.36", 77d), WFI2033-4723 (1.66", 125d), J1206+4332 (1.37", 111d), HE1104-1805 (3.19", 162d), SDSS0246-0825 (1.04"), HS2209+1914 (1.20"). All inaccessible to ZTF due to resolution limit.

### 6.3 ANTARES Broker Filter
Submitted as MR !1 to `gitlab.com/OlderDyad/antares`. Filter pre-screens LSST alert loci for AGN-like variability properties before passing to ZDCF scorer. When merged, provides real-time push of candidates rather than periodic pull queries.

### 6.4 RSP TAP Adapter
`rubin_tap_adapter.py` connects to Rubin Science Platform via TAP protocol. RSP account pending approval from Heather Shaughnessy (SLAC). When available: save token to `~/.rsp-tap.token`, run `python rubin_tap_adapter.py` to verify, switch SETTINGS to SETTINGS_RUBIN.

---

## 7. Literature Context

### 7.1 Bansal & Huterer 2025
Key finding: no late-time expansion modification can simultaneously fit SH0ES + DESI BAO + SNIa. Only trivial solutions (Mstep at z_t~0.01 or Etherington duality breaking) achieve good fits. The intermediate Mstep at z_t~0.15 is a "sweet spot" with modest improvement (Δχ²~-16). Our z_L~0.6 lenses probe exactly the redshift window relevant to this analysis.

Time-delay lensing is fully independent of all these datasets — the one clean probe available.

### 7.2 Wagner 2018–2019 (Heidelberg)
Six-paper series on model-independent strong lensing characterization. Directly relevant:
- Paper IV (A&A 2018): Mass-sheet degeneracy is the fundamental systematic floor. For cluster lenses this is 15–20%; for galaxy lenses breakable with velocity dispersions to ~5%.
- Paper V (MNRAS 2019): Cosmology-free distance measure from SNIa — applicable to our DESI module.
- Paper VI (MNRAS 2019): H0 uniquely determined if total integrated mass along light paths is known. This is what velocity dispersions provide.

### 7.3 TDCOSMO 2025
Eight galaxy lenses, hierarchical Bayesian inference, JWST NIRSpec kinematics. Current result: H0 = 73.3 ± 1.7 km/s/Mpc. Their systematic floor: ~3% from mass model + kinematics. Target for Rubin pipeline: reach similar precision with larger sample.

---

## 8. Pending Items

| Item | Description | ETA |
|---|---|---|
| RSP account | Rubin Science Platform — Heather, SLAC | Days–weeks |
| ANTARES MR !1 | Broker filter review | Weeks |
| Rubin alerts | Week 21 engineering tests in progress | Weeks |
| DOI assignment | IOP publishing for AAS75139 | Days |
| New discovery paper | First unknown system from pipeline | Rubin Year 1 |

---

## 9. File Inventory

| File | Purpose | Status |
|---|---|---|
| `run_pipeline.py` | Master run script | ✓ Production |
| `rubin_survey_v3.py` | ZDCF discovery engine | ✓ Production |
| `lc_utils.py` | ZDCF/scoring utilities | ✓ Production |
| `h0_pipeline.py` | H0 cosmography (5 modules) | ✓ Production |
| `anisotropy_monitor.py` | Cosmological tensions (3 modules) | ✓ Production |
| `mstep_comparison.py` | Bansal & Huterer 2025 scenarios | ✓ Production |
| `rubin_tap_adapter.py` | Rubin RSP connector | ✓ Staged |
| `antares_lens_consumer.py` | ANTARES Kafka consumer | ✓ Staged |
| `rubin_settings_patch.py` | ZTF/Rubin settings + compact catalog | ✓ Production |
| `zdcf_bin_diagnostic.py` | 3-test validation suite | ✓ Archive |
| `seeds/castles_wide_seeds.csv` | 19 CASTLES targets | ✓ Production |
| `outputs/survey_v3/survey_results.db` | SQLite: all results | ✓ Live |
| `outputs/diagnostic/h0_comparison.png` | H0 ladder + DESI bias | ✓ Production |
| `outputs/diagnostic/mstep_comparison.png` | Five-scenario curves | ✓ Production |

---

*David E. McKnight — Independent Researcher — Battle Creek, Michigan, USA*  
*GitHub: https://github.com/OlderDyad/Project-Vera-Rubin*