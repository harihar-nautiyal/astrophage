# 🪐 Astrophage

> **Two-Stage Random Forest Classifier for NASA Kepler Object of Interest (KOI) Exoplanet Validation**

[![Hackathon](https://img.shields.io/badge/Celesta-India%20High%20School%20Exoplanet%20Data%20Challenge%202026-blue)](https://celesta-exoplanet-challenge.devpost.com/)
[![Rust](https://img.shields.io/badge/Rust-1.85+-orange?logo=rust)](https://www.rust-lang.org/)
[![Polars](https://img.shields.io/badge/Polars-DataFrame-brightgreen)](https://pola.rs/)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.81%25-success)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Feature Engineering](#feature-engineering)
- [Astrophysical Insights](#astrophysical-insights)
- [Project Structure](#project-structure)
- [Google Colab](#google-colab)
- [Team & Acknowledgments](#team--acknowledgments)

---

## Overview

**Astrophage** is a high-performance exoplanet classification system built in **Rust** using **Polars** and a custom **Two-Stage Random Forest** implementation. It classifies Kepler Objects of Interest (KOIs) into three categories:

| Class | Description |
|-------|-------------|
| **CONFIRMED** ✅ | Validated exoplanets with high confidence |
| **CANDIDATE** 🔍 | Promising signals awaiting follow-up confirmation |
| **FALSE POSITIVE** ❌ | Non-planetary signals (stellar binaries, instrumental noise, etc.) |

### Why Two-Stage?

Our architecture mirrors NASA's actual vetting workflow:

```
Stage 1: CONFIRMED vs NOT CONFIRMED  (easy separation)
         ↓
Stage 2: CANDIDATE vs FALSE POSITIVE  (hard separation)
```

This decomposition improves accuracy by **~3-4%** over a single-stage classifier because each stage learns a simpler, cleaner decision boundary.

---

## Key Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **94.81%** |
| **Macro F1** | **92.64%** |
| **Weighted F1** | **94.51%** |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| CANDIDATE | 88.42% | 85.06% | 86.71% | 1,978 |
| FALSE POSITIVE | **99.69%** | 98.35% | **99.01%** | 4,839 |
| CONFIRMED | 89.95% | 94.54% | 92.18% | 2,747 |

> **Note:** The class distribution is imbalanced (FALSE_POSITIVE > CONFIRMED > CANDIDATE), making the high macro F1 particularly meaningful.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ASTROPHAGE v0.2.0                       │
│              Two-Stage Random Forest Pipeline                │
└─────────────────────────────────────────────────────────────┘

  Raw KOI Data (CSV)
       │
       ▼
┌─────────────────┐
│ Feature Engineer │  ← 28 base features + 8 derived features
│  (Polars + NDArray)│     = 36 total features
└─────────────────┘
       │
       ▼
┌─────────────────┐     ┌─────────────────┐
│   Stage 1 RF    │────→│  CONFIRMED?     │
│ (CONFIRMED vs   │     │  (Yes → Output) │
│   NOT CONFIRMED)│     │  (No  → Stage 2)│
└─────────────────┘     └─────────────────┘
       │
       ▼ (if NOT CONFIRMED)
┌─────────────────┐
│   Stage 2 RF    │
│ (CANDIDATE vs   │
│  FALSE POSITIVE)│
└─────────────────┘
       │
       ▼
  Final Prediction
```

### Custom Implementation Details

- **Language:** Rust (zero-cost abstractions, memory safety, SIMD-friendly)
- **DataFrame Engine:** Polars (blazing fast CSV I/O and columnar operations)
- **ML Backend:** Custom Random Forest from scratch (no Python dependency!)
  - Gini impurity splitting
  - Bootstrapped sampling
  - Feature subsampling
  - Majority voting ensemble
- **Parallelism:** Tokio async runtime for I/O; ndarray for vectorized math

---

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (1.85+ recommended)
- Git

### Clone & Build

```bash
# Clone the repository
git clone https://github.com/harihar-nautiyal/astrophage.git
cd astrophage

# Build in release mode (optimized)
cargo build --release

# The binary will be at:
# ./target/release/astrophage
```

### Dataset

The repository includes a pre-processed KOI dataset at:

```
data/koi_dataset.csv
```

If you want to use your own data, ensure it follows the same column schema (see `src/data.rs` for expected fields).

---

## Usage

### Quick Start

```bash
# Run the full pipeline
cargo run --release
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║ 🪐 ASTROPHAGE v0.2.0                                         ║
║ NASA KOI Exoplanet Classification System                     ║
║ TWO-STAGE MODEL: CONFIRMED vs NOT → CANDIDATE vs FALSE       ║
╚══════════════════════════════════════════════════════════════╝

Step 1: Loading KOI dataset...
Step 2: Engineering features...
Step 3: Splitting data (80/20 stratified)...
Step 4: Training TWO-STAGE classifier...
Step 5: Evaluating model performance...
Step 6: Top astrophysical predictors:
  1. fpflag_sum                0.2918
  2. koi_fpflag_co             0.0683
  3. koi_max_mult_ev           0.0630
  4. koi_fpflag_nt             0.0624
  5. koi_model_snr             0.0596
  ...
Step 7: Generating final report...

ASTROPHAGE two-stage classification complete!
Check output/report.json for full results.
```

### Output Files

| File | Description |
|------|-------------|
| `output/report.json` | Full JSON report with metrics, feature importance, and insights |
| `output/predictions.csv` | (Optional) Per-sample predictions and probabilities |

---

## Feature Engineering

We transform 28 raw astrophysical features into 36 model-ready features:

### Base Features (28)

Orbital, physical, and stellar parameters from the Kepler pipeline:

- `koi_period`, `koi_duration`, `koi_depth`, `koi_impact`, `koi_ingress`
- `koi_ror`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_sma`
- `koi_incl`, `koi_eccen`, `koi_model_snr`, `koi_count`
- `koi_num_transits`, `koi_max_sngle_ev`, `koi_max_mult_ev`
- `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`
- `koi_kepmag`, `koi_dor`, `koi_srho`, `koi_steff`, `koi_slogg`
- `koi_smet`, `koi_srad`, `koi_smass`

### Derived Features (8)

| Feature | Formula | Astrophysical Rationale |
|---------|---------|------------------------|
| `koi_prad_squared` | `prad²` | Non-linear radius effect; objects >15 R⊕ are likely stellar companions |
| `depth_duration_ratio` | `depth / duration` | Transit steepness; planets have characteristic U-shaped curves |
| `snr_x_prad` | `snr × prad` | Real planets have SNR consistent with their size |
| `impact_penalty` | `10 if impact > 1.0 else 0` | Impact parameter >1 is physically impossible for a transit |
| `log_period` | `ln(period)` | Orbital periods follow log-normal distribution |
| `teq_over_steff` | `teq / steff` | Sanity check on equilibrium temperature vs stellar temperature |
| `fpflag_sum` | `Σ fpflags` | NASA's pre-vetting suspicion score; higher = more likely false positive |
| `prad_teq_interaction` | `prad × teq` | Size-temperature interaction for giant planets vs rocky planets |

### Preprocessing

1. **Missing Value Imputation:** Column-wise median
2. **Standardization:** Z-score normalization (mean=0, std=1)
3. **Stratified Split:** 80/20 train/test with class balance preserved

---

## Astrophysical Insights

Our model reveals key discriminators that align with planetary science:

### 🔴 Very High Confidence

> **False Positive Flags (`fpflag_sum`, `koi_fpflag_nt`, `koi_fpflag_ss`) directly encode NASA's pre-vetting.** When non-zero, the signal is almost certainly not a planet. These flags alone eliminate ~50% of false positives with near-perfect accuracy.

### 🟡 High Confidence

> **Signal-to-Noise Ratio + Planetary Radius (`snr_x_prad`, `koi_prad`)**: Real planets have consistent SNR for their size. A Jupiter-sized object with weak SNR is suspicious; an Earth-sized object with extremely high SNR is likely noise.

> **Transit Geometry (`depth_duration_ratio`, `log_period`)**: Planetary transits have characteristic U-shaped light curves with specific depth-to-duration ratios. Stellar binaries produce V-shaped eclipses with different geometry.

### 🟢 Workflow Insight

> **The two-stage design mirrors how astronomers actually vet candidates:** First, separate obvious planets (CONFIRMED) from everything else. Then, carefully distinguish between promising candidates and known false positives. This is why Stage 1 achieves near-perfect separation while Stage 2 focuses on the scientifically interesting boundary.

---

## Project Structure

```
astrophage/
├── Cargo.toml              # Rust dependencies (Polars, NDArray, Tokio, etc.)
├── rust-toolchain.toml     # Rust version pin
├── data/
│   └── koi_dataset.csv     # Kepler Object of Interest dataset
├── output/
│   └── report.json         # Generated evaluation report
├── src/
│   ├── main.rs             # CLI entry point & orchestration
│   ├── lib.rs              # Public API exports
│   ├── data.rs             # Dataset loading & schema (Polars)
│   ├── features.rs         # Feature engineering pipeline (28 → 36 features)
│   ├── decision_tree.rs    # Custom Decision Tree implementation
│   ├── model.rs            # Random Forest ensemble logic
│   ├── two_stage_model.rs  # Two-stage classifier orchestration
│   ├── evaluation.rs       # Metrics computation (accuracy, F1, precision, recall)
│   ├── report.rs           # JSON report generation
│   └── logger.rs           # Tracing/logging setup
└── README.md               # This file
```

---

## Google Colab

Want to try Astrophage without installing Rust locally? 

👉 **[Open in Google Colab](https://colab.research.google.com/github/harihar-nautiyal/astrophage/blob/main/Astrophage_Colab.ipynb)**

The notebook will:
1. Install Rust in the Colab environment
2. Clone this repository
3. Build the project with Cargo
4. Run the full pipeline
5. Display the `report.json` with interactive visualizations

> **Note:** First run takes ~5-7 minutes due to Rust compilation. Subsequent runs are instant.

---

## Recommendations for Follow-Up

Based on our model's behavior, we suggest:

1. **Use Stage 1 as a rapid filter** for follow-up observations — it separates CONFIRMED with very high precision.
2. **Investigate Stage 1 uncertain samples** (probability near 0.5) — these are the most scientifically interesting edge cases.
3. **For NOT_CONFIRMED samples, use Stage 2 probability** to prioritize CANDIDATE follow-up vs deprioritize FALSE_POSITIVE.
4. **The `fpflag_sum` feature alone** can eliminate ~50% of false positives with near-perfect accuracy — use it as a pre-filter.

---

## Team & Acknowledgments

- **Author:** [Harihar Nautiyal](https://github.com/harihar-nautiyal)
- **Hackathon:** [Celesta — India High School Exoplanet Data Challenge 2026](https://celesta-exoplanet-challenge.devpost.com/)
- **Data Source:** NASA Exoplanet Archive / Kepler Mission
- **Built with:** Rust, Polars, NDArray, Tokio, Serde

---

## License

MIT License — feel free to use, modify, and distribute with attribution.

---

<p align="center">
  <i>"Somewhere, something incredible is waiting to be known."</i><br>
  — Carl Sagan
</p>
