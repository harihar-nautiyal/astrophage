# 🪐 Astrophage

> **Two-Stage Random Forest Classifier Model for NASA Kepler Object of Interest (KOI) Exoplanet Validation**

[![Hackathon](https://img.shields.io/badge/Celesta-India%20High%20School%20Exoplanet%20Data%20Challenge%202026-blue)](https://celesta-exoplanet-challenge.devpost.com/)
[![Rust](https://img.shields.io/badge/Rust-1.85+-orange?logo=rust)](https://www.rust-lang.org/)
[![Polars](https://img.shields.io/badge/Polars-DataFrame-brightgreen)](https://pola.rs/)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.81%25-success)]()
[![Open](https://img.shields.io/badge/Open-Colab-orange)](https://colab.research.google.com/github/harihar-nautiyal/astrophage/blob/main/Astrophage_Colab.ipynb)

---

## What is Astrophage?

Astrophage is a high-performance exoplanet classification system built in **Rust** using **Polars** and a custom **Two-Stage Random Forest** implementation. It classifies Kepler Objects of Interest (KOIs) into three categories:

| Class | Description | Count |
|-------|-------------|-------|
| **CONFIRMED** ✅ | Validated exoplanets with high confidence | 2,747 |
| **CANDIDATE** 🔍 | Promising signals awaiting follow-up confirmation | 1,978 |
| **FALSE POSITIVE** ❌ | Non-planetary signals (stellar binaries, instrumental noise, etc.) | 4,839 |

```mermaid
pie title Class Distribution in KOI Dataset
    "FALSE POSITIVE" : 4839
    "CONFIRMED" : 2747
    "CANDIDATE" : 1978
```

> **Total Samples:** 9,564 | **Features:** 36 (28 base + 8 derived) | **Accuracy:** 94.81%

---

## Why Two-Stage?

Our architecture mirrors NASA's actual vetting workflow. Instead of forcing a single model to learn three classes simultaneously, we decompose the problem into two simpler binary decisions:

```mermaid
graph TD
    A[Raw KOI Data<br/>36 Features] --> B[Stage 1: CONFIRMED vs NOT CONFIRMED]
    B -->|CONFIRMED| C[Output: CONFIRMED ✅]
    B -->|NOT CONFIRMED| D[Stage 2: CANDIDATE vs FALSE POSITIVE]
    D -->|CANDIDATE| E[Output: CANDIDATE 🔍]
    D -->|FALSE POSITIVE| F[Output: FALSE POSITIVE ❌]

    style C fill:#2ecc71,stroke:#27ae60,color:#fff
    style E fill:#3498db,stroke:#2980b9,color:#fff
    style F fill:#e74c3c,stroke:#c0392c,color:#fff
```

This decomposition improves accuracy by **~3-4%** over a single-stage classifier because each stage learns a simpler, cleaner decision boundary.

---

## Key Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **94.81%** |
| **Macro F1** | **92.64%** |
| **Weighted F1** | **94.51%** |

```mermaid
graph LR
    subgraph "Overall Metrics"
        A[Accuracy<br/>94.81%]
        B[Macro F1<br/>92.64%]
        C[Weighted F1<br/>94.51%]
    end

    style A fill:#2ecc71,stroke:#27ae60,color:#fff
    style B fill:#3498db,stroke:#2980b9,color:#fff
    style C fill:#9b59b6,stroke:#8e44ad,color:#fff
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/harihar-nautiyal/astrophage.git
cd astrophage

# Build
cargo build --release

# Run
./target/release/astrophage
```

Or try it in your browser with **Google Colab** — no installation needed!

---

## Project Structure

```mermaid
graph TD
    A[astrophage/] --> B[Cargo.toml]
    A --> C[data/]
    A --> D[src/]
    A --> E[output/]

    C --> C1[koi_dataset.csv]

    D --> D1[main.rs]
    D --> D2[data.rs]
    D --> D3[features.rs]
    D --> D4[decision_tree.rs]
    D --> D5[model.rs]
    D --> D6[two_stage_model.rs]
    D --> D7[evaluation.rs]
    D --> D8[report.rs]

    E --> E1[report.json]

    style D1 fill:#f39c12,stroke:#e67e22,color:#fff
    style D6 fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## Technology Stack

```mermaid
graph LR
    A[Astrophage] --> B[Rust]
    A --> C[Polars]
    A --> D[NDArray]
    A --> E[Tokio]
    A --> F[Serde]

    B --> B1[Memory Safety]
    B --> B2[Zero-Cost Abstractions]
    B --> B3[SIMD-Friendly]

    C --> C1[Fast CSV I/O]
    C --> C2[Columnar Operations]

    D --> D1[Vectorized Math]
    D --> D2[N-Dimensional Arrays]

    E --> E1[Async Runtime]

    F --> F1[JSON Serialization]
```

---

<p align="center">
  <i>"Somewhere, something incredible is waiting to be known."</i><br>
  — Carl Sagan
</p>
