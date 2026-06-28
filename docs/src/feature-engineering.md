# Feature Engineering

## Philosophy

Feature engineering is where astrophysics meets machine learning. We don't just throw raw data at a model — we encode domain knowledge about how planets, stars, and false positives behave.

```mermaid
graph LR
    A[Raw Data] --> B[Domain Knowledge]
    B --> C[Engineered Features]
    C --> D[Better Model]

    style B fill:#f39c12,stroke:#e67e22,color:#fff
    style D fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## Base Features (28)

These are the raw astrophysical measurements from the Kepler pipeline:

```mermaid
graph TB
    subgraph "Orbital Parameters"
        A1[koi_period]
        A2[koi_duration]
        A3[koi_impact]
        A4[koi_ingress]
        A5[koi_incl]
        A6[koi_eccen]
        A7[koi_sma]
    end

    subgraph "Physical Parameters"
        B1[koi_ror]
        B2[koi_prad]
        B3[koi_teq]
        B4[koi_insol]
    end

    subgraph "Signal Quality"
        C1[koi_model_snr]
        C2[koi_count]
        C3[koi_num_transits]
        C4[koi_max_sngle_ev]
        C5[koi_max_mult_ev]
    end

    subgraph "False Positive Flags"
        D1[koi_fpflag_nt]
        D2[koi_fpflag_ss]
        D3[koi_fpflag_co]
        D4[koi_fpflag_ec]
    end

    subgraph "Stellar Parameters"
        E1[koi_kepmag]
        E2[koi_dor]
        E3[koi_srho]
        E4[koi_steff]
        E5[koi_slogg]
        E6[koi_smet]
        E7[koi_srad]
        E8[koi_smass]
    end
```

---

## Derived Features (8)

These are where the magic happens. Each derived feature encodes a specific astrophysical insight:

### 1. `fpflag_sum` — Total Suspicion Score

```mermaid
graph LR
    A[koi_fpflag_nt] -->|+| B[fpflag_sum]
    C[koi_fpflag_ss] -->|+| B
    D[koi_fpflag_co] -->|+| B
    E[koi_fpflag_ec] -->|+| B

    B -->|Value > 0| F[Almost certainly<br/>FALSE POSITIVE]
    B -->|Value = 0| G[Needs further<br/>analysis]

    style F fill:#e74c3c,stroke:#c0392c,color:#fff
    style G fill:#3498db,stroke:#2980b9,color:#fff
```

> **Importance: 0.2918** — The single most important feature. NASA already did the hard work of flagging suspicious signals; we just aggregate those flags.

---

### 2. `snr_x_prad` — Signal Consistency

```mermaid
graph LR
    A[Real Planet] -->|Jupiter-sized| B[High SNR expected]
    A -->|Earth-sized| C[Low SNR expected]

    D[Inconsistent Signal] -->|Large planet<br/>Low SNR| E[FALSE POSITIVE<br/>suspicion]
    D -->|Small planet<br/>Very high SNR| E

    B --> F[snr_x_prad<br/>consistent]
    C --> F
    E --> G[snr_x_prad<br/>inconsistent]

    style F fill:#2ecc71,stroke:#27ae60,color:#fff
    style G fill:#e74c3c,stroke:#c0392c,color:#fff
```

> **Importance: 0.0390** — Real planets have SNR proportional to their size. A Jupiter-sized object with weak SNR is suspicious.

---

### 3. `depth_duration_ratio` — Transit Shape

```mermaid
graph LR
    subgraph "Planet Transit"
        A1[U-shaped curve]
        A2[Specific depth/duration<br/>ratio]
    end

    subgraph "Stellar Eclipse"
        B1[V-shaped curve]
        B2[Different depth/duration<br/>ratio]
    end

    A1 --> C[depth_duration_ratio<br/>~ planet signature]
    B1 --> D[depth_duration_ratio<br/>~ binary signature]

    style C fill:#2ecc71,stroke:#27ae60,color:#fff
    style D fill:#e74c3c,stroke:#c0392c,color:#fff
```

> **Importance: 0.0239** — Planets produce U-shaped transits; stellar binaries produce V-shaped eclipses. The ratio captures this difference.

---

### 4. `koi_prad_squared` — Non-Linear Radius Effect

```mermaid
graph LR
    A[Planetary Radius] --> B[Linear: prad]
    A --> C[Non-linear: prad²]

    B --> D[Gradual increase]
    C --> E[Sharp threshold<br/>at ~15 R⊕]

    E -->|> 15 R⊕| F[Stellar companion<br/>not a planet]
    E -->|< 15 R⊕| G[Could be a planet]

    style F fill:#e74c3c,stroke:#c0392c,color:#fff
    style G fill:#2ecc71,stroke:#27ae60,color:#fff
```

> **Importance: 0.0275** — Objects larger than ~15 Earth radii are almost certainly stellar companions, not planets. The squared term captures this threshold.

---

### 5. `impact_penalty` — Physical Impossibility

```mermaid
graph LR
    A[Impact Parameter] -->|b < 1.0| B[Physical transit<br/>possible]
    A -->|b > 1.0| C[No transit possible<br/>by geometry]

    B --> D[impact_penalty = 0]
    C --> E[impact_penalty = 10]

    E --> F[Strong FALSE<br/>POSITIVE signal]

    style D fill:#2ecc71,stroke:#27ae60,color:#fff
    style E fill:#e74c3c,stroke:#c0392c,color:#fff
    style F fill:#e74c3c,stroke:#c0392c,color:#fff
```

> An impact parameter > 1.0 means the planet would miss the star entirely. Any signal with this value is physically impossible as a transit.

---

### 6. `log_period` — Orbital Distribution

```mermaid
graph LR
    A[Orbital Period] --> B[Linear scale: skewed]
    A --> C[Log scale: normal]

    B --> D[Hard to model]
    C --> E[Log-normal<br/>distribution]

    E --> F[Better model<br/>fit]

    style F fill:#2ecc71,stroke:#27ae60,color:#fff
```

> Planetary orbital periods follow a log-normal distribution. Taking the log makes the feature more Gaussian and easier for the model to learn.

---

### 7. `teq_over_steff` — Temperature Sanity Check

```mermaid
graph LR
    A[Equilibrium Temp] -->|/| B[Stellar Temp]
    B --> C[teq_over_steff]

    C -->|~ 0.1-0.5| D[Plausible]
    C -->|> 1.0| E[Implausible<br/>teq > steff]

    style D fill:#2ecc71,stroke:#27ae60,color:#fff
    style E fill:#e74c3c,stroke:#c0392c,color:#fff
```

> A planet's equilibrium temperature should never exceed its host star's temperature. This ratio is a simple sanity check.

---

### 8. `prad_teq_interaction` — Size-Temperature Relationship

```mermaid
graph LR
    A[Hot Jupiters] -->|Large + Hot| B[High prad_teq]
    C[Rocky Planets] -->|Small + Cool| D[Low prad_teq]

    B --> E[Distinct population]
    D --> E

    style E fill:#3498db,stroke:#2980b9,color:#fff
```

> This interaction helps distinguish between giant planets (large + hot) and rocky planets (small + cool).

---

## Feature Importance Ranking

```mermaid
graph LR
    subgraph "Top 5 Features"
        A1[fpflag_sum<br/>0.2918]
        A2[koi_fpflag_co<br/>0.0683]
        A3[koi_max_mult_ev<br/>0.0630]
        A4[koi_fpflag_nt<br/>0.0624]
        A5[koi_model_snr<br/>0.0596]
    end

    subgraph "Next 5"
        B1[koi_fpflag_ss<br/>0.0450]
        B2[koi_prad<br/>0.0437]
        B3[snr_x_prad<br/>0.0390]
        B4[koi_count<br/>0.0324]
        B5[koi_ror<br/>0.0300]
    end
```

---

## Preprocessing Pipeline

```mermaid
graph LR
    A[Raw Features<br/>28 columns] --> B[Imputation]
    B --> C[Standardization]
    C --> D[Derived Features<br/>8 columns]
    D --> E[Final Feature Matrix<br/>36 columns]

    B -->|Missing values| B1[Column median]
    C -->|Z-score| C1[Mean=0, Std=1]

    style E fill:#2ecc71,stroke:#27ae60,color:#fff
```

### Missing Value Imputation

```mermaid
graph TD
    A[Feature Column] --> B{Contains<br/>NaN?}
    B -->|Yes| C[Collect valid values]
    C --> D[Sort values]
    D --> E[Take median]
    E --> F[Replace NaN with median]
    B -->|No| G[Keep as-is]

    style F fill:#2ecc71,stroke:#27ae60,color:#fff
```

### Z-Score Standardization

```mermaid
graph LR
    A[Raw Value x] --> B[Subtract Mean]
    B --> C[Divide by Std]
    C --> D[Standardized Value<br/>(x - μ) / σ]

    style D fill:#2ecc71,stroke:#27ae60,color:#fff
```

> Standardization ensures all features contribute equally to distance-based calculations. Without it, features with large scales (like period in days) would dominate over small-scale features (like impact parameter).

---

## Feature Correlation Insight

```mermaid
graph TB
    subgraph "Feature Groups"
        A1[FP Flags] --- A2[fpflag_sum]
        A1 --- A3[koi_fpflag_nt]
        A1 --- A4[koi_fpflag_ss]

        B1[Size] --- B2[koi_prad]
        B1 --- B3[koi_ror]
        B1 --- B4[snr_x_prad]

        C1[Signal] --- C2[koi_model_snr]
        C1 --- C3[koi_max_mult_ev]
        C1 --- C4[koi_max_sngle_ev]

        D1[Orbit] --- D2[koi_period]
        D1 --- D3[log_period]
        D1 --- D4[koi_duration]
    end

    A2 --> E[High Importance]
    B2 --> E
    C2 --> E
    D3 --> E
```
