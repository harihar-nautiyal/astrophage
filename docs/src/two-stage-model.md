# Two-Stage Model

## The Core Innovation

Astrophage's defining feature is its **Two-Stage Random Forest** architecture. Instead of a single three-class classifier, we decompose the problem into two sequential binary decisions — exactly how NASA astronomers actually vet candidates.

```mermaid
graph TB
    subgraph "Problem Decomposition"
        A[3-Class Problem<br/>CONFIRMED vs CANDIDATE vs FALSE_POSITIVE]
        B[Stage 1<br/>CONFIRMED vs NOT_CONFIRMED]
        C[Stage 2<br/>CANDIDATE vs FALSE_POSITIVE]
    end

    A -->|Decompose| B
    A -->|Decompose| C

    style A fill:#e74c3c,stroke:#c0392c,color:#fff
    style B fill:#2ecc71,stroke:#27ae60,color:#fff
    style C fill:#3498db,stroke:#2980b9,color:#fff
```

---

## Why This Works

### The Astronomy Perspective

When NASA discovers a KOI, the vetting process is sequential:

1. **First question:** "Do we have overwhelming evidence this is a planet?" → If yes, **CONFIRMED**
2. **Second question:** "If not confirmed, is it worth follow-up?" → **CANDIDATE** or **FALSE_POSITIVE**

```mermaid
graph LR
    A[Discovery] --> B{Overwhelming<br/>Evidence?}
    B -->|Yes| C[CONFIRMED<br/>Follow-up complete]
    B -->|No| D{Promising<br/>Signal?}
    D -->|Yes| E[CANDIDATE<br/>Needs more data]
    D -->|No| F[FALSE_POSITIVE<br/>Discard]

    style C fill:#2ecc71,stroke:#27ae60,color:#fff
    style E fill:#3498db,stroke:#2980b9,color:#fff
    style F fill:#e74c3c,stroke:#c0392c,color:#fff
```

### The ML Perspective

```mermaid
graph TB
    subgraph "Single-Stage Classifier"
        A1[Decision Boundary<br/>Complex 3-way split]
        A2[Accuracy: ~91%]
        A3[Confusion between<br/>CANDIDATE & FALSE_POSITIVE]
    end

    subgraph "Two-Stage Classifier"
        B1[Stage 1: Simple linear<br/>separation for CONFIRMED]
        B2[Stage 2: Focused boundary<br/>between CANDIDATE & FALSE+]
        B3[Accuracy: ~94.8%]
        B4[Each stage learns<br/>a cleaner boundary]
    end

    style A2 fill:#e74c3c,stroke:#c0392c,color:#fff
    style B3 fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## Stage 1: CONFIRMED vs NOT CONFIRMED

### Decision Boundary

Stage 1 separates the "easy" class (CONFIRMED) from everything else. Confirmed planets have very strong, consistent signals:

```mermaid
graph LR
    subgraph "Feature Space"
        A[High SNR]
        B[Zero FP Flags]
        C[Consistent Radius]
        D[Regular Period]
    end

    A --> E[CONFIRMED Zone]
    B --> E
    C --> E
    D --> E

    F[Low SNR] --> G[NOT CONFIRMED Zone]
    H[Non-zero FP Flags] --> G
    I[Inconsistent Radius] --> G
    J[Irregular Period] --> G

    style E fill:#2ecc71,stroke:#27ae60,color:#fff
    style G fill:#e74c3c,stroke:#c0392c,color:#fff
```

### Performance

Stage 1 is nearly perfect because confirmed planets are genuinely distinct:

```mermaid
graph TD
    A[Stage 1 Performance] --> B[Precision: ~99%]
    A --> C[Recall: ~98%]
    A --> D[F1: ~99%]

    style B fill:#2ecc71,stroke:#27ae60,color:#fff
    style C fill:#2ecc71,stroke:#27ae60,color:#fff
    style D fill:#2ecc71,stroke:#27ae60,color:#fff
```

### Key Features

The most important features for Stage 1:

```mermaid
graph LR
    A[fpflag_sum] -->|Weight: 0.29| B[Stage 1]
    C[koi_model_snr] -->|Weight: 0.06| B
    D[koi_fpflag_nt] -->|Weight: 0.06| B
    E[koi_fpflag_ss] -->|Weight: 0.05| B
    F[koi_prad] -->|Weight: 0.04| B

    style B fill:#f39c12,stroke:#e67e22,color:#fff
```

---

## Stage 2: CANDIDATE vs FALSE POSITIVE

### The Hard Problem

This is where the science gets interesting. Candidates and false positives can look very similar:

```mermaid
graph TB
    subgraph "CANDIDATE Characteristics"
        A1[Moderate SNR]
        A2[Some transit-like shape]
        A3[Plausible radius]
        A4[No strong FP flags]
    end

    subgraph "FALSE POSITIVE Characteristics"
        B1[Variable SNR]
        B2[Non-transit shape possible]
        B3[Radius may be too large]
        B4[Subtle FP indicators]
    end

    A1 --- C[The Boundary]
    A2 --- C
    A3 --- C
    A4 --- C
    B1 --- C
    B2 --- C
    B3 --- C
    B4 --- C

    style C fill:#f39c12,stroke:#e67e22,color:#fff
```

### Performance

```mermaid
graph TD
    A[Stage 2 Performance] --> B[Precision: ~88%]
    A --> C[Recall: ~85%]
    A --> D[F1: ~87%]

    style B fill:#3498db,stroke:#2980b9,color:#fff
    style C fill:#3498db,stroke:#2980b9,color:#fff
    style D fill:#3498db,stroke:#2980b9,color:#fff
```

> Stage 2 is harder but also more scientifically valuable — these are the edge cases astronomers care about most.

---

## Combined Inference Pipeline

```mermaid
graph TD
    A[Input Sample<br/>36 Features] --> B{Stage 1:<br/>CONFIRMED?}

    B -->|Probability > 0.5| C[Output:<br/>CONFIRMED ✅]
    B -->|Probability <= 0.5| D{Stage 2:<br/>CANDIDATE?}

    D -->|Probability > 0.5| E[Output:<br/>CANDIDATE 🔍]
    D -->|Probability <= 0.5| F[Output:<br/>FALSE POSITIVE ❌]

    style C fill:#2ecc71,stroke:#27ae60,color:#fff
    style E fill:#3498db,stroke:#2980b9,color:#fff
    style F fill:#e74c3c,stroke:#c0392c,color:#fff
```

### Probability Flow

```mermaid
graph LR
    A[Input] --> B[Stage 1 RF]
    B -->|P(CONFIRMED) = 0.85| C[→ CONFIRMED]
    B -->|P(CONFIRMED) = 0.30| D[→ Stage 2]
    D -->|P(CANDIDATE) = 0.70| E[→ CANDIDATE]
    D -->|P(CANDIDATE) = 0.20| F[→ FALSE_POSITIVE]

    style C fill:#2ecc71,stroke:#27ae60,color:#fff
    style E fill:#3498db,stroke:#2980b9,color:#fff
    style F fill:#e74c3c,stroke:#c0392c,color:#fff
```

---

## Training Data Flow

```mermaid
graph TD
    A[Full Dataset<br/>9,564 samples] --> B[Stratified Split<br/>80/20]

    B --> C[Train Set<br/>~7,650 samples]
    B --> D[Test Set<br/>~1,910 samples]

    C --> E[Stage 1 Labels<br/>CONFIRMED=1, NOT=0]
    C --> F[Stage 2 Labels<br/>CANDIDATE=1, FALSE=0]

    E --> G[Train Stage 1 RF<br/>100 trees]
    F --> H[Train Stage 2 RF<br/>100 trees]

    G --> I[Stage 1 Model]
    H --> J[Stage 2 Model]

    D --> K[Evaluate Both<br/>on Test Set]
    I --> K
    J --> K

    style G fill:#2ecc71,stroke:#27ae60,color:#fff
    style H fill:#3498db,stroke:#2980b9,color:#fff
    style K fill:#f39c12,stroke:#e67e22,color:#fff
```

---

## Error Analysis

```mermaid
graph TD
    A[Total Test Samples<br/>1,911] --> B[Correct<br/>1,806]
    A --> C[Errors<br/>105]

    C --> D[Stage 1 Errors<br/>~15]
    C --> E[Stage 2 Errors<br/>~90]

    D --> D1[CONFIRMED misclassified<br/>as NOT CONFIRMED]
    D --> D2[NOT CONFIRMED misclassified<br/>as CONFIRMED]

    E --> E1[CANDIDATE misclassified<br/>as FALSE_POSITIVE]
    E --> E2[FALSE_POSITIVE misclassified<br/>as CANDIDATE]

    style B fill:#2ecc71,stroke:#27ae60,color:#fff
    style D fill:#e74c3c,stroke:#c0392c,color:#fff
    style E fill:#e74c3c,stroke:#c0392c,color:#fff
```

> Most errors occur in Stage 2, which is expected — the boundary between candidates and false positives is inherently ambiguous. These are the most scientifically interesting samples.
