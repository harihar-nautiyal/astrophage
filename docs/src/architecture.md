# System Architecture

## High-Level Overview

Astrophage follows a clean pipeline architecture from raw data to final predictions:

```mermaid
graph LR
    subgraph "Input"
        A[Raw KOI CSV<br/>28 columns]
    end

    subgraph "Feature Engineering"
        B[Base Features<br/>28 columns]
        C[Derived Features<br/>8 interactions]
        D[Preprocessing<br/>impute + standardize]
    end

    subgraph "Two-Stage Model"
        E[Stage 1 RF<br/>CONFIRMED vs NOT]
        F[Stage 2 RF<br/>CANDIDATE vs FALSE+]
    end

    subgraph "Output"
        G[Predictions<br/>3 classes]
        H[JSON Report<br/>metrics + insights]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E -->|CONFIRMED| G
    E -->|NOT| F
    F -->|CANDIDATE| G
    F -->|FALSE_POSITIVE| G
    G --> H

    style A fill:#e74c3c,stroke:#c0392c,color:#fff
    style D fill:#f39c12,stroke:#e67e22,color:#fff
    style E fill:#2ecc71,stroke:#27ae60,color:#fff
    style F fill:#3498db,stroke:#2980b9,color:#fff
    style H fill:#9b59b6,stroke:#8e44ad,color:#fff
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant User as User
    participant Main as main.rs
    participant Data as data.rs
    participant Features as features.rs
    participant Stage1 as Stage 1 RF
    participant Stage2 as Stage 2 RF
    participant Eval as evaluation.rs
    participant Report as report.rs

    User->>Main: cargo run --release
    Main->>Data: load("data/koi_dataset.csv")
    Data-->>Main: KoiDataset (9,564 samples)

    Main->>Features: process(&dataset)
    Features->>Features: impute_missing()
    Features->>Features: standardize()
    Features->>Features: compute_derived()
    Features-->>Main: ProcessedDataset (36 features)

    Main->>Main: split(0.2, seed=42)
    Note over Main: 80/20 stratified split

    Main->>Stage1: train(&train_data)
    Note over Stage1: Binary: CONFIRMED=1, NOT=0
    Stage1-->>Main: Stage 1 trained

    Main->>Stage2: train(&train_stage2)
    Note over Stage2: Binary: CANDIDATE=1, FALSE_POSITIVE=0
    Stage2-->>Main: Stage 2 trained

    Main->>Eval: evaluate(&classifier, &test)
    Eval-->>Main: Metrics (accuracy, F1, etc.)

    Main->>Report: generate_report(&metrics, &classifier)
    Report-->>Main: output/report.json
    Main-->>User: Done!
```

---

## Random Forest Internals

### Single Decision Tree

```mermaid
graph TD
    A[Root Node<br/>Gini = 0.65] -->|fpflag_sum < 0.5| B[Left: Gini = 0.15]
    A -->|fpflag_sum >= 0.5| C[Right: Gini = 0.05]

    B -->|koi_model_snr < 2.0| D[Leaf: CANDIDATE]
    B -->|koi_model_snr >= 2.0| E[Leaf: CONFIRMED]

    C -->|koi_prad < 15.0| F[Leaf: FALSE_POSITIVE]
    C -->|koi_prad >= 15.0| G[Leaf: FALSE_POSITIVE]

    style D fill:#3498db,stroke:#2980b9,color:#fff
    style E fill:#2ecc71,stroke:#27ae60,color:#fff
    style F fill:#e74c3c,stroke:#c0392c,color:#fff
    style G fill:#e74c3c,stroke:#c0392c,color:#fff
```

### Ensemble Voting

```mermaid
graph TD
    A[Sample Input] --> B[Tree 1]
    A --> C[Tree 2]
    A --> D[Tree 3]
    A --> E[...]
    A --> F[Tree N]

    B -->|CONFIRMED| G[Voting Box]
    C -->|CONFIRMED| G
    D -->|CANDIDATE| G
    E -->|CONFIRMED| G
    F -->|CONFIRMED| G

    G -->|Majority Vote| H[Final: CONFIRMED]

    style G fill:#f39c12,stroke:#e67e22,color:#fff
    style H fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## Technology Layers

```mermaid
graph TB
    subgraph "Application Layer"
        A1[main.rs - CLI & Orchestration]
        A2[report.rs - JSON Generation]
    end

    subgraph "ML Layer"
        M1[two_stage_model.rs - Pipeline]
        M2[model.rs - Random Forest]
        M3[decision_tree.rs - Trees]
    end

    subgraph "Data Layer"
        D1[features.rs - Engineering]
        D2[data.rs - Loading]
    end

    subgraph "Infrastructure Layer"
        I1[Polars - DataFrame I/O]
        I2[NDArray - Vectorized Math]
        I3[Tokio - Async Runtime]
        I4[Serde - Serialization]
    end

    A1 --> M1
    A1 --> A2
    M1 --> M2
    M2 --> M3
    M1 --> D1
    D1 --> D2
    D2 --> I1
    D1 --> I2
    A1 --> I3
    A2 --> I4
```

---

## Performance Comparison

```mermaid
graph LR
    subgraph "Training Time"
        A[Astrophage<br/>Rust: ~30s]
        B[sklearn RF<br/>Python: ~120s]
    end

    subgraph "Inference Time"
        C[Astrophage<br/>~1ms/sample]
        D[sklearn RF<br/>~10ms/sample]
    end

    subgraph "Binary Size"
        E[Astrophage<br/>~2MB]
        F[sklearn env<br/>~500MB+]
    end

    style A fill:#2ecc71,stroke:#27ae60,color:#fff
    style C fill:#2ecc71,stroke:#27ae60,color:#fff
    style E fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## Memory Layout

```mermaid
graph TD
    subgraph "Training Data"
        A[Features Array2<br/>f64 x (n_samples x 36)]
        B[Labels Array1<br/>u8 x n_samples]
    end

    subgraph "Stage 1 Model"
        C[100 Decision Trees]
        C1[Tree 1: ~50 nodes]
        C2[Tree 2: ~50 nodes]
        C3[Tree N: ~50 nodes]
    end

    subgraph "Stage 2 Model"
        D[100 Decision Trees]
        D1[Tree 1: ~50 nodes]
        D2[Tree 2: ~50 nodes]
        D3[Tree N: ~50 nodes]
    end

    A --> C
    A --> D
    B --> C
    B --> D
```
