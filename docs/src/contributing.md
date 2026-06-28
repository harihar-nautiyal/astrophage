# Contributing to Astrophage

Thank you for your interest in contributing! Astrophage is a Rust-based exoplanet classification project, and we welcome contributions of all kinds.

```mermaid
graph LR
    A[Fork] --> B[Branch]
    B --> C[Code]
    C --> D[Test]
    D --> E[PR]
    E --> F[Merge]

    style F fill:#2ecc71,stroke:#27ae60,color:#fff
```

---

## Getting Started

### Prerequisites

- Rust 1.85+ (install via [rustup](https://rustup.rs/))
- Git

### Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/astrophage.git
cd astrophage

# Build
cargo build --release

# Run tests
cargo test
```

---

## Development Workflow

```mermaid
graph TD
    A[Issue/Feature Request] --> B[Create Branch]
    B --> C[Implement]
    C --> D[Format & Lint]
    D --> E[Test]
    E --> F[Commit]
    F --> G[Push]
    G --> H[Pull Request]
    H --> I[Review]
    I -->|Approved| J[Merge]
    I -->|Changes| C

    style J fill:#2ecc71,stroke:#27ae60,color:#fff
```

### Code Style

We follow standard Rust conventions:

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Generate docs
cargo doc --open
```

---

## Areas for Contribution

```mermaid
graph TB
    subgraph "High Priority"
        A1[Hyperparameter Tuning]
        A2[Cross-Validation]
        A3[Feature Selection]
    end

    subgraph "Medium Priority"
        B1[More Derived Features]
        B2[Model Serialization]
        B3[Prediction API]
    end

    subgraph "Low Priority"
        C1[Web Dashboard]
        C2[NASA Archive Integration]
        C3[GPU Acceleration]
    end
```

### High Priority

1. **Hyperparameter tuning** — Grid search over tree depth, n_estimators, max_features
2. **Cross-validation** — K-fold stratified CV implementation
3. **Feature selection** — Recursive feature elimination to find optimal subset

### Medium Priority

4. **Additional derived features** — More astrophysical interactions
5. **Model serialization** — Save/load trained models to avoid retraining
6. **Prediction API** — REST API for real-time classification

### Low Priority

7. **Web dashboard** — Visualize predictions and feature importance
8. **NASA Archive integration** — Direct API connection for live data
9. **GPU acceleration** — CUDA kernels for tree training

---

## Submitting Changes

```mermaid
sequenceDiagram
    participant C as Contributor
    participant R as Repo
    participant M as Maintainer

    C->>R: Fork repository
    C->>C: git checkout -b feature/amazing
    C->>C: git commit -m "Add amazing feature"
    C->>R: git push origin feature/amazing
    C->>R: Open Pull Request
    M->>R: Review code
    M->>C: Request changes (if needed)
    C->>R: Push updates
    M->>R: Approve & Merge
```

### Pull Request Guidelines

- Describe what changed and why
- Reference any related issues
- Include test results
- Keep changes focused and atomic

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn
- Credit original authors

## Questions?

Open an issue or reach out to [@harihar-nautiyal](https://github.com/harihar-nautiyal).
