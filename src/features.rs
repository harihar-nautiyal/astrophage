//! Feature engineering for KOI classification

use crate::data::KoiDataset;
use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use tracing::info;

/// I am using total 32 features from the dataset to train the model
pub const BASE_FEATURES: &[&str] = &[
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_ingress",
    "koi_ror",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_sma",
    "koi_incl",
    "koi_eccen",
    "koi_model_snr",
    "koi_count",
    "koi_num_transits",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_kepmag",
    "koi_dor",
    "koi_srho",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
];

/// These are derived features to improve accuracy of the model
pub const DERIVED_FEATURES: &[(&str, &str)] = &[
    ("koi_prad_squared", "koi_prad^2 — non-linear radius effect"),
    (
        "depth_duration_ratio",
        "koi_depth/koi_duration — transit steepness",
    ),
    (
        "snr_x_prad",
        "koi_model_snr * koi_prad — SNR weighted by size",
    ),
    (
        "impact_penalty",
        "penalty if koi_impact > 1.0 — physical impossibility",
    ),
    (
        "log_period",
        "ln(koi_period) — orbital period is log-normal",
    ),
    (
        "teq_over_steff",
        "koi_teq/koi_steff — temperature ratio sanity check",
    ),
    ("fpflag_sum", "sum of all fpflags — total suspicion score"),
    (
        "prad_teq_interaction",
        "koi_prad * koi_teq — size-temperature interaction",
    ),
];

/// Processed dataset ready for modeling
pub struct ProcessedDataset {
    features: Array2<f64>,
    labels: Array1<u8>,
    feature_names: Vec<String>,
    label_names: HashMap<u8, String>,
}

impl ProcessedDataset {
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }
    pub fn features(&self) -> &Array2<f64> {
        &self.features
    }
    pub fn labels(&self) -> &Array1<u8> {
        &self.labels
    }
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Split dataset into train and test sets (stratified)
    pub fn split(&self, test_ratio: f64, seed: u64) -> (ProcessedDataset, ProcessedDataset) {
        let n_classes = 3;

        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (idx, &label) in self.labels.iter().enumerate() {
            if (label as usize) < n_classes {
                class_indices[label as usize].push(idx);
            }
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        for class_idx in class_indices {
            let test_size = (class_idx.len() as f64 * test_ratio) as usize;
            let mut shuffled = class_idx.clone();
            shuffled.shuffle(&mut rng);

            let (test, train) = shuffled.split_at(test_size);
            test_indices.extend_from_slice(test);
            train_indices.extend_from_slice(train);
        }

        let train = self.subset(&train_indices);
        let test = self.subset(&test_indices);

        (train, test)
    }

    fn subset(&self, indices: &[usize]) -> ProcessedDataset {
        let n = indices.len();
        let n_features = self.n_features();
        let mut features = Array2::zeros((n, n_features));
        let mut labels = Array1::zeros(n);

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for col in 0..n_features {
                features[[new_idx, col]] = self.features[[old_idx, col]];
            }
            labels[new_idx] = self.labels[old_idx];
        }

        ProcessedDataset {
            features,
            labels,
            feature_names: self.feature_names.clone(),
            label_names: self.label_names.clone(),
        }
    }
}

/// Feature engineering pipeline
pub struct FeatureEngineer {
    base_features: Vec<String>,
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureEngineer {
    pub fn new() -> Self {
        Self {
            base_features: BASE_FEATURES.iter().map(|s| s.to_string()).collect(),
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }

    /// Process raw dataset into model-ready features
    pub fn process(&mut self, dataset: &KoiDataset) -> Result<ProcessedDataset> {
        info!("   Selecting {} base features...", self.base_features.len());

        let mut selected_indices = Vec::new();
        let mut selected_names = Vec::new();

        for name in &self.base_features {
            if let Some(idx) = dataset.feature_index(name) {
                selected_indices.push(idx);
                selected_names.push(name.clone());
            }
        }

        info!(
            "Found {} of {} base features",
            selected_indices.len(),
            self.base_features.len()
        );

        let n_samples = dataset.n_samples();
        let n_base = selected_indices.len();
        let mut features = Array2::from_elem((n_samples, n_base), f64::NAN);

        for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
            for row in 0..n_samples {
                features[[row, new_idx]] = dataset.features()[[row, old_idx]];
            }
        }

        info!("Imputing missing values with column medians...");
        Self::impute_missing(&mut features);

        info!("Standardizing features (z-score normalization)...");
        self.standardize(&mut features);

        info!(
            "Computing {} derived astrophysical features...",
            DERIVED_FEATURES.len()
        );
        let derived = self.compute_derived_features(&features, &selected_names);

        let n_total = n_base + derived.ncols();
        let mut all_features = Array2::zeros((n_samples, n_total));

        for row in 0..n_samples {
            for col in 0..n_base {
                all_features[[row, col]] = features[[row, col]];
            }
            for col in 0..derived.ncols() {
                all_features[[row, n_base + col]] = derived[[row, col]];
            }
        }

        let mut all_names = selected_names.clone();
        for (name, _) in DERIVED_FEATURES {
            all_names.push(name.to_string());
        }

        info!(
            "Total features: {} ({} base + {} derived)",
            all_names.len(),
            n_base,
            derived.ncols()
        );

        let label_names = HashMap::from([
            (0u8, "CONFIRMED".to_string()),
            (1u8, "CANDIDATE".to_string()),
            (2u8, "FALSE POSITIVE".to_string()),
        ]);

        Ok(ProcessedDataset {
            features: all_features,
            labels: dataset.labels().clone(),
            feature_names: all_names,
            label_names,
        })
    }

    fn compute_derived_features(&self, features: &Array2<f64>, names: &[String]) -> Array2<f64> {
        let n_samples = features.nrows();
        let n_derived = DERIVED_FEATURES.len();
        let mut derived = Array2::zeros((n_samples, n_derived));

        let col_idx = |name: &str| -> Option<usize> { names.iter().position(|n| n == name) };

        for row in 0..n_samples {
            if let Some(idx) = col_idx("koi_prad") {
                let prad = features[[row, idx]];
                derived[[row, 0]] = prad * prad;
            }

            if let (Some(d_idx), Some(dur_idx)) = (col_idx("koi_depth"), col_idx("koi_duration")) {
                let depth = features[[row, d_idx]];
                let duration = features[[row, dur_idx]];
                derived[[row, 1]] = if duration.abs() > 1e-10 {
                    depth / duration
                } else {
                    0.0
                };
            }

            if let (Some(snr_idx), Some(prad_idx)) = (col_idx("koi_model_snr"), col_idx("koi_prad"))
            {
                derived[[row, 2]] = features[[row, snr_idx]] * features[[row, prad_idx]];
            }

            if let Some(idx) = col_idx("koi_impact") {
                let impact = features[[row, idx]];
                derived[[row, 3]] = if impact > 1.0 { 10.0 } else { 0.0 };
            }

            if let Some(idx) = col_idx("koi_period") {
                let period = features[[row, idx]];
                derived[[row, 4]] = (period.abs() + 1.0).ln();
            }

            if let (Some(teq_idx), Some(steff_idx)) = (col_idx("koi_teq"), col_idx("koi_steff")) {
                let teq = features[[row, teq_idx]];
                let steff = features[[row, steff_idx]];
                derived[[row, 5]] = if steff.abs() > 1e-10 {
                    teq / steff
                } else {
                    0.0
                };
            }

            let mut fpflag_sum = 0.0;
            for fp_name in &[
                "koi_fpflag_nt",
                "koi_fpflag_ss",
                "koi_fpflag_co",
                "koi_fpflag_ec",
            ] {
                if let Some(idx) = col_idx(fp_name) {
                    fpflag_sum += features[[row, idx]];
                }
            }
            derived[[row, 6]] = fpflag_sum;

            if let (Some(prad_idx), Some(teq_idx)) = (col_idx("koi_prad"), col_idx("koi_teq")) {
                derived[[row, 7]] = features[[row, prad_idx]] * features[[row, teq_idx]];
            }
        }

        derived
    }

    fn impute_missing(features: &mut Array2<f64>) {
        let n_cols = features.ncols();
        for col in 0..n_cols {
            let column = features.column(col);
            let valid_values: Vec<f64> = column.iter().filter(|&&v| !v.is_nan()).cloned().collect();

            if valid_values.is_empty() {
                continue;
            }

            let mut sorted = valid_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];

            for row in 0..features.nrows() {
                if features[[row, col]].is_nan() {
                    features[[row, col]] = median;
                }
            }
        }
    }

    fn standardize(&mut self, features: &mut Array2<f64>) {
        let n_cols = features.ncols();
        self.feature_means.clear();
        self.feature_stds.clear();

        for col in 0..n_cols {
            let column = features.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(0.0).max(1e-10);

            self.feature_means.push(mean);
            self.feature_stds.push(std);

            for row in 0..features.nrows() {
                features[[row, col]] = (features[[row, col]] - mean) / std;
            }
        }
    }
}
