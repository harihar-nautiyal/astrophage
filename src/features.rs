use crate::data::KoiDataset;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use std::collections::HashMap;
use tracing::info;

pub const SELECTED_FEATURES: &[&str] = &[
    // Transit geometry (primary discriminators)
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_ingress",
    "koi_ror",
    // Physical properties
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_sma",
    "koi_incl",
    "koi_eccen",
    // Signal quality
    "koi_model_snr",
    "koi_count",
    "koi_num_transits",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    // False positive flags (pre-computed indicators)
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    // Photometric properties
    "koi_kepmag",
    "koi_dor",
    "koi_srho",
    // Stellar properties
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
];

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

    pub fn split(&self, test_ration: f64, seed: u64) -> (ProcessedDataset, ProcessedDataset) {
        let n = self.n_samples();
        let test_size = (n as f64 * test_ration) as usize;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let test_indices: std::collections::HashSet<usize> =
            indices[..test_size].iter().cloned().collect();

        let mut train_features = Vec::new();
        let mut train_labels = Vec::new();
        let mut test_features = Vec::new();
        let mut test_labels = Vec::new();

        for i in 0..n {
            let row: Vec<f64> = self.features.row(i).to_vec();
            if test_indices.contains(&i) {
                test_features.push(row);
                test_labels.push(self.labels[i]);
            } else {
                train_features.push(row);
                train_labels.push(self.labels[i]);
            }
        }

        let train = ProcessedDataset {
            features: Array2::from_shape_vec(
                (train_features.len(), self.n_features()),
                train_features.into_iter().flatten().collect(),
            )
            .unwrap(),
            labels: Array1::from(train_labels),
            feature_names: self.feature_names.clone(),
            label_names: self.label_names.clone(),
        };

        let test = ProcessedDataset {
            features: Array2::from_shape_vec(
                (test_features.len(), self.n_features()),
                test_features.into_iter().flatten().collect(),
            )
            .unwrap(),
            labels: Array1::from(test_labels),
            feature_names: self.feature_names.clone(),
            label_names: self.label_names.clone(),
        };

        (train, test)
    }
}

/// Feature engineering pipeline
pub struct FeatureEngineer {
    selected_features: Vec<String>,
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
}

impl FeatureEngineer {
    pub fn new() -> Self {
        Self {
            selected_features: SELECTED_FEATURES.iter().map(|s| s.to_string()).collect(),
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }

    /// Process raw dataset into model-ready features
    pub fn process(&mut self, dataset: &KoiDataset) -> Result<ProcessedDataset> {
        info!(
            "Selecting {} astrophysical features...",
            self.selected_features.len()
        );

        let mut selected_indices = Vec::new();
        let mut selected_names = Vec::new();

        for name in &self.selected_features {
            if let Some(idx) = dataset.feature_index(name) {
                selected_indices.push(idx);
                selected_names.push(name.clone());
            }
        }

        info!(
            "Found {} of {} requested features",
            selected_indices.len(),
            self.selected_features.len()
        );

        let n_samples = dataset.n_samples();
        let n_selected = selected_indices.len();
        let mut features = Array2::from_elem((n_samples, n_selected), f64::NAN);

        for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
            for row in 0..n_samples {
                features[[row, new_idx]] = dataset.features()[[row, old_idx]];
            }
        }

        info!("Imputing missing values with column medians...");
        Self::impute_missing(&mut features);

        info!("   Standardizing features (z-score normalization)...");
        self.standardize(&mut features);

        let label_names = HashMap::from([
            (0u8, "CONFIRMED".to_string()),
            (1u8, "CANDIDATE".to_string()),
            (2u8, "FALSE POSITIVE".to_string()),
        ]);

        Ok(ProcessedDataset {
            features,
            labels: dataset.labels().clone(),
            feature_names: selected_names,
            label_names,
        })
    }

    /// Impute missing values with column median
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

    /// Standardize features (z-score normalization)
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
