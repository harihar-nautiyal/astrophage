//! Exoplanet classification model
//!
//! Uses custom Random Forest implementation
//! for multi-class classification of KOI signals.

use crate::decision_tree::RandomForest;
use crate::features::ProcessedDataset;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct ExoplanetClassifier {
    forest: Option<RandomForest>,
    feature_names: Vec<String>,
    feature_importance: Vec<(String, f64)>,
    class_weights: HashMap<u8, f64>,
}

impl ExoplanetClassifier {
    pub fn new() -> Self {
        let class_weights = HashMap::from([(0u8, 1.76), (1u8, 2.45), (2u8, 1.0)]);

        Self {
            forest: None,
            feature_names: Vec::new(),
            feature_importance: Vec::new(),
            class_weights,
        }
    }

    pub fn train(&mut self, train_data: &ProcessedDataset) -> Result<()> {
        println!(
            "   Training Random Forest with {} samples, {} features...",
            train_data.n_samples(),
            train_data.n_features()
        );

        self.feature_names = train_data.feature_names().to_vec();

        // OPTIMIZED hyperparameters
        let n_trees = 20;
        let max_depth = 12;
        let min_samples_split = 2;
        let min_samples_leaf = 1;

        println!("   Hyperparameters:");
        println!("     - Trees: {}", n_trees);
        println!("     - Max depth: {}", max_depth);
        println!("     - Min samples split: {}", min_samples_split);
        println!("     - Min samples leaf: {}", min_samples_leaf);
        println!(
            "     - Class weights: CONFIRMED={:.2}, CANDIDATE={:.2}, FALSE_POS={:.2}",
            self.class_weights[&0], self.class_weights[&1], self.class_weights[&2]
        );

        let mut forest = RandomForest::new(
            n_trees,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            self.class_weights.clone(),
        );
        forest.fit(train_data.features(), train_data.labels())?;

        let importance = forest.feature_importance();
        let mut feature_importance: Vec<(String, f64)> = self
            .feature_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), *importance.get(i).unwrap_or(&0.0)))
            .collect();

        feature_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        self.feature_importance = feature_importance;
        self.forest = Some(forest);

        println!("   ✓ Model training complete!");

        Ok(())
    }

    pub fn predict(&self, features: &Array2<f64>) -> Result<Array1<u8>> {
        match &self.forest {
            Some(forest) => Ok(forest.predict(features)),
            None => Err(anyhow!(
                "Model not trained yet. Call train() first.".to_string(),
            )),
        }
    }

    pub fn predict_proba(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        match &self.forest {
            Some(forest) => Ok(forest.predict_proba(features)),
            None => Err(anyhow!(
                "Model not trained yet. Call train() first.".to_string(),
            )),
        }
    }

    pub fn feature_importance(&self) -> &[(String, f64)] {
        &self.feature_importance
    }

    pub fn is_trained(&self) -> bool {
        self.forest.is_some()
    }

    pub fn n_trees(&self) -> usize {
        self.forest.as_ref().map(|f| f.n_trees()).unwrap_or(0)
    }
}

pub struct ModelConfig {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub class_weights: HashMap<u8, f64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_trees: 20,
            max_depth: 12,
            min_samples_split: 2,
            min_samples_leaf: 1,
            class_weights: HashMap::from([(0u8, 1.76), (1u8, 2.45), (2u8, 1.0)]),
        }
    }
}
