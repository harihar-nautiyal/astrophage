//! Two-stage classification model
//! 
//! Stage 1: CONFIRMED vs NOT_CONFIRMED (binary)
//! Stage 2: If NOT_CONFIRMED, CANDIDATE vs FALSE_POSITIVE (binary)
//! 
//! This often outperforms direct 3-class classification because
//! each binary decision is simpler and more robust.

use crate::decision_tree::RandomForest;
use crate::features::ProcessedDataset;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Two-stage exoplanet classifier
pub struct TwoStageClassifier {
    stage1: Option<RandomForest>,  // CONFIRMED vs NOT_CONFIRMED
    stage2: Option<RandomForest>,  // CANDIDATE vs FALSE_POSITIVE
    feature_names: Vec<String>,
    feature_importance: Vec<(String, f64)>,
}

impl TwoStageClassifier {
    pub fn new() -> Self {
        Self {
            stage1: None,
            stage2: None,
            feature_names: Vec::new(),
            feature_importance: Vec::new(),
        }
    }

    pub fn train(&mut self, train_data: &ProcessedDataset) -> Result<()> {
        println!("   Training TWO-STAGE classifier...");
        println!("   Stage 1: CONFIRMED vs NOT_CONFIRMED");
        println!("   Stage 2: CANDIDATE vs FALSE_POSITIVE");

        self.feature_names = train_data.feature_names().to_vec();

        // ===== STAGE 1: Binary classification =====
        // Labels: CONFIRMED = 0, everything else = 1 (NOT_CONFIRMED)
        let stage1_labels: Array1<u8> = train_data.labels().iter()
            .map(|&label| if label == 0 { 0 } else { 1 })
            .collect();

        println!("   Stage 1 training on {} samples...", train_data.n_samples());
        let mut stage1_forest = RandomForest::new(
            50,  // n_trees
            15,  // max_depth
            2,   // min_samples_split
            1,   // min_samples_leaf
            HashMap::from([(0u8, 1.0), (1u8, 1.0)]),  // equal weights for binary
        );
        stage1_forest.fit(train_data.features(), &stage1_labels)?;

        // ===== STAGE 2: Binary classification on NOT_CONFIRMED only =====
        // Labels: CANDIDATE = 0, FALSE_POSITIVE = 1
        let mut stage2_indices = Vec::new();
        for (idx, &label) in train_data.labels().iter().enumerate() {
            if label != 0 {  // NOT CONFIRMED
                stage2_indices.push(idx);
            }
        }

        println!("   Stage 2 training on {} NOT_CONFIRMED samples...", stage2_indices.len());

        // Extract subset for stage 2
        let n_stage2 = stage2_indices.len();
        let n_features = train_data.n_features();
        let mut stage2_features = Array2::zeros((n_stage2, n_features));
        let mut stage2_labels = Array1::zeros(n_stage2);

        for (new_idx, &old_idx) in stage2_indices.iter().enumerate() {
            for col in 0..n_features {
                stage2_features[[new_idx, col]] = train_data.features()[[old_idx, col]];
            }
            // Remap: CANDIDATE (1) -> 0, FALSE_POSITIVE (2) -> 1
            stage2_labels[new_idx] = if train_data.labels()[old_idx] == 1 { 0 } else { 1 };
        }

        let mut stage2_forest = RandomForest::new(
            50,  // n_trees
            15,  // max_depth
            2,   // min_samples_split
            1,   // min_samples_leaf
            HashMap::from([(0u8, 1.5), (1u8, 1.0)]),  // weight CANDIDATE higher (rarer in stage 2)
        );
        stage2_forest.fit(&stage2_features, &stage2_labels)?;

        self.stage1 = Some(stage1_forest);
        self.stage2 = Some(stage2_forest);

        // Combine feature importance (weighted by stage)
        let mut combined_importance = HashMap::new();

        if let Some(ref s1) = self.stage1 {
            for (i, &imp) in s1.feature_importance().iter().enumerate() {
                *combined_importance.entry(i).or_insert(0.0) += imp * 0.6;  // Stage 1 is 60% of decision
            }
        }
        if let Some(ref s2) = self.stage2 {
            for (i, &imp) in s2.feature_importance().iter().enumerate() {
                *combined_importance.entry(i).or_insert(0.0) += imp * 0.4;  // Stage 2 is 40%
            }
        }

        let mut feature_importance: Vec<(String, f64)> = combined_importance.iter()
            .map(|(&i, &imp)| (self.feature_names[i].clone(), imp))
            .collect();
        feature_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.feature_importance = feature_importance;

        println!("   ✓ Two-stage training complete!");

        Ok(())
    }

    pub fn predict(&self, features: &Array2<f64>) -> Result<Array1<u8>> {
        let s1 = self.stage1.as_ref()
            .ok_or_else(|| anyhow!("Stage 1 not trained"))?;
        let s2 = self.stage2.as_ref()
            .ok_or_else(|| anyhow!("Stage 2 not trained"))?;

        let n = features.nrows();
        let mut predictions = Array1::zeros(n);

        // Stage 1: Predict CONFIRMED vs NOT_CONFIRMED
        let stage1_preds = s1.predict(features);

        // Collect indices that need Stage 2
        let mut stage2_indices = Vec::new();
        for (i, &pred) in stage1_preds.iter().enumerate() {
            if pred == 0 {
                // Stage 1 says CONFIRMED
                predictions[i] = 0;
            } else {
                // Stage 1 says NOT_CONFIRMED, need Stage 2
                stage2_indices.push(i);
            }
        }

        // Stage 2: Predict CANDIDATE vs FALSE_POSITIVE for NOT_CONFIRMED samples
        if !stage2_indices.is_empty() {
            let n_stage2 = stage2_indices.len();
            let n_features = features.ncols();
            let mut stage2_features = Array2::zeros((n_stage2, n_features));

            for (new_idx, &old_idx) in stage2_indices.iter().enumerate() {
                for col in 0..n_features {
                    stage2_features[[new_idx, col]] = features[[old_idx, col]];
                }
            }

            let stage2_preds = s2.predict(&stage2_features);

            // Remap back: 0 -> CANDIDATE (1), 1 -> FALSE_POSITIVE (2)
            for (new_idx, &old_idx) in stage2_indices.iter().enumerate() {
                predictions[old_idx] = if stage2_preds[new_idx] == 0 { 1 } else { 2 };
            }
        }

        Ok(predictions)
    }

    pub fn predict_proba(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let s1 = self.stage1.as_ref()
            .ok_or_else(|| anyhow!("Stage 1 not trained"))?;
        let s2 = self.stage2.as_ref()
            .ok_or_else(|| anyhow!("Stage 2 not trained"))?;

        let n = features.nrows();
        let mut probs = Array2::zeros((n, 3));  // 3 classes

        let stage1_probs = s1.predict_proba(features);  // [P(CONFIRMED), P(NOT_CONFIRMED)]

        let mut stage2_indices = Vec::new();
        for i in 0..n {
            let p_confirmed = stage1_probs[[i, 0]];
            let p_not_confirmed = stage1_probs[[i, 1]];

            probs[[i, 0]] = p_confirmed;  // P(CONFIRMED)

            if p_not_confirmed > 0.01 {
                stage2_indices.push(i);
            }
        }

        // Stage 2 probabilities for NOT_CONFIRMED samples
        if !stage2_indices.is_empty() {
            let n_stage2 = stage2_indices.len();
            let n_features = features.ncols();
            let mut stage2_features = Array2::zeros((n_stage2, n_features));

            for (new_idx, &old_idx) in stage2_indices.iter().enumerate() {
                for col in 0..n_features {
                    stage2_features[[new_idx, col]] = features[[old_idx, col]];
                }
            }

            let stage2_probs = s2.predict_proba(&stage2_features);  // [P(CANDIDATE), P(FALSE_POS)]

            for (new_idx, &old_idx) in stage2_indices.iter().enumerate() {
                let p_not_confirmed = stage1_probs[[old_idx, 1]];
                let p_candidate = stage2_probs[[new_idx, 0]];
                let p_false_pos = stage2_probs[[new_idx, 1]];

                probs[[old_idx, 1]] = p_not_confirmed * p_candidate;   // P(CANDIDATE)
                probs[[old_idx, 2]] = p_not_confirmed * p_false_pos;   // P(FALSE_POSITIVE)
            }
        }

        // Normalize rows to sum to 1
        for i in 0..n {
            let sum: f64 = probs.row(i).sum();
            if sum > 0.0 {
                for j in 0..3 {
                    probs[[i, j]] /= sum;
                }
            }
        }

        Ok(probs)
    }

    pub fn feature_importance(&self) -> &[(String, f64)] {
        &self.feature_importance
    }

    pub fn is_trained(&self) -> bool {
        self.stage1.is_some() && self.stage2.is_some()
    }
}
