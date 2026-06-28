//! Custom Decision Tree and Random Forest — OPTIMIZED
//!
//! Key optimizations:
//! 1. Limit thresholds to 20 per feature (not every unique value)
//! 2. Early stopping when all labels are identical
//! 3. Reduced default trees (20) for faster iteration
//! 4. Progress indicators

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::RngExt;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;

/// A single node in the decision tree
#[derive(Debug, Clone)]
pub enum TreeNode {
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    Leaf {
        class_counts: HashMap<u8, usize>,
        prediction: u8,
        total: usize,
    },
}

impl TreeNode {
    pub fn predict(&self, sample: &[f64]) -> u8 {
        match self {
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] <= *threshold {
                    left.predict(sample)
                } else {
                    right.predict(sample)
                }
            }
            TreeNode::Leaf { prediction, .. } => *prediction,
        }
    }

    pub fn predict_proba(&self, sample: &[f64]) -> HashMap<u8, f64> {
        match self {
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] <= *threshold {
                    left.predict_proba(sample)
                } else {
                    right.predict_proba(sample)
                }
            }
            TreeNode::Leaf {
                class_counts,
                total,
                ..
            } => {
                let mut probs = HashMap::new();
                for (&class, &count) in class_counts.iter() {
                    probs.insert(class, count as f64 / *total as f64);
                }
                probs
            }
        }
    }

    pub fn count_nodes(&self) -> usize {
        match self {
            TreeNode::Split { left, right, .. } => 1 + left.count_nodes() + right.count_nodes(),
            TreeNode::Leaf { .. } => 1,
        }
    }

    pub fn count_leaves(&self) -> usize {
        match self {
            TreeNode::Split { left, right, .. } => left.count_leaves() + right.count_leaves(),
            TreeNode::Leaf { .. } => 1,
        }
    }

    pub fn max_depth(&self) -> usize {
        match self {
            TreeNode::Split { left, right, .. } => 1 + left.max_depth().max(right.max_depth()),
            TreeNode::Leaf { .. } => 1,
        }
    }
}

pub struct DecisionTree {
    root: Option<TreeNode>,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    class_weights: HashMap<u8, f64>,
    feature_importance: Vec<f64>,
    n_features: usize,
    n_samples: usize,
}

impl DecisionTree {
    pub fn new(
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        class_weights: HashMap<u8, f64>,
    ) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            class_weights,
            feature_importance: Vec::new(),
            n_features: 0,
            n_samples: 0,
        }
    }

    pub fn fit(&mut self, features: &Array2<f64>, labels: &Array1<u8>) -> Result<()> {
        self.n_features = features.ncols();
        self.n_samples = features.nrows();
        self.feature_importance = vec![0.0; self.n_features];

        let indices: Vec<usize> = (0..self.n_samples).collect();

        self.root = Some(self.build_tree(features, labels, &indices, 0));

        let total: f64 = self.feature_importance.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importance {
                *imp /= total;
            }
        }

        Ok(())
    }

    fn build_tree(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<u8>,
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        if indices.len() < self.min_samples_split || depth >= self.max_depth {
            return self.create_leaf(labels, indices);
        }

        let first_label = labels[indices[0]];
        let all_same = indices.iter().all(|&i| labels[i] == first_label);
        if all_same {
            return self.create_leaf(labels, indices);
        }

        let (best_feature, best_threshold, best_gain) =
            self.find_best_split_fast(features, labels, indices);

        if best_gain <= 1e-10 || best_feature.is_none() {
            return self.create_leaf(labels, indices);
        }

        let feature = best_feature.unwrap();
        let threshold = best_threshold.unwrap();

        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if features[[idx, feature]] <= threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        if left_indices.len() < self.min_samples_leaf || right_indices.len() < self.min_samples_leaf
        {
            return self.create_leaf(labels, indices);
        }

        self.feature_importance[feature] +=
            best_gain * (indices.len() as f64 / self.n_samples as f64);

        let left = self.build_tree(features, labels, &left_indices, depth + 1);
        let right = self.build_tree(features, labels, &right_indices, depth + 1);

        TreeNode::Split {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn find_best_split_fast(
        &self,
        features: &Array2<f64>,
        labels: &Array1<u8>,
        indices: &[usize],
    ) -> (Option<usize>, Option<f64>, f64) {
        let mut best_gain = 0.0;
        let mut best_feature = None;
        let mut best_threshold = None;

        let parent_gini = self.weighted_gini(labels, indices);
        let n = indices.len() as f64;

        let n_features_to_try = (self.n_features as f64).sqrt() as usize;
        let mut feature_indices: Vec<usize> = (0..self.n_features).collect();
        let mut rng = rand::rng();
        feature_indices.shuffle(&mut rng);

        for &feature in feature_indices.iter().take(n_features_to_try) {
            let mut values: Vec<f64> = indices
                .iter()
                .map(|&i| features[[i, feature]])
                .filter(|&v| !v.is_nan())
                .collect();

            if values.len() < 2 {
                continue;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

            let max_thresholds = 20usize.min(values.len().saturating_sub(1));
            if max_thresholds == 0 {
                continue;
            }

            let step = (values.len() - 1) / max_thresholds;
            let step = step.max(1);

            for i in (0..values.len() - 1).step_by(step) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                let mut left = Vec::new();
                let mut right = Vec::new();

                for &idx in indices {
                    if features[[idx, feature]] <= threshold {
                        left.push(idx);
                    } else {
                        right.push(idx);
                    }
                }

                if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
                    continue;
                }

                let left_gini = self.weighted_gini(labels, &left);
                let right_gini = self.weighted_gini(labels, &right);

                let gain = parent_gini
                    - ((left.len() as f64 / n) * left_gini + (right.len() as f64 / n) * right_gini);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = Some(feature);
                    best_threshold = Some(threshold);
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn weighted_gini(&self, labels: &Array1<u8>, indices: &[usize]) -> f64 {
        let mut weighted_counts = HashMap::new();
        let mut total_weight = 0.0;

        for &idx in indices {
            let class = labels[idx];
            let weight = *self.class_weights.get(&class).unwrap_or(&1.0);
            *weighted_counts.entry(class).or_insert(0.0) += weight;
            total_weight += weight;
        }

        let mut impurity = 1.0;
        for &count in weighted_counts.values() {
            let p = count / total_weight;
            impurity -= p * p;
        }

        impurity
    }

    fn create_leaf(&self, labels: &Array1<u8>, indices: &[usize]) -> TreeNode {
        let mut class_counts = HashMap::new();
        for &idx in indices {
            *class_counts.entry(labels[idx]).or_insert(0) += 1;
        }

        let prediction = class_counts
            .iter()
            .map(|(&class, &count)| {
                let weight = self.class_weights.get(&class).unwrap_or(&1.0);
                (class, count as f64 * weight)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class)
            .unwrap_or(0);

        TreeNode::Leaf {
            class_counts,
            prediction,
            total: indices.len(),
        }
    }

    pub fn predict(&self, features: &Array2<f64>) -> Array1<u8> {
        let n = features.nrows();
        let mut predictions = Array1::zeros(n);

        if let Some(ref root) = self.root {
            for i in 0..n {
                let sample: Vec<f64> = features.row(i).to_vec();
                predictions[i] = root.predict(&sample);
            }
        }

        predictions
    }

    pub fn feature_importance(&self) -> &[f64] {
        &self.feature_importance
    }

    pub fn tree_stats(&self) -> Option<(usize, usize, usize)> {
        self.root
            .as_ref()
            .map(|root| (root.count_nodes(), root.count_leaves(), root.max_depth()))
    }
}

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    class_weights: HashMap<u8, f64>,
    feature_importance: Vec<f64>,
    n_features: usize,
}

impl RandomForest {
    pub fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        class_weights: HashMap<u8, f64>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            class_weights,
            feature_importance: Vec::new(),
            n_features: 0,
        }
    }

    pub fn fit(&mut self, features: &Array2<f64>, labels: &Array1<u8>) -> Result<()> {
        self.n_features = features.ncols();
        self.feature_importance = vec![0.0; self.n_features];

        let n_samples = features.nrows();
        let mut rng = rand::rng();

        info!("Growing {} trees (optimized)...", self.n_trees);
        let start = Instant::now();

        for i in 0..self.n_trees {
            let mut indices: Vec<usize> = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                indices.push(rng.random_range(0..n_samples));
            }

            let mut tree = DecisionTree::new(
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                self.class_weights.clone(),
            );

            tree.fit(features, labels)?;

            self.trees.push(tree);

            if (i + 1) % 5 == 0 || i == self.n_trees - 1 {
                let elapsed = start.elapsed();
                info!(
                    "Tree {}/{} grown ({:.1}s)",
                    i + 1,
                    self.n_trees,
                    elapsed.as_secs_f64()
                );
            }
        }

        for tree in &self.trees {
            for (i, &imp) in tree.feature_importance().iter().enumerate() {
                self.feature_importance[i] += imp;
            }
        }
        let total: f64 = self.feature_importance.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importance {
                *imp /= total;
            }
        }

        let total_time = start.elapsed();
        info!(
            "Forest grown in {:.1}s! {} trees total",
            total_time.as_secs_f64(),
            self.trees.len()
        );

        Ok(())
    }

    pub fn predict(&self, features: &Array2<f64>) -> Array1<u8> {
        let n = features.nrows();
        let mut predictions = Array1::zeros(n);

        for i in 0..n {
            let sample: Vec<f64> = features.row(i).to_vec();
            let mut votes = HashMap::new();

            for tree in &self.trees {
                if let Some(ref root) = tree.root {
                    let pred = root.predict(&sample);
                    *votes.entry(pred).or_insert(0.0) += 1.0;
                }
            }

            predictions[i] = votes
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(&class, _)| class)
                .unwrap_or(0);
        }

        predictions
    }

    pub fn predict_proba(&self, features: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();
        let mut probs = Array2::zeros((n, self.n_classes()));

        for i in 0..n {
            let sample: Vec<f64> = features.row(i).to_vec();
            let mut class_probs: HashMap<u8, f64> = HashMap::new();

            for tree in &self.trees {
                if let Some(ref root) = tree.root {
                    let tree_probs = root.predict_proba(&sample);
                    for (&class, &p) in &tree_probs {
                        *class_probs.entry(class).or_insert(0.0) += p;
                    }
                }
            }

            for (&class, &sum_p) in &class_probs {
                probs[[i, class as usize]] = sum_p / self.trees.len() as f64;
            }
        }

        probs
    }

    pub fn n_classes(&self) -> usize {
        3
    }

    pub fn feature_importance(&self) -> &[f64] {
        &self.feature_importance
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}
