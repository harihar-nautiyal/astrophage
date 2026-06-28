//! Custom Decision Tree and Random Forest Implementation
//!
//! Built from scratch for Astrophage - No external ML dependencies is in use.

use crate::features::ProcessedDataset;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};
use rand::RngExt;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use tracing::info;

/// A single node in the decision tree
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Internal node: splits on a feature and threshold
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    /// Leaf node: holds the predicted class distribution
    Leaf {
        /// Class -> count
        class_counts: HashMap<u8, usize>,
        /// Most common class
        prediction: u8,
        /// Total samples in this leaf
        total: usize,
    },
}

impl TreeNode {
    /// Predict a single sample
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

    /// Predict with probability distribution
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

    /// Count total nodes in tree
    pub fn count_nodes(&self) -> usize {
        match self {
            TreeNode::Split { left, right, .. } => 1 + left.count_nodes() + right.count_nodes(),
            TreeNode::Leaf { .. } => 1,
        }
    }

    /// Count leaf nodes
    pub fn count_leaves(&self) -> usize {
        match self {
            TreeNode::Split { left, right, .. } => left.count_leaves() + right.count_leaves(),
            TreeNode::Leaf { .. } => 1,
        }
    }

    /// Get max depth
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
    n_classes: usize,
    feature_importance: Vec<f64>,
    n_features: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            n_classes: 3,
            feature_importance: Vec::new(),
            n_features: 0,
        }
    }

    /// Train the decision tree
    pub fn fit(&mut self, features: &Array2<f64>, labels: &Array1<u8>) -> Result<()> {
        self.n_features = features.ncols();
        self.feature_importance = vec![0.0; self.n_features];

        let n_samples = features.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();

        self.root = Some(self.build_tree(features, labels, &indices, 0));

        // Normalize feature importance
        let total: f64 = self.feature_importance.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importance {
                *imp /= total;
            }
        }

        Ok(())
    }

    /// Recursively build the tree
    fn build_tree(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<u8>,
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        // Check stopping conditions
        if indices.len() < self.min_samples_split || depth >= self.max_depth {
            return self.create_leaf(labels, indices);
        }

        // Check if all labels are the same
        let first_label = labels[indices[0]];
        let all_same = indices.iter().all(|&i| labels[i] == first_label);
        if all_same {
            return self.create_leaf(labels, indices);
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) =
            self.find_best_split(features, labels, indices);

        // If no good split found, create leaf
        if best_gain <= 0.0 || best_feature.is_none() {
            return self.create_leaf(labels, indices);
        }

        let feature = best_feature.unwrap();
        let threshold = best_threshold.unwrap();

        // Split indices
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if features[[idx, feature]] <= threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Check min_samples_leaf
        if left_indices.len() < self.min_samples_leaf || right_indices.len() < self.min_samples_leaf
        {
            return self.create_leaf(labels, indices);
        }

        // Update feature importance
        self.feature_importance[feature] +=
            best_gain * (indices.len() as f64 / features.nrows() as f64);

        // Recursively build subtrees
        let left = self.build_tree(features, labels, &left_indices, depth + 1);
        let right = self.build_tree(features, labels, &right_indices, depth + 1);

        TreeNode::Split {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Find the best split for a set of samples
    fn find_best_split(
        &self,
        features: &Array2<f64>,
        labels: &Array1<u8>,
        indices: &[usize],
    ) -> (Option<usize>, Option<f64>, f64) {
        let mut best_gain = 0.0;
        let mut best_feature = None;
        let mut best_threshold = None;

        let parent_gini = self.gini(labels, indices);
        let n = indices.len() as f64;

        for feature in 0..self.n_features {
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

            // Try thresholds between unique values
            for i in 0..values.len() - 1 {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                let mut left_indices = Vec::new();
                let mut right_indices = Vec::new();

                for &idx in indices {
                    if features[[idx, feature]] <= threshold {
                        left_indices.push(idx);
                    } else {
                        right_indices.push(idx);
                    }
                }

                if left_indices.len() < self.min_samples_leaf
                    || right_indices.len() < self.min_samples_leaf
                {
                    continue;
                }

                let left_gini = self.gini(labels, &left_indices);
                let right_gini = self.gini(labels, &right_indices);

                let left_weight = left_indices.len() as f64 / n;
                let right_weight = right_indices.len() as f64 / n;

                let gain = parent_gini - (left_weight * left_gini + right_weight * right_gini);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = Some(feature);
                    best_threshold = Some(threshold);
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    /// Calculate Gini impurity
    fn gini(&self, labels: &Array1<u8>, indices: &[usize]) -> f64 {
        let mut counts = HashMap::new();
        for &idx in indices {
            *counts.entry(labels[idx]).or_insert(0) += 1;
        }

        let n = indices.len() as f64;
        let mut impurity = 1.0;

        for &count in counts.values() {
            let p = count as f64 / n;
            impurity -= p * p;
        }

        impurity
    }

    /// Create a leaf node
    fn create_leaf(&self, labels: &Array1<u8>, indices: &[usize]) -> TreeNode {
        let mut class_counts = HashMap::new();
        for &idx in indices {
            *class_counts.entry(labels[idx]).or_insert(0) += 1;
        }

        let prediction = class_counts
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(&class, _)| class)
            .unwrap_or(0);

        TreeNode::Leaf {
            class_counts,
            prediction,
            total: indices.len(),
        }
    }

    /// Predict labels for a feature matrix
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

    /// Get tree statistics
    pub fn tree_stats(&self) -> Option<(usize, usize, usize)> {
        self.root
            .as_ref()
            .map(|root| (root.count_nodes(), root.count_leaves(), root.max_depth()))
    }
}

/// Random Forest classifier
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    feature_importance: Vec<f64>,
    n_features: usize,
}

impl RandomForest {
    pub fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            feature_importance: Vec::new(),
            n_features: 0,
        }
    }

    /// Train the random forest with bootstrap sampling
    pub fn fit(&mut self, features: &Array2<f64>, labels: &Array1<u8>) -> Result<()> {
        self.n_features = features.ncols();
        self.feature_importance = vec![0.0; self.n_features];

        let n_samples = features.nrows();
        let mut rng = rand::rng();

        info!("Growing {} trees...", self.n_trees);

        for i in 0..self.n_trees {
            // Bootstrap sample
            let mut indices: Vec<usize> = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                indices.push(rng.random_range(0..n_samples));
            }

            // Create subsampled feature set for this tree (feature bagging)
            let n_features_subset = (self.n_features as f64).sqrt() as usize;
            let mut feature_indices: Vec<usize> = (0..self.n_features).collect();
            feature_indices.shuffle(&mut rng);
            let feature_subset: std::collections::HashSet<usize> = feature_indices
                [..n_features_subset]
                .iter()
                .cloned()
                .collect();

            let mut tree = DecisionTree::new(
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
            );

            // Build tree on bootstrap sample
            tree.fit(features, labels)?;

            self.trees.push(tree);

            if (i + 1) % 10 == 0 || i == 0 {
                info!("Tree {}/{} grown", i + 1, self.n_trees);
            }
        }

        // Aggregate feature importance
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

        println!("   ✓ Forest grown! {} trees total", self.trees.len());

        Ok(())
    }

    /// Predict by majority voting
    pub fn predict(&self, features: &Array2<f64>) -> Array1<u8> {
        let n = features.nrows();
        let mut predictions = Array1::zeros(n);

        for i in 0..n {
            let sample: Vec<f64> = features.row(i).to_vec();
            let mut votes = HashMap::new();

            for tree in &self.trees {
                if let Some(ref root) = tree.root {
                    let pred = root.predict(&sample);
                    *votes.entry(pred).or_insert(0) += 1;
                }
            }

            predictions[i] = votes
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(&class, _)| class)
                .unwrap_or(0);
        }

        predictions
    }

    /// Predict with probability (average across trees)
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

    /// Number of classes
    pub fn n_classes(&self) -> usize {
        3
    }

    /// Get feature importance
    pub fn feature_importance(&self) -> &[f64] {
        &self.feature_importance
    }

    /// Get tree count
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}
