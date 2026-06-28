//! Model evaluation for exoplanet classification
//!
//! Computes:
//! - Accuracy, Precision, Recall, F1-Score
//! - Confusion Matrix
//! - Per-class metrics

use crate::features::ProcessedDataset;
use crate::two_stage_model::TwoStageClassifier;
use anyhow::{Result, anyhow};
use ndarray::Array2;
use std::collections::HashMap;

/// Evaluation metrics for the classifier
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub accuracy: f64,
    pub precision: HashMap<u8, f64>,
    pub recall: HashMap<u8, f64>,
    pub f1_score: HashMap<u8, f64>,
    pub macro_f1: f64,
    pub weighted_f1: f64,
    pub confusion_matrix: Array2<usize>,
    pub n_samples: usize,
}

impl EvaluationMetrics {
    pub fn new(n_classes: usize) -> Self {
        Self {
            accuracy: 0.0,
            precision: HashMap::new(),
            recall: HashMap::new(),
            f1_score: HashMap::new(),
            macro_f1: 0.0,
            weighted_f1: 0.0,
            confusion_matrix: Array2::zeros((n_classes, n_classes)),
            n_samples: 0,
        }
    }

    pub fn display(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    EVALUATION RESULTS                          ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        println!("📊 Overall Accuracy:  {:.4}", self.accuracy);
        println!("📊 Macro F1-Score:     {:.4}", self.macro_f1);
        println!("📊 Weighted F1-Score:  {:.4}", self.weighted_f1);
        println!();

        println!("Per-Class Metrics:");
        println!("┌─────────────────┬──────────┬──────────┬──────────┐");
        println!(
            "│ {:<15} │ {:>8} │ {:>8} │ {:>8} │",
            "Class", "Precision", "Recall", "F1-Score"
        );
        println!("├─────────────────┼──────────┼──────────┼──────────┤");

        for class in 0..3 {
            let name = match class {
                0 => "CONFIRMED",
                1 => "CANDIDATE",
                2 => "FALSE_POS",
                _ => "UNKNOWN",
            };
            let p = self.precision.get(&class).unwrap_or(&0.0);
            let r = self.recall.get(&class).unwrap_or(&0.0);
            let f1 = self.f1_score.get(&class).unwrap_or(&0.0);
            println!("│ {:<15} │ {:>8.4} │ {:>8.4} │ {:>8.4} │", name, p, r, f1);
        }
        println!("└─────────────────┴──────────┴──────────┴──────────┘");

        println!();
        println!("Confusion Matrix:");
        println!("                    Predicted");
        println!("              CONF    CAND    FALSE");
        for true_class in 0..3 {
            let name = match true_class {
                0 => "CONF    ",
                1 => "CAND    ",
                2 => "FALSE   ",
                _ => "UNKNOWN ",
            };
            print!("{} {:>5}", name, self.confusion_matrix[[true_class, 0]]);
            for pred_class in 1..3 {
                print!(" {:>5}", self.confusion_matrix[[true_class, pred_class]]);
            }
            println!();
        }
    }
}

/// Model evaluator for TwoStageClassifier
pub struct ModelEvaluator<'a> {
    classifier: &'a TwoStageClassifier,
    test_data: &'a ProcessedDataset,
}

impl<'a> ModelEvaluator<'a> {
    pub fn new(classifier: &'a TwoStageClassifier, test_data: &'a ProcessedDataset) -> Self {
        Self {
            classifier,
            test_data,
        }
    }

    pub fn evaluate(&self) -> Result<EvaluationMetrics> {
        let predictions = self
            .classifier
            .predict(self.test_data.features())
            .map_err(|e| anyhow!("Prediction error: {}", e))?;
        let true_labels = self.test_data.labels();

        let n = predictions.len();
        let mut metrics = EvaluationMetrics::new(3);
        metrics.n_samples = n;

        // Build confusion matrix
        for i in 0..n {
            let true_label = true_labels[i] as usize;
            let pred_label = predictions[i] as usize;
            metrics.confusion_matrix[[true_label, pred_label]] += 1;
        }

        // Calculate per-class metrics
        let mut macro_f1 = 0.0;
        let mut weighted_f1 = 0.0;
        let mut total_support = 0;

        for class in 0..3 {
            let tp = metrics.confusion_matrix[[class, class]] as f64;
            let fp: f64 = (0..3)
                .filter(|&c| c != class)
                .map(|c| metrics.confusion_matrix[[c, class]] as f64)
                .sum();
            let fn_: f64 = (0..3)
                .filter(|&c| c != class)
                .map(|c| metrics.confusion_matrix[[class, c]] as f64)
                .sum();
            let support = metrics.confusion_matrix.row(class).sum();

            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            metrics.precision.insert(class as u8, precision);
            metrics.recall.insert(class as u8, recall);
            metrics.f1_score.insert(class as u8, f1);

            macro_f1 += f1;
            weighted_f1 += f1 * support as f64;
            total_support += support;
        }

        metrics.macro_f1 = macro_f1 / 3.0;
        metrics.weighted_f1 = if total_support > 0 {
            weighted_f1 / total_support as f64
        } else {
            0.0
        };

        // Overall accuracy
        let correct: usize = (0..3).map(|c| metrics.confusion_matrix[[c, c]]).sum();
        metrics.accuracy = correct as f64 / n as f64;

        metrics.display();

        Ok(metrics)
    }
}
