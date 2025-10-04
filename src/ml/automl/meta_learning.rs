//! Meta-Learning Module
//!
//! Implements meta-learning for model recommendation:
//! - Dataset feature extraction
//! - Similarity-based recommendations
//! - Performance prediction based on historical data

use super::types::*;
use super::selection::extract_dataset_features;
use crate::timeseries::TimeSeries;
use crate::ml::MLResult;
use std::collections::HashMap;

// ================================================================================================
// Meta-Learner
// ================================================================================================

pub struct MetaLearner {
    performance_database: PerformanceDatabase,
    similarity_metric: SimilarityMetric,
}

/// Database storing historical performance results
#[derive(Debug, Clone)]
pub struct PerformanceDatabase {
    entries: Vec<PerformanceEntry>,
}

#[derive(Debug, Clone)]
struct PerformanceEntry {
    dataset_features: DatasetFeatures,
    model_type: ModelType,
    hyperparameters: ParameterConfiguration,
    performance: f64,
}

/// Similarity metrics for dataset comparison
#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    Euclidean,
    Cosine,
    Manhattan,
}

impl MetaLearner {
    pub fn new(similarity_metric: SimilarityMetric) -> Self {
        Self {
            performance_database: PerformanceDatabase {
                entries: Vec::new(),
            },
            similarity_metric,
        }
    }

    /// Add performance result to database
    pub fn add_performance_entry(
        &mut self,
        features: DatasetFeatures,
        model_type: ModelType,
        hyperparameters: ParameterConfiguration,
        performance: f64,
    ) {
        self.performance_database.entries.push(PerformanceEntry {
            dataset_features: features,
            model_type,
            hyperparameters,
            performance,
        });
    }

    /// Recommend models based on dataset features
    pub fn recommend_models(&self, features: &DatasetFeatures, top_k: usize) -> Vec<ModelRecommendation> {
        if self.performance_database.entries.is_empty() {
            return vec![];
        }

        // Find similar datasets
        let similar_datasets = self.find_similar_datasets(features, 10);

        // Aggregate performance by model type
        let mut model_performances: HashMap<ModelType, Vec<f64>> = HashMap::new();

        for (entry, _similarity) in &similar_datasets {
            model_performances
                .entry(entry.model_type)
                .or_insert_with(Vec::new)
                .push(entry.performance);
        }

        // Compute average performance and confidence for each model
        let mut recommendations: Vec<_> = model_performances
            .iter()
            .map(|(model_type, performances)| {
                let avg_performance = performances.iter().sum::<f64>() / performances.len() as f64;
                let std = {
                    let mean = avg_performance;
                    let variance = performances.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                        / performances.len() as f64;
                    variance.sqrt()
                };

                let confidence = 1.0 / (1.0 + std);

                ModelRecommendation {
                    model_type: *model_type,
                    confidence,
                    expected_performance: avg_performance,
                    reasoning: format!(
                        "Based on {} similar datasets with average performance {:.3}",
                        performances.len(),
                        avg_performance
                    ),
                }
            })
            .collect();

        // Sort by confidence * expected_performance
        recommendations.sort_by(|a, b| {
            let score_a = a.confidence * a.expected_performance;
            let score_b = b.confidence * b.expected_performance;
            score_b.partial_cmp(&score_a).unwrap()
        });

        recommendations.into_iter().take(top_k).collect()
    }

    /// Find similar datasets from database
    fn find_similar_datasets(
        &self,
        features: &DatasetFeatures,
        k: usize,
    ) -> Vec<(&PerformanceEntry, f64)> {
        let mut similarities: Vec<_> = self
            .performance_database
            .entries
            .iter()
            .map(|entry| {
                let similarity = self.compute_similarity(features, &entry.dataset_features);
                (entry, similarity)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.into_iter().take(k).collect()
    }

    /// Compute similarity between two datasets
    fn compute_similarity(&self, features1: &DatasetFeatures, features2: &DatasetFeatures) -> f64 {
        let vec1 = features_to_vector(features1);
        let vec2 = features_to_vector(features2);

        match self.similarity_metric {
            SimilarityMetric::Euclidean => {
                let dist: f64 = vec1
                    .iter()
                    .zip(vec2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                1.0 / (1.0 + dist)
            }

            SimilarityMetric::Cosine => {
                let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
                let norm1: f64 = vec1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm2: f64 = vec2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

                if norm1 > 0.0 && norm2 > 0.0 {
                    dot_product / (norm1 * norm2)
                } else {
                    0.0
                }
            }

            SimilarityMetric::Manhattan => {
                let dist: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| (a - b).abs()).sum();
                1.0 / (1.0 + dist)
            }
        }
    }

    /// Predict performance for model on new dataset
    pub fn predict_performance(
        &self,
        features: &DatasetFeatures,
        model_type: ModelType,
    ) -> Option<f64> {
        let similar = self.find_similar_datasets(features, 5);

        let relevant: Vec<_> = similar
            .iter()
            .filter(|(entry, _)| entry.model_type == model_type)
            .collect();

        if relevant.is_empty() {
            return None;
        }

        // Weighted average by similarity
        let total_similarity: f64 = relevant.iter().map(|(_, sim)| sim).sum();
        let weighted_performance: f64 = relevant
            .iter()
            .map(|(entry, sim)| entry.performance * sim)
            .sum();

        Some(weighted_performance / total_similarity)
    }
}

// ================================================================================================
// Feature Vector Conversion
// ================================================================================================

/// Convert dataset features to vector for distance computation
fn features_to_vector(features: &DatasetFeatures) -> Vec<f64> {
    let mut vec = Vec::new();

    // Statistical features
    vec.push(features.statistical.mean);
    vec.push(features.statistical.std);
    vec.push(features.statistical.skewness);
    vec.push(features.statistical.kurtosis);
    vec.extend(&features.statistical.autocorrelation);

    // Temporal features
    vec.push(features.temporal.trend_strength);
    vec.push(features.temporal.seasonality_strength);
    vec.push(features.temporal.n_observations as f64);

    // Complexity features
    vec.push(features.complexity.entropy);
    vec.push(features.complexity.lempel_ziv_complexity);
    vec.push(features.complexity.approximate_entropy);

    // Normalize vector
    normalize_vector(&vec)
}

/// Normalize vector to [0, 1] range
fn normalize_vector(vec: &[f64]) -> Vec<f64> {
    let min = vec.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-10 {
        return vec![0.5; vec.len()];
    }

    vec.iter().map(|x| (x - min) / (max - min)).collect()
}

// ================================================================================================
// Public API
// ================================================================================================

/// Create meta-learner and get model recommendations
pub fn recommend_models_for_dataset(
    data: &TimeSeries,
    performance_history: &[(DatasetFeatures, ModelType, ParameterConfiguration, f64)],
    top_k: usize,
) -> MLResult<Vec<ModelRecommendation>> {
    let mut meta_learner = MetaLearner::new(SimilarityMetric::Euclidean);

    // Populate database
    for (features, model_type, hyperparams, performance) in performance_history {
        meta_learner.add_performance_entry(
            features.clone(),
            *model_type,
            hyperparams.clone(),
            *performance,
        );
    }

    // Extract features from new dataset
    let features = extract_dataset_features(data);

    // Get recommendations
    Ok(meta_learner.recommend_models(&features, top_k))
}
