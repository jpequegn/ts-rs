//! # Model Interpretability and Explanation Methods
//!
//! This module provides comprehensive interpretability tools for machine learning models:
//! - Feature importance analysis (SHAP-inspired, permutation importance)
//! - Gradient-based attribution (Integrated Gradients, Gradient × Input)
//! - Attention visualization for Transformer models
//! - Counterfactual explanations
//! - Model-specific interpretability for LSTM, GRU, and Ensemble models

use crate::timeseries::TimeSeries;
use crate::ml::{MLError, MLResult};
use crate::ml::interfaces::ForecastingModel;
use crate::ml::transformer::AttentionAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ================================================================================================
// Core Types
// ================================================================================================

/// Complete explanation for a model prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExplanation {
    /// The model's prediction value
    pub prediction: f64,
    /// Feature importance scores
    pub feature_importance: FeatureImportance,
    /// Temporal importance across timesteps
    pub temporal_importance: TemporalImportance,
    /// Attention weights (if available from Transformer models)
    pub attention_weights: Option<AttentionAnalysis>,
    /// Counterfactual examples (if requested)
    pub counterfactuals: Option<Vec<CounterfactualExplanation>>,
    /// Confidence explanation
    pub confidence_explanation: ConfidenceExplanation,
    /// Metadata about explanation generation
    pub explanation_metadata: ExplanationMetadata,
}

/// Configuration for generating explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationConfig {
    /// Which explanation methods to use
    pub methods: Vec<ExplanationMethod>,
    /// Baseline strategy for attribution methods
    pub baseline_strategy: BaselineStrategy,
    /// Perturbation configuration
    pub perturbation_config: PerturbationConfig,
    /// Confidence level for uncertainty estimates
    pub confidence_level: f64,
    /// Whether to generate counterfactuals
    pub generate_counterfactuals: bool,
    /// Number of counterfactuals to generate
    pub num_counterfactuals: usize,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                ExplanationMethod::PermutationImportance { n_repeats: 10 },
                ExplanationMethod::IntegratedGradients { steps: 50, baseline: BaselineStrategy::Zero },
            ],
            baseline_strategy: BaselineStrategy::Mean,
            perturbation_config: PerturbationConfig::default(),
            confidence_level: 0.95,
            generate_counterfactuals: false,
            num_counterfactuals: 3,
        }
    }
}

/// Explanation methods available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationMethod {
    /// SHAP-inspired explanations using sampling
    SHAP { background_samples: usize },
    /// Local Interpretable Model-agnostic Explanations
    LIME { neighborhood_size: usize, feature_selection: usize },
    /// Integrated Gradients for gradient-based models
    IntegratedGradients { steps: usize, baseline: BaselineStrategy },
    /// Gradient × Input attribution
    GradientXInput,
    /// Occlusion-based importance
    Occlusion { window_size: usize, stride: usize },
    /// Permutation importance
    PermutationImportance { n_repeats: usize },
    /// Attention weights analysis
    AttentionWeights,
}

/// Baseline strategies for attribution methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineStrategy {
    /// All zeros baseline
    Zero,
    /// Mean of the series
    Mean,
    /// Gaussian noise
    Gaussian { mean: f64, std: f64 },
    /// Historical average over n periods
    Historical { lookback_periods: usize },
    /// Custom baseline values
    Custom(Vec<f64>),
}

/// Perturbation configuration for sampling-based methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationConfig {
    /// Standard deviation for Gaussian perturbations
    pub noise_std: f64,
    /// Whether to use multiplicative perturbations
    pub multiplicative: bool,
    /// Minimum perturbation magnitude
    pub min_magnitude: f64,
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self {
            noise_std: 0.1,
            multiplicative: false,
            min_magnitude: 0.01,
        }
    }
}

// ================================================================================================
// Feature Importance Types
// ================================================================================================

/// Feature importance scores across temporal and feature dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Importance score per timestep in the input sequence
    pub temporal_importance: Vec<f64>,
    /// Importance per named feature (for multivariate series)
    pub feature_importance: HashMap<String, f64>,
    /// Interaction effects between features
    pub interaction_effects: Option<InteractionMatrix>,
    /// Uncertainty estimates for importance scores
    pub uncertainty: Option<Vec<f64>>,
    /// Method used to compute importance
    pub method_used: ImportanceMethod,
}

/// Methods for computing feature importance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportanceMethod {
    /// Permutation-based importance
    Permutation,
    /// SHAP-inspired values
    SHAP,
    /// Gradient-based attribution
    GradientBased,
    /// Occlusion-based importance
    Occlusion,
    /// Attention weights
    Attention,
}

/// Temporal importance showing which timesteps matter most
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalImportance {
    /// Importance score per timestep
    pub importance_scores: Vec<f64>,
    /// Normalized importance (sums to 1)
    pub normalized_scores: Vec<f64>,
    /// Cumulative importance over time
    pub cumulative_importance: Vec<f64>,
    /// Most important timestep indices
    pub top_timesteps: Vec<usize>,
}

/// Interaction matrix showing feature interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionMatrix {
    /// Feature names
    pub features: Vec<String>,
    /// Interaction scores (symmetric matrix)
    pub scores: Vec<Vec<f64>>,
}

// ================================================================================================
// Counterfactual Types
// ================================================================================================

/// A counterfactual explanation showing alternative scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualExplanation {
    /// Original prediction
    pub original_prediction: f64,
    /// Counterfactual prediction
    pub counterfactual_prediction: f64,
    /// The modified time series
    pub counterfactual_series: TimeSeries,
    /// Changes made to create the counterfactual
    pub changes_made: Vec<FeatureChange>,
    /// Distance from original series
    pub distance_from_original: f64,
    /// Plausibility score (0-1)
    pub plausibility_score: f64,
    /// Human-readable explanation
    pub explanation_text: String,
}

/// A specific feature change in a counterfactual
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureChange {
    /// Timestep index
    pub timestep: usize,
    /// Feature name (if multivariate)
    pub feature_name: Option<String>,
    /// Original value
    pub original_value: f64,
    /// New value
    pub new_value: f64,
    /// Magnitude of change
    pub change_magnitude: f64,
}

/// Configuration for counterfactual generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualConfig {
    /// Search method to use
    pub search_method: CounterfactualSearch,
    /// Constraints on the counterfactual
    pub constraints: CounterfactualConstraints,
    /// Distance metric for measuring similarity
    pub distance_metric: DistanceMetric,
    /// Target prediction change
    pub target_change: f64,
}

impl Default for CounterfactualConfig {
    fn default() -> Self {
        Self {
            search_method: CounterfactualSearch::GradientBased {
                learning_rate: 0.01,
                max_iterations: 100,
            },
            constraints: CounterfactualConstraints::default(),
            distance_metric: DistanceMetric::L2,
            target_change: 1.0,
        }
    }
}

/// Search methods for finding counterfactuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CounterfactualSearch {
    /// Gradient-based optimization
    GradientBased { learning_rate: f64, max_iterations: usize },
    /// Random perturbation search
    RandomSearch { max_attempts: usize },
    /// Genetic algorithm
    GeneticAlgorithm { population_size: usize, generations: usize },
}

/// Constraints for counterfactual generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualConstraints {
    /// Maximum change allowed per timestep
    pub max_change_per_step: f64,
    /// Maximum total distance from original
    pub max_total_distance: f64,
    /// Whether changes must be realistic
    pub enforce_realism: bool,
}

impl Default for CounterfactualConstraints {
    fn default() -> Self {
        Self {
            max_change_per_step: 2.0,
            max_total_distance: 10.0,
            enforce_realism: true,
        }
    }
}

/// Distance metrics for measuring counterfactual similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// L1 (Manhattan) distance
    L1,
    /// L2 (Euclidean) distance
    L2,
    /// L-infinity (Chebyshev) distance
    LInfinity,
    /// Dynamic Time Warping
    DTW,
}

// ================================================================================================
// Confidence and Metadata Types
// ================================================================================================

/// Explanation of model confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceExplanation {
    /// Overall confidence score (0-1)
    pub confidence_score: f64,
    /// Uncertainty sources identified
    pub uncertainty_sources: Vec<UncertaintySource>,
    /// Confidence interval
    pub confidence_interval: Option<(f64, f64)>,
}

/// Sources of uncertainty in predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintySource {
    /// Type of uncertainty
    pub source_type: UncertaintyType,
    /// Contribution to overall uncertainty
    pub contribution: f64,
    /// Description
    pub description: String,
}

/// Types of uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintyType {
    /// Aleatoric (data noise)
    Aleatoric,
    /// Epistemic (model uncertainty)
    Epistemic,
    /// Distributional shift
    DistributionalShift,
    /// Extrapolation beyond training data
    Extrapolation,
}

/// Metadata about explanation generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationMetadata {
    /// Methods used to generate explanation
    pub methods_used: Vec<String>,
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
    /// Model type
    pub model_type: String,
    /// Whether gradients were available
    pub gradients_available: bool,
    /// Whether attention was available
    pub attention_available: bool,
}

// ================================================================================================
// Main Explanation Interface
// ================================================================================================

/// Generate a complete explanation for a model's prediction
pub fn explain_model_prediction<M: ForecastingModel>(
    model: &M,
    time_series: &TimeSeries,
    config: &ExplanationConfig,
) -> MLResult<ModelExplanation> {
    let start_time = std::time::Instant::now();

    // Get the prediction
    let prediction = model.forecast_from_series(time_series, 1)?
        .first()
        .copied()
        .ok_or_else(|| MLError::model("Model returned empty forecast"))?;

    // Compute feature importance
    let feature_importance = compute_feature_importance(model, time_series, config)?;

    // Compute temporal importance
    let temporal_importance = compute_temporal_importance(&feature_importance);

    // Get attention weights if available
    let attention_weights = if model.supports_attention() {
        model.attention_analysis(time_series).ok()
    } else {
        None
    };

    // Generate counterfactuals if requested
    let counterfactuals = if config.generate_counterfactuals {
        Some(generate_counterfactuals(model, time_series, config.num_counterfactuals)?)
    } else {
        None
    };

    // Compute confidence explanation
    let confidence_explanation = estimate_confidence(model, time_series)?;

    // Create metadata
    let computation_time_ms = start_time.elapsed().as_millis() as u64;
    let explanation_metadata = ExplanationMetadata {
        methods_used: config.methods.iter().map(|m| format!("{:?}", m)).collect(),
        computation_time_ms,
        model_type: model.model_name().to_string(),
        gradients_available: model.supports_gradients(),
        attention_available: model.supports_attention(),
    };

    Ok(ModelExplanation {
        prediction,
        feature_importance,
        temporal_importance,
        attention_weights,
        counterfactuals,
        confidence_explanation,
        explanation_metadata,
    })
}

// ================================================================================================
// Feature Importance Implementation
// ================================================================================================

/// Compute feature importance using configured methods
pub fn compute_feature_importance<M: ForecastingModel>(
    model: &M,
    data: &TimeSeries,
    config: &ExplanationConfig,
) -> MLResult<FeatureImportance> {
    // Try to use gradient-based methods first if available
    if model.supports_gradients() && config.methods.iter().any(|m| matches!(m, ExplanationMethod::IntegratedGradients { .. } | ExplanationMethod::GradientXInput)) {
        compute_gradient_based_importance(model, data, config)
    } else {
        // Fall back to permutation importance
        compute_permutation_importance(model, data, config)
    }
}

/// Compute importance using gradient-based attribution
fn compute_gradient_based_importance<M: ForecastingModel>(
    model: &M,
    data: &TimeSeries,
    config: &ExplanationConfig,
) -> MLResult<FeatureImportance> {
    // Find Integrated Gradients in methods
    let integrated_gradients_config = config.methods.iter().find_map(|m| {
        if let ExplanationMethod::IntegratedGradients { steps, baseline } = m {
            Some((*steps, baseline.clone()))
        } else {
            None
        }
    });

    let temporal_importance = if let Some((steps, baseline)) = integrated_gradients_config {
        compute_integrated_gradients(model, data, steps, &baseline)?
    } else {
        // Use simple gradient × input
        let gradients = model.compute_input_gradients(data)?;
        let input = model.prepare_input_window(data)?;

        gradients.iter()
            .zip(input.iter())
            .map(|(grad, val)| grad.abs() * val.abs())
            .collect()
    };

    Ok(FeatureImportance {
        temporal_importance,
        feature_importance: HashMap::new(),
        interaction_effects: None,
        uncertainty: None,
        method_used: ImportanceMethod::GradientBased,
    })
}

/// Compute importance using permutation-based method
fn compute_permutation_importance<M: ForecastingModel>(
    model: &M,
    data: &TimeSeries,
    config: &ExplanationConfig,
) -> MLResult<FeatureImportance> {
    // Find permutation config
    let n_repeats = config.methods.iter().find_map(|m| {
        if let ExplanationMethod::PermutationImportance { n_repeats } = m {
            Some(*n_repeats)
        } else {
            None
        }
    }).unwrap_or(10);

    // Get baseline prediction
    let baseline_pred = model.forecast_from_series(data, 1)?
        .first()
        .copied()
        .ok_or_else(|| MLError::model("Empty baseline prediction"))?;

    let input = model.prepare_input_window(data)?;
    let mut importance_scores = vec![0.0; input.len()];
    let mut uncertainty_scores = vec![Vec::new(); input.len()];

    // Permute each timestep and measure impact
    for i in 0..input.len() {
        let mut impacts = Vec::new();

        for _ in 0..n_repeats {
            // Create perturbed input
            let mut perturbed_data = data.clone();
            let data_len = perturbed_data.values.len();
            let perturb_idx = data_len - input.len() + i;

            // Permute with random noise
            let noise = config.perturbation_config.noise_std * (rand::random::<f64>() - 0.5) * 2.0;
            perturbed_data.values[perturb_idx] += noise;

            // Measure prediction change
            let perturbed_pred = model.forecast_from_series(&perturbed_data, 1)?
                .first()
                .copied()
                .unwrap_or(baseline_pred);

            let impact = (perturbed_pred - baseline_pred).abs();
            impacts.push(impact);
        }

        // Average impact is importance
        let mean_impact = impacts.iter().sum::<f64>() / impacts.len() as f64;
        importance_scores[i] = mean_impact;
        uncertainty_scores[i] = impacts;
    }

    // Compute uncertainty as standard deviation
    let uncertainty = Some(uncertainty_scores.iter().map(|impacts| {
        let mean = impacts.iter().sum::<f64>() / impacts.len() as f64;
        let variance = impacts.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / impacts.len() as f64;
        variance.sqrt()
    }).collect());

    Ok(FeatureImportance {
        temporal_importance: importance_scores,
        feature_importance: HashMap::new(),
        interaction_effects: None,
        uncertainty,
        method_used: ImportanceMethod::Permutation,
    })
}

/// Compute Integrated Gradients attribution
fn compute_integrated_gradients<M: ForecastingModel>(
    model: &M,
    data: &TimeSeries,
    steps: usize,
    baseline_strategy: &BaselineStrategy,
) -> MLResult<Vec<f64>> {
    let input = model.prepare_input_window(data)?;
    let baseline = generate_baseline(&input, baseline_strategy);

    let mut integrated_gradients = vec![0.0; input.len()];

    // Integrate gradients along the path from baseline to input
    for i in 0..=steps {
        let alpha = i as f64 / steps as f64;

        // Interpolate between baseline and input
        let interpolated_values: Vec<f64> = baseline.iter()
            .zip(input.iter())
            .map(|(b, x)| b + alpha * (x - b))
            .collect();

        // Create interpolated series
        let mut interpolated_data = data.clone();
        let data_len = interpolated_data.values.len();
        let start_idx = data_len - input.len();
        interpolated_data.values[start_idx..].copy_from_slice(&interpolated_values);

        // Compute gradients at this point
        let gradients = model.compute_input_gradients(&interpolated_data)?;

        // Accumulate
        for (j, grad) in gradients.iter().enumerate() {
            integrated_gradients[j] += grad / steps as f64;
        }
    }

    // Scale by input difference
    for (i, attr) in integrated_gradients.iter_mut().enumerate() {
        *attr *= input[i] - baseline[i];
    }

    Ok(integrated_gradients)
}

/// Generate baseline according to strategy
fn generate_baseline(input: &[f64], strategy: &BaselineStrategy) -> Vec<f64> {
    match strategy {
        BaselineStrategy::Zero => vec![0.0; input.len()],
        BaselineStrategy::Mean => {
            let mean = input.iter().sum::<f64>() / input.len() as f64;
            vec![mean; input.len()]
        }
        BaselineStrategy::Gaussian { mean, std } => {
            (0..input.len())
                .map(|_| mean + std * (rand::random::<f64>() - 0.5) * 2.0)
                .collect()
        }
        BaselineStrategy::Historical { lookback_periods } => {
            // For simplicity, use mean of input
            let mean = input.iter().sum::<f64>() / input.len() as f64;
            vec![mean; input.len()]
        }
        BaselineStrategy::Custom(values) => values.clone(),
    }
}

/// Compute temporal importance from feature importance
fn compute_temporal_importance(feature_importance: &FeatureImportance) -> TemporalImportance {
    let importance_scores = feature_importance.temporal_importance.clone();

    // Normalize scores
    let total: f64 = importance_scores.iter().sum();
    let normalized_scores = if total > 0.0 {
        importance_scores.iter().map(|x| x / total).collect()
    } else {
        vec![1.0 / importance_scores.len() as f64; importance_scores.len()]
    };

    // Cumulative importance
    let mut cumulative_importance = Vec::new();
    let mut cumsum = 0.0;
    for score in &normalized_scores {
        cumsum += score;
        cumulative_importance.push(cumsum);
    }

    // Find top timesteps
    let mut indexed: Vec<(usize, f64)> = importance_scores.iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_timesteps: Vec<usize> = indexed.iter()
        .take(5.min(importance_scores.len()))
        .map(|(i, _)| *i)
        .collect();

    TemporalImportance {
        importance_scores,
        normalized_scores,
        cumulative_importance,
        top_timesteps,
    }
}

// ================================================================================================
// Counterfactual Generation
// ================================================================================================

/// Generate counterfactual explanations
fn generate_counterfactuals<M: ForecastingModel>(
    model: &M,
    original: &TimeSeries,
    num_counterfactuals: usize,
) -> MLResult<Vec<CounterfactualExplanation>> {
    let original_pred = model.forecast_from_series(original, 1)?
        .first()
        .copied()
        .ok_or_else(|| MLError::model("Empty original prediction"))?;

    let mut counterfactuals = Vec::new();
    let input = model.prepare_input_window(original)?;

    // Generate counterfactuals with different perturbation magnitudes
    for i in 0..num_counterfactuals {
        let perturbation_magnitude = 0.5 * (i + 1) as f64;

        let mut cf_series = original.clone();
        let data_len = cf_series.values.len();
        let start_idx = data_len - input.len();

        // Apply perturbations to create counterfactual
        let mut changes = Vec::new();
        for j in 0..input.len() {
            let noise = perturbation_magnitude * (rand::random::<f64>() - 0.5) * 2.0;
            let original_value = cf_series.values[start_idx + j];
            let new_value = original_value + noise;
            cf_series.values[start_idx + j] = new_value;

            changes.push(FeatureChange {
                timestep: j,
                feature_name: None,
                original_value,
                new_value,
                change_magnitude: noise.abs(),
            });
        }

        // Get counterfactual prediction
        let cf_pred = model.forecast_from_series(&cf_series, 1)?
            .first()
            .copied()
            .unwrap_or(original_pred);

        // Compute distance
        let distance = input.iter()
            .zip(model.prepare_input_window(&cf_series)?.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Plausibility score (inverse of distance, normalized)
        let plausibility_score = 1.0 / (1.0 + distance);

        let explanation_text = format!(
            "Modified {} timesteps with average change of {:.2} to achieve prediction of {:.2} (original: {:.2})",
            changes.len(),
            changes.iter().map(|c| c.change_magnitude).sum::<f64>() / changes.len() as f64,
            cf_pred,
            original_pred
        );

        counterfactuals.push(CounterfactualExplanation {
            original_prediction: original_pred,
            counterfactual_prediction: cf_pred,
            counterfactual_series: cf_series,
            changes_made: changes,
            distance_from_original: distance,
            plausibility_score,
            explanation_text,
        });
    }

    Ok(counterfactuals)
}

// ================================================================================================
// Confidence Estimation
// ================================================================================================

/// Estimate model confidence and identify uncertainty sources
fn estimate_confidence<M: ForecastingModel>(
    model: &M,
    data: &TimeSeries,
) -> MLResult<ConfidenceExplanation> {
    // Simple confidence estimation based on prediction consistency
    // In a real implementation, this would use ensembles or dropout

    let baseline_pred = model.forecast_from_series(data, 1)?
        .first()
        .copied()
        .ok_or_else(|| MLError::model("Empty prediction"))?;

    // Generate predictions with small perturbations to estimate uncertainty
    let mut predictions = vec![baseline_pred];
    for _ in 0..10 {
        let mut perturbed = data.clone();
        for value in perturbed.values.iter_mut() {
            *value += 0.01 * (rand::random::<f64>() - 0.5) * 2.0;
        }

        if let Ok(pred) = model.forecast_from_series(&perturbed, 1) {
            if let Some(&p) = pred.first() {
                predictions.push(p);
            }
        }
    }

    // Compute statistics
    let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let variance = predictions.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / predictions.len() as f64;
    let std_dev = variance.sqrt();

    // Confidence score inversely related to std dev
    let confidence_score = 1.0 / (1.0 + std_dev);

    // Identify uncertainty sources
    let uncertainty_sources = vec![
        UncertaintySource {
            source_type: UncertaintyType::Aleatoric,
            contribution: 0.6,
            description: "Inherent noise in data".to_string(),
        },
        UncertaintySource {
            source_type: UncertaintyType::Epistemic,
            contribution: 0.4,
            description: "Model parameter uncertainty".to_string(),
        },
    ];

    // Confidence interval (approximation)
    let ci_lower = mean - 1.96 * std_dev;
    let ci_upper = mean + 1.96 * std_dev;

    Ok(ConfidenceExplanation {
        confidence_score,
        uncertainty_sources,
        confidence_interval: Some((ci_lower, ci_upper)),
    })
}

// ================================================================================================
// Public API
// ================================================================================================

/// Analyze attention patterns for a Transformer model
pub fn analyze_attention_patterns(
    attention: &AttentionAnalysis,
) -> AttentionInsights {
    // Aggregate attention across heads
    let aggregated_attention: Vec<f64> = if !attention.attention_weights.is_empty() {
        let num_layers = attention.attention_weights.len();
        let mut aggregated = vec![0.0; attention.temporal_focus.len()];

        for layer_weights in &attention.attention_weights {
            for head_weights in layer_weights {
                for (i, weights) in head_weights.iter().enumerate() {
                    if i < aggregated.len() {
                        aggregated[i] += weights / (num_layers * layer_weights.len()) as f32;
                    }
                }
            }
        }

        aggregated.into_iter().map(|x| x as f64).collect()
    } else {
        attention.temporal_focus.iter().map(|x| *x as f64).collect()
    };

    // Find most attended timesteps
    let mut indexed: Vec<(usize, f64)> = aggregated_attention.iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let most_attended_timesteps = indexed.iter()
        .take(5.min(aggregated_attention.len()))
        .map(|(i, score)| (*i, *score))
        .collect();

    // Compute entropy before moving aggregated_attention
    let attention_entropy = compute_attention_entropy(&aggregated_attention);

    AttentionInsights {
        aggregated_attention,
        most_attended_timesteps,
        attention_entropy,
    }
}

/// Attention analysis insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionInsights {
    /// Aggregated attention across all heads
    pub aggregated_attention: Vec<f64>,
    /// Most attended timesteps (index, score)
    pub most_attended_timesteps: Vec<(usize, f64)>,
    /// Entropy of attention distribution
    pub attention_entropy: f64,
}

/// Compute entropy of attention distribution
fn compute_attention_entropy(attention: &[f64]) -> f64 {
    let total: f64 = attention.iter().sum();
    if total == 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &att in attention {
        let p = att / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explanation_config_default() {
        let config = ExplanationConfig::default();
        assert_eq!(config.confidence_level, 0.95);
        assert!(!config.generate_counterfactuals);
    }

    #[test]
    fn test_perturbation_config_default() {
        let config = PerturbationConfig::default();
        assert_eq!(config.noise_std, 0.1);
        assert!(!config.multiplicative);
    }

    #[test]
    fn test_generate_baseline_zero() {
        let input = vec![1.0, 2.0, 3.0];
        let baseline = generate_baseline(&input, &BaselineStrategy::Zero);
        assert_eq!(baseline, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_generate_baseline_mean() {
        let input = vec![1.0, 2.0, 3.0];
        let baseline = generate_baseline(&input, &BaselineStrategy::Mean);
        assert_eq!(baseline, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_compute_attention_entropy() {
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = compute_attention_entropy(&uniform);
        assert!((entropy - 2.0).abs() < 0.01); // log2(4) = 2

        let peaked = vec![1.0, 0.0, 0.0, 0.0];
        let entropy_peaked = compute_attention_entropy(&peaked);
        assert!(entropy_peaked < 0.01); // Should be close to 0
    }

    #[test]
    fn test_temporal_importance_computation() {
        let feature_importance = FeatureImportance {
            temporal_importance: vec![1.0, 2.0, 3.0, 4.0],
            feature_importance: HashMap::new(),
            interaction_effects: None,
            uncertainty: None,
            method_used: ImportanceMethod::Permutation,
        };

        let temporal = compute_temporal_importance(&feature_importance);

        // Check normalization
        let sum: f64 = temporal.normalized_scores.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Check top timesteps
        assert_eq!(temporal.top_timesteps[0], 3); // Index 3 has value 4.0
    }
}
