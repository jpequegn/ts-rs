//! Ensemble Methods for Neural Network and Traditional Forecasting Models
//!
//! This module provides sophisticated ensemble methods that combine multiple ML models
//! (LSTM, GRU, Transformers) with traditional forecasting methods to achieve superior
//! prediction accuracy and robustness through model combination strategies.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::ml::{MLError, MLResult};
use crate::ml::recurrent::{LSTMForecaster, GRUForecaster};
use crate::ml::transformer::TransformerForecaster;

// ============================================================================
// Core Ensemble Types
// ============================================================================

/// Main ensemble forecaster combining multiple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleForecaster {
    /// Neural network models in the ensemble
    pub neural_models: Vec<NeuralModelWrapper>,
    /// Traditional forecasting models (via wrapper)
    pub traditional_models: Vec<TraditionalModelWrapper>,
    /// Combination strategy for aggregating predictions
    pub combination_strategy: CombinationStrategy,
    /// Model weights for weighted combinations
    pub model_weights: Vec<f64>,
    /// Diversity metrics tracking
    pub diversity_metrics: Option<DiversityMetrics>,
    /// Performance history for dynamic weighting
    pub performance_history: PerformanceHistory,
    /// Enable dynamic weight adaptation
    pub dynamic_weighting: bool,
    /// Model metadata
    pub metadata: EnsembleMetadata,
}

/// Wrapper for neural network models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralModelWrapper {
    LSTM {
        forecaster: LSTMForecaster,
        weight: f64,
        performance_score: f64,
    },
    GRU {
        forecaster: GRUForecaster,
        weight: f64,
        performance_score: f64,
    },
    Transformer {
        forecaster: TransformerForecaster,
        weight: f64,
        performance_score: f64,
    },
}

/// Wrapper for traditional forecasting methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraditionalModelWrapper {
    /// Method type identifier
    pub method_type: String,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Weight in ensemble
    pub weight: f64,
    /// Performance score
    pub performance_score: f64,
    /// Uncertainty estimation method
    pub uncertainty_method: UncertaintyMethod,
}

/// Ensemble configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Combination strategy
    pub combination_strategy: CombinationStrategy,
    /// Weight optimization method
    pub weight_optimization: WeightOptimizationMethod,
    /// Minimum diversity threshold for model inclusion
    pub diversity_threshold: f64,
    /// Performance window for dynamic weighting
    pub performance_window: usize,
    /// Enable dynamic adaptation
    pub dynamic_adaptation: bool,
    /// Enable outlier detection in predictions
    pub outlier_detection: bool,
    /// Use confidence-based weighting
    pub confidence_weighting: bool,
    /// Minimum confidence threshold
    pub min_confidence: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            combination_strategy: CombinationStrategy::WeightedAverage,
            weight_optimization: WeightOptimizationMethod::InverseVariance,
            diversity_threshold: 0.3,
            performance_window: 10,
            dynamic_adaptation: true,
            outlier_detection: false,
            confidence_weighting: true,
            min_confidence: 0.5,
        }
    }
}

/// Model combination strategies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CombinationStrategy {
    /// Simple average of all predictions
    SimpleAverage,
    /// Weighted average with fixed or dynamic weights
    WeightedAverage,
    /// Median of all predictions
    Median,
    /// Voting-based combination
    Voting { threshold: f64 },
    /// Stacking with meta-learner
    Stacking,
    /// Dynamic weighting based on recent performance
    DynamicWeighting { adaptation_rate: f64 },
    /// Performance-based selection with lookback
    PerformanceBased { lookback_window: usize },
    /// Confidence-weighted combination
    ConfidenceWeighted,
    /// Bayesian Model Averaging
    BayesianModelAveraging,
}

/// Weight optimization methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WeightOptimizationMethod {
    /// Equal weights for all models
    EqualWeights,
    /// Inverse variance weighting
    InverseVariance,
    /// Returns-based optimization
    ReturnsBasedOptimization,
    /// Convex optimization
    ConvexOptimization,
    /// Genetic algorithm optimization
    GeneticAlgorithm,
    /// Bayesian optimization
    BayesianOptimization,
    /// Online gradient descent
    OnlineGradientDescent { learning_rate: f64 },
}

/// Uncertainty estimation methods for traditional models
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UncertaintyMethod {
    /// Residual-based confidence intervals
    ResidualBased,
    /// Bootstrap-based intervals
    Bootstrap { num_samples: usize },
    /// Quantile regression
    QuantileRegression,
    /// Prediction interval estimation
    PredictionInterval { confidence_level: f64 },
}

// ============================================================================
// Stacking Ensemble
// ============================================================================

/// Stacking ensemble with meta-learner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackingEnsemble {
    /// Base models
    pub base_models: Vec<NeuralModelWrapper>,
    /// Meta-model for combining base predictions
    pub meta_model: Box<NeuralModelWrapper>,
    /// Cross-validation folds for training
    pub cv_folds: usize,
    /// Feature engineering for meta-model
    pub meta_feature_config: MetaFeatureConfig,
    /// Training metadata
    pub training_metadata: HashMap<String, String>,
}

/// Meta-feature configuration for stacking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaFeatureConfig {
    /// Include base model predictions
    pub include_predictions: bool,
    /// Include prediction confidence scores
    pub include_confidence: bool,
    /// Include model agreement metrics
    pub include_model_agreement: bool,
    /// Include prediction variance
    pub include_prediction_variance: bool,
    /// Include temporal features
    pub include_temporal_features: bool,
    /// Include error pattern features
    pub include_error_patterns: bool,
}

impl Default for MetaFeatureConfig {
    fn default() -> Self {
        Self {
            include_predictions: true,
            include_confidence: true,
            include_model_agreement: true,
            include_prediction_variance: true,
            include_temporal_features: false,
            include_error_patterns: false,
        }
    }
}

// ============================================================================
// Dynamic Weight Management
// ============================================================================

/// Dynamic weight adaptation manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWeightManager {
    /// Adaptation method
    pub adaptation_method: AdaptationMethod,
    /// Performance tracking window
    pub performance_window: usize,
    /// Learning rate for gradient-based methods
    pub learning_rate: f64,
    /// Weight bounds
    pub weight_bounds: (f64, f64),
    /// Regularization parameter
    pub regularization: f64,
    /// Historical weights
    pub weight_history: Vec<Vec<f64>>,
}

/// Adaptation methods for dynamic weighting
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdaptationMethod {
    /// Exponential smoothing with alpha parameter
    ExponentialSmoothing { alpha: f64 },
    /// Kalman filter-based adaptation
    KalmanFilter {
        process_noise: f64,
        measurement_noise: f64,
    },
    /// Online gradient descent
    OnlineGradient { learning_rate: f64 },
    /// Reinforcement learning approach
    ReinforcementLearning { exploration_rate: f64 },
}

impl DynamicWeightManager {
    /// Create new dynamic weight manager
    pub fn new(config: &EnsembleConfig) -> Self {
        Self {
            adaptation_method: AdaptationMethod::ExponentialSmoothing { alpha: 0.1 },
            performance_window: config.performance_window,
            learning_rate: 0.01,
            weight_bounds: (0.0, 1.0),
            regularization: 0.01,
            weight_history: Vec::new(),
        }
    }

    /// Update weights based on recent performance
    pub fn update_weights(
        &mut self,
        current_weights: &[f64],
        model_errors: &[f64],
        confidence_scores: &[f64],
    ) -> MLResult<Vec<f64>> {
        match &self.adaptation_method {
            AdaptationMethod::ExponentialSmoothing { alpha } => {
                self.exponential_smoothing_update(current_weights, model_errors, *alpha)
            }
            AdaptationMethod::OnlineGradient { learning_rate } => {
                self.gradient_based_update(current_weights, model_errors, *learning_rate)
            }
            AdaptationMethod::KalmanFilter {
                process_noise,
                measurement_noise,
            } => self.kalman_filter_update(
                current_weights,
                model_errors,
                *process_noise,
                *measurement_noise,
            ),
            AdaptationMethod::ReinforcementLearning { exploration_rate } => {
                self.rl_based_update(current_weights, model_errors, confidence_scores, *exploration_rate)
            }
        }
    }

    fn exponential_smoothing_update(
        &self,
        current_weights: &[f64],
        errors: &[f64],
        alpha: f64,
    ) -> MLResult<Vec<f64>> {
        // Compute inverse error weights
        let inverse_errors: Vec<f64> = errors
            .iter()
            .map(|&e| if e > 1e-10 { 1.0 / e } else { 1e10 })
            .collect();

        let total_inverse_error: f64 = inverse_errors.iter().sum();
        let target_weights: Vec<f64> = inverse_errors
            .iter()
            .map(|&ie| ie / total_inverse_error)
            .collect();

        // Exponential smoothing
        let updated_weights: Vec<f64> = current_weights
            .iter()
            .zip(target_weights.iter())
            .map(|(&current, &target)| alpha * target + (1.0 - alpha) * current)
            .collect();

        // Normalize
        let sum: f64 = updated_weights.iter().sum();
        Ok(updated_weights.iter().map(|&w| w / sum).collect())
    }

    fn gradient_based_update(
        &self,
        current_weights: &[f64],
        errors: &[f64],
        lr: f64,
    ) -> MLResult<Vec<f64>> {
        // Gradient descent to minimize weighted error
        let mut updated_weights = current_weights.to_vec();

        for i in 0..updated_weights.len() {
            let gradient = errors[i];
            updated_weights[i] = (updated_weights[i] - lr * gradient)
                .max(self.weight_bounds.0)
                .min(self.weight_bounds.1);
        }

        // Normalize
        let sum: f64 = updated_weights.iter().sum();
        Ok(updated_weights.iter().map(|&w| w / sum).collect())
    }

    fn kalman_filter_update(
        &self,
        current_weights: &[f64],
        errors: &[f64],
        process_noise: f64,
        measurement_noise: f64,
    ) -> MLResult<Vec<f64>> {
        // Simplified Kalman filter for weight adaptation
        let mut updated_weights = Vec::new();

        for (i, &weight) in current_weights.iter().enumerate() {
            let prediction_error = errors[i];
            let kalman_gain = process_noise / (process_noise + measurement_noise);
            let updated_weight = weight + kalman_gain * prediction_error;

            updated_weights.push(updated_weight.max(0.0).min(1.0));
        }

        // Normalize
        let sum: f64 = updated_weights.iter().sum();
        Ok(updated_weights.iter().map(|&w| w / sum).collect())
    }

    fn rl_based_update(
        &self,
        current_weights: &[f64],
        errors: &[f64],
        confidence_scores: &[f64],
        exploration_rate: f64,
    ) -> MLResult<Vec<f64>> {
        // Reinforcement learning approach with exploration
        let mut updated_weights = current_weights.to_vec();

        for i in 0..updated_weights.len() {
            let reward = -errors[i] * confidence_scores[i];

            // Exploration vs exploitation
            if rand::random::<f64>() < exploration_rate {
                // Explore: random adjustment
                let adjustment = (rand::random::<f64>() - 0.5) * 0.1;
                updated_weights[i] = (updated_weights[i] + adjustment)
                    .max(self.weight_bounds.0)
                    .min(self.weight_bounds.1);
            } else {
                // Exploit: move toward better performance
                updated_weights[i] = updated_weights[i] + self.learning_rate * reward;
            }
        }

        // Normalize
        let sum: f64 = updated_weights.iter().sum();
        Ok(updated_weights.iter().map(|&w| w / sum).collect())
    }
}

// ============================================================================
// Diversity Metrics
// ============================================================================

/// Model diversity metrics for ensemble quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    /// Correlation between model predictions
    pub prediction_correlation: f64,
    /// Correlation between model errors
    pub error_correlation: f64,
    /// Disagreement measure (variance of predictions)
    pub disagreement_measure: f64,
    /// Entropy-based diversity
    pub entropy_based_diversity: f64,
    /// Temporal diversity over time
    pub temporal_diversity: Vec<f64>,
}

impl DiversityMetrics {
    /// Compute diversity metrics from predictions
    pub fn compute_diversity(
        predictions: &[Vec<f64>],
        ground_truth: &[f64],
    ) -> MLResult<Self> {
        if predictions.is_empty() {
            return Err(MLError::invalid_input("No predictions provided"));
        }

        let prediction_correlation = Self::compute_prediction_correlation(predictions)?;
        let error_correlation = Self::compute_error_correlation(predictions, ground_truth)?;
        let disagreement = Self::compute_disagreement_measure(predictions)?;
        let entropy_diversity = Self::compute_entropy_diversity(predictions)?;
        let temporal_diversity = Self::compute_temporal_diversity(predictions)?;

        Ok(Self {
            prediction_correlation,
            error_correlation,
            disagreement_measure: disagreement,
            entropy_based_diversity: entropy_diversity,
            temporal_diversity,
        })
    }

    /// Check if ensemble is diverse enough
    pub fn is_diverse_enough(&self, threshold: f64) -> bool {
        self.disagreement_measure > threshold && self.prediction_correlation < (1.0 - threshold)
    }

    fn compute_prediction_correlation(predictions: &[Vec<f64>]) -> MLResult<f64> {
        if predictions.len() < 2 {
            return Ok(0.0);
        }

        let n_models = predictions.len();
        let mut total_corr = 0.0;
        let mut count = 0;

        for i in 0..n_models {
            for j in (i + 1)..n_models {
                let corr = correlation(&predictions[i], &predictions[j])?;
                total_corr += corr;
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_corr / count as f64
        } else {
            0.0
        })
    }

    fn compute_error_correlation(
        predictions: &[Vec<f64>],
        ground_truth: &[f64],
    ) -> MLResult<f64> {
        if predictions.len() < 2 {
            return Ok(0.0);
        }

        // Compute errors for each model
        let errors: Vec<Vec<f64>> = predictions
            .iter()
            .map(|pred| {
                pred.iter()
                    .zip(ground_truth.iter())
                    .map(|(p, gt)| p - gt)
                    .collect()
            })
            .collect();

        Self::compute_prediction_correlation(&errors)
    }

    fn compute_disagreement_measure(predictions: &[Vec<f64>]) -> MLResult<f64> {
        if predictions.is_empty() || predictions[0].is_empty() {
            return Ok(0.0);
        }

        let n_points = predictions[0].len();
        let mut total_variance = 0.0;

        for i in 0..n_points {
            let values: Vec<f64> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / values.len() as f64;
            total_variance += variance;
        }

        Ok(total_variance / n_points as f64)
    }

    fn compute_entropy_diversity(predictions: &[Vec<f64>]) -> MLResult<f64> {
        if predictions.is_empty() || predictions[0].is_empty() {
            return Ok(0.0);
        }

        // Simplified entropy calculation based on prediction distribution
        let n_models = predictions.len() as f64;
        let n_points = predictions[0].len();
        let mut total_entropy = 0.0;

        for i in 0..n_points {
            let values: Vec<f64> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = values.iter().sum::<f64>() / n_models;
            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_models).sqrt();

            // Entropy approximation
            if std > 1e-10 {
                total_entropy += (std * (2.0 * std::f64::consts::PI * std::f64::consts::E).sqrt()).ln();
            }
        }

        Ok(total_entropy / n_points as f64)
    }

    fn compute_temporal_diversity(predictions: &[Vec<f64>]) -> MLResult<Vec<f64>> {
        if predictions.is_empty() || predictions[0].is_empty() {
            return Ok(Vec::new());
        }

        let n_points = predictions[0].len();
        let mut temporal_div = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let values: Vec<f64> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / values.len() as f64;
            temporal_div.push(variance.sqrt());
        }

        Ok(temporal_div)
    }
}

// ============================================================================
// Performance Tracking
// ============================================================================

/// Performance history for ensemble models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Model performance over time
    pub model_performances: Vec<ModelPerformance>,
    /// Ensemble performance over time
    pub ensemble_performance: Vec<f64>,
    /// Tracking window size
    pub window_size: usize,
}

/// Individual model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Model identifier
    pub model_id: String,
    /// Mean Absolute Error history
    pub mae_history: Vec<f64>,
    /// Root Mean Square Error history
    pub rmse_history: Vec<f64>,
    /// Confidence scores over time
    pub confidence_history: Vec<f64>,
    /// Timestamps for performance records
    pub timestamps: Vec<DateTime<Utc>>,
}

impl PerformanceHistory {
    /// Create new performance history tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            model_performances: Vec::new(),
            ensemble_performance: Vec::new(),
            window_size,
        }
    }

    /// Add performance record
    pub fn add_record(&mut self, model_id: &str, mae: f64, rmse: f64, confidence: f64) {
        // Find or create model performance entry
        if let Some(perf) = self
            .model_performances
            .iter_mut()
            .find(|p| p.model_id == model_id)
        {
            perf.mae_history.push(mae);
            perf.rmse_history.push(rmse);
            perf.confidence_history.push(confidence);
            perf.timestamps.push(Utc::now());

            // Keep only recent history
            if perf.mae_history.len() > self.window_size {
                perf.mae_history.remove(0);
                perf.rmse_history.remove(0);
                perf.confidence_history.remove(0);
                perf.timestamps.remove(0);
            }
        } else {
            self.model_performances.push(ModelPerformance {
                model_id: model_id.to_string(),
                mae_history: vec![mae],
                rmse_history: vec![rmse],
                confidence_history: vec![confidence],
                timestamps: vec![Utc::now()],
            });
        }
    }

    /// Get recent average performance for a model
    pub fn get_recent_performance(&self, model_id: &str, lookback: usize) -> Option<f64> {
        self.model_performances
            .iter()
            .find(|p| p.model_id == model_id)
            .and_then(|perf| {
                let start = perf.mae_history.len().saturating_sub(lookback);
                let recent_mae = &perf.mae_history[start..];
                if !recent_mae.is_empty() {
                    Some(recent_mae.iter().sum::<f64>() / recent_mae.len() as f64)
                } else {
                    None
                }
            })
    }
}

// ============================================================================
// Ensemble Metadata
// ============================================================================

/// Ensemble metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMetadata {
    /// Ensemble creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Number of neural models
    pub num_neural_models: usize,
    /// Number of traditional models
    pub num_traditional_models: usize,
    /// Configuration used
    pub config: EnsembleConfig,
    /// Training statistics
    pub training_stats: HashMap<String, f64>,
}

// ============================================================================
// Confidence-Based Weighting
// ============================================================================

/// Confidence-weighted ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceWeightedEnsemble {
    /// Models with confidence calibration
    pub models: Vec<NeuralModelWrapper>,
    /// Confidence calibration method
    pub confidence_calibration: ConfidenceCalibration,
    /// Minimum confidence threshold
    pub min_confidence_threshold: f64,
    /// Confidence smoothing parameter
    pub confidence_smoothing: f64,
}

/// Confidence calibration methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceCalibration {
    /// Platt scaling
    PlattScaling,
    /// Isotonic regression
    IsotonicRegression,
    /// Temperature scaling
    TemperatureScaling { temperature: f64 },
    /// Beta calibration
    BetaCalibration { alpha: f64, beta: f64 },
}

// ============================================================================
// Forecast Results
// ============================================================================

/// Ensemble forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleForecastResult {
    /// Combined forecast predictions
    pub predictions: Vec<f64>,
    /// Prediction timestamps
    pub timestamps: Vec<DateTime<Utc>>,
    /// Individual model predictions
    pub individual_predictions: Vec<ModelPrediction>,
    /// Confidence intervals (lower, upper)
    pub confidence_intervals: Option<(Vec<f64>, Vec<f64>)>,
    /// Prediction uncertainty
    pub uncertainty: Vec<f64>,
    /// Model weights used
    pub model_weights: Vec<f64>,
    /// Diversity score
    pub diversity_score: f64,
    /// Ensemble metadata
    pub metadata: HashMap<String, String>,
}

/// Individual model prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    /// Model identifier
    pub model_id: String,
    /// Predictions
    pub predictions: Vec<f64>,
    /// Confidence scores
    pub confidence: Vec<f64>,
    /// Model weight in ensemble
    pub weight: f64,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute Pearson correlation between two series
fn correlation(x: &[f64], y: &[f64]) -> MLResult<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err(MLError::invalid_input("Series must have same non-zero length"));
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / n;

    let std_x = (x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n).sqrt();
    let std_y = (y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n).sqrt();

    if std_x < 1e-10 || std_y < 1e-10 {
        return Ok(0.0);
    }

    Ok(cov / (std_x * std_y))
}

/// Training result for ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleTrainingResult {
    /// Final model weights
    pub final_weights: Vec<f64>,
    /// Training metrics
    pub training_metrics: HashMap<String, f64>,
    /// Validation metrics
    pub validation_metrics: HashMap<String, f64>,
    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics,
    /// Training duration in seconds
    pub training_duration: f64,
}

// ============================================================================
// Core Ensemble Functions
// ============================================================================

/// Create an ensemble forecaster from multiple models
pub fn create_ensemble_forecaster(
    neural_models: Vec<NeuralModelWrapper>,
    traditional_models: Vec<TraditionalModelWrapper>,
    config: EnsembleConfig,
) -> MLResult<EnsembleForecaster> {
    let total_models = neural_models.len() + traditional_models.len();

    if total_models == 0 {
        return Err(MLError::invalid_input("At least one model required for ensemble"));
    }

    // Initialize equal weights
    let model_weights = vec![1.0 / total_models as f64; total_models];

    Ok(EnsembleForecaster {
        neural_models,
        traditional_models,
        combination_strategy: config.combination_strategy,
        model_weights,
        diversity_metrics: None,
        performance_history: PerformanceHistory::new(config.performance_window),
        dynamic_weighting: config.dynamic_adaptation,
        metadata: EnsembleMetadata {
            created_at: Utc::now(),
            updated_at: Utc::now(),
            num_neural_models: 0,
            num_traditional_models: 0,
            config,
            training_stats: HashMap::new(),
        },
    })
}

/// Optimize ensemble weights
pub fn optimize_ensemble_weights(
    predictions: &[Vec<f64>],
    ground_truth: &[f64],
    method: WeightOptimizationMethod,
) -> MLResult<Vec<f64>> {
    match method {
        WeightOptimizationMethod::EqualWeights => {
            Ok(vec![1.0 / predictions.len() as f64; predictions.len()])
        }
        WeightOptimizationMethod::InverseVariance => {
            optimize_inverse_variance_weights(predictions, ground_truth)
        }
        WeightOptimizationMethod::OnlineGradientDescent { learning_rate } => {
            optimize_gradient_descent_weights(predictions, ground_truth, learning_rate)
        }
        _ => {
            // Default to inverse variance for unimplemented methods
            optimize_inverse_variance_weights(predictions, ground_truth)
        }
    }
}

fn optimize_inverse_variance_weights(
    predictions: &[Vec<f64>],
    ground_truth: &[f64],
) -> MLResult<Vec<f64>> {
    let mut weights = Vec::new();

    for pred in predictions {
        let mse: f64 = pred
            .iter()
            .zip(ground_truth.iter())
            .map(|(p, gt)| (p - gt).powi(2))
            .sum::<f64>()
            / pred.len() as f64;

        let weight = if mse > 1e-10 { 1.0 / mse } else { 1e10 };
        weights.push(weight);
    }

    let sum: f64 = weights.iter().sum();
    Ok(weights.iter().map(|w| w / sum).collect())
}

fn optimize_gradient_descent_weights(
    predictions: &[Vec<f64>],
    ground_truth: &[f64],
    learning_rate: f64,
) -> MLResult<Vec<f64>> {
    let n_models = predictions.len();
    let mut weights = vec![1.0 / n_models as f64; n_models];

    // Simplified gradient descent for 10 iterations
    for _ in 0..10 {
        let mut gradients = vec![0.0; n_models];

        for (i, pred) in predictions.iter().enumerate() {
            let errors: Vec<f64> = pred
                .iter()
                .zip(ground_truth.iter())
                .map(|(p, gt)| p - gt)
                .collect();

            gradients[i] = errors.iter().sum::<f64>() / errors.len() as f64;
        }

        // Update weights
        for (i, gradient) in gradients.iter().enumerate() {
            weights[i] = (weights[i] - learning_rate * gradient).max(0.0);
        }

        // Normalize
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
    }

    Ok(weights)
}
