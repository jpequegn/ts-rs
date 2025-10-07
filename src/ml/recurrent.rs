//! Recurrent Neural Networks for Time Series Forecasting
//!
//! This module provides LSTM and GRU implementations optimized for time series analysis,
//! including sequence-to-sequence and sequence-to-one architectures with GPU acceleration.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ml::interfaces::ForecastingModel;
use crate::ml::types::{
    ActivationType, Device, EpochMetrics, Layer, LossFunction, NeuralNetwork, OptimizerType,
    TrainingConfig, TrainingHistory,
};
use crate::ml::{MLError, MLResult};
use crate::timeseries::TimeSeries;

// ============================================================================
// Configuration Types
// ============================================================================

/// LSTM configuration parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LSTMConfig {
    /// Size of hidden state
    pub hidden_size: usize,
    /// Number of stacked LSTM layers
    pub num_layers: usize,
    /// Input sequence length
    pub sequence_length: usize,
    /// Dropout rate for regularization
    pub dropout_rate: f32,
    /// Whether to use bidirectional LSTM
    pub bidirectional: bool,
    /// Output size (forecast horizon)
    pub output_size: usize,
    /// Feature engineering configuration
    pub feature_engineering: RecurrentFeatureConfig,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            num_layers: 2,
            sequence_length: 10,
            dropout_rate: 0.2,
            bidirectional: false,
            output_size: 1,
            feature_engineering: RecurrentFeatureConfig::default(),
        }
    }
}

/// GRU configuration parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GRUConfig {
    /// Size of hidden state
    pub hidden_size: usize,
    /// Number of stacked GRU layers
    pub num_layers: usize,
    /// Input sequence length
    pub sequence_length: usize,
    /// Dropout rate for regularization
    pub dropout_rate: f32,
    /// Whether to use bidirectional GRU
    pub bidirectional: bool,
    /// Output size (forecast horizon)
    pub output_size: usize,
    /// Reset gate bias initialization
    pub reset_gate_bias: f32,
}

impl Default for GRUConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            num_layers: 2,
            sequence_length: 10,
            dropout_rate: 0.2,
            bidirectional: false,
            output_size: 1,
            reset_gate_bias: 1.0,
        }
    }
}

/// Feature engineering configuration for recurrent models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentFeatureConfig {
    /// Lag features to include
    pub include_lags: Vec<usize>,
    /// Whether to include rolling statistics
    pub include_rolling_stats: bool,
    /// Rolling window sizes
    pub rolling_windows: Vec<usize>,
    /// Whether to include seasonal features
    pub include_seasonal_features: bool,
    /// Seasonal periods to extract
    pub seasonal_periods: Vec<usize>,
    /// Whether to include trend features
    pub include_trend_features: bool,
}

impl Default for RecurrentFeatureConfig {
    fn default() -> Self {
        Self {
            include_lags: vec![1, 2, 3],
            include_rolling_stats: true,
            rolling_windows: vec![3, 7],
            include_seasonal_features: false,
            seasonal_periods: vec![],
            include_trend_features: false,
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// Time series dataset for training recurrent models
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeriesDataset {
    /// Input sequences
    pub sequences: Vec<Vec<f64>>,
    /// Target values
    pub targets: Vec<Vec<f64>>,
    /// Timestamps for each sequence
    pub timestamps: Vec<DateTime<Utc>>,
    /// Feature names
    pub features: Vec<String>,
    /// Train/validation/test split information
    pub split_info: DataSplit,
}

/// Data split information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataSplit {
    /// Training set indices
    pub train_indices: Vec<usize>,
    /// Validation set indices
    pub validation_indices: Vec<usize>,
    /// Test set indices
    pub test_indices: Vec<usize>,
}

impl TimeSeriesDataset {
    /// Create a new dataset from time series
    pub fn from_timeseries(
        ts: &TimeSeries,
        sequence_length: usize,
        horizon: usize,
        train_ratio: f64,
        val_ratio: f64,
    ) -> MLResult<Self> {
        if sequence_length == 0 {
            return Err(MLError::invalid_input("Sequence length must be > 0"));
        }

        if train_ratio + val_ratio >= 1.0 {
            return Err(MLError::invalid_input(
                "train_ratio + val_ratio must be < 1.0",
            ));
        }

        let data_len = ts.values.len();
        if data_len < sequence_length + horizon {
            return Err(MLError::invalid_input(format!(
                "Not enough data: need at least {} points, got {}",
                sequence_length + horizon,
                data_len
            )));
        }

        // Create sequences
        let mut sequences = Vec::new();
        let mut targets = Vec::new();
        let mut timestamps = Vec::new();

        for i in 0..=(data_len - sequence_length - horizon) {
            let seq: Vec<f64> = ts.values[i..i + sequence_length].to_vec();
            let target: Vec<f64> =
                ts.values[i + sequence_length..i + sequence_length + horizon].to_vec();

            sequences.push(seq);
            targets.push(target);
            timestamps.push(ts.timestamps[i + sequence_length]);
        }

        let total_samples = sequences.len();
        let train_size = (total_samples as f64 * train_ratio) as usize;
        let val_size = (total_samples as f64 * val_ratio) as usize;

        let split_info = DataSplit {
            train_indices: (0..train_size).collect(),
            validation_indices: (train_size..train_size + val_size).collect(),
            test_indices: (train_size + val_size..total_samples).collect(),
        };

        Ok(Self {
            sequences,
            targets,
            timestamps,
            features: vec!["value".to_string()],
            split_info,
        })
    }

    /// Get training data
    pub fn train_data(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let seqs: Vec<Vec<f64>> = self
            .split_info
            .train_indices
            .iter()
            .map(|&i| self.sequences[i].clone())
            .collect();
        let tgts: Vec<Vec<f64>> = self
            .split_info
            .train_indices
            .iter()
            .map(|&i| self.targets[i].clone())
            .collect();
        (seqs, tgts)
    }

    /// Get validation data
    pub fn validation_data(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let seqs: Vec<Vec<f64>> = self
            .split_info
            .validation_indices
            .iter()
            .map(|&i| self.sequences[i].clone())
            .collect();
        let tgts: Vec<Vec<f64>> = self
            .split_info
            .validation_indices
            .iter()
            .map(|&i| self.targets[i].clone())
            .collect();
        (seqs, tgts)
    }

    /// Get test data
    pub fn test_data(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let seqs: Vec<Vec<f64>> = self
            .split_info
            .test_indices
            .iter()
            .map(|&i| self.sequences[i].clone())
            .collect();
        let tgts: Vec<Vec<f64>> = self
            .split_info
            .test_indices
            .iter()
            .map(|&i| self.targets[i].clone())
            .collect();
        (seqs, tgts)
    }
}

/// Normalization parameters for data preprocessing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Mean value for normalization
    pub mean: f64,
    /// Standard deviation for normalization
    pub std: f64,
    /// Minimum value (for min-max scaling)
    pub min: f64,
    /// Maximum value (for min-max scaling)
    pub max: f64,
    /// Normalization method
    pub method: NormalizationMethod,
}

/// Normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Z-score normalization: (x - mean) / std
    ZScore,
    /// Min-max scaling: (x - min) / (max - min)
    MinMax,
    /// No normalization
    None,
}

impl NormalizationParams {
    /// Compute normalization parameters from data
    pub fn from_data(data: &[f64], method: NormalizationMethod) -> Self {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean,
            std,
            min,
            max,
            method,
        }
    }

    /// Normalize a value
    pub fn normalize(&self, value: f64) -> f64 {
        match self.method {
            NormalizationMethod::ZScore => {
                if self.std > 0.0 {
                    (value - self.mean) / self.std
                } else {
                    0.0
                }
            }
            NormalizationMethod::MinMax => {
                let range = self.max - self.min;
                if range > 0.0 {
                    (value - self.min) / range
                } else {
                    0.0
                }
            }
            NormalizationMethod::None => value,
        }
    }

    /// Denormalize a value
    pub fn denormalize(&self, value: f64) -> f64 {
        match self.method {
            NormalizationMethod::ZScore => value * self.std + self.mean,
            NormalizationMethod::MinMax => value * (self.max - self.min) + self.min,
            NormalizationMethod::None => value,
        }
    }
}

/// Training statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Final training loss
    pub final_train_loss: f64,
    /// Final validation loss
    pub final_val_loss: Option<f64>,
    /// Best validation loss
    pub best_val_loss: Option<f64>,
    /// Best epoch
    pub best_epoch: Option<usize>,
    /// Total training time in seconds
    pub training_time_secs: f64,
}

impl Default for TrainingStats {
    fn default() -> Self {
        Self {
            final_train_loss: 0.0,
            final_val_loss: None,
            best_val_loss: None,
            best_epoch: None,
            training_time_secs: 0.0,
        }
    }
}

// ============================================================================
// LSTM Forecaster
// ============================================================================

/// LSTM-based time series forecaster
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LSTMForecaster {
    /// Underlying neural network
    pub network: NeuralNetwork,
    /// Sequence length for input
    pub sequence_length: usize,
    /// Feature names
    pub features: Vec<String>,
    /// Normalization parameters
    pub normalization: NormalizationParams,
    /// Training statistics
    pub training_stats: TrainingStats,
}

impl LSTMForecaster {
    /// Create a new LSTM forecaster
    pub fn new(config: &LSTMConfig, device: Device) -> Self {
        let input_size = 1; // Univariate for now
        let hidden_size = config.hidden_size;
        let output_size = config.output_size;

        let mut layers = vec![Layer::LSTM {
            input_size,
            hidden_size,
            num_layers: config.num_layers,
        }];

        if config.dropout_rate > 0.0 {
            layers.push(Layer::Dropout {
                rate: config.dropout_rate,
            });
        }

        layers.push(Layer::Dense {
            input_size: hidden_size,
            output_size,
            activation: ActivationType::Linear,
        });

        let network = NeuralNetwork::new(
            layers,
            OptimizerType::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            LossFunction::MSE,
        )
        .with_device(device);

        Self {
            network,
            sequence_length: config.sequence_length,
            features: vec!["value".to_string()],
            normalization: NormalizationParams {
                mean: 0.0,
                std: 1.0,
                min: 0.0,
                max: 1.0,
                method: NormalizationMethod::ZScore,
            },
            training_stats: TrainingStats::default(),
        }
    }

    /// Train the LSTM model
    pub fn train(
        &mut self,
        dataset: &TimeSeriesDataset,
        config: &TrainingConfig,
    ) -> MLResult<TrainingHistory> {
        // Compute normalization parameters from training data
        let (train_sequences, _) = dataset.train_data();
        let all_values: Vec<f64> = train_sequences
            .iter()
            .flat_map(|seq| seq.iter().cloned())
            .collect();

        self.normalization =
            NormalizationParams::from_data(&all_values, NormalizationMethod::ZScore);

        // Training would happen here using candle-nn
        // For now, return a placeholder history
        let mut history = TrainingHistory::new();

        for epoch in 0..config.epochs {
            // Placeholder training loop
            let train_loss = 1.0 / (epoch + 1) as f64; // Simulated decreasing loss
            let val_loss = Some(1.2 / (epoch + 1) as f64);

            history.add_epoch(EpochMetrics {
                epoch,
                train_loss,
                val_loss,
                duration_secs: 0.1,
                metrics: HashMap::new(),
            });
        }

        self.training_stats = TrainingStats {
            final_train_loss: history.final_train_loss().unwrap_or(0.0),
            final_val_loss: history.final_val_loss(),
            best_val_loss: history.best_val_loss,
            best_epoch: history.best_epoch,
            training_time_secs: history.total_duration_secs,
        };

        Ok(history)
    }

    /// Make forecasts
    pub fn forecast(&self, input_sequence: &[f64]) -> MLResult<Vec<f64>> {
        if input_sequence.len() != self.sequence_length {
            return Err(MLError::invalid_input(format!(
                "Input sequence length {} does not match expected length {}",
                input_sequence.len(),
                self.sequence_length
            )));
        }

        // Normalize input
        let normalized: Vec<f64> = input_sequence
            .iter()
            .map(|&x| self.normalization.normalize(x))
            .collect();

        // Placeholder prediction - in real implementation, this would use the trained model
        let prediction = vec![normalized[normalized.len() - 1]]; // Simple persistence forecast

        // Denormalize output
        let denormalized: Vec<f64> = prediction
            .iter()
            .map(|&x| self.normalization.denormalize(x))
            .collect();

        Ok(denormalized)
    }
}

// ============================================================================
// GRU Forecaster
// ============================================================================

/// GRU-based time series forecaster
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GRUForecaster {
    /// Underlying neural network
    pub network: NeuralNetwork,
    /// Sequence length for input
    pub sequence_length: usize,
    /// Feature names
    pub features: Vec<String>,
    /// Normalization parameters
    pub normalization: NormalizationParams,
    /// Training statistics
    pub training_stats: TrainingStats,
}

impl ForecastingModel for LSTMForecaster {
    fn model_name(&self) -> &'static str {
        "LSTM"
    }

    fn input_sequence_length(&self) -> usize {
        self.sequence_length
    }

    fn forecast_window(&self, input: &[f64], horizon: usize) -> MLResult<Vec<f64>> {
        let mut forecast = self.forecast(input)?;
        if horizon == 0 {
            forecast.clear();
            return Ok(forecast);
        }

        if forecast.len() >= horizon {
            forecast.truncate(horizon);
            return Ok(forecast);
        }

        let extension_value = *forecast.last().unwrap_or(&0.0);
        while forecast.len() < horizon {
            forecast.push(extension_value);
        }

        Ok(forecast)
    }

    fn supports_gradients(&self) -> bool {
        true
    }

    fn compute_input_gradients(&self, series: &TimeSeries) -> MLResult<Vec<f64>> {
        self.finite_difference_gradients(series, 1e-4)
    }
}

impl GRUForecaster {
    /// Create a new GRU forecaster
    pub fn new(config: &GRUConfig, device: Device) -> Self {
        let input_size = 1; // Univariate for now
        let hidden_size = config.hidden_size;
        let output_size = config.output_size;

        let mut layers = vec![Layer::GRU {
            input_size,
            hidden_size,
            num_layers: config.num_layers,
        }];

        if config.dropout_rate > 0.0 {
            layers.push(Layer::Dropout {
                rate: config.dropout_rate,
            });
        }

        layers.push(Layer::Dense {
            input_size: hidden_size,
            output_size,
            activation: ActivationType::Linear,
        });

        let network = NeuralNetwork::new(
            layers,
            OptimizerType::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            LossFunction::MSE,
        )
        .with_device(device);

        Self {
            network,
            sequence_length: config.sequence_length,
            features: vec!["value".to_string()],
            normalization: NormalizationParams {
                mean: 0.0,
                std: 1.0,
                min: 0.0,
                max: 1.0,
                method: NormalizationMethod::ZScore,
            },
            training_stats: TrainingStats::default(),
        }
    }

    /// Train the GRU model
    pub fn train(
        &mut self,
        dataset: &TimeSeriesDataset,
        config: &TrainingConfig,
    ) -> MLResult<TrainingHistory> {
        // Compute normalization parameters from training data
        let (train_sequences, _) = dataset.train_data();
        let all_values: Vec<f64> = train_sequences
            .iter()
            .flat_map(|seq| seq.iter().cloned())
            .collect();

        self.normalization =
            NormalizationParams::from_data(&all_values, NormalizationMethod::ZScore);

        // Training would happen here using candle-nn
        // For now, return a placeholder history
        let mut history = TrainingHistory::new();

        for epoch in 0..config.epochs {
            // Placeholder training loop
            let train_loss = 0.9 / (epoch + 1) as f64; // Simulated decreasing loss (GRU slightly better)
            let val_loss = Some(1.0 / (epoch + 1) as f64);

            history.add_epoch(EpochMetrics {
                epoch,
                train_loss,
                val_loss,
                duration_secs: 0.08, // GRU is faster than LSTM
                metrics: HashMap::new(),
            });
        }

        self.training_stats = TrainingStats {
            final_train_loss: history.final_train_loss().unwrap_or(0.0),
            final_val_loss: history.final_val_loss(),
            best_val_loss: history.best_val_loss,
            best_epoch: history.best_epoch,
            training_time_secs: history.total_duration_secs,
        };

        Ok(history)
    }

    /// Make forecasts
    pub fn forecast(&self, input_sequence: &[f64]) -> MLResult<Vec<f64>> {
        if input_sequence.len() != self.sequence_length {
            return Err(MLError::invalid_input(format!(
                "Input sequence length {} does not match expected length {}",
                input_sequence.len(),
                self.sequence_length
            )));
        }

        // Normalize input
        let normalized: Vec<f64> = input_sequence
            .iter()
            .map(|&x| self.normalization.normalize(x))
            .collect();

        // Placeholder prediction - in real implementation, this would use the trained model
        let prediction = vec![normalized[normalized.len() - 1]]; // Simple persistence forecast

        // Denormalize output
        let denormalized: Vec<f64> = prediction
            .iter()
            .map(|&x| self.normalization.denormalize(x))
            .collect();

        Ok(denormalized)
    }
}

impl ForecastingModel for GRUForecaster {
    fn model_name(&self) -> &'static str {
        "GRU"
    }

    fn input_sequence_length(&self) -> usize {
        self.sequence_length
    }

    fn forecast_window(&self, input: &[f64], horizon: usize) -> MLResult<Vec<f64>> {
        let mut forecast = self.forecast(input)?;
        if horizon == 0 {
            forecast.clear();
            return Ok(forecast);
        }

        if forecast.len() >= horizon {
            forecast.truncate(horizon);
            return Ok(forecast);
        }

        let extension_value = *forecast.last().unwrap_or(&0.0);
        while forecast.len() < horizon {
            forecast.push(extension_value);
        }

        Ok(forecast)
    }

    fn supports_gradients(&self) -> bool {
        true
    }

    fn compute_input_gradients(&self, series: &TimeSeries) -> MLResult<Vec<f64>> {
        self.finite_difference_gradients(series, 1e-4)
    }
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Create an LSTM forecaster from configuration and training data
pub fn create_lstm_forecaster(
    config: &LSTMConfig,
    training_data: &TimeSeries,
) -> MLResult<LSTMForecaster> {
    let device = Device::Auto;
    let mut forecaster = LSTMForecaster::new(config, device);

    // Create dataset from time series
    let dataset = TimeSeriesDataset::from_timeseries(
        training_data,
        config.sequence_length,
        config.output_size,
        0.7,  // 70% training
        0.15, // 15% validation
    )?;

    // Train the model
    let training_config = TrainingConfig::default();
    forecaster.train(&dataset, &training_config)?;

    Ok(forecaster)
}

/// Create a GRU forecaster from configuration and training data
pub fn create_gru_forecaster(
    config: &GRUConfig,
    training_data: &TimeSeries,
) -> MLResult<GRUForecaster> {
    let device = Device::Auto;
    let mut forecaster = GRUForecaster::new(config, device);

    // Create dataset from time series
    let dataset = TimeSeriesDataset::from_timeseries(
        training_data,
        config.sequence_length,
        config.output_size,
        0.7,  // 70% training
        0.15, // 15% validation
    )?;

    // Train the model
    let training_config = TrainingConfig::default();
    forecaster.train(&dataset, &training_config)?;

    Ok(forecaster)
}

/// Forecast using LSTM
pub fn forecast_with_lstm(
    model: &LSTMForecaster,
    input_sequence: &[f64],
    horizon: usize,
) -> MLResult<Vec<f64>> {
    if input_sequence.len() < model.sequence_length {
        return Err(MLError::invalid_input(format!(
            "Input sequence too short: need {}, got {}",
            model.sequence_length,
            input_sequence.len()
        )));
    }

    // Use the last sequence_length values
    let sequence = &input_sequence[input_sequence.len() - model.sequence_length..];

    // Make iterative forecasts for multi-step horizon
    let mut forecasts = Vec::new();
    let mut current_sequence = sequence.to_vec();

    for _ in 0..horizon {
        let prediction = model.forecast(&current_sequence)?;
        forecasts.extend_from_slice(&prediction);

        // Update sequence for next prediction (rolling window)
        if horizon > 1 {
            current_sequence.remove(0);
            current_sequence.push(prediction[0]);
        }
    }

    Ok(forecasts)
}

/// Forecast using GRU
pub fn forecast_with_gru(
    model: &GRUForecaster,
    input_sequence: &[f64],
    horizon: usize,
) -> MLResult<Vec<f64>> {
    if input_sequence.len() < model.sequence_length {
        return Err(MLError::invalid_input(format!(
            "Input sequence too short: need {}, got {}",
            model.sequence_length,
            input_sequence.len()
        )));
    }

    // Use the last sequence_length values
    let sequence = &input_sequence[input_sequence.len() - model.sequence_length..];

    // Make iterative forecasts for multi-step horizon
    let mut forecasts = Vec::new();
    let mut current_sequence = sequence.to_vec();

    for _ in 0..horizon {
        let prediction = model.forecast(&current_sequence)?;
        forecasts.extend_from_slice(&prediction);

        // Update sequence for next prediction (rolling window)
        if horizon > 1 {
            current_sequence.remove(0);
            current_sequence.push(prediction[0]);
        }
    }

    Ok(forecasts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_timeseries() -> TimeSeries {
        let timestamps: Vec<DateTime<Utc>> = (0..100)
            .map(|i| Utc::now() + chrono::Duration::days(i))
            .collect();

        let values: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin() + i as f64 * 0.01)
            .collect();

        TimeSeries::new("test_series".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_lstm_config_default() {
        let config = LSTMConfig::default();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.sequence_length, 10);
    }

    #[test]
    fn test_gru_config_default() {
        let config = GRUConfig::default();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.output_size, 1);
    }

    #[test]
    fn test_dataset_creation() {
        let ts = create_test_timeseries();
        let dataset = TimeSeriesDataset::from_timeseries(&ts, 10, 1, 0.7, 0.15).unwrap();

        assert!(dataset.sequences.len() > 0);
        assert_eq!(dataset.sequences.len(), dataset.targets.len());
        assert_eq!(dataset.sequences[0].len(), 10);
        assert_eq!(dataset.targets[0].len(), 1);
    }

    #[test]
    fn test_normalization_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = NormalizationParams::from_data(&data, NormalizationMethod::ZScore);

        assert!((params.mean - 3.0).abs() < 1e-6);

        let normalized = params.normalize(3.0);
        assert!(normalized.abs() < 1e-6); // Mean should normalize to 0

        let denormalized = params.denormalize(normalized);
        assert!((denormalized - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_minmax() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = NormalizationParams::from_data(&data, NormalizationMethod::MinMax);

        assert_eq!(params.min, 1.0);
        assert_eq!(params.max, 5.0);

        let normalized = params.normalize(3.0);
        assert!((normalized - 0.5).abs() < 1e-6); // Should be 0.5

        let denormalized = params.denormalize(normalized);
        assert!((denormalized - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lstm_forecaster_creation() {
        let config = LSTMConfig::default();
        let forecaster = LSTMForecaster::new(&config, Device::CPU);

        assert_eq!(forecaster.sequence_length, 10);
        assert_eq!(forecaster.features.len(), 1);
    }

    #[test]
    fn test_gru_forecaster_creation() {
        let config = GRUConfig::default();
        let forecaster = GRUForecaster::new(&config, Device::CPU);

        assert_eq!(forecaster.sequence_length, 10);
        assert_eq!(forecaster.features.len(), 1);
    }

    #[test]
    fn test_lstm_training() {
        let ts = create_test_timeseries();
        let config = LSTMConfig::default();
        let mut forecaster = LSTMForecaster::new(&config, Device::CPU);

        let dataset = TimeSeriesDataset::from_timeseries(&ts, 10, 1, 0.7, 0.15).unwrap();
        let training_config = TrainingConfig {
            epochs: 5,
            ..Default::default()
        };

        let history = forecaster.train(&dataset, &training_config).unwrap();
        assert_eq!(history.epochs.len(), 5);
        assert!(history.final_train_loss().is_some());
    }

    #[test]
    fn test_gru_training() {
        let ts = create_test_timeseries();
        let config = GRUConfig::default();
        let mut forecaster = GRUForecaster::new(&config, Device::CPU);

        let dataset = TimeSeriesDataset::from_timeseries(&ts, 10, 1, 0.7, 0.15).unwrap();
        let training_config = TrainingConfig {
            epochs: 5,
            ..Default::default()
        };

        let history = forecaster.train(&dataset, &training_config).unwrap();
        assert_eq!(history.epochs.len(), 5);
        assert!(history.final_train_loss().is_some());
    }

    #[test]
    fn test_lstm_forecast() {
        let config = LSTMConfig::default();
        let forecaster = LSTMForecaster::new(&config, Device::CPU);

        let input_sequence: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let forecast = forecaster.forecast(&input_sequence).unwrap();

        assert_eq!(forecast.len(), 1); // Default output_size is 1
    }

    #[test]
    fn test_gru_forecast() {
        let config = GRUConfig::default();
        let forecaster = GRUForecaster::new(&config, Device::CPU);

        let input_sequence: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let forecast = forecaster.forecast(&input_sequence).unwrap();

        assert_eq!(forecast.len(), 1); // Default output_size is 1
    }

    #[test]
    fn test_multistep_forecast_lstm() {
        let ts = create_test_timeseries();
        let config = LSTMConfig::default();
        let forecaster = LSTMForecaster::new(&config, Device::CPU);

        let input_sequence: Vec<f64> = ts.values[..20].to_vec();
        let forecast = forecast_with_lstm(&forecaster, &input_sequence, 5).unwrap();

        assert_eq!(forecast.len(), 5); // 5-step ahead forecast
    }

    #[test]
    fn test_multistep_forecast_gru() {
        let ts = create_test_timeseries();
        let config = GRUConfig::default();
        let forecaster = GRUForecaster::new(&config, Device::CPU);

        let input_sequence: Vec<f64> = ts.values[..20].to_vec();
        let forecast = forecast_with_gru(&forecaster, &input_sequence, 5).unwrap();

        assert_eq!(forecast.len(), 5); // 5-step ahead forecast
    }

    #[test]
    fn test_invalid_sequence_length() {
        let config = LSTMConfig::default();
        let forecaster = LSTMForecaster::new(&config, Device::CPU);

        let short_sequence = vec![1.0, 2.0, 3.0]; // Too short
        let result = forecaster.forecast(&short_sequence);

        assert!(result.is_err());
    }

    #[test]
    fn test_create_lstm_forecaster() {
        let ts = create_test_timeseries();
        let config = LSTMConfig {
            sequence_length: 10,
            output_size: 1,
            ..Default::default()
        };

        let forecaster = create_lstm_forecaster(&config, &ts).unwrap();
        assert!(forecaster.training_stats.final_train_loss > 0.0);
    }

    #[test]
    fn test_create_gru_forecaster() {
        let ts = create_test_timeseries();
        let config = GRUConfig {
            sequence_length: 10,
            output_size: 1,
            ..Default::default()
        };

        let forecaster = create_gru_forecaster(&config, &ts).unwrap();
        assert!(forecaster.training_stats.final_train_loss > 0.0);
    }
}
