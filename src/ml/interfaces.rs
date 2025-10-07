use std::fmt;

use crate::timeseries::TimeSeries;

use super::transformer::AttentionAnalysis;
use super::{MLError, MLResult};

/// Common interface for forecasting models exposing capabilities needed by interpretability features.
pub trait ForecastingModel {
    /// Name used for diagnostics and error reporting.
    fn model_name(&self) -> &'static str;

    /// Number of input timesteps consumed by the model.
    fn input_sequence_length(&self) -> usize;

    /// Core forecasting routine operating on a prepared input window.
    fn forecast_window(&self, input: &[f64], horizon: usize) -> MLResult<Vec<f64>>;

    /// Convenience helper to forecast directly from a time series.
    fn forecast_from_series(&self, series: &TimeSeries, horizon: usize) -> MLResult<Vec<f64>> {
        let window = self.prepare_input_window(series)?;
        self.forecast_window(&window, horizon)
    }

    /// Indicates whether the model can provide gradient-based explanations.
    fn supports_gradients(&self) -> bool {
        false
    }

    /// Compute input gradients for attribution.
    fn compute_input_gradients(&self, series: &TimeSeries) -> MLResult<Vec<f64>> {
        Err(MLError::model(format!(
            "{} does not support gradient-based introspection",
            self.model_name()
        )))
    }

    /// Indicates whether attention-based insights are available.
    fn supports_attention(&self) -> bool {
        false
    }

    /// Retrieve attention analysis for the provided series.
    fn attention_analysis(&self, series: &TimeSeries) -> MLResult<AttentionAnalysis> {
        Err(MLError::model(format!(
            "{} does not expose attention patterns",
            self.model_name()
        )))
    }

    /// Extract the most recent input window from the series.
    fn prepare_input_window(&self, series: &TimeSeries) -> MLResult<Vec<f64>> {
        let seq_len = self.input_sequence_length();
        let total = series.values.len();
        if total < seq_len {
            return Err(MLError::invalid_input(format!(
                "{} requires at least {} observations but found {}",
                self.model_name(),
                seq_len,
                total
            )));
        }

        Ok(series.values[total - seq_len..].to_vec())
    }

    /// Utility for models that approximate gradients through finite differences.
    fn finite_difference_gradients(&self, series: &TimeSeries, epsilon: f64) -> MLResult<Vec<f64>> {
        let base_input = self.prepare_input_window(series)?;
        if base_input.is_empty() {
            return Err(MLError::invalid_input(format!(
                "{} received an empty input window",
                self.model_name()
            )));
        }

        let base_forecast = self.forecast_window(&base_input, 1)?;
        let reference = *base_forecast.first().ok_or_else(|| {
            MLError::model(format!(
                "{} returned an empty forecast for gradient computation",
                self.model_name()
            ))
        })?;

        let mut gradients = Vec::with_capacity(base_input.len());
        for index in 0..base_input.len() {
            let mut perturbed = base_input.clone();
            perturbed[index] += epsilon;
            let forecast = self.forecast_window(&perturbed, 1)?;
            let value = *forecast.first().unwrap_or(&reference);
            gradients.push((value - reference) / epsilon);
        }

        Ok(gradients)
    }
}

/// Summary of model capabilities used by higher-level components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub gradients: bool,
    pub attention: bool,
}

impl fmt::Display for ModelCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "gradients: {}, attention: {}",
            self.gradients, self.attention
        )
    }
}

impl ModelCapabilities {
    pub fn new(gradients: bool, attention: bool) -> Self {
        Self {
            gradients,
            attention,
        }
    }
}
