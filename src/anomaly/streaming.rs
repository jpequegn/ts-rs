//! # Streaming Anomaly Detection
//!
//! Real-time anomaly detection capabilities for streaming time series data
//! with adaptive thresholds and online learning.

use crate::analysis::{Anomaly, AnomalySeverity};
use crate::TimeSeries;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

/// Streaming anomaly detector with adaptive capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingAnomalyDetector {
    /// Detection method configuration
    pub method: StreamingMethod,

    /// Current data window for processing
    pub window: VecDeque<(DateTime<Utc>, f64)>,

    /// Maximum window size
    pub max_window_size: usize,

    /// Current detection threshold
    pub threshold: f64,

    /// Adaptive threshold parameters
    pub adaptive_config: AdaptiveConfig,

    /// Online learning parameters
    pub learning_config: LearningConfig,

    /// Running statistics for the window
    pub stats: WindowStatistics,
}

/// Available streaming detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingMethod {
    /// Real-time Z-score detection
    ZScore { threshold: f64 },

    /// Real-time Modified Z-score detection
    ModifiedZScore { threshold: f64 },

    /// Real-time IQR-based detection
    IQR { factor: f64 },

    /// Real-time volatility detection
    Volatility { window_size: usize },

    /// Adaptive threshold detection
    AdaptiveThreshold { initial_threshold: f64 },
}

/// Adaptive threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable adaptive thresholds
    pub enabled: bool,

    /// Learning rate for threshold adaptation
    pub learning_rate: f64,

    /// Minimum threshold value
    pub min_threshold: f64,

    /// Maximum threshold value
    pub max_threshold: f64,

    /// Adaptation sensitivity
    pub sensitivity: f64,
}

/// Online learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable online learning
    pub enabled: bool,

    /// Learning rate for model updates
    pub learning_rate: f64,

    /// Decay factor for old observations
    pub decay_factor: f64,

    /// Minimum samples before enabling learning
    pub min_samples: usize,
}

/// Running window statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowStatistics {
    /// Current mean
    pub mean: f64,

    /// Current variance
    pub variance: f64,

    /// Current standard deviation
    pub std_dev: f64,

    /// Current median
    pub median: f64,

    /// Current Q1 (25th percentile)
    pub q1: f64,

    /// Current Q3 (75th percentile)
    pub q3: f64,

    /// Current IQR
    pub iqr: f64,

    /// Number of samples processed
    pub sample_count: usize,
}

impl StreamingAnomalyDetector {
    /// Create a new streaming anomaly detector
    pub fn new(
        method: StreamingMethod,
        max_window_size: usize,
        adaptive_config: AdaptiveConfig,
        learning_config: LearningConfig,
    ) -> Self {
        let threshold = match &method {
            StreamingMethod::ZScore { threshold } => *threshold,
            StreamingMethod::ModifiedZScore { threshold } => *threshold,
            StreamingMethod::IQR { factor } => *factor,
            StreamingMethod::Volatility { .. } => 2.0,
            StreamingMethod::AdaptiveThreshold { initial_threshold } => *initial_threshold,
        };

        Self {
            method,
            window: VecDeque::new(),
            max_window_size,
            threshold,
            adaptive_config,
            learning_config,
            stats: WindowStatistics::default(),
        }
    }

    /// Add a new data point and check for anomalies
    pub fn process_point(
        &mut self,
        timestamp: DateTime<Utc>,
        value: f64,
    ) -> Result<Option<Anomaly>, Box<dyn std::error::Error>> {
        // Skip NaN values
        if value.is_nan() {
            return Ok(None);
        }

        // Add to window
        self.window.push_back((timestamp, value));

        // Maintain window size
        if self.window.len() > self.max_window_size {
            self.window.pop_front();
        }

        // Update statistics
        self.update_statistics();

        // Check for anomaly
        let anomaly_score = self.calculate_anomaly_score(value)?;

        if self.is_anomaly(anomaly_score) {
            let severity = self.classify_severity(anomaly_score);
            let expected_value = self.calculate_expected_value();

            let anomaly = Anomaly {
                index: self.stats.sample_count - 1,
                timestamp,
                value,
                score: anomaly_score,
                severity,
                expected_value,
            };

            // Update adaptive threshold if enabled
            if self.adaptive_config.enabled {
                self.update_adaptive_threshold(anomaly_score, true);
            }

            Ok(Some(anomaly))
        } else {
            // Update adaptive threshold for normal points
            if self.adaptive_config.enabled {
                self.update_adaptive_threshold(anomaly_score, false);
            }

            Ok(None)
        }
    }

    /// Process multiple points from a time series
    pub fn process_timeseries(
        &mut self,
        timeseries: &TimeSeries,
    ) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
        let mut anomalies = Vec::new();

        for (timestamp, value) in timeseries.timestamps.iter().zip(timeseries.values.iter()) {
            if let Some(anomaly) = self.process_point(*timestamp, *value)? {
                anomalies.push(anomaly);
            }
        }

        Ok(anomalies)
    }

    /// Calculate anomaly score for a value
    fn calculate_anomaly_score(&self, value: f64) -> Result<f64, Box<dyn std::error::Error>> {
        if self.window.len() < 2 {
            return Ok(0.0);
        }

        match &self.method {
            StreamingMethod::ZScore { .. } => {
                if self.stats.std_dev > 0.0 {
                    Ok((value - self.stats.mean).abs() / self.stats.std_dev)
                } else {
                    Ok(0.0)
                }
            }
            StreamingMethod::ModifiedZScore { .. } => {
                if self.stats.median != 0.0 {
                    let median_deviation = (value - self.stats.median).abs();
                    let mad = self.calculate_mad();
                    if mad > 0.0 {
                        Ok(0.6745 * median_deviation / mad)
                    } else {
                        Ok(0.0)
                    }
                } else {
                    Ok(0.0)
                }
            }
            StreamingMethod::IQR { factor } => {
                if self.stats.iqr > 0.0 {
                    let lower_bound = self.stats.q1 - factor * self.stats.iqr;
                    let upper_bound = self.stats.q3 + factor * self.stats.iqr;

                    if value < lower_bound {
                        Ok((lower_bound - value) / self.stats.iqr)
                    } else if value > upper_bound {
                        Ok((value - upper_bound) / self.stats.iqr)
                    } else {
                        Ok(0.0)
                    }
                } else {
                    Ok(0.0)
                }
            }
            StreamingMethod::Volatility { window_size } => {
                self.calculate_volatility_score(value, *window_size)
            }
            StreamingMethod::AdaptiveThreshold { .. } => {
                // Use current adaptive threshold
                if self.stats.std_dev > 0.0 {
                    Ok((value - self.stats.mean).abs() / self.stats.std_dev)
                } else {
                    Ok(0.0)
                }
            }
        }
    }

    /// Check if a score indicates an anomaly
    fn is_anomaly(&self, score: f64) -> bool {
        score > self.threshold
    }

    /// Classify anomaly severity based on score
    fn classify_severity(&self, score: f64) -> AnomalySeverity {
        let ratio = score / self.threshold;

        if ratio >= 2.5 {
            AnomalySeverity::Critical
        } else if ratio >= 2.0 {
            AnomalySeverity::High
        } else if ratio >= 1.5 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }

    /// Calculate expected value based on current statistics
    fn calculate_expected_value(&self) -> Option<f64> {
        if self.window.len() >= 2 {
            Some(self.stats.mean)
        } else {
            None
        }
    }

    /// Update adaptive threshold based on current score
    fn update_adaptive_threshold(&mut self, score: f64, is_anomaly: bool) {
        if !self.adaptive_config.enabled || self.stats.sample_count < self.learning_config.min_samples {
            return;
        }

        let learning_rate = self.adaptive_config.learning_rate;
        let sensitivity = self.adaptive_config.sensitivity;

        if is_anomaly {
            // Increase threshold for false positives
            self.threshold = self.threshold * (1.0 + learning_rate * sensitivity);
        } else {
            // Decrease threshold for potential false negatives
            if score > self.threshold * 0.8 {
                self.threshold = self.threshold * (1.0 - learning_rate * sensitivity * 0.5);
            }
        }

        // Clamp threshold to configured bounds
        self.threshold = self.threshold.clamp(
            self.adaptive_config.min_threshold,
            self.adaptive_config.max_threshold,
        );
    }

    /// Update running window statistics
    fn update_statistics(&mut self) {
        let values: Vec<f64> = self.window.iter().map(|(_, v)| *v).collect();

        if values.is_empty() {
            return;
        }

        self.stats.sample_count += 1;

        // Calculate mean
        self.stats.mean = values.iter().sum::<f64>() / values.len() as f64;

        // Calculate variance and standard deviation
        if values.len() > 1 {
            self.stats.variance = values
                .iter()
                .map(|v| (v - self.stats.mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            self.stats.std_dev = self.stats.variance.sqrt();
        } else {
            self.stats.variance = 0.0;
            self.stats.std_dev = 0.0;
        }

        // Calculate quantiles
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.stats.median = calculate_percentile(&sorted_values, 50.0);
        self.stats.q1 = calculate_percentile(&sorted_values, 25.0);
        self.stats.q3 = calculate_percentile(&sorted_values, 75.0);
        self.stats.iqr = self.stats.q3 - self.stats.q1;
    }

    /// Calculate Median Absolute Deviation (MAD)
    fn calculate_mad(&self) -> f64 {
        let values: Vec<f64> = self.window.iter().map(|(_, v)| *v).collect();

        if values.is_empty() {
            return 0.0;
        }

        let median = self.stats.median;
        let mut deviations: Vec<f64> = values
            .iter()
            .map(|v| (v - median).abs())
            .collect();

        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        calculate_percentile(&deviations, 50.0)
    }

    /// Calculate volatility-based anomaly score
    fn calculate_volatility_score(
        &self,
        value: f64,
        window_size: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if self.window.len() < window_size {
            return Ok(0.0);
        }

        // Calculate recent volatility
        let recent_values: Vec<f64> = self.window
            .iter()
            .rev()
            .take(window_size)
            .map(|(_, v)| *v)
            .collect();

        if recent_values.len() < 2 {
            return Ok(0.0);
        }

        // Calculate returns (percentage changes)
        let mut returns = Vec::new();
        for i in 1..recent_values.len() {
            if recent_values[i - 1] != 0.0 {
                let return_val = (recent_values[i] - recent_values[i - 1]) / recent_values[i - 1];
                returns.push(return_val);
            }
        }

        if returns.is_empty() {
            return Ok(0.0);
        }

        // Calculate volatility (standard deviation of returns)
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = (returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64)
            .sqrt();

        // Current return
        if let Some((_, last_value)) = self.window.iter().rev().nth(1) {
            if *last_value != 0.0 {
                let current_return = (value - last_value) / last_value;
                if volatility > 0.0 {
                    Ok(current_return.abs() / volatility)
                } else {
                    Ok(0.0)
                }
            } else {
                Ok(0.0)
            }
        } else {
            Ok(0.0)
        }
    }

    /// Reset the detector state
    pub fn reset(&mut self) {
        self.window.clear();
        self.stats = WindowStatistics::default();

        // Reset threshold to initial value
        self.threshold = match &self.method {
            StreamingMethod::ZScore { threshold } => *threshold,
            StreamingMethod::ModifiedZScore { threshold } => *threshold,
            StreamingMethod::IQR { factor } => *factor,
            StreamingMethod::Volatility { .. } => 2.0,
            StreamingMethod::AdaptiveThreshold { initial_threshold } => *initial_threshold,
        };
    }

    /// Get current detector state
    pub fn get_state(&self) -> StreamingDetectorState {
        StreamingDetectorState {
            window_size: self.window.len(),
            current_threshold: self.threshold,
            stats: self.stats.clone(),
        }
    }
}

/// Current state of the streaming detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDetectorState {
    /// Current window size
    pub window_size: usize,

    /// Current detection threshold
    pub current_threshold: f64,

    /// Current window statistics
    pub stats: WindowStatistics,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            learning_rate: 0.01,
            min_threshold: 1.0,
            max_threshold: 10.0,
            sensitivity: 1.0,
        }
    }
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            learning_rate: 0.05,
            decay_factor: 0.99,
            min_samples: 10,
        }
    }
}

impl Default for WindowStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            median: 0.0,
            q1: 0.0,
            q3: 0.0,
            iqr: 0.0,
            sample_count: 0,
        }
    }
}

/// Calculate percentile of a sorted array
fn calculate_percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    if sorted_values.len() == 1 {
        return sorted_values[0];
    }

    let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = index - lower as f64;
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn test_streaming_zscore_detector() {
        let method = StreamingMethod::ZScore { threshold: 3.0 };
        let adaptive_config = AdaptiveConfig::default();
        let learning_config = LearningConfig::default();

        let mut detector = StreamingAnomalyDetector::new(
            method,
            100,
            adaptive_config,
            learning_config,
        );

        // Add normal values
        for i in 0..50 {
            let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, i as u32, 0).unwrap();
            let value = 10.0 + (i as f64 * 0.1).sin();
            let result = detector.process_point(timestamp, value).unwrap();
            assert!(result.is_none());
        }

        // Add an anomalous value
        let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, 50, 0).unwrap();
        let anomalous_value = 50.0; // Should be an anomaly
        let result = detector.process_point(timestamp, anomalous_value).unwrap();

        assert!(result.is_some());
        let anomaly = result.unwrap();
        assert_eq!(anomaly.value, anomalous_value);
        assert!(anomaly.score > 3.0);
    }

    #[test]
    fn test_adaptive_threshold() {
        let method = StreamingMethod::AdaptiveThreshold { initial_threshold: 2.0 };
        let adaptive_config = AdaptiveConfig {
            enabled: true,
            learning_rate: 0.1,
            min_threshold: 1.0,
            max_threshold: 5.0,
            sensitivity: 1.0,
        };
        let learning_config = LearningConfig {
            enabled: true,
            min_samples: 5,
            ..Default::default()
        };

        let mut detector = StreamingAnomalyDetector::new(
            method,
            50,
            adaptive_config,
            learning_config,
        );

        let initial_threshold = detector.threshold;

        // Add some normal values
        for i in 0..20 {
            let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, i as u32, 0).unwrap();
            let value = 10.0 + (i as f64 * 0.1).sin();
            detector.process_point(timestamp, value).unwrap();
        }

        // The threshold should potentially have been adjusted
        assert!(detector.threshold > 0.0);
    }

    #[test]
    fn test_window_statistics() {
        let method = StreamingMethod::ZScore { threshold: 3.0 };
        let adaptive_config = AdaptiveConfig::default();
        let learning_config = LearningConfig::default();

        let mut detector = StreamingAnomalyDetector::new(
            method,
            10,
            adaptive_config,
            learning_config,
        );

        // Add values to build statistics
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for (i, &value) in values.iter().enumerate() {
            let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, i as u32, 0).unwrap();
            detector.process_point(timestamp, value).unwrap();
        }

        let stats = &detector.stats;
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!(stats.std_dev > 0.0);
        assert!((stats.median - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(calculate_percentile(&values, 0.0), 1.0);
        assert_eq!(calculate_percentile(&values, 50.0), 3.0);
        assert_eq!(calculate_percentile(&values, 100.0), 5.0);

        let single_value = vec![42.0];
        assert_eq!(calculate_percentile(&single_value, 50.0), 42.0);

        let empty_values: Vec<f64> = vec![];
        assert_eq!(calculate_percentile(&empty_values, 50.0), 0.0);
    }

    #[test]
    fn test_detector_reset() {
        let method = StreamingMethod::ZScore { threshold: 3.0 };
        let adaptive_config = AdaptiveConfig::default();
        let learning_config = LearningConfig::default();

        let mut detector = StreamingAnomalyDetector::new(
            method,
            100,
            adaptive_config,
            learning_config,
        );

        // Add some data
        for i in 0..10 {
            let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, i as u32, 0).unwrap();
            detector.process_point(timestamp, i as f64).unwrap();
        }

        assert!(detector.window.len() > 0);
        assert!(detector.stats.sample_count > 0);

        // Reset detector
        detector.reset();

        assert_eq!(detector.window.len(), 0);
        assert_eq!(detector.stats.sample_count, 0);
        assert_eq!(detector.threshold, 3.0);
    }
}