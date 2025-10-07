//! Early Stopping Module
//!
//! Intelligent early stopping for efficient hyperparameter optimization:
//! - Adaptive patience based on optimization progress
//! - Learning curve prediction
//! - Resource-aware stopping criteria

use super::types::*;

// ================================================================================================
// Early Stopping Manager
// ================================================================================================

pub struct EarlyStoppingManager {
    config: EarlyStoppingConfig,
    performance_history: Vec<f64>,
    best_performance: f64,
    epochs_since_improvement: usize,
}

impl EarlyStoppingManager {
    pub fn new(config: EarlyStoppingConfig) -> Self {
        let best_performance = match config.mode {
            EarlyStoppingMode::Minimize => f64::INFINITY,
            EarlyStoppingMode::Maximize => f64::NEG_INFINITY,
        };

        Self {
            config,
            performance_history: Vec::new(),
            best_performance,
            epochs_since_improvement: 0,
        }
    }

    /// Update with new performance and check if should stop
    pub fn should_stop(&mut self, current_performance: f64, epoch: usize) -> bool {
        self.performance_history.push(current_performance);

        // Skip warmup period
        if epoch < self.config.warmup_epochs {
            return false;
        }

        // Check for improvement
        let is_improvement = match self.config.mode {
            EarlyStoppingMode::Minimize => {
                current_performance < self.best_performance - self.config.min_delta
            }
            EarlyStoppingMode::Maximize => {
                current_performance > self.best_performance + self.config.min_delta
            }
        };

        if is_improvement {
            self.best_performance = current_performance;
            self.epochs_since_improvement = 0;
            false
        } else {
            self.epochs_since_improvement += 1;
            self.epochs_since_improvement >= self.config.patience
        }
    }

    /// Get adaptive patience based on optimization progress
    pub fn adaptive_patience(&self) -> usize {
        if self.performance_history.len() < 10 {
            return self.config.patience;
        }

        // Compute improvement rate
        let recent_improvement = self.compute_improvement_rate(10);

        if recent_improvement > 0.01 {
            // Good progress, increase patience
            (self.config.patience as f64 * 1.5) as usize
        } else if recent_improvement < 0.001 {
            // Slow progress, decrease patience
            (self.config.patience as f64 * 0.7) as usize
        } else {
            self.config.patience
        }
    }

    /// Compute improvement rate over last n epochs
    fn compute_improvement_rate(&self, n: usize) -> f64 {
        if self.performance_history.len() < n {
            return 0.0;
        }

        let start_idx = self.performance_history.len() - n;
        let start_perf = self.performance_history[start_idx];
        let end_perf = self.performance_history[self.performance_history.len() - 1];

        (end_perf - start_perf).abs() / n as f64
    }

    /// Predict if training will improve
    pub fn predict_improvement(&self, future_epochs: usize) -> bool {
        if self.performance_history.len() < 5 {
            return true; // Not enough data
        }

        // Simple linear extrapolation
        let recent_slope = self.estimate_trend_slope(5);

        match self.config.mode {
            EarlyStoppingMode::Minimize => {
                let predicted = self.best_performance + recent_slope * future_epochs as f64;
                predicted < self.best_performance - self.config.min_delta
            }
            EarlyStoppingMode::Maximize => {
                let predicted = self.best_performance + recent_slope * future_epochs as f64;
                predicted > self.best_performance + self.config.min_delta
            }
        }
    }

    /// Estimate trend slope using linear regression
    fn estimate_trend_slope(&self, n: usize) -> f64 {
        if self.performance_history.len() < n {
            return 0.0;
        }

        let start_idx = self.performance_history.len() - n;
        let data: Vec<_> = self.performance_history[start_idx..].to_vec();

        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = data.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Reset early stopping state
    pub fn reset(&mut self) {
        self.performance_history.clear();
        self.best_performance = match self.config.mode {
            EarlyStoppingMode::Minimize => f64::INFINITY,
            EarlyStoppingMode::Maximize => f64::NEG_INFINITY,
        };
        self.epochs_since_improvement = 0;
    }

    /// Get best performance achieved
    pub fn best_performance(&self) -> f64 {
        self.best_performance
    }

    /// Get performance history
    pub fn history(&self) -> &[f64] {
        &self.performance_history
    }
}

// ================================================================================================
// Trial Pruning
// ================================================================================================

/// Prune unpromising trials early based on performance
pub struct TrialPruner {
    baseline_performance: f64,
    pruning_threshold: f64,
}

impl TrialPruner {
    pub fn new(baseline_performance: f64, pruning_threshold: f64) -> Self {
        Self {
            baseline_performance,
            pruning_threshold,
        }
    }

    /// Check if trial should be pruned
    pub fn should_prune(&self, current_performance: f64, epoch: usize, max_epochs: usize) -> bool {
        // Don't prune too early
        if epoch < max_epochs / 4 {
            return false;
        }

        // Prune if significantly worse than baseline
        let gap = (self.baseline_performance - current_performance).abs();
        gap > self.pruning_threshold
    }

    /// Update baseline with new best performance
    pub fn update_baseline(&mut self, performance: f64) {
        if performance > self.baseline_performance {
            self.baseline_performance = performance;
        }
    }
}

// ================================================================================================
// Public API
// ================================================================================================

/// Create early stopping manager with default config
pub fn create_early_stopping_manager(
    patience: usize,
    min_delta: f64,
    mode: EarlyStoppingMode,
) -> EarlyStoppingManager {
    EarlyStoppingManager::new(EarlyStoppingConfig {
        patience,
        min_delta,
        mode,
        warmup_epochs: 5,
    })
}

/// Check if training should stop based on performance history
pub fn check_early_stopping(
    performance_history: &[f64],
    patience: usize,
    min_delta: f64,
    mode: EarlyStoppingMode,
) -> bool {
    if performance_history.len() <= patience {
        return false;
    }

    let recent = &performance_history[performance_history.len() - patience..];
    let best_recent = match mode {
        EarlyStoppingMode::Minimize => recent.iter().cloned().fold(f64::INFINITY, f64::min),
        EarlyStoppingMode::Maximize => recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    };

    let current = performance_history[performance_history.len() - 1];

    match mode {
        EarlyStoppingMode::Minimize => current >= best_recent - min_delta,
        EarlyStoppingMode::Maximize => current <= best_recent + min_delta,
    }
}
