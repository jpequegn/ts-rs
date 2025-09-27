//! # Progress Tracking Module
//!
//! Provides progress bars, ETA estimation, and cancellation support for long operations.

use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use std::thread;
use indicatif::{ProgressBar as IndicatifBar, ProgressStyle, MultiProgress};
use crate::config::PerformanceConfig;
use crate::Result;
use super::PerformanceError;

/// Progress tracking configuration
#[derive(Debug, Clone)]
pub struct ProgressConfig {
    pub enable_progress_bars: bool,
    pub enable_eta: bool,
    pub enable_cancellation: bool,
    pub update_interval_ms: u64,
    pub progress_threshold: usize,
}

impl From<&PerformanceConfig> for ProgressConfig {
    fn from(config: &PerformanceConfig) -> Self {
        Self {
            enable_progress_bars: true,
            enable_eta: true,
            enable_cancellation: true,
            update_interval_ms: 100,
            progress_threshold: config.progress_threshold,
        }
    }
}

/// Progress tracker for long-running operations
#[derive(Debug)]
pub struct ProgressTracker {
    config: ProgressConfig,
    multi_progress: Arc<MultiProgress>,
    active_operations: Arc<Mutex<Vec<OperationHandle>>>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(config: &PerformanceConfig) -> Result<Self> {
        let progress_config = ProgressConfig::from(config);

        Ok(Self {
            config: progress_config,
            multi_progress: Arc::new(MultiProgress::new()),
            active_operations: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Create a new progress bar for an operation
    pub fn create_progress_bar(&self, operation_name: &str, total_items: u64) -> ProgressBar {
        if !self.config.enable_progress_bars || total_items < self.config.progress_threshold as u64 {
            return ProgressBar::new_silent(operation_name, total_items);
        }

        let pb = self.multi_progress.add(IndicatifBar::new(total_items));

        let style = if self.config.enable_eta {
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg} (ETA: {eta})")
                .unwrap()
                .progress_chars("##-")
        } else {
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("##-")
        };

        pb.set_style(style);
        pb.set_message(operation_name.to_string());

        let progress_bar = ProgressBar::new_active(
            operation_name,
            total_items,
            pb,
            self.config.enable_cancellation,
        );

        // Track active operation
        {
            let mut operations = self.active_operations.lock().unwrap();
            operations.push(OperationHandle {
                name: operation_name.to_string(),
                start_time: Instant::now(),
                progress_bar: progress_bar.clone(),
            });
        }

        progress_bar
    }

    /// Create a spinner for indeterminate progress
    pub fn create_spinner(&self, operation_name: &str) -> ProgressSpinner {
        if !self.config.enable_progress_bars {
            return ProgressSpinner::new_silent(operation_name);
        }

        let pb = self.multi_progress.add(IndicatifBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        pb.set_message(operation_name.to_string());

        ProgressSpinner::new_active(operation_name, pb)
    }

    /// Get all active operations
    pub fn get_active_operations(&self) -> Vec<OperationStatus> {
        let operations = self.active_operations.lock().unwrap();
        operations.iter().map(|op| {
            let elapsed = op.start_time.elapsed();
            let progress = op.progress_bar.get_progress();

            OperationStatus {
                name: op.name.clone(),
                progress_percentage: progress.percentage,
                elapsed_time: elapsed,
                estimated_remaining: progress.eta,
                is_cancellable: op.progress_bar.is_cancellable(),
                is_cancelled: op.progress_bar.is_cancelled(),
            }
        }).collect()
    }

    /// Cancel an operation by name
    pub fn cancel_operation(&self, operation_name: &str) -> bool {
        let operations = self.active_operations.lock().unwrap();
        for op in operations.iter() {
            if op.name == operation_name {
                op.progress_bar.cancel();
                return true;
            }
        }
        false
    }

    /// Cancel all active operations
    pub fn cancel_all(&self) {
        let operations = self.active_operations.lock().unwrap();
        for op in operations.iter() {
            op.progress_bar.cancel();
        }
    }

    /// Clean up completed operations
    pub fn cleanup_completed(&self) {
        let mut operations = self.active_operations.lock().unwrap();
        operations.retain(|op| !op.progress_bar.is_finished());
    }
}

/// Progress bar for deterministic operations
#[derive(Debug, Clone)]
pub struct ProgressBar {
    name: String,
    total: u64,
    current: Arc<AtomicU64>,
    start_time: Instant,
    cancelled: Arc<AtomicBool>,
    cancellable: bool,
    pb: Option<IndicatifBar>,
}

impl ProgressBar {
    /// Create a new active progress bar
    fn new_active(name: &str, total: u64, pb: IndicatifBar, cancellable: bool) -> Self {
        Self {
            name: name.to_string(),
            total,
            current: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            cancelled: Arc::new(AtomicBool::new(false)),
            cancellable,
            pb: Some(pb),
        }
    }

    /// Create a silent progress bar (no UI)
    fn new_silent(name: &str, total: u64) -> Self {
        Self {
            name: name.to_string(),
            total,
            current: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            cancelled: Arc::new(AtomicBool::new(false)),
            cancellable: false,
            pb: None,
        }
    }

    /// Increment progress by 1
    pub fn inc(&self) {
        self.add(1);
    }

    /// Add to progress
    pub fn add(&self, delta: u64) {
        let new_value = self.current.fetch_add(delta, Ordering::Relaxed) + delta;

        if let Some(ref pb) = self.pb {
            pb.set_position(new_value);
        }
    }

    /// Set absolute progress position
    pub fn set_position(&self, position: u64) {
        self.current.store(position, Ordering::Relaxed);

        if let Some(ref pb) = self.pb {
            pb.set_position(position);
        }
    }

    /// Set progress message
    pub fn set_message(&self, message: &str) {
        if let Some(ref pb) = self.pb {
            pb.set_message(message.to_string());
        }
    }

    /// Check if operation is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Check if operation is cancellable
    pub fn is_cancellable(&self) -> bool {
        self.cancellable
    }

    /// Cancel the operation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        if let Some(ref pb) = self.pb {
            pb.abandon_with_message("Cancelled");
        }
    }

    /// Finish the progress bar
    pub fn finish(&self) {
        if let Some(ref pb) = self.pb {
            pb.finish_with_message("Completed");
        }
    }

    /// Finish with custom message
    pub fn finish_with_message(&self, message: &str) {
        if let Some(ref pb) = self.pb {
            pb.finish_with_message(message.to_string());
        }
    }

    /// Check if operation is finished
    pub fn is_finished(&self) -> bool {
        if let Some(ref pb) = self.pb {
            pb.is_finished()
        } else {
            self.current.load(Ordering::Relaxed) >= self.total
        }
    }

    /// Get current progress information
    pub fn get_progress(&self) -> ProgressInfo {
        let current = self.current.load(Ordering::Relaxed);
        let percentage = if self.total > 0 {
            (current as f64 / self.total as f64) * 100.0
        } else {
            0.0
        };

        let elapsed = self.start_time.elapsed();
        let eta = if current > 0 && current < self.total {
            let rate = current as f64 / elapsed.as_secs_f64();
            let remaining_items = self.total - current;
            Some(Duration::from_secs_f64(remaining_items as f64 / rate))
        } else {
            None
        };

        ProgressInfo {
            current,
            total: self.total,
            percentage,
            elapsed,
            eta,
        }
    }
}

/// Progress spinner for indeterminate operations
#[derive(Debug)]
pub struct ProgressSpinner {
    name: String,
    start_time: Instant,
    pb: Option<IndicatifBar>,
}

impl ProgressSpinner {
    fn new_active(name: &str, pb: IndicatifBar) -> Self {
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
            pb: Some(pb),
        }
    }

    fn new_silent(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
            pb: None,
        }
    }

    /// Set spinner message
    pub fn set_message(&self, message: &str) {
        if let Some(ref pb) = self.pb {
            pb.set_message(message.to_string());
        }
    }

    /// Finish the spinner
    pub fn finish(&self) {
        if let Some(ref pb) = self.pb {
            pb.finish_with_message("Completed");
        }
    }

    /// Finish with custom message
    pub fn finish_with_message(&self, message: &str) {
        if let Some(ref pb) = self.pb {
            pb.finish_with_message(message.to_string());
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Operation handle for tracking
#[derive(Debug)]
struct OperationHandle {
    name: String,
    start_time: Instant,
    progress_bar: ProgressBar,
}

/// Current operation status
#[derive(Debug, Clone)]
pub struct OperationStatus {
    pub name: String,
    pub progress_percentage: f64,
    pub elapsed_time: Duration,
    pub estimated_remaining: Option<Duration>,
    pub is_cancellable: bool,
    pub is_cancelled: bool,
}

/// Progress information
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub current: u64,
    pub total: u64,
    pub percentage: f64,
    pub elapsed: Duration,
    pub eta: Option<Duration>,
}

/// Progress-aware operation trait
pub trait ProgressAware {
    /// Execute operation with progress tracking
    fn execute_with_progress<F, R>(&self, operation_name: &str, total_items: u64, operation: F) -> Result<R>
    where
        F: FnOnce(&ProgressBar) -> Result<R>;

    /// Execute operation with spinner
    fn execute_with_spinner<F, R>(&self, operation_name: &str, operation: F) -> Result<R>
    where
        F: FnOnce(&ProgressSpinner) -> Result<R>;
}

impl ProgressAware for ProgressTracker {
    fn execute_with_progress<F, R>(&self, operation_name: &str, total_items: u64, operation: F) -> Result<R>
    where
        F: FnOnce(&ProgressBar) -> Result<R>
    {
        let progress_bar = self.create_progress_bar(operation_name, total_items);

        let result = operation(&progress_bar);

        match result {
            Ok(r) => {
                progress_bar.finish();
                Ok(r)
            }
            Err(e) => {
                progress_bar.finish_with_message("Failed");
                Err(e)
            }
        }
    }

    fn execute_with_spinner<F, R>(&self, operation_name: &str, operation: F) -> Result<R>
    where
        F: FnOnce(&ProgressSpinner) -> Result<R>
    {
        let spinner = self.create_spinner(operation_name);

        let result = operation(&spinner);

        match result {
            Ok(r) => {
                spinner.finish();
                Ok(r)
            }
            Err(e) => {
                spinner.finish_with_message("Failed");
                Err(e)
            }
        }
    }
}

/// Helper for progress-aware loops
pub struct ProgressLoop {
    progress_bar: ProgressBar,
    batch_size: usize,
    current_batch: usize,
}

impl ProgressLoop {
    /// Create a new progress loop
    pub fn new(progress_bar: ProgressBar, batch_size: usize) -> Self {
        Self {
            progress_bar,
            batch_size,
            current_batch: 0,
        }
    }

    /// Update progress for current iteration
    pub fn update(&mut self) -> Result<()> {
        if self.progress_bar.is_cancelled() {
            return Err(PerformanceError::ProgressError("Operation cancelled".to_string()).into());
        }

        self.current_batch += 1;

        if self.current_batch >= self.batch_size {
            self.progress_bar.add(self.batch_size as u64);
            self.current_batch = 0;
        }

        Ok(())
    }

    /// Check if operation should continue
    pub fn should_continue(&self) -> bool {
        !self.progress_bar.is_cancelled()
    }

    /// Finish the loop
    pub fn finish(self) {
        if self.current_batch > 0 {
            self.progress_bar.add(self.current_batch as u64);
        }
        self.progress_bar.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PerformanceConfig;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_progress_tracker_creation() {
        let config = PerformanceConfig::default();
        let tracker = ProgressTracker::new(&config).unwrap();

        assert_eq!(tracker.get_active_operations().len(), 0);
    }

    #[test]
    fn test_progress_bar_silent() {
        let pb = ProgressBar::new_silent("test", 100);
        assert!(!pb.is_cancellable());
        assert!(!pb.is_cancelled());

        pb.inc();
        let progress = pb.get_progress();
        assert_eq!(progress.current, 1);
        assert_eq!(progress.percentage, 1.0);
    }

    #[test]
    fn test_progress_bar_operations() {
        let config = PerformanceConfig::default();
        let tracker = ProgressTracker::new(&config).unwrap();

        let pb = tracker.create_progress_bar("test_operation", 100);

        pb.add(50);
        let progress = pb.get_progress();
        assert_eq!(progress.current, 50);
        assert_eq!(progress.percentage, 50.0);

        pb.set_message("Half done");
        pb.finish();
        assert!(pb.is_finished());
    }

    #[test]
    fn test_cancellation() {
        let config = PerformanceConfig::default();
        let tracker = ProgressTracker::new(&config).unwrap();

        let pb = tracker.create_progress_bar("cancellable_op", 100);
        assert!(pb.is_cancellable());
        assert!(!pb.is_cancelled());

        pb.cancel();
        assert!(pb.is_cancelled());
    }

    #[test]
    fn test_progress_aware_execution() {
        let config = PerformanceConfig::default();
        let tracker = ProgressTracker::new(&config).unwrap();

        let result = tracker.execute_with_progress("test_op", 10, |pb| {
            for i in 0..10 {
                pb.inc();
                thread::sleep(Duration::from_millis(10));
            }
            Ok(42)
        }).unwrap();

        assert_eq!(result, 42);
    }
}