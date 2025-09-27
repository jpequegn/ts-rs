//! # Parallel Processing Module
//!
//! Provides multi-threaded analysis operations and parallel forecasting capabilities.

use std::sync::Arc;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::config::PerformanceConfig;
use crate::{Result, TimeSeries, TimeSeriesError};
use super::PerformanceError;

/// Parallel processing configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub num_threads: Option<usize>,
    pub chunk_size: usize,
    pub enable_simd: bool,
    pub task_queue_size: usize,
}

impl From<&PerformanceConfig> for ParallelConfig {
    fn from(config: &PerformanceConfig) -> Self {
        Self {
            num_threads: config.num_threads,
            chunk_size: config.chunk_size,
            enable_simd: true, // Enable SIMD optimizations where available
            task_queue_size: 1000,
        }
    }
}

/// Parallel processor for multi-threaded operations
#[derive(Debug)]
pub struct ParallelProcessor {
    config: ParallelConfig,
    thread_pool: rayon::ThreadPool,
}

impl ParallelProcessor {
    /// Create a new parallel processor
    pub fn new(config: &PerformanceConfig) -> Result<Self> {
        let parallel_config = ParallelConfig::from(config);

        let mut builder = ThreadPoolBuilder::new();

        if let Some(num_threads) = parallel_config.num_threads {
            builder = builder.num_threads(num_threads);
        }

        let thread_pool = builder
            .build()
            .map_err(|e| PerformanceError::ParallelProcessingError(e.to_string()))?;

        Ok(Self {
            config: parallel_config,
            thread_pool,
        })
    }

    /// Prepare time series for parallel processing
    pub fn prepare_timeseries(&self, _ts: &mut TimeSeries) -> Result<()> {
        // Prepare data structures for optimal parallel access
        // For now, this is a placeholder
        Ok(())
    }

    /// Process multiple time series in parallel
    pub fn process_multiple<T, R, F>(&self, data: Vec<T>, processor: F) -> Result<Vec<R>>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(T) -> Result<R> + Send + Sync,
    {
        let results: Result<Vec<_>> = self.thread_pool.install(|| {
            data.into_par_iter()
                .map(|item| processor(item))
                .collect()
        });

        results
    }

    /// Parallel map operation on time series values
    pub fn parallel_map<F>(&self, ts: &TimeSeries, mapper: F) -> Result<Vec<f64>>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        let results: Vec<f64> = self.thread_pool.install(|| {
            ts.values.par_iter()
                .map(|&value| mapper(value))
                .collect()
        });

        Ok(results)
    }

    /// Parallel reduce operation on time series
    pub fn parallel_reduce<F, R>(&self, ts: &TimeSeries, init: R, reducer: F) -> Result<R>
    where
        F: Fn(R, f64) -> R + Send + Sync,
        R: Send + Sync + Clone,
    {
        let values = ts.values;

        let result = self.thread_pool.install(|| {
            values.par_iter()
                .fold(|| init.clone(), |acc, &value| reducer(acc, value))
                .reduce(|| init.clone(), |a, b| reducer(a, 0.0)) // Simplified reduction
        });

        Ok(result)
    }

    /// Parallel windowed operation
    pub fn parallel_windowed<F, R>(&self, ts: &TimeSeries, window_size: usize, processor: F) -> Result<Vec<R>>
    where
        F: Fn(&[f64]) -> R + Send + Sync,
        R: Send,
    {
        let values = ts.values;

        if values.len() < window_size {
            return Err(TimeSeriesError::invalid_input("Window size larger than data").into());
        }

        let results: Vec<R> = self.thread_pool.install(|| {
            (0..=values.len() - window_size)
                .into_par_iter()
                .map(|i| {
                    let window = &values[i..i + window_size];
                    processor(window)
                })
                .collect()
        });

        Ok(results)
    }

    /// Parallel correlation calculation between multiple series
    pub fn parallel_correlations(&self, series: &[TimeSeries]) -> Result<Vec<Vec<f64>>> {
        if series.is_empty() {
            return Ok(vec![]);
        }

        let n = series.len();
        let mut correlation_matrix = vec![vec![0.0; n]; n];

        // Generate all (i,j) pairs for upper triangle
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i..n {
                pairs.push((i, j));
            }
        }

        // Calculate correlations in parallel
        let correlations: Vec<_> = self.thread_pool.install(|| {
            pairs.par_iter().map(|&(i, j)| {
                let corr = calculate_correlation(&series[i], &series[j]).unwrap_or(0.0);
                (i, j, corr)
            }).collect()
        });

        // Fill in the matrix with calculated correlations
        for (i, j, corr) in correlations {
            correlation_matrix[i][j] = corr;
            if i != j {
                correlation_matrix[j][i] = corr; // Symmetric matrix
            }
        }

        Ok(correlation_matrix)
    }

    /// Parallel forecasting for multiple series
    pub fn parallel_forecast<F, R>(&self, series: Vec<TimeSeries>, forecaster: F) -> Result<Vec<R>>
    where
        F: Fn(&TimeSeries) -> Result<R> + Send + Sync,
        R: Send,
    {
        let results: Result<Vec<_>> = self.thread_pool.install(|| {
            series.par_iter()
                .map(|ts| forecaster(ts))
                .collect()
        });

        results
    }

    /// Parallel statistical analysis
    pub fn parallel_statistics(&self, series: &[TimeSeries]) -> Result<Vec<SeriesStatistics>> {
        let results: Vec<SeriesStatistics> = self.thread_pool.install(|| {
            series.par_iter()
                .map(|ts| calculate_statistics(ts))
                .collect()
        });

        Ok(results)
    }

    /// Get thread pool information
    pub fn thread_info(&self) -> ThreadInfo {
        ThreadInfo {
            num_threads: self.thread_pool.current_num_threads(),
            queue_size: 0, // Not directly available from rayon
            active_tasks: 0, // Not directly available from rayon
        }
    }
}

/// Task processor for batch operations
#[derive(Debug)]
pub struct TaskProcessor {
    parallel_processor: Arc<ParallelProcessor>,
}

impl TaskProcessor {
    /// Create a new task processor
    pub fn new(parallel_processor: Arc<ParallelProcessor>) -> Self {
        Self {
            parallel_processor,
        }
    }

    /// Process batch of tasks in parallel
    pub fn process_batch<T, R, F>(&self, tasks: Vec<T>, processor: F) -> Result<Vec<R>>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(T) -> Result<R> + Send + Sync,
    {
        self.parallel_processor.process_multiple(tasks, processor)
    }

    /// Process tasks with different priorities
    pub fn process_prioritized<T, R, F>(&self, high_priority: Vec<T>, low_priority: Vec<T>, processor: F) -> Result<(Vec<R>, Vec<R>)>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(T) -> Result<R> + Send + Sync + Clone,
    {
        // Process high priority tasks first
        let high_results = self.parallel_processor.process_multiple(high_priority, processor.clone())?;
        let low_results = self.parallel_processor.process_multiple(low_priority, processor)?;

        Ok((high_results, low_results))
    }
}

/// Series statistics computed in parallel
#[derive(Debug, Clone)]
pub struct SeriesStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

/// Thread pool information
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    pub num_threads: usize,
    pub queue_size: usize,
    pub active_tasks: usize,
}

/// Calculate correlation between two time series
fn calculate_correlation(ts1: &TimeSeries, ts2: &TimeSeries) -> Result<f64> {
    let values1 = ts1.values;
    let values2 = ts2.values;

    if values1.len() != values2.len() {
        return Err(TimeSeriesError::invalid_input("Series must have same length").into());
    }

    if values1.is_empty() {
        return Ok(0.0);
    }

    let n = values1.len() as f64;
    let mean1: f64 = values1.iter().sum::<f64>() / n;
    let mean2: f64 = values2.iter().sum::<f64>() / n;

    let numerator: f64 = values1.iter().zip(values2.iter())
        .map(|(v1, v2)| (v1 - mean1) * (v2 - mean2))
        .sum();

    let sum_sq1: f64 = values1.iter().map(|v| (v - mean1).powi(2)).sum();
    let sum_sq2: f64 = values2.iter().map(|v| (v - mean2).powi(2)).sum();

    let denominator = (sum_sq1 * sum_sq2).sqrt();

    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

/// Calculate basic statistics for a time series
fn calculate_statistics(ts: &TimeSeries) -> SeriesStatistics {
    let values = ts.values;

    if values.is_empty() {
        return SeriesStatistics {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            count: 0,
        };
    }

    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    SeriesStatistics {
        mean,
        std_dev,
        min,
        max,
        count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PerformanceConfig;
    use chrono::{DateTime, Utc};

    fn create_test_timeseries(size: usize) -> TimeSeries {
        let timestamps: Vec<DateTime<Utc>> = (0..size)
            .map(|i| Utc::now() + chrono::Duration::seconds(i as i64))
            .collect();
        let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_parallel_processor_creation() {
        let config = PerformanceConfig::default();
        let processor = ParallelProcessor::new(&config).unwrap();
        let info = processor.thread_info();
        assert!(info.num_threads > 0);
    }

    #[test]
    fn test_parallel_map() {
        let config = PerformanceConfig::default();
        let processor = ParallelProcessor::new(&config).unwrap();
        let ts = create_test_timeseries(100);

        let results = processor.parallel_map(&ts, |x| x * 2.0).unwrap();
        assert_eq!(results.len(), 100);
        assert_eq!(results[0], 0.0);
        assert_eq!(results[99], 198.0);
    }

    #[test]
    fn test_parallel_statistics() {
        let config = PerformanceConfig::default();
        let processor = ParallelProcessor::new(&config).unwrap();
        let series = vec![
            create_test_timeseries(100),
            create_test_timeseries(50),
        ];

        let stats = processor.parallel_statistics(&series).unwrap();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].count, 100);
        assert_eq!(stats[1].count, 50);
    }

    #[test]
    fn test_correlation_calculation() {
        let ts1 = create_test_timeseries(10);
        let ts2 = create_test_timeseries(10);

        let correlation = calculate_correlation(&ts1, &ts2).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10); // Should be perfectly correlated
    }
}