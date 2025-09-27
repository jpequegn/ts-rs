//! Integration tests for the performance optimization system

use chronos_time_series::*;
use chronos_time_series::performance::*;
use chronos_time_series::config::PerformanceConfig;
use std::time::Duration;
use tempfile::TempDir;
use chrono::{DateTime, Utc};

fn create_test_config() -> PerformanceConfig {
    PerformanceConfig {
        enable_optimization: true,
        max_memory_mb: 100,
        chunk_size: 1000,
        num_threads: Some(2),
        max_cache_size_mb: 50,
        progress_threshold: 100,
        enable_database: true,
        cache_directory: Some(std::env::temp_dir().join("chronos_test_cache")),
    }
}

fn create_test_timeseries(size: usize) -> TimeSeries {
    let timestamps: Vec<DateTime<Utc>> = (0..size)
        .map(|i| Utc::now() + chrono::Duration::seconds(i as i64))
        .collect();
    let values: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();

    TimeSeries::new(timestamps, values).expect("Failed to create test time series")
}

#[tokio::test]
async fn test_performance_optimizer_creation() {
    let config = create_test_config();
    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let metrics = optimizer.get_metrics();
    assert_eq!(metrics.memory_usage_mb, 0.0);
    assert_eq!(metrics.cache_hit_rate, 0.0);
}

#[tokio::test]
async fn test_memory_optimization() {
    let config = create_test_config();
    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let mut ts = create_test_timeseries(1000);
    optimizer.optimize_memory(&mut ts).expect("Memory optimization failed");

    let metrics = optimizer.get_metrics();
    assert!(metrics.memory_usage_mb > 0.0);
}

#[tokio::test]
async fn test_parallel_processing() {
    let config = create_test_config();
    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let ts = create_test_timeseries(1000);

    // Test parallel computation
    let result = optimizer.execute_parallel(&ts, |values| {
        values.iter().map(|&x| x * 2.0).collect::<Vec<f64>>()
    }).expect("Parallel execution failed");

    assert_eq!(result.len(), 1000);
    assert_eq!(result[0], ts.values()[0] * 2.0);
}

#[tokio::test]
async fn test_caching_system() {
    let config = create_test_config();
    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let ts = create_test_timeseries(500);
    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Test cache operations
    optimizer.cache_result("test_key", &test_data).expect("Cache set failed");

    let cached: Option<Vec<f64>> = optimizer.get_cached("test_key");
    assert_eq!(cached, Some(test_data));

    let metrics = optimizer.get_metrics();
    assert!(metrics.cache_hit_rate >= 0.0);
}

#[tokio::test]
async fn test_database_integration() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let ts = create_test_timeseries(100);

    // Test database operations
    optimizer.store_large_dataset("test_series", &ts).expect("Database store failed");

    let loaded = optimizer.load_large_dataset("test_series").expect("Database load failed");
    assert!(loaded.is_some());

    let loaded_ts = loaded.unwrap();
    assert_eq!(loaded_ts.len(), ts.len());
}

#[tokio::test]
async fn test_progress_tracking() {
    let config = create_test_config();
    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    // Test progress-aware operation
    let result = optimizer.execute_with_progress("test_operation", 100, |progress| {
        for i in 0..100 {
            progress.inc();
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(42)
    }).expect("Progress operation failed");

    assert_eq!(result, 42);
}

#[tokio::test]
async fn test_comprehensive_optimization() {
    let config = create_test_config();
    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let mut ts = create_test_timeseries(2000);

    // Run comprehensive optimization
    optimizer.optimize_comprehensive(&mut ts).expect("Comprehensive optimization failed");

    let metrics = optimizer.get_metrics();
    assert!(metrics.memory_usage_mb > 0.0);
    assert!(metrics.operations_per_second > 0.0);
}

#[tokio::test]
async fn test_performance_metrics() {
    let config = create_test_config();
    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let ts = create_test_timeseries(1000);

    // Perform some operations to generate metrics
    optimizer.execute_parallel(&ts, |values| {
        values.iter().sum::<f64>()
    }).expect("Parallel operation failed");

    let metrics = optimizer.get_metrics();
    assert!(metrics.total_operations > 0);
    assert!(metrics.average_operation_time.as_millis() >= 0);
}

#[tokio::test]
async fn test_memory_pressure_handling() {
    let mut config = create_test_config();
    config.max_memory_mb = 10; // Very low limit to trigger pressure handling

    let optimizer = PerformanceOptimizer::new(&config).expect("Failed to create optimizer");

    let mut large_ts = create_test_timeseries(10000);

    // This should trigger memory optimization
    let result = optimizer.optimize_memory(&mut large_ts);
    assert!(result.is_ok());

    let metrics = optimizer.get_metrics();
    assert!(metrics.memory_pressure_events > 0);
}

#[tokio::test]
async fn test_concurrent_operations() {
    let config = create_test_config();
    let optimizer = std::sync::Arc::new(PerformanceOptimizer::new(&config).expect("Failed to create optimizer"));

    let mut handles = vec![];

    // Spawn multiple concurrent operations
    for i in 0..5 {
        let opt = optimizer.clone();
        let handle = tokio::spawn(async move {
            let ts = create_test_timeseries(500);
            let key = format!("concurrent_test_{}", i);

            opt.cache_result(&key, &ts.values()).expect("Cache failed");
            let cached: Option<Vec<f64>> = opt.get_cached(&key);
            assert!(cached.is_some());
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.await.expect("Concurrent operation failed");
    }

    let metrics = optimizer.get_metrics();
    assert!(metrics.total_operations >= 5);
}