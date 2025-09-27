//! Unit tests for database integration module

use chronos_time_series::performance::database::*;
use chronos_time_series::config::PerformanceConfig;
use chronos_time_series::TimeSeries;
use chrono::{DateTime, Utc};
use tempfile::TempDir;

fn create_test_config() -> PerformanceConfig {
    PerformanceConfig {
        enable_optimization: true,
        max_memory_mb: 100,
        chunk_size: 1000,
        num_threads: Some(2),
        max_cache_size_mb: 50,
        progress_threshold: 100,
        enable_database: true,
        cache_directory: Some(std::env::temp_dir().join("chronos_db_test")),
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
async fn test_database_manager_creation() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    // Test basic connection
    let conn = db_manager.get_connection().await.expect("Failed to get connection");
    assert!(conn.is_some());
}

#[tokio::test]
async fn test_timeseries_storage_and_retrieval() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    let ts = create_test_timeseries(100);
    let series_id = "test_series_1";

    // Store the time series
    db_manager.store_timeseries(series_id, &ts).await.expect("Failed to store time series");

    // Retrieve the time series
    let retrieved = db_manager.load_timeseries(series_id).await.expect("Failed to load time series");

    assert!(retrieved.is_some());
    let retrieved_ts = retrieved.unwrap();
    assert_eq!(retrieved_ts.len(), ts.len());

    // Compare values (allowing for small floating point differences)
    for (i, (&original, &retrieved)) in ts.values().iter().zip(retrieved_ts.values().iter()).enumerate() {
        assert!((original - retrieved).abs() < 1e-10, "Mismatch at index {}: {} vs {}", i, original, retrieved);
    }
}

#[tokio::test]
async fn test_timeseries_deletion() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    let ts = create_test_timeseries(50);
    let series_id = "test_series_to_delete";

    // Store and verify
    db_manager.store_timeseries(series_id, &ts).await.expect("Failed to store time series");
    let retrieved = db_manager.load_timeseries(series_id).await.expect("Failed to load time series");
    assert!(retrieved.is_some());

    // Delete and verify
    db_manager.delete_timeseries(series_id).await.expect("Failed to delete time series");
    let retrieved = db_manager.load_timeseries(series_id).await.expect("Failed to load after deletion");
    assert!(retrieved.is_none());
}

#[tokio::test]
async fn test_analysis_result_caching() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    let cache_key = "test_analysis_result";
    let test_result = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Cache the result
    db_manager.cache_analysis_result(cache_key, &test_result).await.expect("Failed to cache analysis result");

    // Retrieve the result
    let retrieved: Option<Vec<f64>> = db_manager.get_cached_analysis_result(cache_key).await.expect("Failed to get cached result");

    assert_eq!(retrieved, Some(test_result));
}

#[tokio::test]
async fn test_analysis_result_cache_miss() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    // Try to get non-existent result
    let retrieved: Option<Vec<f64>> = db_manager.get_cached_analysis_result("nonexistent_key").await.expect("Failed to handle cache miss");

    assert_eq!(retrieved, None);
}

#[tokio::test]
async fn test_performance_metrics_tracking() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    let record = PerformanceRecord {
        operation: "test_operation".to_string(),
        duration_ms: 150,
        memory_usage_mb: 25.5,
        success: true,
        error_message: None,
    };

    // Record performance metric
    db_manager.record_performance(&record).await.expect("Failed to record performance");

    // Query performance metrics
    let metrics = db_manager.get_performance_metrics("test_operation", 10).await.expect("Failed to get performance metrics");

    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].operation, "test_operation");
    assert_eq!(metrics[0].duration_ms, 150);
    assert_eq!(metrics[0].memory_usage_mb, 25.5);
    assert!(metrics[0].success);
}

#[tokio::test]
async fn test_query_optimization() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    // Store multiple time series
    for i in 0..10 {
        let ts = create_test_timeseries(100);
        let series_id = format!("benchmark_series_{}", i);
        db_manager.store_timeseries(&series_id, &ts).await.expect("Failed to store time series");
    }

    // Test batch loading (this tests query optimization)
    let series_ids: Vec<String> = (0..10).map(|i| format!("benchmark_series_{}", i)).collect();

    // Load all series (this should benefit from query optimization)
    let start_time = std::time::Instant::now();
    for series_id in &series_ids {
        let _retrieved = db_manager.load_timeseries(series_id).await.expect("Failed to load time series");
    }
    let duration = start_time.elapsed();

    // Just verify the operation completed (actual optimization testing would require more sophisticated benchmarking)
    assert!(duration.as_millis() < 5000); // Should complete reasonably quickly
}

#[tokio::test]
async fn test_database_connection_pooling() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    // Get multiple connections concurrently
    let mut handles = vec![];

    for i in 0..5 {
        let db_clone = db_manager.clone();
        let handle = tokio::spawn(async move {
            let conn = db_clone.get_connection().await.expect("Failed to get connection");
            assert!(conn.is_some());

            // Store a small time series to test the connection
            let ts = create_test_timeseries(10);
            let series_id = format!("concurrent_test_{}", i);
            db_clone.store_timeseries(&series_id, &ts).await.expect("Failed to store time series");
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.await.expect("Concurrent operation failed");
    }
}

#[tokio::test]
async fn test_large_timeseries_storage() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    // Create a large time series
    let large_ts = create_test_timeseries(10000);
    let series_id = "large_test_series";

    // Store and retrieve
    db_manager.store_timeseries(series_id, &large_ts).await.expect("Failed to store large time series");
    let retrieved = db_manager.load_timeseries(series_id).await.expect("Failed to load large time series");

    assert!(retrieved.is_some());
    let retrieved_ts = retrieved.unwrap();
    assert_eq!(retrieved_ts.len(), large_ts.len());

    // Verify a few sample values
    assert_eq!(retrieved_ts.values()[0], large_ts.values()[0]);
    assert_eq!(retrieved_ts.values()[5000], large_ts.values()[5000]);
    assert_eq!(retrieved_ts.values()[9999], large_ts.values()[9999]);
}

#[tokio::test]
async fn test_database_error_handling() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    // Test handling of invalid series ID
    let result = db_manager.load_timeseries("").await;
    // Should handle gracefully and return None rather than error
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), None);
}

#[tokio::test]
async fn test_database_cache_statistics() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    // Record some performance metrics
    for i in 0..5 {
        let record = PerformanceRecord {
            operation: "cache_stat_test".to_string(),
            duration_ms: 100 + i * 10,
            memory_usage_mb: 10.0 + i as f64,
            success: i % 2 == 0, // Mix of success and failure
            error_message: if i % 2 == 0 { None } else { Some("Test error".to_string()) },
        };
        db_manager.record_performance(&record).await.expect("Failed to record performance");
    }

    // Get metrics for analysis
    let metrics = db_manager.get_performance_metrics("cache_stat_test", 10).await.expect("Failed to get metrics");
    assert_eq!(metrics.len(), 5);

    // Verify we have both successes and failures
    let successes = metrics.iter().filter(|m| m.success).count();
    let failures = metrics.iter().filter(|m| !m.success).count();
    assert!(successes > 0);
    assert!(failures > 0);
}

#[tokio::test]
async fn test_timeseries_db_integration() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let timeseries_db = TimeSeriesDb::new(&config).await.expect("Failed to create TimeSeriesDb");

    let ts = create_test_timeseries(200);
    let table_name = "integration_test";

    // Store time series
    timeseries_db.store(&table_name, &ts).await.expect("Failed to store to TimeSeriesDb");

    // Load time series
    let loaded = timeseries_db.load(&table_name).await.expect("Failed to load from TimeSeriesDb");

    assert!(loaded.is_some());
    let loaded_ts = loaded.unwrap();
    assert_eq!(loaded_ts.len(), ts.len());
}

#[tokio::test]
async fn test_concurrent_database_operations() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let db_manager = DatabaseManager::new(&config).await.expect("Failed to create database manager");

    let mut handles = vec![];

    // Spawn multiple concurrent database operations
    for i in 0..10 {
        let db_clone = db_manager.clone();
        let handle = tokio::spawn(async move {
            let ts = create_test_timeseries(50);
            let series_id = format!("concurrent_db_test_{}", i);

            // Store
            db_clone.store_timeseries(&series_id, &ts).await.expect("Failed to store");

            // Load
            let loaded = db_clone.load_timeseries(&series_id).await.expect("Failed to load");
            assert!(loaded.is_some());

            // Cache analysis result
            let cache_key = format!("analysis_{}", i);
            let result = vec![i as f64; 10];
            db_clone.cache_analysis_result(&cache_key, &result).await.expect("Failed to cache");

            // Get cached result
            let cached: Option<Vec<f64>> = db_clone.get_cached_analysis_result(&cache_key).await.expect("Failed to get cached");
            assert_eq!(cached, Some(result));
        });
        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        handle.await.expect("Concurrent database operation failed");
    }
}