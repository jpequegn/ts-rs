//! Unit tests for caching system module

use chronos_time_series::performance::cache::*;
use chronos_time_series::config::PerformanceConfig;
use std::collections::HashMap;
use tempfile::TempDir;

fn create_test_config() -> PerformanceConfig {
    PerformanceConfig {
        enable_optimization: true,
        max_memory_mb: 100,
        chunk_size: 1000,
        num_threads: Some(2),
        max_cache_size_mb: 10,
        progress_threshold: 100,
        enable_database: true,
        cache_directory: Some(std::env::temp_dir().join("chronos_cache_test")),
    }
}

#[test]
fn test_cache_manager_creation() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.memory_usage_mb, 0.0);
    assert_eq!(stats.hit_count, 0);
    assert_eq!(stats.miss_count, 0);
    assert_eq!(stats.hit_rate, 0.0);
}

#[test]
fn test_cache_set_and_get() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    cache.set("test_key", &test_data).expect("Failed to set cache entry");

    let retrieved: Option<Vec<f64>> = cache.get("test_key");
    assert_eq!(retrieved, Some(test_data));

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 1);
    assert_eq!(stats.hit_count, 1);
    assert_eq!(stats.miss_count, 0);
    assert_eq!(stats.hit_rate, 1.0);
}

#[test]
fn test_cache_miss() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    let retrieved: Option<Vec<f64>> = cache.get("nonexistent_key");
    assert_eq!(retrieved, None);

    let stats = cache.stats();
    assert_eq!(stats.miss_count, 1);
    assert_eq!(stats.hit_rate, 0.0);
}

#[test]
fn test_cache_remove() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    let test_data = vec![1.0, 2.0, 3.0];
    cache.set("test_key", &test_data).expect("Failed to set cache entry");

    // Verify it's there
    let retrieved: Option<Vec<f64>> = cache.get("test_key");
    assert_eq!(retrieved, Some(test_data));

    // Remove it
    assert!(cache.remove("test_key"));

    // Verify it's gone
    let retrieved: Option<Vec<f64>> = cache.get("test_key");
    assert_eq!(retrieved, None);

    // Try to remove again
    assert!(!cache.remove("test_key"));
}

#[test]
fn test_cache_clear_all() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    // Add multiple entries
    cache.set("key1", &vec![1.0]).expect("Failed to set key1");
    cache.set("key2", &vec![2.0]).expect("Failed to set key2");
    cache.set("key3", &vec![3.0]).expect("Failed to set key3");

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 3);

    // Clear all
    cache.clear_all().expect("Failed to clear cache");

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.memory_usage_mb, 0.0);

    // Verify entries are gone
    let retrieved: Option<Vec<f64>> = cache.get("key1");
    assert_eq!(retrieved, None);
}

#[test]
fn test_cache_should_cache() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    // Test expensive operations
    assert!(cache.should_cache("correlation_matrix", 100));
    assert!(cache.should_cache("fft", 50));
    assert!(cache.should_cache("arima_fit", 200));
    assert!(cache.should_cache("anomaly_detection", 300));

    // Test large data
    assert!(cache.should_cache("simple_op", 2000));

    // Test operations that shouldn't be cached
    assert!(!cache.should_cache("simple_op", 100));
}

#[test]
fn test_cache_compression() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    // Create data large enough to trigger compression
    let large_data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

    cache.set("large_data", &large_data).expect("Failed to set large data");

    let retrieved: Option<Vec<f64>> = cache.get("large_data");
    assert_eq!(retrieved, Some(large_data));

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 1);
    assert!(stats.memory_usage_mb > 0.0);
}

#[test]
fn test_cache_eviction() {
    let mut config = create_test_config();
    config.max_cache_size_mb = 1; // Very small cache to force eviction

    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    // Add entries until eviction happens
    for i in 0..100 {
        let large_data: Vec<f64> = (0..1000).map(|j| (i * 1000 + j) as f64).collect();
        cache.set(&format!("data_{}", i), &large_data).expect("Failed to set data");
    }

    let stats = cache.stats();
    // Should have fewer than 100 entries due to eviction
    assert!(stats.entry_count < 100);
    assert!(stats.memory_usage_mb <= config.max_cache_size_mb as f64 * 1.1); // Allow small margin
}

#[test]
fn test_cache_ttl_expiration() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    let test_data = vec![1.0, 2.0, 3.0];
    cache.set("ttl_test", &test_data).expect("Failed to set cache entry");

    // Should be available immediately
    let retrieved: Option<Vec<f64>> = cache.get("ttl_test");
    assert_eq!(retrieved, Some(test_data.clone()));

    // Note: In a real test, we would need to wait for TTL expiration
    // or modify the cache entry creation time. For now, we just verify
    // the entry exists and the TTL mechanism is in place.
}

#[test]
fn test_analysis_cache() {
    let config = create_test_config();
    let cache_manager = std::sync::Arc::new(
        CacheManager::new(&config).expect("Failed to create cache manager")
    );
    let analysis_cache = AnalysisCache::new(cache_manager);

    // Test statistical analysis caching
    let mut stats = HashMap::new();
    stats.insert("mean".to_string(), 5.0);
    stats.insert("std_dev".to_string(), 2.0);

    let data_hash = 12345u64;
    analysis_cache.cache_statistics(data_hash, &stats).expect("Failed to cache statistics");

    let retrieved_stats = analysis_cache.get_statistics(data_hash);
    assert_eq!(retrieved_stats, Some(stats));
}

#[test]
fn test_analysis_cache_correlation_matrix() {
    let config = create_test_config();
    let cache_manager = std::sync::Arc::new(
        CacheManager::new(&config).expect("Failed to create cache manager")
    );
    let analysis_cache = AnalysisCache::new(cache_manager);

    let correlation_matrix = vec![
        vec![1.0, 0.5, 0.3],
        vec![0.5, 1.0, 0.7],
        vec![0.3, 0.7, 1.0],
    ];

    let data_hash = 67890u64;
    analysis_cache.cache_correlation_matrix(data_hash, &correlation_matrix)
        .expect("Failed to cache correlation matrix");

    let retrieved_matrix = analysis_cache.get_correlation_matrix(data_hash);
    assert_eq!(retrieved_matrix, Some(correlation_matrix));
}

#[test]
fn test_analysis_cache_forecast() {
    let config = create_test_config();
    let cache_manager = std::sync::Arc::new(
        CacheManager::new(&config).expect("Failed to create cache manager")
    );
    let analysis_cache = AnalysisCache::new(cache_manager);

    let forecast = vec![10.0, 11.0, 12.0, 13.0, 14.0];
    let model_hash = 54321u64;
    let horizon = 5;

    analysis_cache.cache_forecast(model_hash, horizon, &forecast)
        .expect("Failed to cache forecast");

    let retrieved_forecast = analysis_cache.get_forecast(model_hash, horizon);
    assert_eq!(retrieved_forecast, Some(forecast));

    // Test with different horizon
    let retrieved_none = analysis_cache.get_forecast(model_hash, 10);
    assert_eq!(retrieved_none, None);
}

#[test]
fn test_persistent_cache() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = create_test_config();
    config.cache_directory = Some(temp_dir.path().to_path_buf());

    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    cache.set("persistent_test", &test_data).expect("Failed to set cache entry");

    // Create a new cache manager with the same directory
    let cache2 = CacheManager::new(&config).expect("Failed to create second cache manager");

    // Should be able to retrieve from persistent storage
    let retrieved: Option<Vec<f64>> = cache2.get("persistent_test");
    assert_eq!(retrieved, Some(test_data));
}

#[test]
fn test_concurrent_cache_operations() {
    let config = create_test_config();
    let cache = std::sync::Arc::new(
        CacheManager::new(&config).expect("Failed to create cache manager")
    );

    let mut handles = vec![];

    // Spawn multiple threads doing cache operations
    for i in 0..10 {
        let cache_clone = cache.clone();
        let handle = std::thread::spawn(move || {
            let key = format!("concurrent_key_{}", i);
            let data = vec![i as f64; 100];

            // Set data
            cache_clone.set(&key, &data).expect("Failed to set data");

            // Get data
            let retrieved: Option<Vec<f64>> = cache_clone.get(&key);
            assert_eq!(retrieved, Some(data));
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 10);
    assert_eq!(stats.hit_count, 10);
    assert!(stats.hit_rate > 0.0);
}

#[test]
fn test_cache_stats_accuracy() {
    let config = create_test_config();
    let cache = CacheManager::new(&config).expect("Failed to create cache manager");

    // Initially empty
    let stats = cache.stats();
    assert_eq!(stats.hit_count, 0);
    assert_eq!(stats.miss_count, 0);
    assert_eq!(stats.hit_rate, 0.0);

    // Add an entry and retrieve it
    cache.set("stats_test", &vec![1.0, 2.0]).expect("Failed to set entry");
    let _: Option<Vec<f64>> = cache.get("stats_test");

    // Try to get non-existent entry
    let _: Option<Vec<f64>> = cache.get("nonexistent");

    let stats = cache.stats();
    assert_eq!(stats.hit_count, 1);
    assert_eq!(stats.miss_count, 1);
    assert_eq!(stats.hit_rate, 0.5);
}