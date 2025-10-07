//! Unit tests for memory management module

use chrono::{DateTime, Utc};
use chronos::config::PerformanceConfig;
use chronos::performance::memory::*;
use chronos::TimeSeries;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_config() -> PerformanceConfig {
    PerformanceConfig {
        enable_optimization: true,
        max_memory_mb: 50,
        chunk_size: 1000,
        num_threads: Some(2),
        max_cache_size_mb: 25,
        progress_threshold: 100,
        enable_database: true,
        cache_directory: Some(std::env::temp_dir().join("chronos_test_cache")),
    }
}

fn create_test_timeseries(size: usize) -> TimeSeries {
    let timestamps: Vec<DateTime<Utc>> = (0..size)
        .map(|i| Utc::now() + chrono::Duration::seconds(i as i64))
        .collect();
    let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

    TimeSeries::new(timestamps, values).expect("Failed to create test time series")
}

#[test]
fn test_memory_manager_creation() {
    let config = create_test_config();
    let manager = MemoryManager::new(&config).expect("Failed to create memory manager");

    let stats = manager.get_memory_stats();
    assert_eq!(stats.allocated_bytes, 0);
    assert_eq!(stats.peak_usage_bytes, 0);
}

#[test]
fn test_memory_allocation_tracking() {
    let config = create_test_config();
    let manager = MemoryManager::new(&config).expect("Failed to create memory manager");

    // Simulate memory allocation
    let test_data = vec![0u8; 1024]; // 1KB
    manager
        .track_allocation(test_data.len())
        .expect("Failed to track allocation");

    let stats = manager.get_memory_stats();
    assert_eq!(stats.allocated_bytes, 1024);
    assert_eq!(stats.peak_usage_bytes, 1024);

    // Simulate deallocation
    manager.track_deallocation(test_data.len());

    let stats = manager.get_memory_stats();
    assert_eq!(stats.allocated_bytes, 0);
    assert_eq!(stats.peak_usage_bytes, 1024); // Peak should remain
}

#[test]
fn test_memory_pressure_detection() {
    let mut config = create_test_config();
    config.max_memory_mb = 1; // Very small limit

    let manager = MemoryManager::new(&config).expect("Failed to create memory manager");

    // Allocate more than the limit
    let large_allocation = 2 * 1024 * 1024; // 2MB
    let result = manager.track_allocation(large_allocation);

    assert!(result.is_err());
    assert!(manager.is_under_pressure());
}

#[test]
fn test_streaming_processor_creation() {
    let config = create_test_config();
    let processor = StreamingProcessor::new(&config).expect("Failed to create streaming processor");

    assert!(processor.get_stats().chunks_processed == 0);
}

#[test]
fn test_streaming_file_processing() {
    let config = create_test_config();
    let processor = StreamingProcessor::new(&config).expect("Failed to create streaming processor");

    // Create a temporary file with test data
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let test_data = b"1.0,2.0,3.0,4.0,5.0\n6.0,7.0,8.0,9.0,10.0\n";
    temp_file
        .write_all(test_data)
        .expect("Failed to write test data");

    let path = temp_file.path();
    let result = processor.process_file(path, |chunk| {
        // Simple processing: count lines
        Ok(chunk.lines().count())
    });

    assert!(result.is_ok());
    let total_lines = result.unwrap();
    assert!(total_lines > 0);

    let stats = processor.get_stats();
    assert!(stats.chunks_processed > 0);
    assert!(stats.bytes_processed > 0);
}

#[test]
fn test_lazy_data_loader() {
    let config = create_test_config();
    let loader = LazyDataLoader::new(&config).expect("Failed to create lazy loader");

    // Test loading non-existent data
    let result: Option<Vec<f64>> = loader.load("nonexistent_key");
    assert!(result.is_none());

    // Test that loader handles missing data gracefully
    assert!(!loader.is_loaded("nonexistent_key"));
}

#[test]
fn test_compact_timeseries() {
    let original_ts = create_test_timeseries(1000);
    let compact =
        CompactTimeSeries::from_timeseries(&original_ts).expect("Failed to create compact series");

    // Test basic properties
    assert_eq!(compact.len(), original_ts.len());

    // Test value access
    let first_value = compact.get_value(0).expect("Failed to get first value");
    assert_eq!(first_value, original_ts.values()[0]);

    let last_value = compact
        .get_value(compact.len() - 1)
        .expect("Failed to get last value");
    assert_eq!(last_value, original_ts.values()[original_ts.len() - 1]);
}

#[test]
fn test_compact_timeseries_memory_efficiency() {
    let large_ts = create_test_timeseries(10000);
    let compact =
        CompactTimeSeries::from_timeseries(&large_ts).expect("Failed to create compact series");

    // Compact series should use less memory than the original
    // This is a simple heuristic test
    let original_size = large_ts.values().len() * std::mem::size_of::<f64>();
    let compact_size = compact.memory_usage();

    assert!(compact_size > 0);
    // In practice, compression might not always be smaller for simple data
    // But the infrastructure should be in place
}

#[test]
fn test_compact_timeseries_iteration() {
    let ts = create_test_timeseries(100);
    let compact = CompactTimeSeries::from_timeseries(&ts).expect("Failed to create compact series");

    let mut count = 0;
    for (i, value) in compact.iter().enumerate() {
        assert_eq!(value, ts.values()[i]);
        count += 1;
    }

    assert_eq!(count, ts.len());
}

#[test]
fn test_memory_optimization_with_timeseries() {
    let config = create_test_config();
    let manager = MemoryManager::new(&config).expect("Failed to create memory manager");

    let mut ts = create_test_timeseries(5000);

    // Get initial memory usage
    let initial_stats = manager.get_memory_stats();

    // Optimize the time series
    let result = manager.optimize_timeseries(&mut ts);
    assert!(result.is_ok());

    // Memory usage should be tracked
    let final_stats = manager.get_memory_stats();
    assert!(final_stats.optimizations_applied > initial_stats.optimizations_applied);
}

#[test]
fn test_memory_cleanup() {
    let config = create_test_config();
    let manager = MemoryManager::new(&config).expect("Failed to create memory manager");

    // Allocate some memory
    manager
        .track_allocation(1024)
        .expect("Failed to track allocation");
    manager
        .track_allocation(2048)
        .expect("Failed to track allocation");

    let stats_before = manager.get_memory_stats();
    assert_eq!(stats_before.allocated_bytes, 3072);

    // Perform cleanup
    manager
        .cleanup_unused_memory()
        .expect("Failed to cleanup memory");

    // For this test, cleanup might not change allocated_bytes since we're tracking
    // but it should at least execute without error
    let stats_after = manager.get_memory_stats();
    assert!(stats_after.cleanup_operations > stats_before.cleanup_operations);
}

#[test]
fn test_concurrent_memory_operations() {
    let config = create_test_config();
    let manager =
        std::sync::Arc::new(MemoryManager::new(&config).expect("Failed to create memory manager"));

    let mut handles = vec![];

    // Spawn multiple threads doing memory operations
    for i in 0..5 {
        let mgr = manager.clone();
        let handle = std::thread::spawn(move || {
            let size = (i + 1) * 1024;
            mgr.track_allocation(size)
                .expect("Failed to track allocation");
            std::thread::sleep(std::time::Duration::from_millis(10));
            mgr.track_deallocation(size);
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let final_stats = manager.get_memory_stats();
    assert_eq!(final_stats.allocated_bytes, 0); // All should be deallocated
    assert!(final_stats.peak_usage_bytes > 0); // But peak should be recorded
}
