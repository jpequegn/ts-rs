//! Unit tests for parallel processing module

use chronos_time_series::performance::parallel::*;
use chronos_time_series::config::PerformanceConfig;
use chronos_time_series::TimeSeries;
use chrono::{DateTime, Utc};

fn create_test_config() -> PerformanceConfig {
    PerformanceConfig {
        enable_optimization: true,
        max_memory_mb: 100,
        chunk_size: 100,
        num_threads: Some(4),
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
    let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

    TimeSeries::new(timestamps, values).expect("Failed to create test time series")
}

#[test]
fn test_parallel_processor_creation() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let info = processor.thread_info();
    assert!(info.num_threads > 0);
    assert!(info.num_threads <= 4); // Should respect our config
}

#[test]
fn test_parallel_map_operation() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let ts = create_test_timeseries(1000);

    // Test parallel map: square all values
    let results = processor.parallel_map(&ts, |x| x * x).expect("Parallel map failed");

    assert_eq!(results.len(), ts.len());
    for (i, &result) in results.iter().enumerate() {
        let expected = ts.values()[i] * ts.values()[i];
        assert!((result - expected).abs() < f64::EPSILON);
    }
}

#[test]
fn test_parallel_reduce_operation() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let ts = create_test_timeseries(1000);

    // Test parallel reduce: sum all values
    let sum = processor.parallel_reduce(&ts, 0.0, |acc, x| acc + x).expect("Parallel reduce failed");

    let expected_sum: f64 = ts.values().iter().sum();
    assert!((sum - expected_sum).abs() < 1e-10);
}

#[test]
fn test_parallel_windowed_operation() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let ts = create_test_timeseries(1000);
    let window_size = 10;

    // Test parallel windowed operation: calculate window means
    let window_means = processor.parallel_windowed(&ts, window_size, |window| {
        window.iter().sum::<f64>() / window.len() as f64
    }).expect("Parallel windowed operation failed");

    let expected_windows = ts.len() - window_size + 1;
    assert_eq!(window_means.len(), expected_windows);

    // Verify first window mean
    let first_window_sum: f64 = ts.values()[0..window_size].iter().sum();
    let first_window_mean = first_window_sum / window_size as f64;
    assert!((window_means[0] - first_window_mean).abs() < f64::EPSILON);
}

#[test]
fn test_parallel_windowed_invalid_window() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let ts = create_test_timeseries(100);
    let window_size = 200; // Larger than data

    // Should return error for invalid window size
    let result = processor.parallel_windowed(&ts, window_size, |window| window.len());
    assert!(result.is_err());
}

#[test]
fn test_parallel_correlations() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let ts1 = create_test_timeseries(100);
    let ts2 = create_test_timeseries(100);
    let ts3 = create_test_timeseries(100);

    let series = vec![ts1, ts2, ts3];

    let correlation_matrix = processor.parallel_correlations(&series).expect("Parallel correlations failed");

    assert_eq!(correlation_matrix.len(), 3);
    assert_eq!(correlation_matrix[0].len(), 3);

    // Diagonal should be 1.0 (perfect self-correlation)
    for i in 0..3 {
        assert!((correlation_matrix[i][i] - 1.0).abs() < 1e-10);
    }

    // Matrix should be symmetric
    for i in 0..3 {
        for j in 0..3 {
            assert!((correlation_matrix[i][j] - correlation_matrix[j][i]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_parallel_correlations_empty() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let series = vec![];
    let correlation_matrix = processor.parallel_correlations(&series).expect("Empty correlations should work");

    assert_eq!(correlation_matrix.len(), 0);
}

#[test]
fn test_parallel_statistics() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let ts1 = create_test_timeseries(100);
    let ts2 = create_test_timeseries(200);

    let series = vec![ts1, ts2];
    let stats = processor.parallel_statistics(&series).expect("Parallel statistics failed");

    assert_eq!(stats.len(), 2);

    // Check first series stats
    assert_eq!(stats[0].count, 100);
    assert!(stats[0].mean >= 0.0);
    assert!(stats[0].std_dev >= 0.0);
    assert!(stats[0].min <= stats[0].max);

    // Check second series stats
    assert_eq!(stats[1].count, 200);
    assert!(stats[1].mean >= 0.0);
    assert!(stats[1].std_dev >= 0.0);
    assert!(stats[1].min <= stats[1].max);
}

#[test]
fn test_process_multiple_generic() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let input_data = vec![1, 2, 3, 4, 5];

    let results = processor.process_multiple(input_data, |x| {
        Ok(x * x)
    }).expect("Process multiple failed");

    assert_eq!(results, vec![1, 4, 9, 16, 25]);
}

#[test]
fn test_process_multiple_with_errors() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let input_data = vec![1, 2, 3, 4, 5];

    // Processor that fails on even numbers
    let result = processor.process_multiple(input_data, |x| {
        if x % 2 == 0 {
            Err("Even number error".into())
        } else {
            Ok(x * x)
        }
    });

    // Should fail because some items error
    assert!(result.is_err());
}

#[test]
fn test_parallel_forecast() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let ts1 = create_test_timeseries(50);
    let ts2 = create_test_timeseries(60);

    let series = vec![ts1, ts2];

    // Simple forecast function: predict next value as mean
    let forecasts = processor.parallel_forecast(series, |ts| {
        let mean = ts.values().iter().sum::<f64>() / ts.len() as f64;
        Ok(vec![mean]) // Forecast one step ahead
    }).expect("Parallel forecast failed");

    assert_eq!(forecasts.len(), 2);
    assert_eq!(forecasts[0].len(), 1);
    assert_eq!(forecasts[1].len(), 1);
}

#[test]
fn test_task_processor() {
    let config = create_test_config();
    let parallel_processor = std::sync::Arc::new(
        ParallelProcessor::new(&config).expect("Failed to create parallel processor")
    );
    let task_processor = TaskProcessor::new(parallel_processor);

    let tasks = vec![1, 2, 3, 4, 5];

    let results = task_processor.process_batch(tasks, |x| {
        Ok(x * 2)
    }).expect("Task batch processing failed");

    assert_eq!(results, vec![2, 4, 6, 8, 10]);
}

#[test]
fn test_task_processor_prioritized() {
    let config = create_test_config();
    let parallel_processor = std::sync::Arc::new(
        ParallelProcessor::new(&config).expect("Failed to create parallel processor")
    );
    let task_processor = TaskProcessor::new(parallel_processor);

    let high_priority = vec![1, 2];
    let low_priority = vec![3, 4, 5];

    let (high_results, low_results) = task_processor.process_prioritized(
        high_priority,
        low_priority,
        |x| Ok(x * 10)
    ).expect("Prioritized processing failed");

    assert_eq!(high_results, vec![10, 20]);
    assert_eq!(low_results, vec![30, 40, 50]);
}

#[test]
fn test_thread_info() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let info = processor.thread_info();

    assert!(info.num_threads > 0);
    // Note: queue_size and active_tasks are not directly available from rayon
    // so they'll be 0 in our implementation
    assert_eq!(info.queue_size, 0);
    assert_eq!(info.active_tasks, 0);
}

#[test]
fn test_prepare_timeseries() {
    let config = create_test_config();
    let processor = ParallelProcessor::new(&config).expect("Failed to create parallel processor");

    let mut ts = create_test_timeseries(100);

    // This is currently a placeholder operation
    let result = processor.prepare_timeseries(&mut ts);
    assert!(result.is_ok());
}

#[test]
fn test_concurrent_parallel_operations() {
    let config = create_test_config();
    let processor = std::sync::Arc::new(
        ParallelProcessor::new(&config).expect("Failed to create parallel processor")
    );

    let mut handles = vec![];

    // Spawn multiple threads doing parallel operations
    for i in 0..3 {
        let proc = processor.clone();
        let handle = std::thread::spawn(move || {
            let ts = create_test_timeseries(100 + i * 10);
            let results = proc.parallel_map(&ts, |x| x + i as f64).expect("Parallel map failed");
            results.len()
        });
        handles.push(handle);
    }

    // Wait for all threads
    let mut total_results = 0;
    for handle in handles {
        let result_count = handle.join().expect("Thread panicked");
        total_results += result_count;
    }

    assert!(total_results > 0);
}