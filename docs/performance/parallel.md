# Parallel Processing

The parallel processing system in Chronos leverages multi-core CPUs to accelerate time series analysis operations through efficient thread management and parallel algorithms.

## Overview

The parallel processing system provides:

1. **ParallelProcessor** - Core parallel execution engine
2. **TaskProcessor** - Batch and prioritized task processing
3. **Thread Pool Management** - Efficient worker thread allocation
4. **Parallel Algorithms** - Optimized implementations for common operations

## ParallelProcessor

The `ParallelProcessor` is the main component that handles parallel execution of time series operations.

### Features

- **Configurable Thread Pools**: Control the number of worker threads
- **Parallel Map/Reduce**: Functional programming patterns for data processing
- **Windowed Operations**: Parallel processing of sliding windows
- **Correlation Analysis**: Efficient parallel correlation matrix computation
- **SIMD Optimizations**: Automatic vectorization where possible

### Basic Usage

```rust
use chronos_time_series::performance::parallel::ParallelProcessor;
use chronos_time_series::config::PerformanceConfig;

let config = PerformanceConfig {
    num_threads: Some(8),    // Use 8 worker threads
    chunk_size: 10000,       // Process in 10K item chunks
    ..PerformanceConfig::default()
};

let processor = ParallelProcessor::new(&config)?;

// Get thread pool information
let info = processor.thread_info();
println!("Using {} threads", info.num_threads);
```

## Parallel Operations

### Map Operations

Apply a function to every element in parallel:

```rust
// Square all values in parallel
let squared = processor.parallel_map(&timeseries, |x| x * x)?;

// Apply complex transformations
let transformed = processor.parallel_map(&timeseries, |x| {
    // More complex calculation
    (x * 2.0).sin() + (x / 3.0).cos()
})?;

// Type transformations
let classifications = processor.parallel_map(&timeseries, |x| {
    if x > 0.5 { "high" } else { "low" }
})?;
```

### Reduce Operations

Aggregate values across the time series:

```rust
// Sum all values
let total = processor.parallel_reduce(&timeseries, 0.0, |acc, x| acc + x)?;

// Find maximum value
let max_value = processor.parallel_reduce(&timeseries, f64::NEG_INFINITY, |acc, x| acc.max(x))?;

// Calculate custom statistics
let variance = processor.parallel_reduce(&timeseries, (0.0, 0.0), |(sum, sum_sq), x| {
    (sum + x, sum_sq + x * x)
})?;
```

### Windowed Operations

Process sliding windows in parallel:

```rust
// Calculate moving averages
let window_size = 10;
let moving_averages = processor.parallel_windowed(&timeseries, window_size, |window| {
    window.iter().sum::<f64>() / window.len() as f64
})?;

// Moving standard deviations
let moving_std_devs = processor.parallel_windowed(&timeseries, window_size, |window| {
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    let variance = window.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / window.len() as f64;
    variance.sqrt()
})?;

// Custom window analysis
let trend_indicators = processor.parallel_windowed(&timeseries, 20, |window| {
    let first_half: f64 = window[..10].iter().sum::<f64>() / 10.0;
    let second_half: f64 = window[10..].iter().sum::<f64>() / 10.0;

    if second_half > first_half * 1.05 {
        "increasing"
    } else if second_half < first_half * 0.95 {
        "decreasing"
    } else {
        "stable"
    }
})?;
```

## Correlation Analysis

Efficiently compute correlation matrices for multiple time series:

```rust
let series_collection = vec![ts1, ts2, ts3, ts4, ts5];

// Compute full correlation matrix in parallel
let correlation_matrix = processor.parallel_correlations(&series_collection)?;

// The matrix is symmetric: correlation_matrix[i][j] == correlation_matrix[j][i]
println!("Correlation between series 0 and 2: {:.3}", correlation_matrix[0][2]);

// Find highly correlated pairs
for i in 0..series_collection.len() {
    for j in (i+1)..series_collection.len() {
        let corr = correlation_matrix[i][j];
        if corr.abs() > 0.8 {
            println!("High correlation between series {} and {}: {:.3}", i, j, corr);
        }
    }
}
```

## Batch Processing

Process multiple time series or tasks efficiently:

```rust
// Parallel statistics for multiple series
let series_vec = vec![ts1, ts2, ts3];
let all_stats = processor.parallel_statistics(&series_vec)?;

for (i, stats) in all_stats.iter().enumerate() {
    println!("Series {}: mean={:.2}, std={:.2}, min={:.2}, max={:.2}",
             i, stats.mean, stats.std_dev, stats.min, stats.max);
}

// Generic batch processing
let input_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let results = processor.process_multiple(input_data, |x| {
    // Expensive computation for each item
    Ok(expensive_calculation(x))
})?;
```

## Forecasting

Generate forecasts for multiple time series in parallel:

```rust
let forecast_results = processor.parallel_forecast(series_vec, |ts| {
    // Apply forecasting algorithm to each series
    let model = fit_arima_model(ts)?;
    let forecast = model.forecast(12)?; // 12 periods ahead
    Ok(forecast)
})?;

for (i, forecast) in forecast_results.iter().enumerate() {
    println!("Series {} forecast: {:?}", i, forecast);
}
```

## TaskProcessor

The `TaskProcessor` provides higher-level task management with prioritization:

```rust
use chronos_time_series::performance::parallel::TaskProcessor;

let task_processor = TaskProcessor::new(Arc::new(processor));

// Process batch of tasks
let tasks = vec!["task1", "task2", "task3", "task4"];
let results = task_processor.process_batch(tasks, |task| {
    // Process each task
    Ok(format!("Processed: {}", task))
})?;

// Prioritized processing
let high_priority = vec!["urgent1", "urgent2"];
let low_priority = vec!["normal1", "normal2", "normal3"];

let (urgent_results, normal_results) = task_processor.process_prioritized(
    high_priority,
    low_priority,
    |task| Ok(process_task(task))
)?;
```

## Configuration and Tuning

### Thread Pool Configuration

```rust
let config = PerformanceConfig {
    // Explicit thread count
    num_threads: Some(8),    // Use exactly 8 threads

    // Auto-detection (recommended)
    num_threads: None,       // Use all available CPU cores

    // Conservative setting
    num_threads: Some(num_cpus::get() - 1), // Leave one core free

    ..PerformanceConfig::default()
};
```

### Chunk Size Optimization

```rust
// Small chunks: better load balancing, higher overhead
config.chunk_size = 1000;

// Large chunks: lower overhead, potential load imbalance
config.chunk_size = 100000;

// Adaptive chunk sizing
let optimal_chunk = if dataset_size < 10000 {
    dataset_size / num_threads
} else {
    10000 // Fixed size for large datasets
};
```

## Performance Optimization

### 1. Choose Optimal Thread Count

```rust
// Rule of thumb: number of CPU cores for CPU-bound tasks
let cpu_cores = num_cpus::get();
let config = PerformanceConfig {
    num_threads: Some(cpu_cores),
    ..PerformanceConfig::default()
};

// For I/O-bound tasks, you might use more threads
let config = PerformanceConfig {
    num_threads: Some(cpu_cores * 2),
    ..PerformanceConfig::default()
};
```

### 2. Minimize Data Movement

```rust
// Instead of returning large data structures
let results = processor.parallel_map(&ts, |x| expensive_computation(x))?;

// Use side effects for accumulation
use std::sync::{Arc, Mutex};
let accumulator = Arc::new(Mutex::new(Vec::new()));

processor.parallel_map(&ts, |x| {
    let result = expensive_computation(x);
    accumulator.lock().unwrap().push(result);
    () // Return unit type
})?;
```

### 3. Balance Work Distribution

```rust
// For uneven workloads, use smaller chunks
let processor = if has_uneven_workload {
    ParallelProcessor::new(&PerformanceConfig {
        chunk_size: 100,  // Small chunks for better distribution
        ..config
    })?
} else {
    ParallelProcessor::new(&config)?
};
```

## Error Handling

Handle errors in parallel operations:

```rust
use chronos_time_series::performance::PerformanceError;

match processor.parallel_map(&ts, |x| fallible_operation(x)) {
    Ok(results) => println!("All operations succeeded"),
    Err(e) => match e.downcast_ref::<PerformanceError>() {
        Some(PerformanceError::ParallelProcessingError(msg)) => {
            eprintln!("Parallel processing failed: {}", msg);
            // Handle by reducing parallelism or fixing the operation
        }
        _ => eprintln!("Other error: {}", e),
    }
}

// For operations that may fail, collect successes and failures
let results: Vec<Result<f64, _>> = ts.values().par_iter()
    .map(|&x| fallible_computation(x))
    .collect();

let (successes, failures): (Vec<_>, Vec<_>) = results.into_iter()
    .partition(|r| r.is_ok());

println!("Successes: {}, Failures: {}", successes.len(), failures.len());
```

## Advanced Usage

### Custom Parallel Algorithms

```rust
// Implement custom parallel reduction
fn parallel_geometric_mean(processor: &ParallelProcessor, ts: &TimeSeries) -> Result<f64> {
    // First pass: compute sum of logarithms
    let log_sum = processor.parallel_reduce(ts, 0.0, |acc, x| {
        acc + x.ln()
    })?;

    // Result: exponential of average logarithm
    Ok((log_sum / ts.len() as f64).exp())
}

// Parallel convolution
fn parallel_convolution(
    processor: &ParallelProcessor,
    signal: &[f64],
    kernel: &[f64]
) -> Result<Vec<f64>> {
    let output_size = signal.len() + kernel.len() - 1;
    let indices: Vec<usize> = (0..output_size).collect();

    processor.process_multiple(indices, |i| {
        let mut sum = 0.0;
        for j in 0..kernel.len() {
            if i >= j && i - j < signal.len() {
                sum += signal[i - j] * kernel[j];
            }
        }
        Ok(sum)
    })
}
```

### Integration with Other Systems

```rust
// Integration with progress tracking
use chronos_time_series::performance::progress::ProgressTracker;

let progress_tracker = ProgressTracker::new(&config)?;

let result = progress_tracker.execute_with_progress("parallel_computation", 1000, |progress| {
    let chunk_size = 100;
    let mut results = Vec::new();

    for chunk_start in (0..1000).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, 1000);
        let chunk_data = &data[chunk_start..chunk_end];

        let chunk_result = processor.parallel_map(&chunk_data, |x| process_item(x))?;
        results.extend(chunk_result);

        progress.add(chunk_size as u64);
    }

    Ok(results)
})?;
```

## Best Practices

### 1. Profile Before Parallelizing

```rust
// Measure serial performance first
let start = std::time::Instant::now();
let serial_result = ts.values().iter().map(|&x| expensive_op(x)).collect::<Vec<_>>();
let serial_time = start.elapsed();

// Then measure parallel performance
let start = std::time::Instant::now();
let parallel_result = processor.parallel_map(&ts, |x| expensive_op(x))?;
let parallel_time = start.elapsed();

println!("Speedup: {:.2}x", serial_time.as_secs_f64() / parallel_time.as_secs_f64());
```

### 2. Consider Memory Usage

```rust
// For memory-intensive operations, limit parallelism
let memory_per_thread = estimated_memory_per_operation();
let max_threads = (available_memory() / memory_per_thread).min(num_cpus::get());

let config = PerformanceConfig {
    num_threads: Some(max_threads),
    ..PerformanceConfig::default()
};
```

### 3. Handle Cancellation

```rust
use std::sync::atomic::{AtomicBool, Ordering};

let should_cancel = Arc::new(AtomicBool::new(false));
let cancel_flag = should_cancel.clone();

// Set up cancellation handler
std::thread::spawn(move || {
    std::thread::sleep(std::time::Duration::from_secs(10));
    cancel_flag.store(true, Ordering::Relaxed);
});

// Check cancellation in parallel operations
let results = processor.parallel_map(&ts, |x| {
    if should_cancel.load(Ordering::Relaxed) {
        return Err("Operation cancelled".into());
    }
    Ok(expensive_operation(x))
})?;
```

### 4. Optimize for Cache Locality

```rust
// Process data in cache-friendly chunks
let cache_line_size = 64; // bytes
let items_per_cache_line = cache_line_size / std::mem::size_of::<f64>();

let config = PerformanceConfig {
    chunk_size: items_per_cache_line * 8, // 8 cache lines per chunk
    ..PerformanceConfig::default()
};
```

## Troubleshooting

### Common Issues

1. **Poor Speedup**
   - Check if operation is CPU-bound vs I/O-bound
   - Verify chunk sizes aren't too small
   - Ensure sufficient work per thread

2. **Thread Contention**
   - Reduce number of threads
   - Increase chunk sizes
   - Minimize shared state

3. **Memory Issues**
   - Monitor memory usage during parallel operations
   - Reduce number of threads if memory-bound
   - Use streaming for large datasets

4. **Load Imbalance**
   - Use smaller chunk sizes for uneven workloads
   - Consider work-stealing algorithms
   - Profile individual operations

### Performance Monitoring

```rust
// Monitor thread utilization
let info = processor.thread_info();
println!("Active threads: {}/{}", info.active_tasks, info.num_threads);

// Measure actual vs theoretical speedup
let theoretical_speedup = info.num_threads as f64;
let actual_speedup = serial_time / parallel_time;
let efficiency = actual_speedup / theoretical_speedup;

println!("Parallel efficiency: {:.1}%", efficiency * 100.0);
```