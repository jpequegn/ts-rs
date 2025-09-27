# Performance Optimization System

The Chronos Time Series Analysis Library includes a comprehensive performance optimization system designed to handle large-scale time series data efficiently. This system provides memory management, parallel processing, intelligent caching, database integration, and progress tracking capabilities.

## Overview

The performance optimization system is built around five core components:

1. **Memory Management** - Efficient memory usage, streaming processing, and lazy loading
2. **Parallel Processing** - Multi-threaded operations and SIMD optimizations
3. **Caching System** - Intelligent result caching with compression and persistence
4. **Database Integration** - SQLite support for large datasets and analysis result storage
5. **Progress Tracking** - Progress bars, ETA estimation, and cancellation support

## Quick Start

### Basic Configuration

```rust
use chronos_time_series::config::PerformanceConfig;
use chronos_time_series::performance::PerformanceOptimizer;

let config = PerformanceConfig {
    enable_optimization: true,
    max_memory_mb: 1024,        // 1GB memory limit
    chunk_size: 10000,          // Process in 10K item chunks
    num_threads: Some(8),       // Use 8 threads for parallel processing
    max_cache_size_mb: 256,     // 256MB cache limit
    progress_threshold: 1000,   // Show progress for operations > 1000 items
    enable_database: true,      // Enable database for large datasets
    cache_directory: Some("/tmp/chronos_cache".into()),
};

let optimizer = PerformanceOptimizer::new(&config)?;
```

### Basic Usage

```rust
// Create a time series
let mut ts = TimeSeries::new(timestamps, values)?;

// Optimize memory usage
optimizer.optimize_memory(&mut ts)?;

// Perform parallel computation
let results = optimizer.execute_parallel(&ts, |values| {
    values.iter().map(|&x| x * 2.0).collect()
})?;

// Cache results for future use
optimizer.cache_result("computation_key", &results)?;

// Execute with progress tracking
let result = optimizer.execute_with_progress("long_operation", 10000, |progress| {
    // Your long-running operation here
    for i in 0..10000 {
        progress.inc();
        // ... do work ...
    }
    Ok(42)
})?;
```

## Core Components

### Memory Management

The memory management system provides efficient handling of large datasets through:

- **Memory Tracking**: Real-time monitoring of memory usage and limits
- **Streaming Processing**: Process large files without loading everything into memory
- **Lazy Loading**: Load data on-demand to minimize memory footprint
- **Compact Data Structures**: Compressed representations for time series data

#### Example: Streaming Large Files

```rust
use chronos_time_series::performance::memory::{MemoryManager, StreamingProcessor};

let memory_manager = Arc::new(MemoryManager::new(&config)?);
let processor = StreamingProcessor::new(memory_manager.clone(), 1024 * 1024); // 1MB chunks

let results = processor.process_file(Path::new("large_dataset.csv"), |chunk| {
    // Process each chunk
    let lines = std::str::from_utf8(chunk)?.lines().count();
    Ok(lines)
})?;
```

#### Example: Compact Time Series

```rust
use chronos_time_series::performance::memory::CompactTimeSeries;

// Create a compact representation that uses less memory
let compact_ts = CompactTimeSeries::from_timeseries(&original_ts)?;

// Use the compact version for storage or transmission
let compression_ratio = compact_ts.compression_ratio(original_ts.len() * 16); // 16 bytes per point

// Convert back when needed for analysis
let restored_ts = compact_ts.to_timeseries()?;
```

### Parallel Processing

The parallel processing system leverages multi-core CPUs for faster analysis:

- **Thread Pool Management**: Configurable number of worker threads
- **Parallel Map/Reduce**: Functional programming patterns for time series operations
- **Batch Processing**: Efficient processing of multiple time series
- **Correlation Analysis**: Parallel computation of correlation matrices

#### Example: Parallel Operations

```rust
use chronos_time_series::performance::parallel::ParallelProcessor;

let processor = ParallelProcessor::new(&config)?;

// Parallel map operation
let squared_values = processor.parallel_map(&ts, |x| x * x)?;

// Parallel reduce operation
let sum = processor.parallel_reduce(&ts, 0.0, |acc, x| acc + x)?;

// Parallel windowed operation
let window_means = processor.parallel_windowed(&ts, 10, |window| {
    window.iter().sum::<f64>() / window.len() as f64
})?;

// Parallel correlation matrix
let correlation_matrix = processor.parallel_correlations(&series_vec)?;
```

### Caching System

The intelligent caching system speeds up repeated operations:

- **Memory Caching**: Fast in-memory cache with LRU eviction
- **Persistent Caching**: Disk-based cache that survives application restarts
- **Compression**: LZ4 compression for efficient storage
- **Smart Invalidation**: Automatic cache cleanup and TTL expiration

#### Example: Using the Cache

```rust
use chronos_time_series::performance::cache::{CacheManager, AnalysisCache};

let cache_manager = Arc::new(CacheManager::new(&config)?);
let analysis_cache = AnalysisCache::new(cache_manager.clone());

// Cache expensive computation results
let stats = compute_expensive_statistics(&ts);
analysis_cache.cache_statistics(ts_hash, &stats)?;

// Retrieve from cache in future runs
if let Some(cached_stats) = analysis_cache.get_statistics(ts_hash) {
    println!("Using cached statistics: {:?}", cached_stats);
} else {
    // Compute and cache
    let stats = compute_expensive_statistics(&ts);
    analysis_cache.cache_statistics(ts_hash, &stats)?;
}
```

### Database Integration

For very large datasets that don't fit in memory:

- **SQLite Backend**: Efficient storage and retrieval of time series data
- **Query Optimization**: Indexed queries for fast data access
- **Performance Metrics**: Track and analyze operation performance
- **Connection Pooling**: Efficient database connection management

#### Example: Database Storage

```rust
use chronos_time_series::performance::database::DatabaseManager;

let db_manager = DatabaseManager::new(&config).await?;

// Store large time series in database
db_manager.store_timeseries("monthly_sales", &large_timeseries).await?;

// Load when needed
let loaded_ts = db_manager.load_timeseries("monthly_sales").await?;

// Cache analysis results
let forecast = generate_forecast(&ts);
db_manager.cache_analysis_result("sales_forecast", &forecast).await?;
```

### Progress Tracking

Keep users informed during long-running operations:

- **Progress Bars**: Visual progress indicators with ETA
- **Cancellation Support**: Allow users to cancel long operations
- **Operation Management**: Track multiple concurrent operations
- **Spinner Indicators**: For indeterminate progress operations

#### Example: Progress Tracking

```rust
use chronos_time_series::performance::progress::{ProgressTracker, ProgressAware};

let tracker = ProgressTracker::new(&config)?;

// Execute with progress bar
let result = tracker.execute_with_progress("data_processing", 10000, |progress| {
    for i in 0..10000 {
        // Check if operation was cancelled
        if progress.is_cancelled() {
            return Err("Operation cancelled".into());
        }

        // Update progress
        progress.inc();
        progress.set_message(&format!("Processing item {}", i));

        // Do actual work
        process_item(i)?;
    }
    Ok("Processing complete")
})?;

// Use spinner for indeterminate operations
let result = tracker.execute_with_spinner("model_training", |spinner| {
    spinner.set_message("Training neural network...");
    train_model()
})?;
```

## Performance Metrics

The system provides comprehensive performance monitoring:

```rust
// Get performance metrics
let metrics = optimizer.get_metrics();

println!("Memory usage: {:.2} MB", metrics.memory_usage_mb);
println!("Cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0);
println!("Operations per second: {:.0}", metrics.operations_per_second);
println!("Average operation time: {:?}", metrics.average_operation_time);
```

## Best Practices

### Memory Management

1. **Set appropriate memory limits** to prevent out-of-memory errors
2. **Use streaming processing** for files larger than available RAM
3. **Choose optimal chunk sizes** based on your data and memory constraints
4. **Monitor memory usage** regularly in production environments

### Parallel Processing

1. **Configure thread count** based on your CPU cores (typically cores - 1)
2. **Use parallel operations** for computationally intensive tasks
3. **Batch similar operations** together for better throughput
4. **Profile your code** to identify parallelization opportunities

### Caching

1. **Cache expensive computations** that are likely to be repeated
2. **Set appropriate cache sizes** to balance memory usage and hit rates
3. **Use persistent caching** for results that should survive restarts
4. **Monitor cache performance** and adjust policies as needed

### Database Integration

1. **Use databases** for datasets larger than available memory
2. **Index frequently queried columns** for better performance
3. **Batch database operations** to reduce connection overhead
4. **Monitor query performance** and optimize slow queries

### Progress Tracking

1. **Show progress** for operations taking more than a few seconds
2. **Provide meaningful progress messages** to keep users informed
3. **Support cancellation** for long-running operations
4. **Use appropriate progress indicators** (bars vs spinners) based on operation type

## Error Handling

The performance system provides detailed error information:

```rust
use chronos_time_series::performance::PerformanceError;

match optimizer.execute_parallel(&ts, computation) {
    Ok(result) => println!("Success: {:?}", result),
    Err(e) => match e.downcast_ref::<PerformanceError>() {
        Some(PerformanceError::MemoryLimitExceeded(mb)) => {
            eprintln!("Out of memory: {} MB exceeded", mb);
        }
        Some(PerformanceError::ParallelProcessingError(msg)) => {
            eprintln!("Parallel processing failed: {}", msg);
        }
        Some(PerformanceError::CacheError(msg)) => {
            eprintln!("Cache error: {}", msg);
        }
        _ => eprintln!("Other error: {}", e),
    }
}
```

## Integration with Analysis Operations

The performance system integrates seamlessly with all analysis operations:

```rust
// Trend analysis with performance optimization
let trend_config = TrendAnalysisConfig::default();
let trend_result = analyze_trend_comprehensive(&ts, &trend_config, Some(&optimizer))?;

// Seasonality analysis with caching
let seasonality_result = analyze_comprehensive_seasonality(&ts, &config, Some(&optimizer))?;

// Forecasting with parallel processing
let forecast_result = forecast_timeseries(&ts, &forecast_config, Some(&optimizer))?;

// Anomaly detection with progress tracking
let anomaly_result = detect_anomalies(&ts, &anomaly_config, Some(&optimizer))?;
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_optimization` | `bool` | `true` | Enable performance optimizations |
| `max_memory_mb` | `usize` | `1024` | Maximum memory usage in MB |
| `chunk_size` | `usize` | `10000` | Processing chunk size |
| `num_threads` | `Option<usize>` | `None` | Number of threads (auto if None) |
| `max_cache_size_mb` | `usize` | `256` | Maximum cache size in MB |
| `progress_threshold` | `usize` | `1000` | Show progress for operations > threshold |
| `enable_database` | `bool` | `false` | Enable database integration |
| `cache_directory` | `Option<PathBuf>` | `None` | Directory for persistent cache |

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `max_memory_mb` setting
   - Use streaming processing for large files
   - Increase chunk size for better memory efficiency

2. **Slow Performance**
   - Increase number of threads for parallel operations
   - Enable caching for repeated computations
   - Profile code to identify bottlenecks

3. **Cache Misses**
   - Increase cache size if memory allows
   - Check cache key consistency
   - Monitor cache hit rates

4. **Database Errors**
   - Ensure database directory exists and is writable
   - Check disk space availability
   - Verify SQLite version compatibility

### Debugging

Enable debug logging to troubleshoot performance issues:

```rust
env_logger::init();
log::debug!("Performance optimizer initialized with config: {:?}", config);
```

The performance system logs detailed information about:
- Memory allocation and deallocation
- Cache hits and misses
- Parallel operation scheduling
- Database query performance
- Progress tracking events

For more detailed information about specific components, see the individual documentation files:
- [Memory Management](memory.md)
- [Parallel Processing](parallel.md)
- [Caching System](caching.md)
- [Database Integration](database.md)
- [Progress Tracking](progress.md)