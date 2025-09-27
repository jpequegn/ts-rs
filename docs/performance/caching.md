# Caching System

The caching system in Chronos provides intelligent result caching with compression, persistence, and automatic invalidation to accelerate repeated time series analysis operations.

## Overview

The caching system consists of:

1. **CacheManager** - Core caching engine with memory and disk storage
2. **AnalysisCache** - Domain-specific cache for time series analysis results
3. **Compression** - LZ4 compression for efficient storage
4. **Persistence** - Disk-based caching that survives application restarts

## CacheManager

The `CacheManager` is the core component that handles caching operations with intelligent storage strategies.

### Features

- **Memory Caching**: Fast in-memory cache with LRU eviction
- **Persistent Storage**: Disk-based cache for long-term storage
- **Compression**: Automatic LZ4 compression for large entries
- **Statistics**: Detailed cache performance metrics
- **Thread Safety**: Concurrent access from multiple threads

### Basic Usage

```rust
use chronos_time_series::performance::cache::CacheManager;
use chronos_time_series::config::PerformanceConfig;

let config = PerformanceConfig {
    max_cache_size_mb: 256,     // 256MB cache limit
    cache_directory: Some("/tmp/chronos_cache".into()),
    ..PerformanceConfig::default()
};

let cache = CacheManager::new(&config)?;

// Store data in cache
let expensive_result = vec![1.0, 2.0, 3.0, 4.0, 5.0];
cache.set("computation_key", &expensive_result)?;

// Retrieve from cache
let cached_result: Option<Vec<f64>> = cache.get("computation_key");
match cached_result {
    Some(data) => println!("Cache hit: {:?}", data),
    None => println!("Cache miss"),
}

// Get cache statistics
let stats = cache.stats();
println!("Hit rate: {:.1}%", stats.hit_rate * 100.0);
```

### Cache Operations

#### Storing Data

```rust
// Simple value storage
cache.set("key1", &42)?;
cache.set("key2", &vec![1, 2, 3, 4])?;
cache.set("key3", &"Hello, World!")?;

// Complex data structures
use std::collections::HashMap;
let mut analysis_results = HashMap::new();
analysis_results.insert("mean", 5.5);
analysis_results.insert("std_dev", 2.1);
cache.set("stats_result", &analysis_results)?;

// Large data will be automatically compressed
let large_dataset: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
cache.set("large_data", &large_dataset)?; // Automatically compressed
```

#### Retrieving Data

```rust
// Type-safe retrieval
let value: Option<i32> = cache.get("key1");
let vector: Option<Vec<i32>> = cache.get("key2");
let text: Option<String> = cache.get("key3");

// Handle cache misses
let result = cache.get::<Vec<f64>>("computation_result")
    .unwrap_or_else(|| {
        // Compute if not cached
        let result = expensive_computation();
        cache.set("computation_result", &result).ok();
        result
    });
```

#### Cache Management

```rust
// Remove specific entries
cache.remove("outdated_key");

// Clear entire cache
cache.clear_all()?;

// Check if caching is worthwhile
if cache.should_cache("expensive_operation", data_size) {
    cache.set("operation_result", &result)?;
}
```

### Cache Statistics

Monitor cache performance with detailed metrics:

```rust
let stats = cache.stats();

println!("Memory usage: {:.2}/{:.2} MB", stats.memory_usage_mb, stats.max_size_mb);
println!("Entries: {}", stats.entry_count);
println!("Hit rate: {:.1}%", stats.hit_rate * 100.0);
println!("Total requests: {}", stats.hit_count + stats.miss_count);

// Performance analysis
if stats.hit_rate < 0.5 {
    println!("Warning: Low cache hit rate, consider adjusting cache size or TTL");
}

if stats.memory_usage_mb > stats.max_size_mb * 0.9 {
    println!("Warning: Cache nearly full, entries may be evicted frequently");
}
```

## AnalysisCache

The `AnalysisCache` provides domain-specific caching for time series analysis operations.

### Features

- **Statistical Analysis Caching**: Cache descriptive statistics, correlations, etc.
- **Forecast Result Caching**: Store forecast results with model and horizon information
- **Smart Key Generation**: Automatic key generation based on data characteristics
- **Type Safety**: Strongly typed caching for analysis results

### Usage

```rust
use chronos_time_series::performance::cache::AnalysisCache;
use std::collections::HashMap;

let cache_manager = Arc::new(CacheManager::new(&config)?);
let analysis_cache = AnalysisCache::new(cache_manager);

// Cache statistical analysis
let mut stats = HashMap::new();
stats.insert("mean".to_string(), 5.5);
stats.insert("std_dev".to_string(), 2.1);
stats.insert("min".to_string(), 1.0);
stats.insert("max".to_string(), 10.0);

let data_hash = compute_data_hash(&timeseries);
analysis_cache.cache_statistics(data_hash, &stats)?;

// Retrieve statistics
if let Some(cached_stats) = analysis_cache.get_statistics(data_hash) {
    println!("Using cached statistics: {:?}", cached_stats);
} else {
    // Compute and cache
    let stats = compute_descriptive_statistics(&timeseries)?;
    analysis_cache.cache_statistics(data_hash, &stats)?;
}
```

### Correlation Matrix Caching

```rust
// Cache expensive correlation computations
let correlation_matrix = compute_correlation_matrix(&series_collection)?;
let matrix_hash = compute_matrix_hash(&series_collection);

analysis_cache.cache_correlation_matrix(matrix_hash, &correlation_matrix)?;

// Later retrieval
if let Some(cached_matrix) = analysis_cache.get_correlation_matrix(matrix_hash) {
    println!("Using cached correlation matrix");
    use_correlation_matrix(&cached_matrix);
} else {
    println!("Computing correlation matrix...");
    let matrix = compute_correlation_matrix(&series_collection)?;
    analysis_cache.cache_correlation_matrix(matrix_hash, &matrix)?;
}
```

### Forecast Caching

```rust
// Cache forecast results by model and horizon
let model_hash = compute_model_hash(&fitted_model);
let horizon = 12;

let forecast = generate_forecast(&model, horizon)?;
analysis_cache.cache_forecast(model_hash, horizon, &forecast)?;

// Retrieve forecast
if let Some(cached_forecast) = analysis_cache.get_forecast(model_hash, horizon) {
    println!("Using cached forecast: {:?}", cached_forecast);
} else {
    println!("Generating new forecast...");
    let forecast = generate_forecast(&model, horizon)?;
    analysis_cache.cache_forecast(model_hash, horizon, &forecast)?;
}
```

## Compression and Storage

### Automatic Compression

The cache automatically applies LZ4 compression to large entries:

```rust
// Compression is transparent to the user
let large_data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();

// This will be automatically compressed
cache.set("large_dataset", &large_data)?;

// Decompression happens automatically on retrieval
let retrieved: Option<Vec<f64>> = cache.get("large_dataset");
assert_eq!(retrieved, Some(large_data));
```

### Persistent Storage

Enable persistent caching to retain cached data across application restarts:

```rust
let config = PerformanceConfig {
    cache_directory: Some("/var/cache/chronos".into()),
    ..PerformanceConfig::default()
};

let cache = CacheManager::new(&config)?;

// Data persisted to disk automatically
cache.set("persistent_data", &expensive_result)?;

// Available after application restart
let cache2 = CacheManager::new(&config)?;
let restored: Option<Vec<f64>> = cache2.get("persistent_data");
```

## Cache Eviction and TTL

### LRU Eviction

The cache uses Least Recently Used (LRU) eviction when memory limits are reached:

```rust
// Configure cache size to trigger eviction
let config = PerformanceConfig {
    max_cache_size_mb: 10, // Small cache to demonstrate eviction
    ..PerformanceConfig::default()
};

let cache = CacheManager::new(&config)?;

// Add entries until eviction occurs
for i in 0..1000 {
    let large_data: Vec<f64> = (0..10_000).map(|j| (i * 10_000 + j) as f64).collect();
    cache.set(&format!("data_{}", i), &large_data)?;

    let stats = cache.stats();
    println!("Entries: {}, Memory: {:.2}MB", stats.entry_count, stats.memory_usage_mb);
}
```

### Time-Based Expiration

Entries automatically expire after a configurable time-to-live (TTL):

```rust
// Entries expire after 1 hour by default
cache.set("temporary_result", &computation_result)?;

// After TTL expires, entry is no longer available
std::thread::sleep(std::time::Duration::from_secs(3600));
let expired: Option<Vec<f64>> = cache.get("temporary_result"); // Returns None
```

## Advanced Usage

### Custom Cache Keys

Generate intelligent cache keys based on data characteristics:

```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn compute_timeseries_hash(ts: &TimeSeries) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash data characteristics
    ts.len().hash(&mut hasher);
    ts.frequency().hash(&mut hasher);

    // Hash sample of values for uniqueness
    for &value in ts.values().iter().step_by(ts.len() / 100 + 1) {
        value.to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

// Use hash as cache key
let key = format!("analysis_{}", compute_timeseries_hash(&ts));
cache.set(&key, &analysis_result)?;
```

### Conditional Caching

Cache only when it's beneficial:

```rust
fn should_cache_operation(operation: &str, data_size: usize, computation_time: Duration) -> bool {
    // Cache expensive operations
    if computation_time > Duration::from_secs(1) {
        return true;
    }

    // Cache operations on large datasets
    if data_size > 10_000 {
        return true;
    }

    // Cache specific operation types
    let expensive_ops = ["correlation_matrix", "fft", "arima_fit"];
    expensive_ops.contains(&operation)
}

// Conditional caching
if should_cache_operation("correlation_matrix", data.len(), computation_time) {
    cache.set("correlation_result", &result)?;
}
```

### Cache Warming

Pre-populate cache with commonly needed data:

```rust
fn warm_cache(cache: &CacheManager, timeseries: &[TimeSeries]) -> Result<()> {
    for (i, ts) in timeseries.iter().enumerate() {
        let key = format!("series_{}_stats", i);

        // Pre-compute and cache basic statistics
        if cache.get::<HashMap<String, f64>>(&key).is_none() {
            let stats = compute_basic_statistics(ts)?;
            cache.set(&key, &stats)?;
        }

        // Pre-compute common transformations
        let log_key = format!("series_{}_log", i);
        if cache.get::<Vec<f64>>(&log_key).is_none() {
            let log_values: Vec<f64> = ts.values().iter().map(|&x| x.ln()).collect();
            cache.set(&log_key, &log_values)?;
        }
    }
    Ok(())
}
```

## Configuration

### Cache Size Management

```rust
let config = PerformanceConfig {
    max_cache_size_mb: 512,    // 512MB memory limit
    cache_directory: Some("/fast/ssd/cache".into()), // Use SSD for persistence
    ..PerformanceConfig::default()
};

// Monitor and adjust cache size dynamically
let cache = CacheManager::new(&config)?;

// Periodic cache size monitoring
std::thread::spawn(move || {
    loop {
        std::thread::sleep(Duration::from_secs(60));
        let stats = cache.stats();

        if stats.memory_usage_mb > stats.max_size_mb * 0.9 {
            println!("Cache nearly full, consider increasing size");
        }
    }
});
```

### Cache Directory Setup

```rust
use std::fs;

// Ensure cache directory exists and is writable
fn setup_cache_directory(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }

    // Test write permissions
    let test_file = path.join(".write_test");
    fs::write(&test_file, "test")?;
    fs::remove_file(&test_file)?;

    Ok(())
}

let cache_dir = Path::new("/var/cache/chronos");
setup_cache_directory(&cache_dir)?;

let config = PerformanceConfig {
    cache_directory: Some(cache_dir.to_path_buf()),
    ..PerformanceConfig::default()
};
```

## Error Handling

Handle caching errors gracefully:

```rust
use chronos_time_series::performance::PerformanceError;

match cache.set("key", &data) {
    Ok(_) => println!("Data cached successfully"),
    Err(e) => match e.downcast_ref::<PerformanceError>() {
        Some(PerformanceError::CacheError(msg)) => {
            eprintln!("Cache error: {}", msg);
            // Continue without caching
        }
        _ => return Err(e), // Propagate other errors
    }
}

// Robust caching function
fn cache_with_fallback<T>(cache: &CacheManager, key: &str, data: &T) -> T
where
    T: Clone + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    // Try to get from cache first
    if let Some(cached) = cache.get(key) {
        return cached;
    }

    // If not in cache, try to store (but don't fail if caching fails)
    if let Err(e) = cache.set(key, data) {
        log::warn!("Failed to cache data for key '{}': {}", key, e);
    }

    data.clone()
}
```

## Best Practices

### 1. Choose Appropriate Cache Keys

```rust
// Good: Specific and meaningful keys
let key = format!("correlation_{}_{}_{}",
                  series1_hash, series2_hash, method);

// Bad: Vague or collision-prone keys
let key = "correlation_result".to_string();
```

### 2. Monitor Cache Performance

```rust
// Regular performance monitoring
fn monitor_cache_performance(cache: &CacheManager) {
    let stats = cache.stats();

    // Log cache metrics
    log::info!("Cache stats - Hit rate: {:.1}%, Memory: {:.2}MB, Entries: {}",
               stats.hit_rate * 100.0, stats.memory_usage_mb, stats.entry_count);

    // Alert on poor performance
    if stats.hit_rate < 0.3 {
        log::warn!("Low cache hit rate: {:.1}%", stats.hit_rate * 100.0);
    }

    if stats.memory_usage_mb > stats.max_size_mb * 0.95 {
        log::warn!("Cache nearly full: {:.2}/{:.2}MB",
                   stats.memory_usage_mb, stats.max_size_mb);
    }
}
```

### 3. Cache Appropriate Data

```rust
// Cache expensive computations
if computation_time > Duration::from_millis(100) {
    cache.set(&key, &result)?;
}

// Don't cache trivial operations
if computation_time < Duration::from_millis(1) {
    // Skip caching overhead
}

// Consider data size
if serialized_size > 1024 * 1024 { // 1MB
    // Large data - consider if caching is worth it
    if access_frequency > 3 {
        cache.set(&key, &result)?;
    }
}
```

### 4. Handle Cache Invalidation

```rust
// Invalidate cache when underlying data changes
fn update_timeseries(ts: &mut TimeSeries, cache: &CacheManager) -> Result<()> {
    // Update the time series
    ts.add_point(timestamp, value)?;

    // Invalidate related cache entries
    let base_key = format!("series_{}", compute_timeseries_hash(ts));
    cache.remove(&format!("{}_stats", base_key));
    cache.remove(&format!("{}_correlation", base_key));
    cache.remove(&format!("{}_forecast", base_key));

    Ok(())
}
```

## Performance Tips

### 1. Optimize Serialization

```rust
// Use efficient serialization for frequently cached data
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct AnalysisResult {
    #[serde(with = "serde_arrays")]
    correlation_matrix: [[f64; 100]; 100], // Fixed-size arrays are faster

    statistics: HashMap<String, f64>,
}
```

### 2. Batch Cache Operations

```rust
// Batch multiple cache operations
let mut operations = Vec::new();
for result in analysis_results {
    operations.push((format!("result_{}", result.id), result));
}

// Execute all operations together (conceptual - actual implementation may vary)
for (key, data) in operations {
    cache.set(&key, &data)?;
}
```

### 3. Use Appropriate Cache Sizes

```rust
// Rule of thumb: 10-25% of available RAM for cache
let total_ram_mb = 8192; // 8GB system
let cache_size_mb = total_ram_mb / 5; // 20% for cache

let config = PerformanceConfig {
    max_cache_size_mb: cache_size_mb,
    ..PerformanceConfig::default()
};
```

## Troubleshooting

### Common Issues

1. **Low Hit Rate**
   - Check if keys are consistent
   - Verify TTL settings aren't too short
   - Monitor access patterns

2. **High Memory Usage**
   - Reduce cache size limit
   - Enable compression for large entries
   - Implement more aggressive eviction

3. **Slow Cache Operations**
   - Use SSD for persistent storage
   - Reduce serialization overhead
   - Consider cache partitioning

4. **Cache Misses After Restart**
   - Verify persistent storage is enabled
   - Check cache directory permissions
   - Ensure disk space is available

### Debugging

```rust
// Enable cache debugging
log::debug!("Cache operation: SET key='{}', size={}bytes", key, data_size);
log::debug!("Cache stats: {:?}", cache.stats());

// Trace cache keys
fn trace_cache_keys(cache: &CacheManager) {
    let stats = cache.stats();
    log::debug!("Active cache entries: {}", stats.entry_count);

    // In debug builds, you might want to list all keys
    #[cfg(debug_assertions)]
    {
        // This would require additional API to enumerate keys
        for key in cache.list_keys() {
            log::debug!("Cached key: {}", key);
        }
    }
}
```