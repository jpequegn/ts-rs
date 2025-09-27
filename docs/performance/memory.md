# Memory Management

The memory management system in Chronos provides efficient handling of large time series datasets through intelligent memory allocation, streaming processing, and compact data structures.

## Overview

The memory management system consists of four main components:

1. **MemoryManager** - Tracks and controls memory usage
2. **StreamingProcessor** - Processes large files without loading everything into memory
3. **LazyDataLoader** - Loads data on-demand to minimize memory footprint
4. **CompactTimeSeries** - Memory-efficient time series representation

## MemoryManager

The `MemoryManager` is the core component that tracks memory usage and enforces limits.

### Features

- **Real-time Memory Tracking**: Monitor current and peak memory usage
- **Memory Limits**: Prevent out-of-memory errors with configurable limits
- **Memory Statistics**: Detailed metrics on memory efficiency
- **Allocation Management**: Track memory allocations and deallocations

### Usage

```rust
use chronos_time_series::performance::memory::MemoryManager;
use chronos_time_series::config::PerformanceConfig;

let config = PerformanceConfig {
    max_memory_mb: 512,  // 512MB limit
    chunk_size: 10000,
    ..PerformanceConfig::default()
};

let manager = MemoryManager::new(&config)?;

// Track memory allocation
manager.allocate(1024 * 1024)?; // Allocate 1MB

// Get memory statistics
let stats = manager.get_stats();
println!("Memory usage: {:.2} MB", stats.used_memory_mb);
println!("Peak usage: {:.2} MB", stats.peak_memory_mb);
println!("Memory efficiency: {:.1}%", stats.memory_efficiency * 100.0);

// Deallocate when done
manager.deallocate(1024 * 1024);
```

### Memory Statistics

The `MemoryStats` structure provides detailed information about memory usage:

```rust
pub struct MemoryStats {
    pub used_memory_mb: f64,        // Current memory usage in MB
    pub available_memory_mb: f64,   // Available memory in MB
    pub cache_memory_mb: f64,       // Memory used by cache in MB
    pub peak_memory_mb: f64,        // Peak memory usage in MB
    pub memory_efficiency: f64,     // Ratio of current to peak usage
}
```

### Memory Limits

Set memory limits to prevent your application from consuming too much memory:

```rust
let mut config = PerformanceConfig::default();
config.max_memory_mb = 1024; // 1GB limit

let manager = MemoryManager::new(&config)?;

// This will succeed if within limits
manager.allocate(512 * 1024 * 1024)?; // 512MB

// This will fail if it exceeds the limit
match manager.allocate(600 * 1024 * 1024) { // 600MB
    Ok(_) => println!("Allocation succeeded"),
    Err(e) => println!("Allocation failed: {}", e),
}
```

## StreamingProcessor

The `StreamingProcessor` enables processing of large files that don't fit in memory by reading and processing data in chunks.

### Features

- **Memory-Mapped Files**: Efficient file access using memory mapping
- **Chunked Processing**: Process files in configurable chunk sizes
- **Memory Monitoring**: Ensure chunks don't exceed memory limits
- **Error Handling**: Robust error handling for file operations

### Usage

```rust
use chronos_time_series::performance::memory::StreamingProcessor;
use std::path::Path;
use std::sync::Arc;

let memory_manager = Arc::new(MemoryManager::new(&config)?);
let processor = StreamingProcessor::new(memory_manager, 1024 * 1024); // 1MB chunks

// Process a large CSV file
let results = processor.process_file(Path::new("large_dataset.csv"), |chunk| {
    // Process each chunk
    let text = std::str::from_utf8(chunk)?;
    let lines = text.lines().count();
    Ok(lines)
})?;

println!("Total chunks processed: {}", results.len());
println!("Total lines: {}", results.iter().sum::<usize>());
```

### Stream Processing Data

Process large datasets in memory-efficient chunks:

```rust
let large_dataset: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();

processor.stream_process(large_dataset, |chunk| {
    // Process each chunk
    let sum: f64 = chunk.iter().sum();
    println!("Chunk sum: {}", sum);
    Ok(())
})?;
```

## LazyDataLoader

The `LazyDataLoader` implements on-demand data loading to minimize memory usage.

### Features

- **Lazy Loading**: Load data only when needed
- **Caching**: Cache frequently accessed data
- **Background Preloading**: Load data in background threads
- **Memory Monitoring**: Respect memory limits when caching

### Usage

```rust
use chronos_time_series::performance::memory::LazyDataLoader;

let memory_manager = Arc::new(MemoryManager::new(&config)?);
let loader = LazyDataLoader::new(memory_manager, 100 * 1024 * 1024); // 100MB cache limit

// Load data lazily
let data = loader.load_lazy("dataset_key", || {
    // This closure is called only if data isn't cached
    expensive_data_loading_operation()
})?;

// Preload data in background
loader.preload("future_dataset".to_string(), || {
    // This runs in a background thread
    load_dataset_from_network()
})?;
```

## CompactTimeSeries

The `CompactTimeSeries` provides a memory-efficient representation of time series data using compression techniques.

### Features

- **Timestamp Compression**: Store timestamp deltas instead of absolute values
- **Value Quantization**: Compress values using 16-bit quantization
- **Memory Footprint Reduction**: Significant memory savings for large datasets
- **Lossless Reconstruction**: Reconstruct original time series with minimal loss

### Usage

```rust
use chronos_time_series::performance::memory::CompactTimeSeries;

// Create a compact representation
let compact_ts = CompactTimeSeries::from_timeseries(&original_ts)?;

// Check compression ratio
let original_size = original_ts.len() * 16; // 8 bytes timestamp + 8 bytes value
let compression_ratio = compact_ts.compression_ratio(original_size);
println!("Compression ratio: {:.1}%", compression_ratio * 100.0);

// Use compact version for storage or transmission
let serialized = bincode::serialize(&compact_ts)?;

// Reconstruct when needed
let restored_ts = compact_ts.to_timeseries()?;
```

### Compression Details

The compression algorithm uses:

1. **Delta Encoding**: Store time differences instead of absolute timestamps
2. **Quantization**: Map float values to 16-bit integers
3. **Range Preservation**: Store min/max values for accurate reconstruction

```rust
// Example compression stats
let stats = compact_ts.compression_stats();
println!("Original size: {} bytes", stats.original_size);
println!("Compressed size: {} bytes", stats.compressed_size);
println!("Space saved: {} bytes", stats.bytes_saved);
println!("Compression ratio: {:.1}%", stats.compression_ratio * 100.0);
```

## Memory Optimization Strategies

### 1. Choose Appropriate Chunk Sizes

```rust
// Get optimal chunk size based on available memory
let optimal_chunk = manager.get_optimal_chunk_size(dataset_size);
println!("Recommended chunk size: {}", optimal_chunk);
```

### 2. Use Compact Representations

```rust
// For large datasets, use compact representation
if original_ts.len() > 100_000 {
    let compact = CompactTimeSeries::from_timeseries(&original_ts)?;
    // Use compact version for storage/processing
}
```

### 3. Stream Large Files

```rust
// Instead of loading entire file into memory
let processor = StreamingProcessor::new(memory_manager, chunk_size);
processor.process_file(large_file_path, |chunk| {
    // Process incrementally
    process_chunk(chunk)
})?;
```

### 4. Monitor Memory Usage

```rust
// Regular memory monitoring
let stats = manager.get_stats();
if stats.memory_efficiency < 0.5 {
    println!("Warning: Low memory efficiency, consider optimization");
}

if stats.used_memory_mb > stats.available_memory_mb * 0.8 {
    println!("Warning: High memory usage, consider freeing memory");
}
```

## Configuration Options

### MemoryManager Configuration

```rust
let config = PerformanceConfig {
    max_memory_mb: 1024,        // Maximum memory usage in MB
    chunk_size: 10000,          // Default chunk size for processing
    ..PerformanceConfig::default()
};
```

### StreamingProcessor Configuration

```rust
// Chunk size affects memory usage vs. processing efficiency
let small_chunks = StreamingProcessor::new(manager, 64 * 1024);    // 64KB - low memory
let large_chunks = StreamingProcessor::new(manager, 10 * 1024 * 1024); // 10MB - high performance
```

## Error Handling

The memory management system provides specific error types:

```rust
use chronos_time_series::performance::PerformanceError;

match manager.allocate(size) {
    Ok(_) => println!("Allocation successful"),
    Err(e) => match e.downcast_ref::<PerformanceError>() {
        Some(PerformanceError::MemoryLimitExceeded(mb)) => {
            eprintln!("Memory limit exceeded: {} MB", mb);
            // Handle by reducing memory usage or increasing limit
        }
        Some(PerformanceError::MemoryError(msg)) => {
            eprintln!("Memory error: {}", msg);
        }
        _ => eprintln!("Other error: {}", e),
    }
}
```

## Best Practices

### 1. Set Realistic Memory Limits

```rust
// Leave some memory for the OS and other applications
let total_system_memory = 8192; // 8GB
let config = PerformanceConfig {
    max_memory_mb: total_system_memory * 3 / 4, // Use 75% of available memory
    ..PerformanceConfig::default()
};
```

### 2. Use Streaming for Large Files

```rust
// Rule of thumb: use streaming if file size > 1/4 of available memory
let file_size = std::fs::metadata(file_path)?.len() as usize;
let available_memory = manager.get_stats().available_memory_mb as usize * 1024 * 1024;

if file_size > available_memory / 4 {
    // Use streaming processor
    processor.process_file(file_path, |chunk| process_chunk(chunk))?;
} else {
    // Safe to load entire file
    let content = std::fs::read(file_path)?;
}
```

### 3. Monitor Memory Regularly

```rust
// Set up periodic memory monitoring
use std::time::{Duration, Instant};

let mut last_check = Instant::now();
let check_interval = Duration::from_secs(30); // Check every 30 seconds

// In your processing loop
if last_check.elapsed() >= check_interval {
    let stats = manager.get_stats();
    log::info!("Memory usage: {:.2}MB, efficiency: {:.1}%",
               stats.used_memory_mb, stats.memory_efficiency * 100.0);
    last_check = Instant::now();
}
```

### 4. Clean Up Properly

```rust
// Implement proper cleanup
struct DataProcessor {
    manager: Arc<MemoryManager>,
    allocated_size: usize,
}

impl Drop for DataProcessor {
    fn drop(&mut self) {
        if self.allocated_size > 0 {
            self.manager.deallocate(self.allocated_size);
        }
    }
}
```

## Performance Tips

### 1. Optimal Chunk Sizes

- **Small chunks (64KB-1MB)**: Low memory usage, higher overhead
- **Medium chunks (1MB-10MB)**: Balanced performance and memory usage
- **Large chunks (10MB+)**: High performance, high memory usage

### 2. Memory-Mapped Files

Memory-mapped files are most effective for:
- Sequential access patterns
- Large files (>100MB)
- Read-heavy workloads

### 3. Compact Time Series

Use `CompactTimeSeries` when:
- Dataset size > 1M points
- Memory is limited
- Data will be stored/transmitted
- Some precision loss is acceptable

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```rust
   // Reduce memory usage
   config.max_memory_mb = 512; // Reduce limit
   config.chunk_size = 5000;   // Smaller chunks
   ```

2. **Poor Performance**
   ```rust
   // Increase chunk size for better throughput
   config.chunk_size = 50000; // Larger chunks
   ```

3. **Memory Leaks**
   ```rust
   // Ensure proper deallocation
   manager.deallocate(previously_allocated_size);
   ```

4. **Fragmentation**
   ```rust
   // Use consistent allocation sizes
   let chunk_size = manager.get_optimal_chunk_size(data_size);
   ```

### Debug Information

Enable detailed memory logging:

```rust
log::debug!("Memory stats: {:?}", manager.get_stats());
log::debug!("Allocating {} bytes", size);
log::debug!("Current memory usage: {:.2}MB",
           manager.get_stats().used_memory_mb);
```