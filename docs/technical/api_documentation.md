# API Documentation

Complete API reference for using Chronos as a Rust library in your applications.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Data Structures](#core-data-structures)
3. [Import and Preprocessing](#import-and-preprocessing)
4. [Statistical Analysis](#statistical-analysis)
5. [Time Series Analysis](#time-series-analysis)
6. [Visualization](#visualization)
7. [Configuration](#configuration)
8. [Performance](#performance)
9. [Error Handling](#error-handling)
10. [Examples](#examples)

## Getting Started

Add Chronos to your `Cargo.toml`:

```toml
[dependencies]
chronos = { git = "https://github.com/jpequegn/ts-rs" }

# Optional features
chronos = { git = "https://github.com/jpequegn/ts-rs", features = ["plotting", "parallel"] }
```

Basic usage:

```rust
use chronos::{TimeSeries, import_csv, analyze_timeseries};
use anyhow::Result;

fn main() -> Result<()> {
    // Import data
    let ts = import_csv("data.csv", "timestamp", "value", None)?;

    // Perform analysis
    let stats = analyze_timeseries(&ts, None)?;
    println!("Mean: {:.2}", stats.descriptive.mean);

    Ok(())
}
```

## Core Data Structures

### TimeSeries

The fundamental data structure representing a time series.

```rust
use chronos::{TimeSeries, TimeSeriesError};
use chrono::{DateTime, Utc};

// Create from vectors
let timestamps: Vec<DateTime<Utc>> = vec![/* ... */];
let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
let ts = TimeSeries::new(timestamps, values)?;

// Access properties
println!("Length: {}", ts.len());
println!("Frequency: {:?}", ts.frequency());
println!("Start: {}", ts.start_time());
println!("End: {}", ts.end_time());

// Access data
let (times, vals) = ts.data();
for (time, value) in times.iter().zip(vals.iter()) {
    println!("{}: {}", time, value);
}
```

**Methods:**

| Method | Description | Return Type |
|--------|-------------|-------------|
| `new(timestamps, values)` | Create new time series | `Result<TimeSeries>` |
| `len()` | Number of data points | `usize` |
| `is_empty()` | Check if empty | `bool` |
| `frequency()` | Detected frequency | `Option<Frequency>` |
| `start_time()` | First timestamp | `DateTime<Utc>` |
| `end_time()` | Last timestamp | `DateTime<Utc>` |
| `data()` | Get timestamps and values | `(&[DateTime<Utc>], &[f64])` |
| `values()` | Get values only | `&[f64]` |
| `timestamps()` | Get timestamps only | `&[DateTime<Utc>]` |
| `subset(start, end)` | Extract time range | `Result<TimeSeries>` |
| `resample(frequency)` | Resample to new frequency | `Result<TimeSeries>` |

### Frequency

Enumeration of supported time frequencies.

```rust
use chronos::Frequency;

// Common frequencies
let freq = Frequency::Daily;
let freq = Frequency::Hourly;
let freq = Frequency::Minutely;

// Custom intervals
let freq = Frequency::Custom {
    interval: 15,
    unit: "minutes".to_string()
};

// Parse from string
let freq: Frequency = "1H".parse()?;
let freq: Frequency = "5min".parse()?;
```

### MissingValuePolicy

Strategy for handling missing values.

```rust
use chronos::MissingValuePolicy;

let policy = MissingValuePolicy::Drop;           // Remove missing values
let policy = MissingValuePolicy::Interpolate;    // Linear interpolation
let policy = MissingValuePolicy::ForwardFill;    // Carry forward last value
let policy = MissingValuePolicy::BackwardFill;   // Carry backward next value
let policy = MissingValuePolicy::Mean;           // Fill with mean
let policy = MissingValuePolicy::Zero;           // Fill with zero
```

## Import and Preprocessing

### Importing Data

```rust
use chronos::{import_csv, import_json, ImportConfig, ImportResult};

// Simple CSV import
let ts = import_csv("data.csv", "timestamp", "value", None)?;

// Advanced CSV import with configuration
let config = ImportConfig {
    time_format: Some("%Y-%m-%d %H:%M:%S".to_string()),
    frequency: Some(Frequency::Hourly),
    missing_value_policy: MissingValuePolicy::Interpolate,
    validate: true,
    ..Default::default()
};

let result: ImportResult = import_csv("data.csv", "timestamp", "value", Some(config))?;
let ts = result.time_series;
println!("Imported {} points", result.stats.total_rows);
println!("Skipped {} invalid rows", result.stats.invalid_rows);

// JSON import
let ts = import_json("data.json", "timestamp", "value", None)?;
```

**ImportConfig Options:**

```rust
pub struct ImportConfig {
    pub time_format: Option<String>,
    pub frequency: Option<Frequency>,
    pub missing_value_policy: MissingValuePolicy,
    pub timezone: Option<String>,
    pub validate: bool,
    pub skip_errors: bool,
    pub max_errors: usize,
    pub sample_rate: Option<f64>,
    pub decimal_separator: char,
    pub thousands_separator: Option<char>,
}
```

### Preprocessing

```rust
use chronos::{preprocess_timeseries, PreprocessingConfig, PreprocessingResult};

let config = PreprocessingConfig {
    remove_outliers: true,
    outlier_method: OutlierMethod::IQR,
    normalize: true,
    normalization_method: NormalizationMethod::ZScore,
    detrend: false,
    seasonal_adjust: false,
    ..Default::default()
};

let result: PreprocessingResult = preprocess_timeseries(&ts, config)?;
let preprocessed_ts = result.time_series;
println!("Removed {} outliers", result.outliers_removed);
```

## Statistical Analysis

### Descriptive Statistics

```rust
use chronos::{compute_descriptive_stats, DescriptiveStats};

let stats: DescriptiveStats = compute_descriptive_stats(&ts.values())?;

println!("Mean: {:.4}", stats.mean);
println!("Median: {:.4}", stats.median);
println!("Std Dev: {:.4}", stats.std_dev);
println!("Skewness: {:.4}", stats.skewness);
println!("Kurtosis: {:.4}", stats.kurtosis);
println!("Min: {:.4}", stats.min);
println!("Max: {:.4}", stats.max);
```

### Distribution Analysis

```rust
use chronos::{compute_distribution_analysis, DistributionAnalysis};

let analysis: DistributionAnalysis = compute_distribution_analysis(&ts.values(), 0.95)?;

println!("Normality test p-value: {:.4}", analysis.normality_test.p_value);
println!("Is normal: {}", analysis.normality_test.is_normal);

// Quantiles
for (percentile, value) in analysis.quantiles.iter() {
    println!("{}th percentile: {:.4}", percentile * 100.0, value);
}
```

### Time Series Statistics

```rust
use chronos::{compute_autocorrelation, compute_partial_autocorrelation,
              test_stationarity, StationarityTest};

// Autocorrelation
let acf = compute_autocorrelation(&ts.values(), 20)?; // 20 lags
for (lag, correlation) in acf.iter().enumerate() {
    println!("Lag {}: {:.4}", lag, correlation);
}

// Partial autocorrelation
let pacf = compute_partial_autocorrelation(&ts.values(), 20)?;

// Stationarity testing
let stationarity: StationarityTest = test_stationarity(&ts.values())?;
println!("ADF statistic: {:.4}", stationarity.adf_statistic);
println!("P-value: {:.4}", stationarity.p_value);
println!("Is stationary: {}", stationarity.is_stationary);
```

### Comprehensive Analysis

```rust
use chronos::{analyze_timeseries, StatisticalAnalysisResult, AnalysisMetadata};

// Complete statistical analysis
let metadata = AnalysisMetadata {
    confidence_level: 0.95,
    include_distributions: true,
    include_time_series_stats: true,
    ..Default::default()
};

let result: StatisticalAnalysisResult = analyze_timeseries(&ts, Some(metadata))?;

// Access all results
println!("Descriptive stats: {:?}", result.descriptive);
println!("Distribution analysis: {:?}", result.distribution);
println!("Time series stats: {:?}", result.time_series);
println!("Stationarity: {:?}", result.stationarity);
```

## Time Series Analysis

### Trend Analysis

```rust
use chronos::trend::{analyze_comprehensive, TrendAnalysisConfig, DetrendingMethod,
                    DecompositionMethod, ComprehensiveTrendAnalysis};

let config = TrendAnalysisConfig {
    decomposition_method: DecompositionMethod::STL,
    detrending_method: DetrendingMethod::Linear,
    detect_change_points: true,
    test_significance: true,
    confidence_level: 0.95,
    ..Default::default()
};

let analysis: ComprehensiveTrendAnalysis = analyze_comprehensive(&ts, config)?;

println!("Trend direction: {:?}", analysis.summary.direction);
println!("Trend strength: {:.4}", analysis.summary.strength);

if let Some(decomp) = analysis.decomposition {
    println!("Trend component available");
    println!("Seasonal component available");
    println!("Residual component available");
}
```

### Seasonality Analysis

```rust
use chronos::seasonality::{analyze_comprehensive_seasonality, SeasonalityAnalysisConfig,
                          SeasonalityMethod, ComprehensiveSeasonalityAnalysis};

let config = SeasonalityAnalysisConfig {
    methods: vec![SeasonalityMethod::Fourier, SeasonalityMethod::STL],
    test_periods: vec![7, 30, 365], // Daily, monthly, yearly patterns
    confidence_level: 0.95,
    min_strength_threshold: 0.1,
    ..Default::default()
};

let analysis: ComprehensiveSeasonalityAnalysis =
    analyze_comprehensive_seasonality(&ts, config)?;

for period in analysis.detected_periods {
    println!("Period: {} days, Strength: {:.4}", period.period, period.strength);
}
```

### Anomaly Detection

```rust
use chronos::anomaly::{detect_anomalies, AnomalyDetectionConfig, AnomalyMethod,
                      ThresholdConfig, Anomaly};

let config = AnomalyDetectionConfig {
    methods: vec![AnomalyMethod::IsolationForest, AnomalyMethod::StatisticalThreshold],
    threshold: ThresholdConfig {
        percentile: 95.0,
        sensitivity: 0.05,
        ..Default::default()
    },
    seasonal_adjustment: true,
    contextual: true,
    ..Default::default()
};

let anomalies: Vec<Anomaly> = detect_anomalies(&ts, config)?;

for anomaly in anomalies {
    println!("Anomaly at {}: value={:.4}, score={:.4}",
             anomaly.timestamp, anomaly.value, anomaly.score);
}
```

### Forecasting

```rust
use chronos::forecasting::{forecast_timeseries, ForecastConfig, ForecastMethod,
                          ForecastResult, EvaluationConfig};

let config = ForecastConfig {
    method: ForecastMethod::ARIMA,
    horizon: 30,
    confidence_level: 0.95,
    include_intervals: true,
    seasonal_periods: vec![7], // Weekly seasonality
    ..Default::default()
};

let forecast: ForecastResult = forecast_timeseries(&ts, config)?;

println!("Forecast values: {:?}", forecast.values);
println!("Lower bounds: {:?}", forecast.lower_bounds);
println!("Upper bounds: {:?}", forecast.upper_bounds);

// Evaluate forecast quality
if let Some(evaluation) = forecast.evaluation {
    println!("MAPE: {:.2}%", evaluation.mape);
    println!("RMSE: {:.4}", evaluation.rmse);
}
```

### Correlation Analysis

```rust
use chronos::correlation::{analyze_correlations, AnalysisConfig, CorrelationType,
                          CorrelationAnalysisResult};

let ts1 = import_csv("series1.csv", "timestamp", "value", None)?;
let ts2 = import_csv("series2.csv", "timestamp", "value", None)?;
let series = vec![ts1, ts2];

let config = AnalysisConfig {
    correlation_type: CorrelationType::Pearson,
    include_rolling_correlation: true,
    rolling_window: 30,
    test_granger_causality: true,
    test_cointegration: true,
    ..Default::default()
};

let result: CorrelationAnalysisResult = analyze_correlations(&series, config)?;

println!("Correlation matrix: {:?}", result.correlation_matrix);

if let Some(granger) = result.granger_causality {
    println!("Granger causality p-value: {:.4}", granger.p_value);
}
```

## Visualization

### Basic Plotting

```rust
use chronos::plotting::{plot, PlotConfig, PlotType, Theme, ExportFormat};

let config = PlotConfig {
    plot_type: PlotType::Line,
    theme: Theme::Dark,
    width: 1200,
    height: 800,
    title: Some("Time Series Analysis".to_string()),
    ..Default::default()
};

// Create and save plot
plot(&ts, config, "output.png", ExportFormat::PNG)?;
```

### Advanced Plotting

```rust
use chronos::plotting::{create_line_plot, create_decomposition_plot, create_forecast_plot,
                       PlotData, PlotSeries, customize_styling};

// Multiple series plot
let series1 = PlotSeries {
    name: "Series 1".to_string(),
    data: ts1.into(),
    color: Some("#1f77b4".to_string()),
};

let series2 = PlotSeries {
    name: "Series 2".to_string(),
    data: ts2.into(),
    color: Some("#ff7f0e".to_string()),
};

let plot_data = PlotData {
    series: vec![series1, series2],
    title: "Multiple Time Series".to_string(),
    ..Default::default()
};

create_line_plot(&plot_data, "multi_series.png")?;

// Forecast visualization
if let Some(forecast_result) = forecast_result {
    create_forecast_plot(&ts, &forecast_result, "forecast.png")?;
}
```

## Configuration

### Configuration Management

```rust
use chronos::config::{Config, ConfigLoader, AnalysisConfig, VisualizationConfig};

// Load configuration
let loader = ConfigLoader::new();
let config: Config = loader.load()?; // Load from default location

// Or load from specific file
let config: Config = loader.load_with_file("custom_config.toml")?;

// Access configuration sections
let analysis_config = &config.analysis;
let viz_config = &config.visualization;

println!("Default confidence level: {}", analysis_config.default_confidence_level);
println!("Default theme: {:?}", viz_config.default_theme);
```

### Custom Configuration

```rust
use chronos::config::{Config, ConfigMetadata, AnalysisConfig, OutputConfig};

let config = Config {
    metadata: ConfigMetadata {
        active_profile: "custom".to_string(),
        version: "0.1.0".to_string(),
    },
    analysis: AnalysisConfig {
        default_confidence_level: 0.99,
        auto_detect_frequency: true,
        handle_missing_values: "interpolate".to_string(),
        ..Default::default()
    },
    output: OutputConfig {
        default_directory: "/tmp/chronos".to_string(),
        default_format: "json".to_string(),
        ..Default::default()
    },
    ..Default::default()
};

// Save configuration
config.save_to_file("my_config.toml")?;
```

## Performance

### Parallel Processing

```rust
use chronos::performance::{ParallelProcessor, ParallelConfig};

let config = ParallelConfig {
    max_threads: Some(8),
    chunk_size: 10000,
    ..Default::default()
};

let processor = ParallelProcessor::new(config);

// Parallel statistical analysis
let results = processor.analyze_multiple_series(&time_series_vec)?;
```

### Memory Management

```rust
use chronos::performance::{MemoryManager, StreamingProcessor, LazyDataLoader};

// For large datasets, use streaming
let processor = StreamingProcessor::new(chunk_size: 50000);
let stats = processor.compute_stats_streaming("large_file.csv")?;

// Lazy loading for memory efficiency
let loader = LazyDataLoader::new();
let ts = loader.load_lazy("huge_dataset.csv")?;
```

### Caching

```rust
use chronos::performance::{CacheManager, CacheConfig};

let cache_config = CacheConfig {
    max_size_mb: 1024,
    ttl_seconds: 3600,
    ..Default::default()
};

let cache = CacheManager::new(cache_config);

// Cache analysis results
let cache_key = format!("stats_{}", file_hash);
let stats = cache.get_or_compute(&cache_key, || {
    analyze_timeseries(&ts, None)
})?;
```

## Error Handling

### Error Types

```rust
use chronos::{TimeSeriesError, Result};

fn handle_errors() -> Result<()> {
    match some_operation() {
        Ok(result) => {
            println!("Success: {:?}", result);
        }
        Err(TimeSeriesError::Validation(msg)) => {
            eprintln!("Validation error: {}", msg);
        }
        Err(TimeSeriesError::DataInconsistency(msg)) => {
            eprintln!("Data inconsistency: {}", msg);
        }
        Err(TimeSeriesError::InvalidTimestamp(msg)) => {
            eprintln!("Invalid timestamp: {}", msg);
        }
        Err(TimeSeriesError::MissingData(msg)) => {
            eprintln!("Missing data: {}", msg);
        }
        Err(TimeSeriesError::Analysis(msg)) => {
            eprintln!("Analysis error: {}", msg);
        }
        Err(e) => {
            eprintln!("Other error: {}", e);
        }
    }
    Ok(())
}
```

### Error Creation

```rust
use chronos::TimeSeriesError;

// Create custom errors
let error = TimeSeriesError::validation("Invalid data format");
let error = TimeSeriesError::missing_data("No data points found");
let error = TimeSeriesError::analysis("Convergence failed");

return Err(error.into());
```

## Examples

### Complete Analysis Pipeline

```rust
use chronos::*;
use anyhow::Result;

fn complete_analysis_example() -> Result<()> {
    // 1. Import data
    let config = ImportConfig {
        missing_value_policy: MissingValuePolicy::Interpolate,
        validate: true,
        ..Default::default()
    };

    let import_result = import_csv("data.csv", "timestamp", "value", Some(config))?;
    let ts = import_result.time_series;

    // 2. Preprocess
    let preprocess_config = PreprocessingConfig {
        remove_outliers: true,
        normalize: true,
        ..Default::default()
    };

    let preprocess_result = preprocess_timeseries(&ts, preprocess_config)?;
    let clean_ts = preprocess_result.time_series;

    // 3. Statistical analysis
    let stats = analyze_timeseries(&clean_ts, None)?;
    println!("Mean: {:.4}, Std Dev: {:.4}", stats.descriptive.mean, stats.descriptive.std_dev);

    // 4. Trend analysis
    let trend_config = TrendAnalysisConfig::default();
    let trend_analysis = analyze_comprehensive(&clean_ts, trend_config)?;
    println!("Trend strength: {:.4}", trend_analysis.summary.strength);

    // 5. Seasonality detection
    let seasonal_config = SeasonalityAnalysisConfig::default();
    let seasonality = analyze_comprehensive_seasonality(&clean_ts, seasonal_config)?;
    println!("Detected {} seasonal periods", seasonality.detected_periods.len());

    // 6. Anomaly detection
    let anomaly_config = AnomalyDetectionConfig::default();
    let anomalies = detect_anomalies(&clean_ts, anomaly_config)?;
    println!("Found {} anomalies", anomalies.len());

    // 7. Forecasting
    let forecast_config = ForecastConfig {
        horizon: 30,
        confidence_level: 0.95,
        ..Default::default()
    };
    let forecast = forecast_timeseries(&clean_ts, forecast_config)?;
    println!("Generated forecast for {} periods", forecast.values.len());

    // 8. Visualization
    let plot_config = PlotConfig {
        plot_type: PlotType::Line,
        theme: Theme::Dark,
        ..Default::default()
    };
    plot(&clean_ts, plot_config, "analysis_result.png", ExportFormat::PNG)?;

    Ok(())
}
```

### Custom Analysis Function

```rust
use chronos::*;

fn custom_analysis(ts: &TimeSeries) -> Result<f64> {
    // Custom metric: ratio of volatility to trend strength

    // Get basic statistics
    let stats = compute_descriptive_stats(ts.values())?;
    let volatility = stats.std_dev;

    // Get trend analysis
    let trend_config = TrendAnalysisConfig::default();
    let trend = analyze_comprehensive(ts, trend_config)?;
    let trend_strength = trend.summary.strength;

    // Avoid division by zero
    if trend_strength < 0.001 {
        return Ok(f64::INFINITY);
    }

    let volatility_ratio = volatility / trend_strength;
    Ok(volatility_ratio)
}

// Usage
let ratio = custom_analysis(&time_series)?;
println!("Volatility to trend ratio: {:.4}", ratio);
```

### Streaming Analysis

```rust
use chronos::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn streaming_analysis_example() -> Result<()> {
    let file = File::open("large_dataset.csv")?;
    let reader = BufReader::new(file);

    let mut running_mean = 0.0;
    let mut count = 0;
    let chunk_size = 1000;
    let mut chunk_values = Vec::with_capacity(chunk_size);

    for line in reader.lines().skip(1) { // Skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if let Ok(value) = parts[1].parse::<f64>() {
            chunk_values.push(value);

            if chunk_values.len() == chunk_size {
                // Process chunk
                let chunk_stats = compute_descriptive_stats(&chunk_values)?;

                // Update running statistics
                running_mean = (running_mean * count as f64 + chunk_stats.mean * chunk_size as f64)
                               / (count + chunk_size) as f64;
                count += chunk_size;

                chunk_values.clear();

                println!("Processed {} points, running mean: {:.4}", count, running_mean);
            }
        }
    }

    // Process remaining values
    if !chunk_values.is_empty() {
        let chunk_stats = compute_descriptive_stats(&chunk_values)?;
        running_mean = (running_mean * count as f64 + chunk_stats.mean * chunk_values.len() as f64)
                       / (count + chunk_values.len()) as f64;
    }

    println!("Final mean: {:.4}", running_mean);
    Ok(())
}
```

## Additional Resources

- **Source Code**: Browse the implementation in `src/lib.rs` and module files
- **Tests**: See `tests/` directory for more usage examples
- **CLI Source**: Check `src/cli/` for command-line interface implementation
- **Examples**: Complete examples in `examples/` directory

For the latest API documentation, run:
```bash
cargo doc --open
```

This will generate and open the full API documentation with all available methods and their signatures.