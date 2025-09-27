# Frequently Asked Questions (FAQ)

Common questions and answers about using Chronos for time series analysis.

## Table of Contents

1. [General Usage](#general-usage)
2. [Data Import and Format Issues](#data-import-and-format-issues)
3. [Analysis and Interpretation](#analysis-and-interpretation)
4. [Performance and Memory](#performance-and-memory)
5. [Configuration and Setup](#configuration-and-setup)
6. [Troubleshooting](#troubleshooting)

## General Usage

### Q: What file formats does Chronos support?

**A:** Chronos supports several common data formats:

- **CSV** - Most common, with configurable delimiters
- **JSON** - Structured data with nested time series
- **TSV** - Tab-separated values
- **Parquet** - High-performance columnar format (planned)
- **Excel** - .xlsx files (planned)

**Example CSV format:**
```csv
timestamp,value,category
2023-01-01 00:00:00,100.5,sales
2023-01-01 01:00:00,102.3,sales
```

### Q: How do I handle different time formats?

**A:** Chronos automatically detects most common time formats, including:

- ISO 8601: `2023-01-01T10:00:00Z`
- US format: `01/01/2023 10:00:00`
- European format: `01.01.2023 10:00:00`
- Unix timestamps: `1672574400`
- Date only: `2023-01-01`

If auto-detection fails, you can specify the format in your configuration:

```toml
[import]
time_format = "%Y-%m-%d %H:%M:%S"
```

### Q: Can I analyze multiple time series simultaneously?

**A:** Yes! Chronos supports several approaches:

1. **Multiple files**: Use the `--files` option with comma-separated file paths
2. **Multiple columns**: Specify multiple value columns in a single file
3. **Grouped data**: Use the `--group-by` option to analyze by categories

```bash
# Multiple files
chronos correlate --files "stock1.csv,stock2.csv,stock3.csv"

# Multiple columns in one file
chronos stats --file data.csv --value-columns "temperature,humidity,pressure"

# Grouped analysis
chronos stats --file sensors.csv --group-by sensor_id
```

### Q: What's the difference between `chronos` and other time series tools?

**A:** Chronos is designed for:

- **CLI-first workflow** - Perfect for automation and scripting
- **Comprehensive analysis** - Statistics, trends, seasonality, anomalies, forecasting in one tool
- **Performance** - Built with Rust for speed and memory efficiency
- **Extensibility** - Plugin system for custom analysis methods
- **Reproducibility** - Configuration-driven analysis for consistent results

## Data Import and Format Issues

### Q: How do I handle missing values in my data?

**A:** Chronos provides several strategies for missing values:

```bash
# Drop rows with missing values
chronos import --file data.csv --missing-value-policy drop

# Interpolate missing values
chronos import --file data.csv --missing-value-policy interpolate --interpolation-method linear

# Forward fill
chronos import --file data.csv --missing-value-policy forward_fill

# Fill with mean
chronos import --file data.csv --missing-value-policy mean
```

**Interpolation methods:**
- `linear` - Linear interpolation
- `spline` - Cubic spline interpolation
- `polynomial` - Polynomial interpolation
- `seasonal` - Seasonal interpolation

### Q: My timestamps are not being parsed correctly. What should I do?

**A:** Try these solutions:

1. **Check your data format**:
```bash
head -5 your_data.csv
```

2. **Specify the time column explicitly**:
```bash
chronos import --file data.csv --time-column "Date Time" --value-column value
```

3. **Set a custom time format** in your config:
```toml
[import]
time_format = "%d/%m/%Y %H:%M"
timezone = "UTC"
```

4. **Use verbose mode** to see parsing details:
```bash
chronos --verbose import --file data.csv --time-column timestamp --value-column value
```

### Q: How do I handle irregular time intervals?

**A:** For irregular data:

1. **Auto-detect frequency**:
```bash
chronos import --file data.csv --frequency auto
```

2. **Specify irregular frequency**:
```bash
chronos import --file data.csv --frequency irregular
```

3. **Resample to regular intervals**:
```bash
chronos import --file data.csv --resample-frequency 1H --resample-method mean
```

### Q: Can I import data from databases?

**A:** Currently, Chronos works with file-based data. For database integration:

1. **Export from database** to CSV:
```sql
-- PostgreSQL example
COPY (SELECT timestamp, value FROM my_table) TO '/tmp/data.csv' CSV HEADER;
```

2. **Use database client tools**:
```bash
# MySQL example
mysql -e "SELECT timestamp, value FROM my_table" --batch > data.tsv
chronos import --file data.tsv --format tsv
```

3. **Database plugin** (coming soon) will support direct connections.

## Analysis and Interpretation

### Q: How do I interpret the statistical results?

**A:** Key statistics and their meanings:

**Descriptive Statistics:**
- **Mean**: Average value
- **Median**: Middle value (less affected by outliers)
- **Standard Deviation**: Measure of variability
- **Skewness**: Asymmetry (positive = right tail, negative = left tail)
- **Kurtosis**: Tail heaviness (>3 = heavy tails, <3 = light tails)

**Time Series Specific:**
- **Autocorrelation**: Correlation with lagged versions
- **Partial Autocorrelation**: Direct correlation at specific lags
- **Stationarity**: Whether statistical properties change over time

**Example interpretation:**
```json
{
  "mean": 100.5,
  "std_dev": 15.2,
  "skewness": 0.3,
  "kurtosis": 2.1,
  "stationarity": {
    "adf_statistic": -3.45,
    "p_value": 0.01,
    "is_stationary": true
  }
}
```

This shows: average value 100.5, moderate variability, slightly right-skewed distribution, normal tail behavior, and stationary series.

### Q: What does "stationary" mean and why does it matter?

**A:** Stationarity means the time series has constant statistical properties over time:

- **Mean doesn't change**: No long-term trend
- **Variance is constant**: Consistent variability
- **Covariance depends only on lag**: Pattern repeats similarly

**Why it matters:**
- Many forecasting models assume stationarity
- Non-stationary data can lead to spurious correlations
- Differencing or detrending can make data stationary

**Check stationarity:**
```bash
chronos stats --file data.csv --stationarity-test
```

**Make data stationary:**
```bash
# First difference
chronos trend --file data.csv --detrend --detrending-method difference

# Remove trend
chronos trend --file data.csv --detrend --detrending-method linear
```

### Q: How do I choose the right forecasting method?

**A:** Choose based on your data characteristics:

| Data Type | Recommended Method | Notes |
|-----------|-------------------|-------|
| Simple trend, no seasonality | Linear Regression | Fast, interpretable |
| Trend + seasonality | ARIMA or ETS | Classical time series |
| Complex patterns, holidays | Prophet | Handles irregularities well |
| Multiple related series | VAR/VECM | Captures relationships |
| High frequency, streaming | Simple exponential smoothing | Fast updates |

**Start with ensemble forecasting** to compare methods:
```bash
chronos forecast --file data.csv --method ensemble --horizon 30
```

### Q: How sensitive are anomaly detection results?

**A:** Anomaly detection sensitivity depends on:

1. **Method choice**:
   - `statistical` - More false positives, catches subtle anomalies
   - `isolation_forest` - Balanced, good for multivariate
   - `ensemble` - Most robust, recommended

2. **Sensitivity parameter**:
```bash
# Conservative (fewer anomalies)
chronos anomaly --file data.csv --sensitivity 0.01

# Aggressive (more anomalies)
chronos anomaly --file data.csv --sensitivity 0.10
```

3. **Seasonal adjustment**:
```bash
# Adjust for known patterns first
chronos anomaly --file data.csv --seasonal-adjustment
```

**Recommendation**: Start with ensemble method and 0.05 sensitivity, then adjust based on domain knowledge.

## Performance and Memory

### Q: Chronos is running slowly on my large dataset. How can I improve performance?

**A:** Several optimization strategies:

1. **Enable parallel processing**:
```bash
chronos config set performance.parallel_processing true
chronos config set performance.max_threads 8
```

2. **Use streaming mode for large files**:
```bash
chronos stats --file large_data.csv --streaming --chunk-size 50000
```

3. **Increase memory limits**:
```bash
chronos config set performance.memory_limit_mb 8192
```

4. **Enable caching**:
```bash
chronos config set performance.enable_caching true
```

5. **Process in chunks**:
```bash
# Split large file and process separately
split -l 100000 large_data.csv chunk_
for chunk in chunk_*; do
    chronos stats --file "$chunk" --output "${chunk}_stats.json"
done
```

### Q: I'm getting out of memory errors. What should I do?

**A:** Memory management strategies:

1. **Use streaming mode**:
```bash
chronos import --file large_data.csv --streaming
```

2. **Reduce memory usage**:
```bash
chronos config set performance.memory_limit_mb 2048
chronos config set performance.use_memory_mapping true
```

3. **Process smaller chunks**:
```bash
chronos stats --file data.csv --streaming --chunk-size 10000
```

4. **Sample large datasets**:
```bash
# Analyze every 10th row
chronos import --file large_data.csv --sample-rate 0.1
```

### Q: How can I monitor Chronos performance?

**A:** Performance monitoring options:

1. **Verbose output**:
```bash
chronos --verbose stats --file data.csv
```

2. **Enable performance tracking**:
```bash
chronos config set performance.enable_profiling true
```

3. **Check resource usage**:
```bash
# Linux/macOS
time chronos stats --file data.csv
/usr/bin/time -v chronos stats --file data.csv
```

4. **Use performance reporting**:
```bash
chronos report --file data.csv --include-performance-metrics
```

## Configuration and Setup

### Q: Where is my configuration file located?

**A:** Default locations:

- **Linux**: `~/.config/chronos/config.toml`
- **macOS**: `~/.config/chronos/config.toml`
- **Windows**: `%APPDATA%\chronos\config.toml`

**Find your config**:
```bash
chronos config show | head -5
```

### Q: How do I create different analysis profiles?

**A:** Profiles allow different settings for different use cases:

1. **Create a new profile**:
```toml
# ~/.config/chronos/config.toml
[profiles.financial]
[profiles.financial.analysis]
default_confidence_level = 0.99
handle_missing_values = "drop"

[profiles.financial.visualization]
default_theme = "professional"

[profiles.iot]
[profiles.iot.analysis]
default_confidence_level = 0.95
handle_missing_values = "interpolate"
```

2. **Use specific profile**:
```bash
chronos --config-profile financial stats --file stock_data.csv
```

3. **Set default profile**:
```bash
chronos config set metadata.active_profile financial
```

### Q: Can I use Chronos in automated workflows?

**A:** Yes! Chronos is designed for automation:

1. **Configuration-driven analysis**:
```toml
# analysis_config.toml
[data]
file = "daily_data.csv"
time_column = "timestamp"
value_column = "value"

[analysis]
statistics = true
trend_detection = true
anomaly_detection = true

[output]
directory = "./reports"
format = "json"
```

```bash
chronos --config analysis_config.toml
```

2. **Scripted workflows**:
```bash
#!/bin/bash
# daily_analysis.sh
TODAY=$(date +%Y-%m-%d)
chronos import --file "data_${TODAY}.csv" --output "processed_${TODAY}.csv"
chronos stats --file "processed_${TODAY}.csv" --output "stats_${TODAY}.json"
chronos report --file "processed_${TODAY}.csv" --output "report_${TODAY}.html"
```

3. **Exit codes for error handling**:
```bash
if chronos stats --file data.csv --output stats.json; then
    echo "Analysis successful"
    chronos report --file data.csv --output report.html
else
    echo "Analysis failed"
    exit 1
fi
```

## Troubleshooting

### Q: I get "command not found" error when running chronos.

**A:** Installation troubleshooting:

1. **Check if chronos is installed**:
```bash
which chronos
ls -la target/release/chronos  # if built from source
```

2. **Add to PATH**:
```bash
export PATH="$PATH:$(pwd)/target/release"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

3. **Rebuild if necessary**:
```bash
cd ts-rs
cargo build --release
```

### Q: I get compilation errors when building from source.

**A:** Common compilation issues:

1. **Update Rust**:
```bash
rustup update stable
```

2. **Install required system libraries**:
```bash
# Ubuntu/Debian
sudo apt install build-essential pkg-config libssl-dev

# macOS
brew install pkg-config openssl
```

3. **Clear cache and rebuild**:
```bash
cargo clean
cargo build --release
```

### Q: The analysis results look wrong or unexpected.

**A:** Validation steps:

1. **Check your data**:
```bash
chronos import --file data.csv --validate
```

2. **Verify time column parsing**:
```bash
chronos --verbose import --file data.csv --time-column timestamp
```

3. **Check for data quality issues**:
```bash
chronos stats --file data.csv --missing-value-analysis --outlier-detection
```

4. **Compare with known results**:
```bash
# Use verbose mode to see intermediate steps
chronos --verbose stats --file data.csv
```

### Q: Interactive mode is not working properly.

**A:** Interactive mode troubleshooting:

1. **Check terminal compatibility**:
```bash
echo $TERM
# Should be xterm-256color or similar
```

2. **Use explicit interactive flag**:
```bash
chronos --interactive
```

3. **Check for conflicting options**:
```bash
# Don't use --quiet with --interactive
chronos --interactive  # (not chronos --quiet --interactive)
```

### Q: How do I report a bug or request a feature?

**A:** Contributing and reporting:

1. **Check existing issues**: [GitHub Issues](https://github.com/jpequegn/ts-rs/issues)

2. **Report bugs with details**:
   - Operating system and version
   - Rust version (`rustc --version`)
   - Chronos version (`chronos --version`)
   - Complete error message
   - Minimal reproducible example
   - Expected vs. actual behavior

3. **Feature requests**:
   - Clear description of the need
   - Use cases and examples
   - Suggested implementation approach

4. **Contributing code**:
   - Fork the repository
   - Create feature branch
   - Add tests for new functionality
   - Submit pull request

## Still Need Help?

- Check the [Troubleshooting Guide](troubleshooting.md) for detailed solutions
- Read the [Tutorial](tutorial.md) for hands-on examples
- Browse the [Examples Gallery](../examples/) for use case specific guidance
- Join our community discussions
- Create an issue on [GitHub](https://github.com/jpequegn/ts-rs/issues)