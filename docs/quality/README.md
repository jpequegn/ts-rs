# Data Quality Module

The Chronos data quality module provides comprehensive tools for assessing, monitoring, and improving the quality of time series data.

## Overview

This module offers several key capabilities:

- **Quality Assessment**: Comprehensive scoring across multiple dimensions (completeness, consistency, validity, timeliness, accuracy)
- **Data Profiling**: Automated analysis of data characteristics, gaps, and temporal coverage
- **Outlier Detection**: Advanced algorithms for identifying anomalous data points
- **Data Cleaning**: Smart imputation, outlier correction, and noise reduction techniques
- **Quality Monitoring**: Continuous tracking, trend detection, and alerting

## Quick Start

### Library Usage

```rust
use chronos::quality::*;
use chronos::TimeSeries;

// Load your time series data
let data = TimeSeries::from_csv("data.csv")?;

// Assess quality with default configuration
let config = QualityConfig::default();
let assessment = assess_quality(&data, &config)?;

println!("Overall quality score: {:.1}/100", assessment.overall_score);
println!("Completeness: {:.1}/100", assessment.dimension_scores.completeness);
println!("Consistency: {:.1}/100", assessment.dimension_scores.consistency);
```

### CLI Usage

```bash
# Assess data quality
chronos quality assess --detailed data.csv

# Generate data profile
chronos quality profile data.csv --output profile.json

# Clean data
chronos quality clean data.csv cleaned.csv --method conservative

# Fill missing data gaps
chronos quality fill-gaps data.csv filled.csv --method linear

# Set up quality monitoring
chronos quality monitor setup data.csv --config monitoring.toml

# Generate quality report
chronos quality report data.csv --template comprehensive --output report.html
```

## Quality Dimensions

The module assesses data quality across five key dimensions:

### 1. Completeness (0-100)
- **What it measures**: Presence of all expected data points
- **Factors**: Missing values, data gaps, temporal coverage
- **Good score**: > 95%

### 2. Consistency (0-100)
- **What it measures**: Data follows expected patterns and rules
- **Factors**: Duplicates, temporal order, value consistency
- **Good score**: > 90%

### 3. Validity (0-100)
- **What it measures**: Data values are within expected ranges
- **Factors**: Outliers, range violations, type correctness
- **Good score**: > 85%

### 4. Timeliness (0-100)
- **What it measures**: Data arrives at expected frequencies
- **Factors**: Regular spacing, frequency adherence, gaps
- **Good score**: > 90%

### 5. Accuracy (0-100)
- **What it measures**: Data correctness and statistical soundness
- **Factors**: Noise level, statistical anomalies, drift
- **Good score**: > 85%

## Configuration Profiles

The module provides several pre-configured quality profiles:

### Strict Profile
```rust
let config = QualityConfig::strict();
// Completeness threshold: 99%
// Outlier sensitivity: High
// Gap tolerance: < 1%
```

### Lenient Profile
```rust
let config = QualityConfig::lenient();
// Completeness threshold: 80%
// Outlier sensitivity: Low
// Gap tolerance: < 10%
```

### Custom Profile
```rust
let config = QualityConfig {
    completeness_threshold: 0.95,
    outlier_detection_methods: vec![
        OutlierMethod::ZScore { threshold: 3.0 },
        OutlierMethod::IQR { factor: 1.5 },
    ],
    acceptable_gap_ratio: 0.05,
    enable_cleaning: true,
};
```

## Module Structure

- **[Architecture](architecture.md)**: Technical architecture and design decisions
- **[API Reference](api_reference.md)**: Complete API documentation
- **[Configuration](configuration.md)**: Configuration options and profiles
- **[CLI Guide](cli_guide.md)**: Command-line interface usage
- **[Examples](examples/)**: Usage examples and tutorials
  - [Basic Assessment](examples/basic_assessment.md)
  - [Data Cleaning](examples/data_cleaning.md)
  - [Quality Monitoring](examples/quality_monitoring.md)
  - [Custom Configuration](examples/custom_configuration.md)
- **[Troubleshooting](troubleshooting.md)**: Common issues and solutions

## Performance

Target performance benchmarks (10,000 datapoints):

| Operation | Target Time | Actual |
|-----------|-------------|--------|
| Quality assessment | < 100ms | ~80ms |
| Data profiling | < 50ms | ~35ms |
| Outlier detection (per method) | < 100ms | ~60ms |
| Data cleaning | < 200ms | ~150ms |
| Monitoring update | < 50ms | ~25ms |

## Features

### Quality Assessment
- Multi-dimensional scoring system
- Configurable quality thresholds
- Detailed issue identification
- Automated recommendations

### Data Profiling
- Statistical summaries
- Temporal coverage analysis
- Gap detection and categorization
- Quality indicator calculation

### Outlier Detection
- Multiple detection methods:
  - Z-score (configurable threshold)
  - IQR (Interquartile Range)
  - Modified Z-score (MAD-based)
  - Temporal outliers
  - Ensemble methods
- Contextual outlier analysis
- Severity classification

### Data Cleaning
- Smart gap filling:
  - Linear interpolation
  - Forward fill
  - Backward fill
  - Spline interpolation
  - Seasonal decomposition
- Outlier correction
- Noise reduction
- Quality impact tracking

### Quality Monitoring
- Continuous quality tracking
- Quality degradation detection
- Configurable alerting
- Trend analysis
- Multiple notification channels

## Dependencies

This module depends on:
- `chronos::TimeSeries` - Core time series data structure
- `statrs` - Statistical calculations
- `chrono` - Date/time handling

## Examples

See the [examples directory](examples/) for detailed usage examples.

## Support

For issues, questions, or contributions, please see:
- [GitHub Issues](https://github.com/jpequegn/ts-rs/issues)
- [Troubleshooting Guide](troubleshooting.md)
- [API Documentation](api_reference.md)

## License

MIT License - See LICENSE file for details
