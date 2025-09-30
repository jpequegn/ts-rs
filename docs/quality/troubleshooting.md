# Troubleshooting Guide

Common issues and solutions when using the Chronos data quality module.

## Table of Contents

- [Quality Assessment Issues](#quality-assessment-issues)
- [Data Cleaning Issues](#data-cleaning-issues)
- [Outlier Detection Issues](#outlier-detection-issues)
- [Monitoring Issues](#monitoring-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)
- [CLI Issues](#cli-issues)

## Quality Assessment Issues

### Issue: Low Quality Score with Good-Looking Data

**Symptom**: Data appears clean visually but receives low quality score.

**Possible Causes**:
1. Configuration too strict for data characteristics
2. Natural variation mistaken for quality issues
3. Domain-specific patterns not accounted for

**Solutions**:

```rust
// Solution 1: Use lenient profile for initial assessment
let config = QualityConfig::lenient();
let assessment = assess_quality(&data, &config)?;

// Solution 2: Adjust outlier sensitivity
let config = QualityConfig {
    outlier_detection_methods: vec![
        OutlierMethod::ZScore { threshold: 4.0 },  // Less sensitive
    ],
    ..QualityConfig::default()
};

// Solution 3: Adjust dimension weights
let config = QualityConfig {
    weights: QualityWeights {
        completeness: 0.40,  // Emphasize what matters
        validity: 0.10,      // De-emphasize outliers
        ..QualityWeights::default()
    },
    ..QualityConfig::default()
};
```

### Issue: Assessment Fails with "Insufficient Data"

**Symptom**: `QualityError::InvalidData("Insufficient data for assessment")`

**Cause**: Dataset too small for meaningful statistical analysis

**Solution**:

```rust
// Check data size before assessment
if data.len() < 10 {
    eprintln!("Warning: Dataset too small for reliable assessment");
    eprintln!("Minimum recommended: 30 points for basic assessment");
    return Err(QualityError::InvalidData(
        "Dataset too small".to_string()
    ));
}

// Or use minimum_points configuration
let config = QualityConfig {
    minimum_points: 10,  // Lower threshold for small datasets
    ..QualityConfig::default()
};
```

### Issue: Inconsistent Scores Between Runs

**Symptom**: Same data produces different quality scores

**Cause**: Non-deterministic outlier detection or parallel processing

**Solution**:

```rust
// Use deterministic methods
let config = QualityConfig {
    outlier_detection_methods: vec![
        OutlierMethod::IQR { factor: 1.5 },  // Deterministic
    ],
    enable_parallel: false,  // Disable parallel processing
    ..QualityConfig::default()
};
```

## Data Cleaning Issues

### Issue: Cleaning Removes Too Much Data

**Symptom**: Cleaned dataset significantly smaller than original

**Solutions**:

```rust
// Solution 1: Use conservative cleaning
let config = CleaningConfig::conservative();

// Solution 2: Limit maximum modifications
let config = CleaningConfig {
    max_modifications: 0.05,  // Maximum 5% changes
    ..CleaningConfig::default()
};

// Solution 3: Use less aggressive outlier correction
let config = CleaningConfig {
    outlier_correction: OutlierCorrection::Cap,  // Don't remove
    ..CleaningConfig::default()
};
```

### Issue: Cleaning Doesn't Improve Quality Score

**Symptom**: Quality score unchanged or worse after cleaning

**Possible Causes**:
1. Cleaning introduced new issues
2. Wrong cleaning methods for data characteristics
3. Issues not addressable by cleaning

**Solutions**:

```rust
// Solution 1: Profile data first to understand issues
let profile = profile_timeseries(&data, &ProfilingConfig::default())?;
println!("Gaps: {}", profile.gaps.len());
println!("Missing: {}", profile.completeness.missing_count);

// Solution 2: Use method appropriate for data type
let config = if has_seasonal_pattern(&data) {
    CleaningConfig {
        gap_filling: ImputationMethod::Seasonal,
        ..CleaningConfig::default()
    }
} else {
    CleaningConfig {
        gap_filling: ImputationMethod::Linear,
        ..CleaningConfig::default()
    }
};

// Solution 3: Verify before and after
let before = assess_quality(&data, &QualityConfig::default())?;
let result = clean_timeseries(&data, &config)?;
let after = assess_quality(&result.cleaned_data, &QualityConfig::default())?;

println!("Quality change: {:.1} → {:.1}",
    before.overall_score, after.overall_score);

if after.overall_score < before.overall_score {
    eprintln!("Warning: Cleaning reduced quality score");
    // Don't use cleaned data
}
```

### Issue: "Maximum Modifications Exceeded"

**Symptom**: Cleaning fails with modification limit error

**Solutions**:

```rust
// Solution 1: Increase limit
let config = CleaningConfig {
    max_modifications: 0.20,  // Allow 20% modifications
    ..CleaningConfig::default()
};

// Solution 2: Clean in multiple passes
let pass1 = clean_timeseries(&data, &CleaningConfig {
    max_modifications: 0.10,
    ..CleaningConfig::default()
})?;

let pass2 = clean_timeseries(&pass1.cleaned_data, &CleaningConfig {
    max_modifications: 0.10,
    ..CleaningConfig::default()
})?;

// Solution 3: Target specific issues only
let config = CleaningConfig {
    fill_gaps: true,
    correct_outliers: false,  // Skip outlier correction
    reduce_noise: false,      // Skip noise reduction
    max_modifications: 0.05,
    ..CleaningConfig::default()
};
```

## Outlier Detection Issues

### Issue: Too Many Outliers Detected

**Symptom**: Significant portion of data flagged as outliers

**Solutions**:

```rust
// Solution 1: Reduce sensitivity
let config = OutlierConfig {
    method: OutlierMethod::ZScore { threshold: 4.0 },  // More lenient
    ..OutlierConfig::default()
};

// Solution 2: Use robust method
let config = OutlierConfig {
    method: OutlierMethod::ModifiedZScore { threshold: 3.5 },
    ..OutlierConfig::default()
};

// Solution 3: Filter by severity
let outliers = detect_outliers(&data, &config)?;
let severe_only: Vec<_> = outliers.outliers.iter()
    .filter(|o| matches!(o.severity, OutlierSeverity::Severe | OutlierSeverity::Extreme))
    .collect();

println!("Total outliers: {}", outliers.outliers.len());
println!("Severe outliers: {}", severe_only.len());
```

### Issue: Expected Outliers Not Detected

**Symptom**: Known anomalies not flagged as outliers

**Solutions**:

```rust
// Solution 1: Increase sensitivity
let config = OutlierConfig {
    method: OutlierMethod::ZScore { threshold: 2.0 },  // More sensitive
    ..OutlierConfig::default()
};

// Solution 2: Use multiple methods
let config = OutlierConfig {
    method: OutlierMethod::Ensemble,  // Combines multiple methods
    ..OutlierConfig::default()
};

// Solution 3: Try different methods
let methods = vec![
    OutlierMethod::ZScore { threshold: 3.0 },
    OutlierMethod::IQR { factor: 1.5 },
    OutlierMethod::ModifiedZScore { threshold: 3.5 },
    OutlierMethod::Temporal,
];

for method in methods {
    let config = OutlierConfig { method, ..OutlierConfig::default() };
    let outliers = detect_outliers(&data, &config)?;
    println!("{:?}: {} outliers", config.method, outliers.outliers.len());
}
```

## Monitoring Issues

### Issue: Monitoring Doesn't Detect Degradation

**Symptom**: Quality degrading but no alerts generated

**Solutions**:

```rust
// Solution 1: Lower alert thresholds
let config = MonitoringConfig {
    alert_thresholds: QualityThresholds {
        overall_quality: ThresholdConfig {
            warning_threshold: 85.0,  // Higher threshold
            critical_threshold: 70.0,
            degradation_rate_threshold: 3.0,  // More sensitive
        },
        ..QualityThresholds::default()
    },
    ..MonitoringConfig::default()
};

// Solution 2: Increase monitoring frequency
let config = MonitoringConfig {
    tracking_frequency: Duration::from_secs(1800),  // 30 minutes
    ..MonitoringConfig::default()
};

// Solution 3: Check trend detection
let tracker = QualityTracker::new(config);
// ... track data ...
let trend = tracker.detect_quality_degradation(&QualityThresholds::default());
println!("Quality trend: {:?}", trend);
```

### Issue: Too Many False Alert s

**Symptom**: Alerts for normal quality variation

**Solutions**:

```rust
// Solution 1: Increase thresholds
let config = MonitoringConfig {
    alert_thresholds: QualityThresholds {
        overall_quality: ThresholdConfig {
            warning_threshold: 70.0,  // Lower threshold
            degradation_rate_threshold: 10.0,  // Less sensitive
            ..ThresholdConfig::default()
        },
        ..QualityThresholds::default()
    },
    ..MonitoringConfig::default()
};

// Solution 2: Require sustained degradation
let config = MonitoringConfig {
    sustained_degradation_periods: 3,  // 3 consecutive periods
    ..MonitoringConfig::default()
};

// Solution 3: Filter alerts by severity
let alerts = tracker.get_alerts()?;
let critical_only: Vec<_> = alerts.iter()
    .filter(|a| a.severity == AlertSeverity::Critical)
    .collect();
```

## Performance Issues

### Issue: Assessment Takes Too Long

**Symptom**: Quality assessment slower than expected

**Solutions**:

```rust
// Solution 1: Enable parallel processing
let config = QualityConfig {
    enable_parallel: true,
    ..QualityConfig::default()
};

// Solution 2: Reduce outlier detection methods
let config = QualityConfig {
    outlier_detection_methods: vec![
        OutlierMethod::ZScore { threshold: 3.0 },  // Single fast method
    ],
    ..QualityConfig::default()
};

// Solution 3: Use sampling for very large datasets
if data.len() > 100_000 {
    let sample = sample_timeseries(&data, 10_000)?;  // Sample 10K points
    let assessment = assess_quality(&sample, &config)?;
}

// Solution 4: Profile to find bottleneck
use std::time::Instant;

let start = Instant::now();
let assessment = assess_quality(&data, &config)?;
println!("Total: {:?}", start.elapsed());

// Check individual dimensions
for dim in &["completeness", "consistency", "validity"] {
    let start = Instant::now();
    // ... assess individual dimension ...
    println!("{}: {:?}", dim, start.elapsed());
}
```

### Issue: High Memory Usage

**Symptom**: Out of memory errors or excessive memory consumption

**Solutions**:

```rust
// Solution 1: Process in chunks
fn assess_chunked(data: &TimeSeries, chunk_size: usize) -> Result<QualityAssessment> {
    let mut scores = Vec::new();

    for chunk in data.chunks(chunk_size) {
        let assessment = assess_quality(&chunk, &QualityConfig::default())?;
        scores.push(assessment.overall_score);
    }

    // Aggregate results
    let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
    // ... combine assessments ...
}

// Solution 2: Disable caching for large datasets
let config = QualityConfig {
    enable_caching: false,
    ..QualityConfig::default()
};

// Solution 3: Use streaming for very large files
let reader = TimeSeries::stream_from_csv("large_file.csv")?;
let mut tracker = QualityTracker::new(MonitoringConfig::default());

for chunk in reader.chunks(1000) {
    let assessment = assess_quality(&chunk?, &QualityConfig::default())?;
    tracker.track_quality_metrics(&assessment)?;
}
```

## Configuration Issues

### Issue: Configuration File Not Found

**Symptom**: "Configuration file not found" error

**Solutions**:

```bash
# Solution 1: Specify absolute path
chronos quality assess --config /full/path/to/config.toml data.csv

# Solution 2: Use relative path from current directory
chronos quality assess --config ./config/quality.toml data.csv

# Solution 3: Create config in default location
mkdir -p ~/.config/chronos
cp config.toml ~/.config/chronos/quality.toml

# Solution 4: Use built-in profiles instead
chronos quality assess --profile production data.csv
```

### Issue: Configuration Validation Fails

**Symptom**: "Invalid configuration" error

**Solutions**:

```rust
// Check configuration validity
let config = QualityConfig {
    completeness_threshold: 0.95,
    ..QualityConfig::default()
};

// Validate before use
if config.completeness_threshold < 0.0 || config.completeness_threshold > 1.0 {
    return Err(QualityError::Configuration(
        "Completeness threshold must be between 0 and 1".to_string()
    ));
}

// Or use validated constructor
let config = QualityConfig::validated()?;
```

## CLI Issues

### Issue: Command Not Found

**Symptom**: `chronos: command not found`

**Solutions**:

```bash
# Solution 1: Build and install
cargo build --release --bin chronos
cargo install --path .

# Solution 2: Run from target directory
./target/release/chronos quality assess data.csv

# Solution 3: Add to PATH
export PATH="$PATH:$(pwd)/target/release"
```

### Issue: Output Format Not Supported

**Symptom**: "Unsupported output format" error

**Solutions**:

```bash
# Check supported formats
chronos quality assess --help | grep format

# Use supported format
chronos quality assess data.csv --format json
chronos quality assess data.csv --format csv
chronos quality assess data.csv --format text

# For HTML/PDF, install optional dependencies
cargo install chronos --features full
```

## Getting Help

If your issue isn't covered here:

1. **Check the documentation**:
   - [README](README.md) - Module overview
   - [API Reference](api_reference.md) - Complete API docs
   - [CLI Guide](cli_guide.md) - Command-line usage
   - [Examples](examples/) - Usage examples

2. **Enable verbose output**:
   ```bash
   chronos quality assess --verbose data.csv
   ```

3. **Check logs**:
   ```rust
   env_logger::init();  // Enable logging
   log::debug!("Assessment starting...");
   ```

4. **Create a minimal reproduction**:
   ```rust
   use chronos::quality::*;

   fn main() {
       // Simplest possible case that reproduces issue
       let data = /* minimal test data */;
       let result = assess_quality(&data, &QualityConfig::default());
       println!("{:?}", result);
   }
   ```

5. **Report an issue**:
   - GitHub Issues: https://github.com/jpequegn/ts-rs/issues
   - Include: Rust version, OS, error message, minimal reproduction

## Common Error Messages

### "Empty dataset"
```rust
// Ensure data is not empty
if data.is_empty() {
    return Err(QualityError::InvalidData("Empty dataset".to_string()));
}
```

### "Invalid timestamp"
```rust
// Ensure timestamps are sorted
let mut data = data.clone();
data.sort_by_timestamp();
```

### "Insufficient statistical power"
```rust
// Need more data points
if data.len() < 30 {
    eprintln!("Warning: Results may be unreliable with < 30 points");
}
```

### "Configuration conflict"
```rust
// Check for conflicting settings
let config = QualityConfig {
    enable_cleaning: true,
    max_modifications: 0.0,  // Conflict: cleaning enabled but no modifications allowed
    ..QualityConfig::default()
};
// Solution: Set max_modifications > 0
```
