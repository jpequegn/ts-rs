# Basic Quality Assessment Example

This example demonstrates the fundamental workflow for assessing data quality using the Chronos quality module.

## Overview

In this example, we will:
1. Load time series data from a CSV file
2. Perform basic quality assessment
3. Interpret the results
4. Make decisions based on quality scores

## Code Example

```rust
use chronos::quality::*;
use chronos::TimeSeries;
use anyhow::Result;

fn main() -> Result<()> {
    // Step 1: Load your time series data
    println!("Loading data...");
    let data = TimeSeries::from_csv("data/sample.csv")?;

    println!("Loaded {} data points", data.len());
    println!("Time range: {} to {}",
        data.timestamps().first().unwrap(),
        data.timestamps().last().unwrap()
    );

    // Step 2: Configure quality assessment
    // Using default configuration for exploratory analysis
    let config = QualityConfig::default();

    println!("\nAssessing data quality...");

    // Step 3: Perform quality assessment
    let assessment = assess_quality(&data, &config)?;

    // Step 4: Display results
    println!("\n=== Quality Assessment Results ===");
    println!("Overall Quality Score: {:.1}/100", assessment.overall_score);

    println!("\nDimension Scores:");
    println!("  Completeness: {:.1}/100", assessment.dimension_scores.completeness);
    println!("  Consistency:  {:.1}/100", assessment.dimension_scores.consistency);
    println!("  Validity:     {:.1}/100", assessment.dimension_scores.validity);
    println!("  Timeliness:   {:.1}/100", assessment.dimension_scores.timeliness);
    println!("  Accuracy:     {:.1}/100", assessment.dimension_scores.accuracy);

    // Step 5: Review quality metrics
    println!("\nQuality Metrics:");
    println!("  Total data points: {}", assessment.quality_metrics.total_points);
    println!("  Missing points:    {}", assessment.quality_metrics.missing_points);
    println!("  Outliers detected: {}", assessment.quality_metrics.outliers);
    println!("  Gaps found:        {}", assessment.quality_metrics.gaps);
    println!("  Duplicates:        {}", assessment.quality_metrics.duplicates);

    // Step 6: Examine quality issues
    if !assessment.quality_issues.is_empty() {
        println!("\n⚠️  Quality Issues Found ({}):", assessment.quality_issues.len());

        for (i, issue) in assessment.quality_issues.iter().enumerate().take(5) {
            println!("\n{}. {} (Severity: {:?})",
                i + 1,
                issue.description,
                issue.severity
            );

            if let Some(recommendation) = &issue.recommendation {
                println!("   Recommendation: {}", recommendation);
            }
        }

        if assessment.quality_issues.len() > 5 {
            println!("\n   ... and {} more issues", assessment.quality_issues.len() - 5);
        }
    } else {
        println!("\n✅ No quality issues detected!");
    }

    // Step 7: Make decisions based on quality score
    println!("\n=== Quality Assessment ===");

    match assessment.overall_score {
        score if score >= 90.0 => {
            println!("✅ EXCELLENT: Data quality is very high. Ready for production use.");
        }
        score if score >= 75.0 => {
            println!("✓ GOOD: Data quality is acceptable. Minor improvements recommended.");
        }
        score if score >= 60.0 => {
            println!("⚠️  FAIR: Data quality needs improvement. Review issues carefully.");
        }
        _ => {
            println!("❌ POOR: Data quality is insufficient. Cleaning required before use.");
        }
    }

    // Step 8: Get automated recommendations
    if assessment.overall_score < 90.0 {
        println!("\n=== Recommendations ===");
        let recommendations = generate_recommendations(&assessment, &data);

        for (i, rec) in recommendations.iter().enumerate().take(3) {
            println!("\n{}. Priority: {:?}", i + 1, rec.priority);
            println!("   Issue: {}", rec.issue);
            println!("   Recommendation: {}", rec.recommendation);
        }
    }

    Ok(())
}
```

## Sample Output

```
Loading data...
Loaded 1000 data points
Time range: 2023-01-01 00:00:00 UTC to 2023-12-31 23:00:00 UTC

Assessing data quality...

=== Quality Assessment Results ===
Overall Quality Score: 85.2/100

Dimension Scores:
  Completeness: 92.5/100
  Consistency:  88.0/100
  Validity:     78.5/100
  Timeliness:   85.0/100
  Accuracy:     90.0/100

Quality Metrics:
  Total data points: 1000
  Missing points:    15
  Outliers detected: 3
  Gaps found:        2
  Duplicates:        0

⚠️  Quality Issues Found (5):

1. Outlier detected at index 245 (Severity: Moderate)
   Recommendation: Consider correction or removal based on domain knowledge

2. Data gap detected from 2023-03-15 to 2023-03-17 (Severity: Minor)
   Recommendation: Fill gap using linear interpolation

3. Outlier detected at index 567 (Severity: Severe)
   Recommendation: Investigate cause and correct if erroneous

4. Irregular frequency detected (Severity: Minor)
   Recommendation: Review data collection process

5. Missing values detected (15 points) (Severity: Minor)
   Recommendation: Consider imputation or investigate data source

=== Quality Assessment ===
✓ GOOD: Data quality is acceptable. Minor improvements recommended.

=== Recommendations ===

1. Priority: High
   Issue: 3 outliers detected affecting validity score
   Recommendation: Review outliers and apply appropriate correction method

2. Priority: Medium
   Issue: 2 temporal gaps affecting completeness
   Recommendation: Fill gaps using linear or spline interpolation

3. Priority: Low
   Issue: Irregular frequency in some regions
   Recommendation: Review data collection process for consistency
```

## Understanding the Results

### Overall Quality Score

The overall quality score (0-100) is a weighted combination of five dimension scores:

- **90-100**: Excellent - Production ready
- **75-89**: Good - Acceptable with minor issues
- **60-74**: Fair - Needs improvement
- **Below 60**: Poor - Not suitable for critical analysis

### Dimension Scores

1. **Completeness (92.5/100)**
   - Measures presence of expected data points
   - High score indicates few missing values
   - In this example, 98.5% of expected data is present

2. **Consistency (88.0/100)**
   - Evaluates data patterns and rules adherence
   - High score means no duplicates, proper temporal order
   - Minor irregularities may exist

3. **Validity (78.5/100)**
   - Assesses data values within expected ranges
   - Lower score due to 3 detected outliers
   - Indicates some data points need review

4. **Timeliness (85.0/100)**
   - Evaluates data frequency adherence
   - Good score with minor irregular spacing
   - Generally follows expected pattern

5. **Accuracy (90.0/100)**
   - Measures statistical soundness
   - High score indicates low noise level
   - Data appears reliable

### Quality Issues

Each issue includes:
- **Description**: What was detected
- **Severity**: Impact level (Minor, Moderate, Severe, Extreme)
- **Location**: Affected indices or time ranges
- **Recommendation**: Suggested action

## Next Steps

Based on this assessment:

### For Good Quality (75-89)
```rust
// Option 1: Use data as-is for exploratory analysis
let result = analyze_timeseries(&data);

// Option 2: Perform light cleaning for production use
let cleaning_config = CleaningConfig::conservative();
let cleaned = clean_timeseries(&data, &cleaning_config)?;
```

### For Fair Quality (60-74)
```rust
// Clean data before analysis
let cleaning_config = CleaningConfig::default();
let cleaned = clean_timeseries(&data, &cleaning_config)?;

// Re-assess to verify improvement
let new_assessment = assess_quality(&cleaned.cleaned_data, &config)?;
println!("Quality improved: {:.1} → {:.1}",
    assessment.overall_score,
    new_assessment.overall_score
);
```

### For Poor Quality (<60)
```rust
// Aggressive cleaning required
let cleaning_config = CleaningConfig::aggressive();
let cleaned = clean_timeseries(&data, &cleaning_config)?;

// May need multiple cleaning passes
let pass2 = clean_timeseries(&cleaned.cleaned_data, &cleaning_config)?;

// Verify final quality
let final_assessment = assess_quality(&pass2.cleaned_data, &config)?;

if final_assessment.overall_score < 75.0 {
    eprintln!("Warning: Data quality remains low after cleaning");
    eprintln!("Consider reviewing data source or collection process");
}
```

## Custom Configuration

For domain-specific requirements:

```rust
// Financial data - stricter requirements
let financial_config = QualityConfig {
    completeness_threshold: 0.99,  // 99% completeness required
    outlier_detection_methods: vec![
        OutlierMethod::ZScore { threshold: 2.5 },  // More sensitive
        OutlierMethod::IQR { factor: 1.5 },
    ],
    acceptable_gap_ratio: 0.01,  // Max 1% gaps
    weights: QualityWeights {
        completeness: 0.30,  // Higher weight on completeness
        consistency: 0.25,
        validity: 0.25,
        timeliness: 0.10,
        accuracy: 0.10,
    },
    ..QualityConfig::default()
};

let assessment = assess_quality(&data, &financial_config)?;
```

## Error Handling

```rust
use chronos::quality::{assess_quality, QualityConfig, QualityError};

fn safe_assessment(data: &TimeSeries) -> Result<(), QualityError> {
    let config = QualityConfig::default();

    match assess_quality(data, &config) {
        Ok(assessment) => {
            println!("Quality score: {:.1}", assessment.overall_score);
            Ok(())
        }
        Err(QualityError::InvalidData(msg)) => {
            eprintln!("Data validation failed: {}", msg);
            Err(QualityError::InvalidData(msg))
        }
        Err(QualityError::Configuration(msg)) => {
            eprintln!("Configuration error: {}", msg);
            Err(QualityError::Configuration(msg))
        }
        Err(e) => {
            eprintln!("Assessment failed: {}", e);
            Err(e)
        }
    }
}
```

## Performance Considerations

For large datasets:

```rust
use std::time::Instant;

// Measure assessment time
let start = Instant::now();
let assessment = assess_quality(&data, &config)?;
let duration = start.elapsed();

println!("Assessment completed in {:?}", duration);
println!("Performance: {:.2} points/ms",
    data.len() as f64 / duration.as_millis() as f64
);

// Expected performance: ~100-200 points/ms on modern hardware
if duration.as_millis() > 100 && data.len() > 10_000 {
    println!("Warning: Assessment slower than expected for dataset size");
}
```

## See Also

- [Data Cleaning Example](data_cleaning.md) - How to fix quality issues
- [Quality Monitoring](quality_monitoring.md) - Continuous quality tracking
- [Configuration Guide](../configuration.md) - Advanced configuration options
- [API Reference](../api_reference.md) - Complete API documentation
