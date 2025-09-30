# Quality CLI Guide

Complete guide to using the Chronos quality commands from the command line.

## Overview

The `chronos quality` command provides six subcommands for data quality management:

```bash
chronos quality <SUBCOMMAND>
```

**Available subcommands**:
- `assess` - Comprehensive quality assessment
- `profile` - Data profiling and analysis
- `clean` - Data cleaning and repair
- `fill-gaps` - Fill missing data gaps
- `monitor` - Quality monitoring setup and status
- `report` - Generate quality reports

## Global Options

All quality commands support these global options:

```bash
-v, --verbose           Enable verbose output
-q, --quiet             Minimal output mode
--config <FILE>         Configuration file path
--format <FORMAT>       Output format: text, json, csv, markdown, html, pdf
-o, --output-dir <DIR>  Output directory for generated files
```

## Commands

### quality assess

Perform comprehensive quality assessment across all dimensions.

**Usage**:
```bash
chronos quality assess [OPTIONS] <INPUT>
```

**Arguments**:
- `<INPUT>` - Input file path (CSV, JSON)

**Options**:
- `--profile <PROFILE>` - Quality profile to use (exploratory, production, regulatory, realtime)
- `--detailed` - Include detailed quality report
- `--output <FILE>` - Output file for quality report

**Examples**:

```bash
# Basic assessment
chronos quality assess data.csv

# Detailed assessment with specific profile
chronos quality assess --detailed --profile production data.csv

# Save results to JSON
chronos quality assess data.csv --format json --output assessment.json

# Use strict quality requirements
chronos quality assess --profile regulatory financial_data.csv

# Quiet mode with minimal output
chronos quality assess --quiet data.csv
```

**Output**:
```
📊 Assessing Data Quality...
Input: data.csv

Quality Assessment Results
============================

Overall Quality Score: 85.2/100

Dimension Scores:
  Completeness:  92.5/100
  Consistency:   88.0/100
  Validity:      78.5/100
  Timeliness:    85.0/100
  Accuracy:      90.0/100

⚠️  Quality Issues (3 found)
  • 3 outliers detected using Z-score method
  • 2 gaps in temporal coverage
  • 1 consistency issue detected

💡 Recommendations
  • Consider outlier correction for improved validity
  • Fill temporal gaps using linear interpolation
  • Review data collection process for consistency

✅ Quality assessment completed!
```

### quality profile

Generate comprehensive data profile with statistics and quality indicators.

**Usage**:
```bash
chronos quality profile [OPTIONS] <INPUT>
```

**Arguments**:
- `<INPUT>` - Input file path

**Options**:
- `--frequency <FREQ>` - Expected data frequency (auto-detect if not specified)
- `--output <FILE>` - Output profiling report path

**Examples**:

```bash
# Basic profiling
chronos quality profile data.csv

# Specify expected frequency
chronos quality profile --frequency daily sensor_data.csv

# Save profile to JSON
chronos quality profile data.csv --format json --output profile.json

# Verbose mode with detailed statistics
chronos quality profile --verbose data.csv
```

**Output**:
```
📋 Generating Data Profile...
Input: data.csv

Data Profile
============

Basic Statistics:
  Data Points: 1,000
  Time Range: 2023-01-01 to 2023-12-31
  Frequency: Daily (detected)

Completeness:
  Coverage: 98.5%
  Missing Values: 15
  Gaps: 2 periods

Temporal Coverage:
  Expected Points: 365
  Actual Points: 350
  Missing Points: 15
  Irregular Spacing: 3 instances

Statistical Profile:
  Mean: 45.23
  Std Dev: 12.45
  Min: 12.0
  Max: 98.5
  Skewness: 0.23
  Kurtosis: -0.15

Quality Indicators:
  Overall Quality: High
  Outliers: 3 detected
  Consistency: Good
  Trend: Slightly increasing

✅ Profiling completed!
```

### quality clean

Clean data and repair quality issues using configurable strategies.

**Usage**:
```bash
chronos quality clean [OPTIONS] <INPUT> <OUTPUT>
```

**Arguments**:
- `<INPUT>` - Input file path
- `<OUTPUT>` - Output file path for cleaned data

**Options**:
- `--config <FILE>` - Cleaning configuration file
- `--max-modifications <RATIO>` - Maximum modification ratio (default: 0.1)
- `--aggressive` - Use aggressive cleaning (more modifications)

**Examples**:

```bash
# Basic cleaning
chronos quality clean data.csv cleaned.csv

# Conservative cleaning with 5% max modifications
chronos quality clean --max-modifications 0.05 data.csv cleaned.csv

# Aggressive cleaning with custom config
chronos quality clean --aggressive --config clean_config.toml data.csv cleaned.csv

# Clean and save report
chronos quality clean data.csv cleaned.csv --format json --output report.json
```

**Output**:
```
🧹 Cleaning Data...
Input: data.csv
Output: cleaned.csv

Cleaning Operations
==================

Gaps Filled: 15
  Method: Linear interpolation

Outliers Corrected: 3
  Method: Median replacement

Noise Reduction: Applied
  Method: Moving average (window=5)

Quality Improvement
===================

Before: 78.5/100
After:  92.3/100
Improvement: +13.8 points

Modifications: 18/1000 (1.8%)

✅ Data cleaning completed!
```

### quality fill-gaps

Fill missing data gaps using various imputation methods.

**Usage**:
```bash
chronos quality fill-gaps [OPTIONS] <INPUT> <OUTPUT>
```

**Arguments**:
- `<INPUT>` - Input file path
- `<OUTPUT>` - Output file path

**Options**:
- `--method <METHOD>` - Imputation method: linear, forward-fill, backward-fill, spline, seasonal

**Examples**:

```bash
# Linear interpolation (default)
chronos quality fill-gaps data.csv filled.csv

# Forward fill missing values
chronos quality fill-gaps --method forward-fill data.csv filled.csv

# Spline interpolation for smooth curves
chronos quality fill-gaps --method spline data.csv filled.csv

# Seasonal decomposition for periodic data
chronos quality fill-gaps --method seasonal stock_data.csv filled.csv
```

**Output**:
```
🔧 Filling Data Gaps...
Input: data.csv
Output: filled.csv
Method: linear

Gap Filling Results
===================

Gaps Found: 3
  Gap 1: 2023-03-15 to 2023-03-17 (2 points)
  Gap 2: 2023-06-20 to 2023-06-22 (2 points)
  Gap 3: 2023-09-10 to 2023-09-10 (1 point)

Total Points Filled: 5

Quality Improvement
===================

Completeness: 98.5% → 100%

✅ Gap filling completed!
```

### quality monitor

Set up and manage quality monitoring for continuous tracking.

**Usage**:
```bash
chronos quality monitor <ACTION> [OPTIONS]
```

**Actions**:
- `setup` - Set up quality monitoring
- `status` - Check monitoring status
- `alerts` - View quality alerts

#### monitor setup

```bash
chronos quality monitor setup [OPTIONS] <INPUT>

Options:
  --config <FILE>        Monitoring configuration file
  --thresholds <FILE>    Alert threshold configuration file
```

**Examples**:

```bash
# Basic setup
chronos quality monitor setup data.csv

# Setup with custom config
chronos quality monitor setup --config monitoring.toml data.csv

# Setup with alert thresholds
chronos quality monitor setup --thresholds alerts.toml data.csv
```

**Output**:
```
🔍 Setting up Quality Monitoring...
Input: data.csv

Monitoring Configuration
=======================

Tracking Frequency: Hourly
Baseline Quality: 85.2/100

Alert Thresholds:
  Overall Quality:
    Warning:  < 75
    Critical: < 60
  Completeness:
    Warning:  < 90
    Critical: < 80
  Validity:
    Warning:  < 80
    Critical: < 65

✅ Monitoring setup completed!
```

#### monitor status

```bash
chronos quality monitor status [OPTIONS]

Options:
  --detailed    Show detailed status information
```

**Examples**:

```bash
# Check status
chronos quality monitor status

# Detailed status
chronos quality monitor status --detailed
```

**Output**:
```
📊 Quality Monitoring Status

Active Monitors: 3

data.csv
  Status: Active
  Last Check: 2024-01-15 14:30:00
  Current Quality: 87.3/100
  Trend: Stable (+0.5 in 24h)
  Alerts: 0 active

sensor_data.csv
  Status: Active
  Last Check: 2024-01-15 14:28:00
  Current Quality: 72.1/100
  Trend: Declining (-5.2 in 24h)
  Alerts: 1 warning

✅ Status retrieved
```

#### monitor alerts

```bash
chronos quality monitor alerts [OPTIONS]

Options:
  --active-only          Show only active alerts
  --severity <SEVERITY>  Filter by severity: info, warning, critical, emergency
```

**Examples**:

```bash
# View all alerts
chronos quality monitor alerts

# Active alerts only
chronos quality monitor alerts --active-only

# Critical alerts only
chronos quality monitor alerts --severity critical
```

**Output**:
```
🚨 Quality Alerts

sensor_data.csv
  ⚠️  WARNING - Quality degradation detected
  Current: 72.1/100 (was 82.5/100)
  Dimension: Completeness (88% → 78%)
  Time: 2024-01-15 12:00:00

  💡 Recommendation:
     Check data collection process for recent changes

stock_data.csv
  🔴 CRITICAL - Quality below threshold
  Current: 58.3/100
  Dimension: Validity (multiple outliers)
  Time: 2024-01-15 13:15:00

  💡 Recommendation:
     Immediate investigation required

✅ 2 alerts found
```

### quality report

Generate comprehensive quality reports with recommendations.

**Usage**:
```bash
chronos quality report [OPTIONS] <INPUT>
```

**Arguments**:
- `<INPUT>` - Input file path

**Options**:
- `--template <TEMPLATE>` - Report template: brief, standard, comprehensive, technical
- `--recommendations` - Include recommendations section
- `--output <FILE>` - Output report file

**Examples**:

```bash
# Standard report
chronos quality report data.csv

# Comprehensive report with recommendations
chronos quality report --template comprehensive --recommendations data.csv

# Technical report in HTML format
chronos quality report --template technical data.csv --format html --output report.html

# Brief summary
chronos quality report --template brief data.csv
```

**Output**:
```
📄 Generating Quality Report...
Input: data.csv
Template: standard

Quality Report
==============

Executive Summary
-----------------
Overall Quality: 85.2/100 (Good)
Assessment Date: 2024-01-15

Quality Dimensions
------------------
Completeness:  92.5/100 ⭐⭐⭐⭐
Consistency:   88.0/100 ⭐⭐⭐⭐
Validity:      78.5/100 ⭐⭐⭐
Timeliness:    85.0/100 ⭐⭐⭐⭐
Accuracy:      90.0/100 ⭐⭐⭐⭐

Key Findings
------------
✅ High completeness (92.5%)
✅ Good consistency across time periods
⚠️  3 outliers detected (Z-score method)
⚠️  2 temporal gaps found

Recommendations
---------------
1. Investigate and correct 3 outliers
2. Fill temporal gaps using appropriate method
3. Review data collection for consistency

✅ Report generated successfully!
```

## Configuration Files

### Quality Configuration

Create a `quality_config.toml`:

```toml
[quality]
completeness_threshold = 0.95
acceptable_gap_ratio = 0.05
enable_cleaning = true

[outlier_detection]
methods = ["zscore", "iqr"]
zscore_threshold = 3.0
iqr_factor = 1.5

[weights]
completeness = 0.25
consistency = 0.20
validity = 0.25
timeliness = 0.15
accuracy = 0.15
```

### Monitoring Configuration

Create a `monitoring.toml`:

```toml
[monitoring]
tracking_frequency = "1h"
enable_alerts = true

[thresholds.overall_quality]
warning = 75.0
critical = 60.0
degradation_rate = 5.0

[thresholds.completeness]
warning = 90.0
critical = 80.0

[notifications]
channels = ["email", "slack"]
email_recipients = ["ops@company.com"]
slack_webhook = "https://hooks.slack.com/..."
```

### Cleaning Configuration

Create a `cleaning.toml`:

```toml
[cleaning]
max_modifications = 0.10
preserve_characteristics = true
uncertainty_tracking = true

[gap_filling]
default_method = "linear"
seasonal_periods = [7, 30]

[outlier_correction]
method = "median_replace"
window_size = 5

[noise_reduction]
method = "moving_average"
window = 5
```

## Output Formats

### Text (default)
Human-readable console output

### JSON
```bash
chronos quality assess data.csv --format json
```

```json
{
  "overall_score": 85.2,
  "dimension_scores": {
    "completeness": 92.5,
    "consistency": 88.0,
    "validity": 78.5,
    "timeliness": 85.0,
    "accuracy": 90.0
  },
  "quality_issues": [...]
}
```

### CSV
```bash
chronos quality assess data.csv --format csv
```

### Markdown
```bash
chronos quality assess data.csv --format markdown
```

### HTML
```bash
chronos quality report data.csv --format html --output report.html
```

### PDF
```bash
chronos quality report data.csv --format pdf --output report.pdf
```

## Best Practices

### 1. Start with Profiling
Always profile your data first to understand characteristics:
```bash
chronos quality profile data.csv
```

### 2. Use Appropriate Profiles
Choose quality profiles based on your domain:
- **exploratory**: Initial data exploration
- **production**: Production systems
- **regulatory**: Financial/regulated industries
- **realtime**: Real-time data streams

### 3. Conservative Cleaning
Start with conservative cleaning:
```bash
chronos quality clean --max-modifications 0.05 data.csv cleaned.csv
```

### 4. Monitor Continuously
Set up monitoring for production data:
```bash
chronos quality monitor setup --config monitoring.toml data.csv
```

### 5. Validate Results
Always validate cleaned data:
```bash
chronos quality assess cleaned.csv
```

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

## Next Steps

- Read the [API Reference](api_reference.md) for programmatic usage
- Check [Examples](examples/) for detailed use cases
- Review [Configuration](configuration.md) for advanced options
