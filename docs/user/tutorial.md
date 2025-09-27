# Chronos Tutorial: Complete Guide with Real Examples

This comprehensive tutorial will walk you through all major features of Chronos using real-world examples. By the end, you'll be comfortable analyzing time series data for various domains.

## Prerequisites

- Chronos installed and configured (see [Installation Guide](installation.md))
- Basic understanding of time series concepts (see [Time Series Primer](../educational/timeseries_primer.md))
- Sample data files (provided in examples)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Financial Data Analysis](#financial-data-analysis)
3. [IoT Sensor Data](#iot-sensor-data)
4. [Business Metrics](#business-metrics)
5. [Weather and Climate Data](#weather-and-climate-data)
6. [Advanced Analysis Workflows](#advanced-analysis-workflows)
7. [Interactive Mode](#interactive-mode)
8. [Automation and Scripting](#automation-and-scripting)

## Quick Start

Let's start with a simple example using synthetic data to understand the basic workflow.

### Generate Sample Data

```bash
# Create a synthetic time series with trend and seasonality
chronos import --generate \
  --output sample_data.csv \
  --points 365 \
  --frequency daily \
  --trend linear \
  --seasonality 7 \
  --noise 0.1
```

### Basic Analysis

```bash
# Perform quick statistical analysis
chronos stats --file sample_data.csv --time-column date --value-column value

# Detect trend patterns
chronos trend --file sample_data.csv --time-column date --value-column value

# Check for seasonality
chronos seasonal --file sample_data.csv --time-column date --value-column value

# Create basic visualization
chronos plot --file sample_data.csv --time-column date --value-column value \
  --output sample_plot.png --type line
```

### Generate Report

```bash
# Create comprehensive report
chronos report --file sample_data.csv --time-column date --value-column value \
  --output sample_report.html --format html
```

## Financial Data Analysis

This section demonstrates analyzing stock price data, a common use case for time series analysis.

### Example: Stock Price Analysis

#### 1. Data Preparation

```bash
# Import stock price data (CSV format)
# File structure: date,open,high,low,close,volume
chronos import --file stock_prices.csv \
  --time-column date \
  --value-column close \
  --frequency daily \
  --output stock_processed.csv
```

#### 2. Descriptive Statistics

```bash
# Calculate comprehensive statistics
chronos stats --file stock_processed.csv \
  --time-column date \
  --value-column close \
  --include-distributions \
  --confidence-level 0.95 \
  --output stock_stats.json
```

**Expected output includes:**
- Mean, median, standard deviation
- Skewness and kurtosis
- Value at Risk (VaR) calculations
- Distribution fitting results

#### 3. Trend Analysis

```bash
# Analyze price trends
chronos trend --file stock_processed.csv \
  --time-column date \
  --value-column close \
  --method comprehensive \
  --detrend \
  --decomposition stl \
  --output stock_trend.json
```

**Key insights:**
- Trend direction and strength
- Trend change points
- Seasonal vs. irregular components

#### 4. Volatility Analysis

```bash
# Calculate volatility patterns
chronos stats --file stock_processed.csv \
  --time-column date \
  --value-column close \
  --rolling-window 30 \
  --volatility-analysis \
  --output volatility.json
```

#### 5. Anomaly Detection

```bash
# Detect unusual price movements
chronos anomaly --file stock_processed.csv \
  --time-column date \
  --value-column close \
  --method ensemble \
  --sensitivity 0.05 \
  --contextual \
  --output anomalies.json
```

#### 6. Price Forecasting

```bash
# Generate price forecasts
chronos forecast --file stock_processed.csv \
  --time-column date \
  --value-column close \
  --method arima \
  --horizon 30 \
  --confidence-intervals \
  --output forecast.json
```

#### 7. Create Financial Dashboard

```bash
# Generate comprehensive financial report
chronos report --file stock_processed.csv \
  --time-column date \
  --value-column close \
  --template financial \
  --include-forecasts \
  --include-risk-metrics \
  --output financial_dashboard.html
```

### Example: Portfolio Analysis

```bash
# Analyze correlation between multiple assets
chronos correlate --files "stock1.csv,stock2.csv,stock3.csv" \
  --time-column date \
  --value-column close \
  --method pearson \
  --rolling-correlation \
  --window 60 \
  --output correlation_analysis.json

# Create correlation heatmap
chronos plot --files "stock1.csv,stock2.csv,stock3.csv" \
  --time-column date \
  --value-column close \
  --type correlation_heatmap \
  --output portfolio_correlation.png
```

## IoT Sensor Data

Analyzing sensor data from IoT devices, focusing on temperature monitoring.

### Example: Temperature Sensor Monitoring

#### 1. Import Sensor Data

```bash
# Import IoT sensor data
# Format: timestamp,sensor_id,temperature,humidity,pressure
chronos import --file sensor_data.csv \
  --time-column timestamp \
  --value-column temperature \
  --frequency "1min" \
  --group-by sensor_id \
  --output processed_sensors.csv
```

#### 2. Data Quality Assessment

```bash
# Check data quality and missing values
chronos stats --file processed_sensors.csv \
  --time-column timestamp \
  --value-column temperature \
  --missing-value-analysis \
  --outlier-detection \
  --output data_quality.json
```

#### 3. Anomaly Detection for Predictive Maintenance

```bash
# Detect temperature anomalies
chronos anomaly --file processed_sensors.csv \
  --time-column timestamp \
  --value-column temperature \
  --method isolation_forest \
  --seasonal-adjustment \
  --streaming \
  --threshold-percentile 95 \
  --output temperature_anomalies.json
```

#### 4. Trend Analysis for Equipment Health

```bash
# Analyze long-term temperature trends
chronos trend --file processed_sensors.csv \
  --time-column timestamp \
  --value-column temperature \
  --decomposition x13 \
  --change-point-detection \
  --output equipment_health.json
```

#### 5. Seasonal Patterns

```bash
# Detect daily and weekly patterns
chronos seasonal --file processed_sensors.csv \
  --time-column timestamp \
  --value-column temperature \
  --multiple-seasonalities \
  --periods "24h,7d" \
  --strength-analysis \
  --output seasonal_patterns.json
```

#### 6. Predictive Monitoring

```bash
# Forecast next 24 hours
chronos forecast --file processed_sensors.csv \
  --time-column timestamp \
  --value-column temperature \
  --method prophet \
  --horizon 1440 \
  --include-seasonality \
  --output temperature_forecast.json
```

#### 7. Create IoT Dashboard

```bash
# Generate IoT monitoring report
chronos report --file processed_sensors.csv \
  --time-column timestamp \
  --value-column temperature \
  --template iot \
  --real-time-updates \
  --alert-thresholds \
  --output iot_dashboard.html
```

## Business Metrics

Analyzing business KPIs and operational metrics.

### Example: Sales Revenue Analysis

#### 1. Import Sales Data

```bash
# Import daily sales data
chronos import --file daily_sales.csv \
  --time-column date \
  --value-column revenue \
  --frequency daily \
  --currency USD \
  --output sales_processed.csv
```

#### 2. Business Intelligence Analysis

```bash
# Comprehensive business analysis
chronos stats --file sales_processed.csv \
  --time-column date \
  --value-column revenue \
  --growth-rates \
  --year-over-year \
  --seasonality-adjusted \
  --output business_insights.json
```

#### 3. Seasonal Business Patterns

```bash
# Analyze seasonal sales patterns
chronos seasonal --file sales_processed.csv \
  --time-column date \
  --value-column revenue \
  --holiday-effects \
  --weekly-patterns \
  --monthly-patterns \
  --output seasonal_business.json
```

#### 4. Sales Forecasting

```bash
# Generate sales forecasts for planning
chronos forecast --file sales_processed.csv \
  --time-column date \
  --value-column revenue \
  --method ensemble \
  --horizon 90 \
  --scenario-analysis \
  --confidence-intervals \
  --output sales_forecast.json
```

#### 5. Performance Monitoring

```bash
# Monitor for unusual sales patterns
chronos anomaly --file sales_processed.csv \
  --time-column date \
  --value-column revenue \
  --method multivariate \
  --business-context \
  --seasonal-adjustment \
  --output sales_anomalies.json
```

#### 6. Create Business Dashboard

```bash
# Generate executive dashboard
chronos report --file sales_processed.csv \
  --time-column date \
  --value-column revenue \
  --template executive \
  --kpi-metrics \
  --trend-analysis \
  --forecast-integration \
  --output executive_dashboard.html
```

## Weather and Climate Data

Analyzing meteorological data for climate research.

### Example: Temperature Trend Analysis

#### 1. Import Weather Data

```bash
# Import climate station data
chronos import --file weather_station.csv \
  --time-column datetime \
  --value-column temperature \
  --frequency hourly \
  --location "Weather Station ID" \
  --output climate_data.csv
```

#### 2. Climate Change Analysis

```bash
# Long-term trend analysis
chronos trend --file climate_data.csv \
  --time-column datetime \
  --value-column temperature \
  --method mann_kendall \
  --change-point-detection \
  --trend-significance \
  --output climate_trends.json
```

#### 3. Seasonal Climate Patterns

```bash
# Analyze seasonal climate variations
chronos seasonal --file climate_data.csv \
  --time-column datetime \
  --value-column temperature \
  --climate-indices \
  --annual-cycles \
  --multi-year-patterns \
  --output climate_seasonality.json
```

#### 4. Extreme Weather Events

```bash
# Detect extreme weather events
chronos anomaly --file climate_data.csv \
  --time-column datetime \
  --value-column temperature \
  --method extreme_value \
  --return-periods \
  --climate-thresholds \
  --output extreme_events.json
```

#### 5. Climate Forecasting

```bash
# Seasonal climate forecasts
chronos forecast --file climate_data.csv \
  --time-column datetime \
  --value-column temperature \
  --method climate_model \
  --horizon 180 \
  --ensemble-forecasts \
  --output climate_forecast.json
```

## Advanced Analysis Workflows

### Multi-Series Analysis

```bash
# Analyze multiple related time series
chronos correlate --files "series1.csv,series2.csv,series3.csv" \
  --time-column timestamp \
  --value-column value \
  --granger-causality \
  --cointegration \
  --dynamic-correlation \
  --output multi_series_analysis.json
```

### Streaming Analysis

```bash
# Real-time analysis for streaming data
chronos anomaly --file streaming_data.csv \
  --time-column timestamp \
  --value-column value \
  --streaming \
  --adaptive-thresholds \
  --real-time-alerts \
  --output streaming_results.json
```

### Batch Processing

```bash
# Process multiple files with same analysis
chronos report --directory ./data/ \
  --pattern "*.csv" \
  --time-column timestamp \
  --value-column value \
  --batch-processing \
  --template standard \
  --output-directory ./reports/
```

## Interactive Mode

Chronos provides an interactive mode for exploratory analysis:

```bash
# Start interactive session
chronos --interactive

# Or with specific file
chronos --interactive --file data.csv
```

**Interactive commands:**
```
> load data.csv timestamp value
> stats
> plot line
> trend decompose
> seasonal detect
> anomaly detect
> forecast 30
> export results.json
> help
> exit
```

## Automation and Scripting

### Bash Script Example

```bash
#!/bin/bash
# automated_analysis.sh

DATA_FILE="$1"
OUTPUT_DIR="$2"

if [ -z "$DATA_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <data_file> <output_directory>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting automated analysis..."

# Basic statistics
chronos stats --file "$DATA_FILE" \
  --time-column timestamp --value-column value \
  --output "$OUTPUT_DIR/stats.json"

# Trend analysis
chronos trend --file "$DATA_FILE" \
  --time-column timestamp --value-column value \
  --output "$OUTPUT_DIR/trends.json"

# Anomaly detection
chronos anomaly --file "$DATA_FILE" \
  --time-column timestamp --value-column value \
  --output "$OUTPUT_DIR/anomalies.json"

# Generate report
chronos report --file "$DATA_FILE" \
  --time-column timestamp --value-column value \
  --output "$OUTPUT_DIR/report.html"

echo "Analysis complete. Results saved to $OUTPUT_DIR"
```

### Configuration-Driven Analysis

```toml
# analysis_config.toml
[data]
file = "data.csv"
time_column = "timestamp"
value_column = "value"

[analysis]
statistics = true
trend_detection = true
seasonality = true
anomaly_detection = true
forecasting = true

[output]
directory = "./results"
format = "json"
generate_plots = true
create_report = true
```

```bash
# Run with configuration
chronos --config analysis_config.toml
```

## Best Practices

### 1. Data Preparation
- Always validate data quality first
- Handle missing values appropriately
- Ensure consistent time intervals
- Remove obvious data entry errors

### 2. Analysis Workflow
- Start with descriptive statistics
- Visualize data before analysis
- Test for stationarity when needed
- Validate model assumptions

### 3. Interpretation
- Consider domain context
- Validate statistical significance
- Check for confounding factors
- Document assumptions and limitations

### 4. Production Use
- Implement automated validation
- Set up monitoring and alerts
- Version control configurations
- Document analysis decisions

## Next Steps

After completing this tutorial:

1. Explore the [Command Reference](command_reference.md) for detailed options
2. Read the [Time Series Primer](../educational/timeseries_primer.md) for deeper understanding
3. Check out specialized [Examples](../examples/) for your domain
4. Join the community for discussions and support

## Common Troubleshooting

### Memory Issues with Large Files
```bash
# Use streaming mode for large datasets
chronos stats --file large_data.csv \
  --streaming \
  --chunk-size 10000
```

### Performance Optimization
```bash
# Enable parallel processing
chronos --config-set performance.parallel_processing true \
  --config-set performance.max_threads 8
```

### Handling Missing Data
```bash
# Configure missing value handling
chronos import --file data.csv \
  --missing-value-policy interpolate \
  --interpolation-method linear
```

For more help, see the [FAQ](faq.md) and [Troubleshooting Guide](troubleshooting.md).