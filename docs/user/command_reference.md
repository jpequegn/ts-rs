# Command Reference

Complete reference for all Chronos commands with detailed options and examples.

## Table of Contents

1. [Global Options](#global-options)
2. [Data Import and Management](#data-import-and-management)
3. [Statistical Analysis](#statistical-analysis)
4. [Time Series Analysis](#time-series-analysis)
5. [Visualization](#visualization)
6. [Reporting](#reporting)
7. [Configuration](#configuration)
8. [Plugin Management](#plugin-management)

## Global Options

These options are available for all commands:

### Basic Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config <FILE>` | | Configuration file path | `~/.config/chronos/config.toml` |
| `--verbose` | `-v` | Enable verbose output | `false` |
| `--quiet` | `-q` | Enable quiet mode (minimal output) | `false` |
| `--output-dir <DIR>` | `-o` | Output directory for generated files | Current directory |
| `--format <FORMAT>` | | Default output format | `text` |
| `--interactive` | `-i` | Enable interactive mode | `false` |

### Output Formats

- `text` - Plain text output (default)
- `json` - JSON format
- `csv` - CSV format
- `markdown` - Markdown format
- `html` - HTML format
- `pdf` - PDF format

### Examples

```bash
# Use custom configuration
chronos --config ./my-config.toml stats --file data.csv

# Verbose output with JSON format
chronos --verbose --format json stats --file data.csv

# Quiet mode with custom output directory
chronos --quiet --output-dir ./results stats --file data.csv
```

## Data Import and Management

### `chronos import`

Import and preprocess time series data from various formats.

#### Basic Usage

```bash
chronos import --file <FILE> --time-column <COLUMN> --value-column <COLUMN>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--time-column <COLUMN>` | Name of the time/date column | `--time-column timestamp` |
| `--value-column <COLUMN>` | Name of the value column | `--value-column price` |
| `--frequency <FREQ>` | Data frequency (auto-detected if not specified) | `--frequency daily` |
| `--format <FORMAT>` | Input file format (auto-detected) | `--format csv` |
| `--output <FILE>` | Output file path | `--output processed.csv` |
| `--missing-value-policy <POLICY>` | How to handle missing values | `--missing-value-policy interpolate` |
| `--interpolation-method <METHOD>` | Interpolation method for missing values | `--interpolation-method linear` |
| `--validate` | Validate data after import | `--validate` |
| `--generate` | Generate synthetic data instead of importing | `--generate` |

#### Frequency Options

- `minutely` / `1min` / `T`
- `hourly` / `1H` / `H`
- `daily` / `1D` / `D`
- `weekly` / `1W` / `W`
- `monthly` / `1M` / `M`
- `quarterly` / `1Q` / `Q`
- `yearly` / `1Y` / `Y`
- Custom: `5min`, `15min`, `2H`, etc.

#### Missing Value Policies

- `drop` - Remove rows with missing values
- `interpolate` - Interpolate missing values
- `forward_fill` - Forward fill missing values
- `backward_fill` - Backward fill missing values
- `mean` - Fill with column mean
- `zero` - Fill with zero

#### Examples

```bash
# Basic import
chronos import --file sales.csv --time-column date --value-column revenue

# Import with preprocessing
chronos import --file sensor_data.csv \
  --time-column timestamp \
  --value-column temperature \
  --frequency 1min \
  --missing-value-policy interpolate \
  --output processed_sensor_data.csv

# Generate synthetic data
chronos import --generate \
  --output synthetic.csv \
  --points 1000 \
  --frequency daily \
  --trend linear \
  --seasonality 7 \
  --noise 0.1
```

## Statistical Analysis

### `chronos stats`

Perform comprehensive statistical analysis on time series data.

#### Basic Usage

```bash
chronos stats --file <FILE> --time-column <COLUMN> --value-column <COLUMN>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output file for results | `--output stats.json` |
| `--confidence-level <LEVEL>` | Confidence level for intervals | `--confidence-level 0.95` |
| `--include-distributions` | Include distribution analysis | `--include-distributions` |
| `--rolling-window <SIZE>` | Rolling statistics window | `--rolling-window 30` |
| `--volatility-analysis` | Include volatility analysis | `--volatility-analysis` |
| `--outlier-detection` | Detect outliers | `--outlier-detection` |
| `--missing-value-analysis` | Analyze missing values | `--missing-value-analysis` |
| `--autocorrelation` | Compute autocorrelation | `--autocorrelation` |
| `--partial-autocorrelation` | Compute partial autocorrelation | `--partial-autocorrelation` |
| `--stationarity-test` | Test for stationarity | `--stationarity-test` |

#### Output Includes

- **Descriptive Statistics**: mean, median, std dev, skewness, kurtosis
- **Distribution Analysis**: normality tests, histogram, quantiles
- **Time Series Statistics**: autocorrelation, partial autocorrelation
- **Stationarity Tests**: Augmented Dickey-Fuller, Phillips-Perron
- **Missing Value Analysis**: count, patterns, recommendations
- **Outlier Detection**: statistical outliers and anomalies

#### Examples

```bash
# Basic statistics
chronos stats --file stock_prices.csv --time-column date --value-column close

# Comprehensive analysis
chronos stats --file data.csv \
  --time-column timestamp \
  --value-column value \
  --include-distributions \
  --confidence-level 0.99 \
  --volatility-analysis \
  --stationarity-test \
  --output comprehensive_stats.json

# Rolling statistics
chronos stats --file data.csv \
  --time-column date \
  --value-column price \
  --rolling-window 30 \
  --output rolling_stats.json
```

## Time Series Analysis

### `chronos trend`

Analyze trends and perform time series decomposition.

#### Basic Usage

```bash
chronos trend --file <FILE> --time-column <COLUMN> --value-column <COLUMN>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output file | `--output trend_analysis.json` |
| `--method <METHOD>` | Analysis method | `--method comprehensive` |
| `--decomposition <TYPE>` | Decomposition method | `--decomposition stl` |
| `--detrend` | Perform detrending | `--detrend` |
| `--detrending-method <METHOD>` | Detrending method | `--detrending-method linear` |
| `--change-point-detection` | Detect trend changes | `--change-point-detection` |
| `--trend-significance` | Test trend significance | `--trend-significance` |

#### Decomposition Methods

- `stl` - Seasonal and Trend decomposition using Loess
- `x13` - X-13ARIMA-SEATS method
- `classical` - Classical decomposition
- `moving_average` - Moving average decomposition

#### Detrending Methods

- `linear` - Linear detrending
- `difference` - First difference
- `moving_average` - Moving average detrending
- `hp_filter` - Hodrick-Prescott filter

#### Examples

```bash
# Basic trend analysis
chronos trend --file sales.csv --time-column date --value-column revenue

# Comprehensive trend analysis with decomposition
chronos trend --file data.csv \
  --time-column timestamp \
  --value-column value \
  --method comprehensive \
  --decomposition stl \
  --change-point-detection \
  --trend-significance \
  --output trend_results.json

# Detrending
chronos trend --file data.csv \
  --time-column date \
  --value-column price \
  --detrend \
  --detrending-method hp_filter \
  --output detrended.json
```

### `chronos seasonal`

Detect and analyze seasonality patterns.

#### Basic Usage

```bash
chronos seasonal --file <FILE> --time-column <COLUMN> --value-column <COLUMN>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output file | `--output seasonal.json` |
| `--method <METHOD>` | Detection method | `--method fourier` |
| `--multiple-seasonalities` | Detect multiple seasonal patterns | `--multiple-seasonalities` |
| `--periods <PERIODS>` | Specific periods to test | `--periods "7,30,365"` |
| `--strength-analysis` | Analyze seasonal strength | `--strength-analysis` |
| `--holiday-effects` | Detect holiday effects | `--holiday-effects` |
| `--seasonal-adjustment` | Apply seasonal adjustment | `--seasonal-adjustment` |
| `--adjustment-method <METHOD>` | Seasonal adjustment method | `--adjustment-method x13` |

#### Detection Methods

- `fourier` - Fourier analysis
- `periodogram` - Periodogram analysis
- `autocorrelation` - Autocorrelation-based
- `stl` - STL decomposition
- `x13` - X-13ARIMA-SEATS

#### Examples

```bash
# Basic seasonality detection
chronos seasonal --file sales.csv --time-column date --value-column revenue

# Multiple seasonalities
chronos seasonal --file hourly_data.csv \
  --time-column timestamp \
  --value-column value \
  --multiple-seasonalities \
  --periods "24,168,8760" \
  --strength-analysis \
  --output seasonal_analysis.json

# Seasonal adjustment
chronos seasonal --file data.csv \
  --time-column date \
  --value-column value \
  --seasonal-adjustment \
  --adjustment-method x13 \
  --output seasonally_adjusted.json
```

### `chronos anomaly`

Detect anomalies and outliers in time series data.

#### Basic Usage

```bash
chronos anomaly --file <FILE> --time-column <COLUMN> --value-column <COLUMN>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output file | `--output anomalies.json` |
| `--method <METHOD>` | Detection method | `--method isolation_forest` |
| `--sensitivity <LEVEL>` | Sensitivity level (0.0-1.0) | `--sensitivity 0.05` |
| `--contextual` | Enable contextual anomaly detection | `--contextual` |
| `--seasonal-adjustment` | Adjust for seasonality | `--seasonal-adjustment` |
| `--streaming` | Enable streaming detection | `--streaming` |
| `--threshold-percentile <PCT>` | Threshold percentile | `--threshold-percentile 95` |
| `--ensemble` | Use ensemble methods | `--ensemble` |

#### Detection Methods

- `statistical` - Statistical threshold-based
- `isolation_forest` - Isolation Forest algorithm
- `lof` - Local Outlier Factor
- `svm` - One-Class SVM
- `ensemble` - Ensemble of multiple methods
- `streaming` - Real-time streaming detection

#### Examples

```bash
# Basic anomaly detection
chronos anomaly --file sensor_data.csv --time-column timestamp --value-column temperature

# Advanced anomaly detection
chronos anomaly --file data.csv \
  --time-column timestamp \
  --value-column value \
  --method ensemble \
  --sensitivity 0.05 \
  --contextual \
  --seasonal-adjustment \
  --output anomalies.json

# Streaming anomaly detection
chronos anomaly --file streaming_data.csv \
  --time-column timestamp \
  --value-column value \
  --streaming \
  --threshold-percentile 99 \
  --output streaming_anomalies.json
```

### `chronos forecast`

Generate forecasts and predictions.

#### Basic Usage

```bash
chronos forecast --file <FILE> --time-column <COLUMN> --value-column <COLUMN> --horizon <N>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output file | `--output forecast.json` |
| `--method <METHOD>` | Forecasting method | `--method arima` |
| `--horizon <N>` | Forecast horizon (periods) | `--horizon 30` |
| `--confidence-intervals` | Include confidence intervals | `--confidence-intervals` |
| `--confidence-level <LEVEL>` | Confidence level | `--confidence-level 0.95` |
| `--include-seasonality` | Include seasonal components | `--include-seasonality` |
| `--scenario-analysis` | Perform scenario analysis | `--scenario-analysis` |
| `--ensemble-forecasts` | Use ensemble methods | `--ensemble-forecasts` |

#### Forecasting Methods

- `arima` - ARIMA model
- `ets` - Exponential Smoothing (ETS)
- `prophet` - Facebook Prophet
- `linear_regression` - Linear regression
- `ensemble` - Ensemble of multiple methods
- `naive` - Naive forecasting methods

#### Examples

```bash
# Basic forecasting
chronos forecast --file sales.csv \
  --time-column date \
  --value-column revenue \
  --horizon 30

# Advanced forecasting
chronos forecast --file data.csv \
  --time-column timestamp \
  --value-column value \
  --method ensemble \
  --horizon 90 \
  --confidence-intervals \
  --confidence-level 0.95 \
  --include-seasonality \
  --output comprehensive_forecast.json
```

### `chronos correlate`

Analyze correlations and relationships between multiple time series.

#### Basic Usage

```bash
chronos correlate --files <FILES> --time-column <COLUMN> --value-column <COLUMN>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--files <FILES>` | Comma-separated list of input files | `--files "a.csv,b.csv,c.csv"` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output file | `--output correlation.json` |
| `--method <METHOD>` | Correlation method | `--method pearson` |
| `--rolling-correlation` | Compute rolling correlation | `--rolling-correlation` |
| `--window <SIZE>` | Rolling window size | `--window 30` |
| `--granger-causality` | Test Granger causality | `--granger-causality` |
| `--cointegration` | Test for cointegration | `--cointegration` |
| `--dynamic-correlation` | Dynamic correlation analysis | `--dynamic-correlation` |

#### Correlation Methods

- `pearson` - Pearson correlation
- `spearman` - Spearman rank correlation
- `kendall` - Kendall's tau
- `dtw` - Dynamic Time Warping distance
- `mutual_information` - Mutual information

#### Examples

```bash
# Basic correlation analysis
chronos correlate --files "stock1.csv,stock2.csv,stock3.csv" \
  --time-column date --value-column close

# Advanced correlation analysis
chronos correlate --files "series1.csv,series2.csv" \
  --time-column timestamp \
  --value-column value \
  --method pearson \
  --rolling-correlation \
  --window 60 \
  --granger-causality \
  --cointegration \
  --output advanced_correlation.json
```

## Visualization

### `chronos plot`

Create various plots and visualizations.

#### Basic Usage

```bash
chronos plot --file <FILE> --time-column <COLUMN> --value-column <COLUMN> --output <FILE>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--files <FILES>` | Multiple files for comparison | `--files "a.csv,b.csv"` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output image file | `--output plot.png` |
| `--type <TYPE>` | Plot type | `--type line` |
| `--theme <THEME>` | Visual theme | `--theme dark` |
| `--width <WIDTH>` | Plot width in pixels | `--width 1200` |
| `--height <HEIGHT>` | Plot height in pixels | `--height 800` |
| `--title <TITLE>` | Plot title | `--title "Sales Data"` |
| `--format <FORMAT>` | Output format | `--format png` |

#### Plot Types

- `line` - Line plot
- `scatter` - Scatter plot
- `histogram` - Histogram
- `box` - Box plot
- `violin` - Violin plot
- `heatmap` - Heatmap
- `correlation_heatmap` - Correlation heatmap
- `acf` - Autocorrelation function
- `pacf` - Partial autocorrelation function
- `decomposition` - Time series decomposition
- `forecast` - Forecast visualization
- `anomaly` - Anomaly visualization

#### Themes

- `default` - Default theme
- `dark` - Dark theme
- `light` - Light theme
- `minimal` - Minimal theme
- `scientific` - Scientific publication theme

#### Output Formats

- `png` - PNG image
- `svg` - SVG vector image
- `pdf` - PDF document
- `html` - Interactive HTML

#### Examples

```bash
# Basic line plot
chronos plot --file data.csv \
  --time-column date \
  --value-column price \
  --output price_chart.png

# Advanced visualization
chronos plot --file data.csv \
  --time-column timestamp \
  --value-column value \
  --type line \
  --theme dark \
  --width 1600 \
  --height 900 \
  --title "Time Series Analysis" \
  --format svg \
  --output advanced_plot.svg

# Multiple series comparison
chronos plot --files "series1.csv,series2.csv,series3.csv" \
  --time-column date \
  --value-column value \
  --type line \
  --output comparison.png

# Correlation heatmap
chronos plot --files "a.csv,b.csv,c.csv" \
  --time-column date \
  --value-column value \
  --type correlation_heatmap \
  --output correlation.png
```

## Reporting

### `chronos report`

Generate comprehensive reports with insights and recommendations.

#### Basic Usage

```bash
chronos report --file <FILE> --time-column <COLUMN> --value-column <COLUMN> --output <FILE>
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--file <FILE>` | Input data file | `--file data.csv` |
| `--time-column <COLUMN>` | Time column name | `--time-column date` |
| `--value-column <COLUMN>` | Value column name | `--value-column value` |
| `--output <FILE>` | Output report file | `--output report.html` |
| `--template <TEMPLATE>` | Report template | `--template executive` |
| `--format <FORMAT>` | Output format | `--format html` |
| `--include-forecasts` | Include forecast analysis | `--include-forecasts` |
| `--include-risk-metrics` | Include risk analysis | `--include-risk-metrics` |
| `--kpi-metrics` | Include KPI analysis | `--kpi-metrics` |
| `--trend-analysis` | Include trend analysis | `--trend-analysis` |
| `--forecast-integration` | Integrate forecasting | `--forecast-integration` |

#### Report Templates

- `standard` - Standard analysis report
- `executive` - Executive summary report
- `technical` - Technical detailed report
- `financial` - Financial analysis report
- `iot` - IoT monitoring report
- `research` - Scientific research report
- `custom` - Custom template

#### Report Formats

- `html` - Interactive HTML report
- `pdf` - PDF document
- `markdown` - Markdown document
- `json` - Structured JSON data

#### Examples

```bash
# Standard report
chronos report --file data.csv \
  --time-column date \
  --value-column value \
  --output standard_report.html

# Executive dashboard
chronos report --file sales.csv \
  --time-column date \
  --value-column revenue \
  --template executive \
  --kpi-metrics \
  --trend-analysis \
  --forecast-integration \
  --output executive_dashboard.html

# Technical report
chronos report --file sensor_data.csv \
  --time-column timestamp \
  --value-column temperature \
  --template technical \
  --include-forecasts \
  --format pdf \
  --output technical_analysis.pdf
```

## Configuration

### `chronos config`

Manage configuration files, profiles, and settings.

#### Subcommands

- `init` - Initialize configuration
- `show` - Show current configuration
- `set` - Set configuration value
- `get` - Get configuration value
- `profiles` - List available profiles
- `profile` - Show specific profile
- `validate` - Validate configuration

#### Examples

```bash
# Initialize configuration
chronos config init

# Show current configuration
chronos config show

# Set configuration values
chronos config set output.default_format json
chronos config set analysis.default_confidence_level 0.99

# Get configuration values
chronos config get output.default_format

# List profiles
chronos config profiles

# Validate configuration
chronos config validate
```

## Plugin Management

### `chronos plugin`

Manage plugins: install, update, configure, and list plugins.

#### Subcommands

- `list` - List installed plugins
- `install` - Install a plugin
- `update` - Update plugins
- `remove` - Remove a plugin
- `info` - Show plugin information

#### Examples

```bash
# List installed plugins
chronos plugin list

# Install a plugin
chronos plugin install forecasting-advanced

# Update all plugins
chronos plugin update

# Remove a plugin
chronos plugin remove old-plugin

# Show plugin information
chronos plugin info forecasting-advanced
```

## Interactive Mode

Start interactive mode for exploratory analysis:

```bash
chronos --interactive
```

### Interactive Commands

- `load <file> <time_col> <value_col>` - Load data
- `stats` - Show statistics
- `plot <type>` - Create plot
- `trend` - Analyze trends
- `seasonal` - Detect seasonality
- `anomaly` - Detect anomalies
- `forecast <horizon>` - Generate forecast
- `export <file>` - Export results
- `help` - Show help
- `exit` - Exit interactive mode

## Common Patterns

### Basic Analysis Workflow

```bash
# 1. Import and validate data
chronos import --file raw_data.csv --time-column date --value-column value --output clean_data.csv

# 2. Basic statistics
chronos stats --file clean_data.csv --time-column date --value-column value --output stats.json

# 3. Visualize data
chronos plot --file clean_data.csv --time-column date --value-column value --output data_plot.png

# 4. Analyze trends
chronos trend --file clean_data.csv --time-column date --value-column value --output trends.json

# 5. Check seasonality
chronos seasonal --file clean_data.csv --time-column date --value-column value --output seasonal.json

# 6. Detect anomalies
chronos anomaly --file clean_data.csv --time-column date --value-column value --output anomalies.json

# 7. Generate forecasts
chronos forecast --file clean_data.csv --time-column date --value-column value --horizon 30 --output forecast.json

# 8. Create comprehensive report
chronos report --file clean_data.csv --time-column date --value-column value --output final_report.html
```

### Batch Processing

```bash
# Process multiple files
for file in data/*.csv; do
    chronos stats --file "$file" --time-column timestamp --value-column value --output "results/$(basename $file .csv)_stats.json"
done
```

### Configuration-Driven Analysis

```bash
# Use configuration file for consistent analysis
chronos --config analysis_config.toml report --file data.csv --output report.html
```

For more examples and use cases, see the [Tutorial](tutorial.md) and [Examples Gallery](../examples/).

## Getting Help

- Use `--help` with any command for detailed options
- Check the [FAQ](faq.md) for common questions
- See the [Troubleshooting Guide](troubleshooting.md) for solutions
- Visit the [Tutorial](tutorial.md) for hands-on examples