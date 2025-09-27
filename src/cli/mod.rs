//! # Comprehensive CLI Interface Module
//!
//! This module provides the main CLI interface for Chronos time series analysis tool.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Chronos - A powerful CLI tool for comprehensive time series analysis
#[derive(Parser)]
#[clap(name = "chronos")]
#[clap(about = "A comprehensive CLI tool for time series data analysis and visualization")]
#[clap(version = "0.1.0")]
#[clap(author = "Chronos Contributors")]
#[clap(long_about = "
Chronos is a powerful command-line tool for time series analysis that provides:
  • Statistical analysis and metrics
  • Trend detection and decomposition
  • Seasonality analysis
  • Anomaly detection
  • Forecasting capabilities
  • Correlation analysis
  • Beautiful visualizations
  • Comprehensive reporting

Use 'chronos <command> --help' for more information on a specific command.
")]
pub struct Cli {
    /// Configuration file path
    #[clap(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Enable verbose output for detailed information
    #[clap(short, long, global = true)]
    pub verbose: bool,

    /// Enable quiet mode (minimal output)
    #[clap(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Output directory for generated files
    #[clap(short = 'o', long, global = true, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Default output format for results
    #[clap(long, global = true, value_enum, default_value = "text")]
    pub format: OutputFormat,

    /// Enable interactive mode for exploration
    #[clap(short = 'i', long)]
    pub interactive: bool,

    #[clap(subcommand)]
    pub command: Option<Commands>,
}

/// Available output formats
#[derive(Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    /// Plain text output
    Text,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// PDF format
    Pdf,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Import and preprocess time series data
    #[clap(about = "Import data from various formats and perform preprocessing")]
    Import(ImportCommand),

    /// Perform statistical analysis on time series data
    #[clap(about = "Comprehensive statistical analysis including descriptive stats, distributions, and tests")]
    Stats(StatsCommand),

    /// Analyze trends and perform decomposition
    #[clap(about = "Trend detection, analysis, and time series decomposition")]
    Trend(TrendCommand),

    /// Detect and analyze seasonality patterns
    #[clap(about = "Seasonality detection, strength analysis, and seasonal adjustment")]
    Seasonal(SeasonalCommand),

    /// Detect anomalies and outliers
    #[clap(about = "Anomaly detection using various statistical methods")]
    Anomaly(AnomalyCommand),

    /// Generate forecasts and predictions
    #[clap(about = "Time series forecasting using multiple methods")]
    Forecast(ForecastCommand),

    /// Analyze correlations and relationships
    #[clap(about = "Correlation analysis between multiple time series")]
    Correlate(CorrelateCommand),

    /// Generate visualizations and plots
    #[clap(about = "Create various plots and visualizations for time series data")]
    Plot(PlotCommand),

    /// Generate comprehensive analysis reports
    #[clap(about = "Generate detailed reports with insights and recommendations")]
    Report(ReportCommand),
}

/// Import command structure
#[derive(Parser, Clone)]
pub struct ImportCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// File format (auto-detect if not specified)
    #[clap(long, value_enum)]
    pub format: Option<ImportFormat>,

    /// Column name or index for timestamps
    #[clap(short = 't', long)]
    pub time_column: Option<String>,

    /// Column names or indices for values (comma-separated)
    #[clap(short = 'v', long)]
    pub value_columns: Option<String>,

    /// Handle missing values: drop, interpolate, forward, backward
    #[clap(long, default_value = "interpolate")]
    pub missing: String,

    /// Resample to regular frequency
    #[clap(long)]
    pub resample: Option<String>,

    /// Output preprocessed data to file
    #[clap(short, long)]
    pub output: Option<PathBuf>,

    /// Validate data quality after import
    #[clap(long)]
    pub validate: bool,
}

/// Statistics command structure
#[derive(Parser, Clone)]
pub struct StatsCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column name to analyze (analyze all if not specified)
    #[clap(short = 'c', long)]
    pub column: Option<String>,

    /// Include normality tests
    #[clap(long)]
    pub normality: bool,

    /// Include stationarity tests
    #[clap(long)]
    pub stationarity: bool,

    /// Compute autocorrelation with max lags
    #[clap(long, value_name = "LAGS")]
    pub autocorr: Option<usize>,

    /// Detect change points
    #[clap(long)]
    pub changepoints: bool,

    /// Output results to file
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Trend command structure
#[derive(Parser, Clone)]
pub struct TrendCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column name to analyze
    #[clap(short = 'c', long, default_value = "value")]
    pub column: String,

    /// Analysis method: detect, decompose, detrend, all
    #[clap(short = 'm', long, default_value = "all")]
    pub method: String,

    /// Decomposition method: classical, stl, x11
    #[clap(long, default_value = "stl")]
    pub decomposition: String,

    /// Detrending method: linear, polynomial, moving_average, hp_filter
    #[clap(long)]
    pub detrending: Option<String>,

    /// Seasonal period (auto-detect if not specified)
    #[clap(long)]
    pub period: Option<usize>,

    /// Output decomposed components
    #[clap(long)]
    pub export_components: bool,

    /// Output results to file
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Seasonal command structure
#[derive(Parser, Clone)]
pub struct SeasonalCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column name to analyze
    #[clap(short = 'c', long, default_value = "value")]
    pub column: String,

    /// Analysis method: detect, strength, adjust, all
    #[clap(short = 'm', long, default_value = "detect")]
    pub method: String,

    /// Maximum period to test
    #[clap(long, default_value = "365")]
    pub max_period: usize,

    /// Minimum period to test
    #[clap(long, default_value = "2")]
    pub min_period: usize,

    /// Force specific periods (comma-separated)
    #[clap(long)]
    pub periods: Option<String>,

    /// Adjustment method: x13, stl, moving_average
    #[clap(long, default_value = "stl")]
    pub adjustment: String,

    /// Export adjusted series
    #[clap(long)]
    pub export_adjusted: bool,

    /// Analyze calendar effects
    #[clap(long)]
    pub calendar_effects: bool,

    /// Output results to file
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Anomaly command structure
#[derive(Parser, Clone)]
pub struct AnomalyCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column name to analyze
    #[clap(short = 'c', long, default_value = "value")]
    pub column: String,

    /// Detection method: zscore, iqr, isolation_forest, all
    #[clap(short = 'm', long, default_value = "zscore")]
    pub method: String,

    /// Threshold for anomaly detection
    #[clap(short = 't', long)]
    pub threshold: Option<f64>,

    /// Window size for contextual anomaly detection
    #[clap(short = 'w', long)]
    pub window: Option<usize>,

    /// Mark anomalies in output
    #[clap(long)]
    pub mark: bool,

    /// Export anomaly scores
    #[clap(long)]
    pub export_scores: bool,

    /// Output results to file
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Forecast command structure
#[derive(Parser, Clone)]
pub struct ForecastCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column name to forecast
    #[clap(short = 'c', long, default_value = "value")]
    pub column: String,

    /// Forecasting method: arima, exponential, prophet, lstm, ensemble
    #[clap(short = 'm', long, default_value = "arima")]
    pub method: String,

    /// Forecast horizon (number of periods)
    #[clap(short = 'h', long, default_value = "10")]
    pub horizon: usize,

    /// Confidence level for prediction intervals
    #[clap(long, default_value = "0.95")]
    pub confidence: f64,

    /// Include backtesting validation
    #[clap(long)]
    pub backtest: bool,

    /// Number of backtesting windows
    #[clap(long, default_value = "5")]
    pub backtest_windows: usize,

    /// Export forecasts with confidence intervals
    #[clap(long)]
    pub export_forecast: bool,

    /// Output results to file
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Correlate command structure
#[derive(Parser, Clone)]
pub struct CorrelateCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column names to correlate (comma-separated, all if not specified)
    #[clap(short = 'c', long)]
    pub columns: Option<String>,

    /// Correlation method: pearson, spearman, kendall
    #[clap(short = 'm', long, default_value = "pearson")]
    pub method: String,

    /// Include lagged correlations
    #[clap(long)]
    pub lagged: bool,

    /// Maximum lag for lagged correlations
    #[clap(long, default_value = "10")]
    pub max_lag: usize,

    /// Significance level for correlation tests
    #[clap(long, default_value = "0.05")]
    pub alpha: f64,

    /// Generate correlation matrix heatmap
    #[clap(long)]
    pub heatmap: bool,

    /// Output results to file
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Plot command structure
#[derive(Parser, Clone)]
pub struct PlotCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column names to plot (comma-separated)
    #[clap(short = 'c', long)]
    pub columns: Option<String>,

    /// Plot type: line, scatter, histogram, box, heatmap, decomposition
    #[clap(short = 't', long, default_value = "line")]
    pub plot_type: String,

    /// Plot title
    #[clap(long)]
    pub title: Option<String>,

    /// X-axis label
    #[clap(long)]
    pub xlabel: Option<String>,

    /// Y-axis label
    #[clap(long)]
    pub ylabel: Option<String>,

    /// Figure size (width,height)
    #[clap(long, default_value = "12,6")]
    pub size: String,

    /// DPI for output image
    #[clap(long, default_value = "100")]
    pub dpi: u32,

    /// Add grid to plot
    #[clap(long)]
    pub grid: bool,

    /// Interactive plot (opens in browser)
    #[clap(long)]
    pub interactive: bool,

    /// Output file path
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Report command structure
#[derive(Parser, Clone)]
pub struct ReportCommand {
    /// Input file path
    #[clap(short, long, value_name = "FILE")]
    pub file: PathBuf,

    /// Column name to analyze
    #[clap(short = 'c', long)]
    pub column: Option<String>,

    /// Report template: executive, technical, data_quality, comprehensive
    #[clap(short = 't', long, default_value = "comprehensive")]
    pub template: String,

    /// Report sections to include (comma-separated)
    #[clap(long)]
    pub sections: Option<String>,

    /// Report title
    #[clap(long)]
    pub title: Option<String>,

    /// Report author
    #[clap(long)]
    pub author: Option<String>,

    /// Include all analyses
    #[clap(long)]
    pub comprehensive: bool,

    /// Maximum insights to generate
    #[clap(long, default_value = "10")]
    pub max_insights: usize,

    /// Output file path
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

/// Import format options
#[derive(Clone, Debug, ValueEnum, PartialEq)]
pub enum ImportFormat {
    Csv,
    Json,
    Parquet,
    Excel,
    Tsv,
}

impl OutputFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &str {
        match self {
            OutputFormat::Text => "txt",
            OutputFormat::Json => "json",
            OutputFormat::Csv => "csv",
            OutputFormat::Markdown => "md",
            OutputFormat::Html => "html",
            OutputFormat::Pdf => "pdf",
        }
    }
}

// Submodules
pub mod commands;
pub mod help;
pub mod interactive;

#[cfg(test)]
mod tests;