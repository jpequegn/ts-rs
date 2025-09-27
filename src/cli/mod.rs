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

    /// Configuration management
    #[clap(about = "Manage configuration files, profiles, and settings")]
    Config(ConfigCommand),

    /// Plugin management
    #[clap(about = "Manage plugins: install, update, configure, and list plugins")]
    Plugin(PluginCommand),
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

/// Configuration command structure
#[derive(Parser, Clone)]
pub struct ConfigCommand {
    #[clap(subcommand)]
    pub action: ConfigAction,
}

/// Configuration management actions
#[derive(Subcommand, Clone)]
pub enum ConfigAction {
    /// Show current configuration
    #[clap(about = "Display current configuration settings")]
    Show {
        /// Configuration section to show (all if not specified)
        #[clap(short, long)]
        section: Option<String>,

        /// Show sources of configuration values
        #[clap(long)]
        sources: bool,

        /// Output format
        #[clap(short, long, value_enum, default_value = "text")]
        format: ConfigOutputFormat,
    },

    /// Create default configuration file
    #[clap(about = "Create a default configuration file")]
    Init {
        /// Configuration file path (default: ~/.config/chronos/config.toml)
        #[clap(short, long)]
        path: Option<PathBuf>,

        /// Configuration format
        #[clap(short, long, value_enum, default_value = "toml")]
        format: ConfigFormat,

        /// Overwrite existing configuration file
        #[clap(long)]
        force: bool,
    },

    /// Set configuration value
    #[clap(about = "Set a configuration value")]
    Set {
        /// Configuration key (e.g., analysis.statistics.confidence_level)
        key: String,

        /// Configuration value
        value: String,

        /// Configuration file to modify
        #[clap(short, long)]
        config: Option<PathBuf>,

        /// Apply to specific profile
        #[clap(short, long)]
        profile: Option<String>,
    },

    /// Get configuration value
    #[clap(about = "Get a configuration value")]
    Get {
        /// Configuration key (e.g., analysis.statistics.confidence_level)
        key: String,

        /// Configuration file to read from
        #[clap(short, long)]
        config: Option<PathBuf>,

        /// Get from specific profile
        #[clap(short, long)]
        profile: Option<String>,
    },

    /// List available profiles
    #[clap(about = "List available configuration profiles")]
    Profiles {
        /// Show detailed profile information
        #[clap(short, long)]
        detailed: bool,

        /// Configuration file to read from
        #[clap(short, long)]
        config: Option<PathBuf>,
    },

    /// Switch to a different profile
    #[clap(about = "Switch to a different configuration profile")]
    Profile {
        /// Profile name to activate
        name: String,

        /// Configuration file to modify
        #[clap(short, long)]
        config: Option<PathBuf>,
    },

    /// Validate configuration
    #[clap(about = "Validate configuration file and settings")]
    Validate {
        /// Configuration file to validate
        #[clap(short, long)]
        config: Option<PathBuf>,

        /// Profile to validate
        #[clap(short, long)]
        profile: Option<String>,

        /// Show warnings and suggestions
        #[clap(short, long)]
        verbose: bool,
    },

    /// Edit configuration file
    #[clap(about = "Open configuration file in default editor")]
    Edit {
        /// Configuration file to edit
        #[clap(short, long)]
        config: Option<PathBuf>,

        /// Editor to use (default: $EDITOR environment variable)
        #[clap(short, long)]
        editor: Option<String>,
    },
}

/// Configuration output formats
#[derive(Clone, Debug, ValueEnum)]
pub enum ConfigOutputFormat {
    Text,
    Json,
    Yaml,
    Toml,
}

/// Configuration file formats
#[derive(Clone, Debug, ValueEnum)]
pub enum ConfigFormat {
    Toml,
    Yaml,
    Json,
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

/// Plugin command structure
#[derive(Parser, Clone)]
pub struct PluginCommand {
    #[clap(subcommand)]
    pub action: PluginAction,
}

/// Plugin management actions
#[derive(Subcommand, Clone)]
pub enum PluginAction {
    /// List installed plugins
    #[clap(about = "List all installed plugins")]
    List {
        /// Show detailed plugin information
        #[clap(short, long)]
        detailed: bool,

        /// Filter by plugin type
        #[clap(short, long)]
        plugin_type: Option<String>,

        /// Output format
        #[clap(short, long, value_enum, default_value = "text")]
        format: ConfigOutputFormat,
    },

    /// Search for available plugins
    #[clap(about = "Search for plugins in repositories")]
    Search {
        /// Search query (plugin name or keyword)
        query: Option<String>,

        /// Plugin category to search in
        #[clap(short, long)]
        category: Option<String>,

        /// Repository to search in
        #[clap(short, long)]
        repository: Option<String>,

        /// Output format
        #[clap(short, long, value_enum, default_value = "text")]
        format: ConfigOutputFormat,
    },

    /// Install a plugin
    #[clap(about = "Install a plugin from repository")]
    Install {
        /// Plugin ID to install
        plugin_id: String,

        /// Specific version to install
        #[clap(short, long)]
        version: Option<String>,

        /// Repository to install from
        #[clap(short, long)]
        repository: Option<String>,

        /// Force installation (overwrite existing)
        #[clap(long)]
        force: bool,

        /// Plugin configuration file
        #[clap(short, long)]
        config: Option<PathBuf>,
    },

    /// Uninstall a plugin
    #[clap(about = "Uninstall an installed plugin")]
    Uninstall {
        /// Plugin ID to uninstall
        plugin_id: String,

        /// Create backup before uninstalling
        #[clap(short, long)]
        backup: bool,

        /// Remove plugin data directory
        #[clap(long)]
        remove_data: bool,
    },

    /// Update plugins
    #[clap(about = "Update installed plugins")]
    Update {
        /// Plugin ID to update (update all if not specified)
        plugin_id: Option<String>,

        /// Target version to update to
        #[clap(short, long)]
        version: Option<String>,

        /// Force update even if versions match
        #[clap(long)]
        force: bool,

        /// Check for updates without installing
        #[clap(long)]
        check_only: bool,
    },

    /// Configure a plugin
    #[clap(about = "Configure plugin settings")]
    Configure {
        /// Plugin ID to configure
        plugin_id: String,

        /// Configuration key to set
        #[clap(short, long)]
        key: Option<String>,

        /// Configuration value to set
        #[clap(short, long)]
        value: Option<String>,

        /// Configuration file to apply
        #[clap(short, long)]
        config_file: Option<PathBuf>,

        /// Show current configuration
        #[clap(long)]
        show: bool,
    },

    /// Show plugin information
    #[clap(about = "Show detailed information about a plugin")]
    Info {
        /// Plugin ID to show information for
        plugin_id: String,

        /// Show plugin status and health
        #[clap(long)]
        status: bool,

        /// Show plugin configuration
        #[clap(long)]
        config: bool,

        /// Output format
        #[clap(short, long, value_enum, default_value = "text")]
        format: ConfigOutputFormat,
    },

    /// Manage plugin repositories
    #[clap(about = "Manage plugin repositories")]
    Repository {
        #[clap(subcommand)]
        action: RepositoryAction,
    },

    /// Enable or disable plugins
    #[clap(about = "Enable or disable plugins")]
    Toggle {
        /// Plugin ID to toggle
        plugin_id: String,

        /// Enable the plugin
        #[clap(long, conflicts_with = "disable")]
        enable: bool,

        /// Disable the plugin
        #[clap(long, conflicts_with = "enable")]
        disable: bool,
    },

    /// Plugin development tools
    #[clap(about = "Tools for plugin development")]
    Dev {
        #[clap(subcommand)]
        action: DevAction,
    },
}

/// Repository management actions
#[derive(Subcommand, Clone)]
pub enum RepositoryAction {
    /// List configured repositories
    #[clap(about = "List all configured plugin repositories")]
    List {
        /// Show detailed repository information
        #[clap(short, long)]
        detailed: bool,
    },

    /// Add a new repository
    #[clap(about = "Add a new plugin repository")]
    Add {
        /// Repository name
        name: String,

        /// Repository URL
        url: String,

        /// Repository type
        #[clap(short, long, value_enum, default_value = "http")]
        repo_type: RepositoryType,

        /// Repository priority
        #[clap(short, long, default_value = "100")]
        priority: i32,
    },

    /// Remove a repository
    #[clap(about = "Remove a plugin repository")]
    Remove {
        /// Repository name to remove
        name: String,
    },

    /// Update repository information
    #[clap(about = "Update repository plugin listings")]
    Update {
        /// Repository name to update (all if not specified)
        name: Option<String>,
    },
}

/// Repository types for CLI
#[derive(Clone, Debug, ValueEnum)]
pub enum RepositoryType {
    Http,
    Git,
    Local,
    Registry,
}

/// Plugin development actions
#[derive(Subcommand, Clone)]
pub enum DevAction {
    /// Create a new plugin template
    #[clap(about = "Create a new plugin template")]
    New {
        /// Plugin name
        name: String,

        /// Plugin type
        #[clap(short, long, value_enum)]
        plugin_type: PluginType,

        /// Output directory
        #[clap(short, long)]
        output: Option<PathBuf>,

        /// Plugin template to use
        #[clap(short, long)]
        template: Option<String>,
    },

    /// Validate plugin metadata and structure
    #[clap(about = "Validate plugin metadata and structure")]
    Validate {
        /// Plugin directory to validate
        path: PathBuf,
    },

    /// Package plugin for distribution
    #[clap(about = "Package plugin for distribution")]
    Package {
        /// Plugin directory to package
        path: PathBuf,

        /// Output file path
        #[clap(short, long)]
        output: Option<PathBuf>,

        /// Package format
        #[clap(short, long, value_enum, default_value = "zip")]
        format: PackageFormat,
    },

    /// Test plugin functionality
    #[clap(about = "Test plugin functionality")]
    Test {
        /// Plugin directory to test
        path: PathBuf,

        /// Test configuration file
        #[clap(short, long)]
        config: Option<PathBuf>,
    },
}

/// Plugin types for CLI
#[derive(Clone, Debug, ValueEnum)]
pub enum PluginType {
    DataSource,
    Analysis,
    Visualization,
}

/// Package formats for plugin distribution
#[derive(Clone, Debug, ValueEnum)]
pub enum PackageFormat {
    Zip,
    Tar,
    TarGz,
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