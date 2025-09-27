//! # Comprehensive Help System
//!
//! This module provides detailed help documentation with examples for all commands.

use colored::Colorize;

/// Display comprehensive help for a specific command
pub fn show_command_help(command: &str) {
    match command {
        "import" => show_import_help(),
        "stats" => show_stats_help(),
        "trend" => show_trend_help(),
        "seasonal" => show_seasonal_help(),
        "anomaly" => show_anomaly_help(),
        "forecast" => show_forecast_help(),
        "correlate" => show_correlate_help(),
        "plot" => show_plot_help(),
        "report" => show_report_help(),
        _ => show_general_help(),
    }
}

/// Show general help
pub fn show_general_help() {
    println!("\n{}", "Chronos - Time Series Analysis Tool".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos [OPTIONS] <COMMAND> [ARGS]");

    println!("\n{}", "GLOBAL OPTIONS:".yellow().bold());
    println!("  {:<20} {}", "--config <FILE>".green(), "Use configuration file");
    println!("  {:<20} {}", "-v, --verbose".green(), "Enable verbose output");
    println!("  {:<20} {}", "-q, --quiet".green(), "Enable quiet mode");
    println!("  {:<20} {}", "-o, --output-dir".green(), "Output directory");
    println!("  {:<20} {}", "--format <FORMAT>".green(), "Output format (text, json, csv, markdown, html, pdf)");
    println!("  {:<20} {}", "-i, --interactive".green(), "Enter interactive mode");

    println!("\n{}", "COMMANDS:".yellow().bold());
    println!("  {:<20} {}", "import".cyan(), "Import and preprocess data");
    println!("  {:<20} {}", "stats".cyan(), "Statistical analysis");
    println!("  {:<20} {}", "trend".cyan(), "Trend analysis and decomposition");
    println!("  {:<20} {}", "seasonal".cyan(), "Seasonality detection and analysis");
    println!("  {:<20} {}", "anomaly".cyan(), "Anomaly detection");
    println!("  {:<20} {}", "forecast".cyan(), "Time series forecasting");
    println!("  {:<20} {}", "correlate".cyan(), "Correlation analysis");
    println!("  {:<20} {}", "plot".cyan(), "Data visualization");
    println!("  {:<20} {}", "report".cyan(), "Generate comprehensive reports");

    println!("\n{}", "EXAMPLES:".yellow().bold());
    println!("  # Import and validate data");
    println!("  {}", "chronos import -f data.csv --validate".green());

    println!("\n  # Run statistical analysis");
    println!("  {}", "chronos stats -f data.csv --normality --stationarity".green());

    println!("\n  # Detect seasonality");
    println!("  {}", "chronos seasonal -f data.csv --method detect".green());

    println!("\n  # Generate forecast");
    println!("  {}", "chronos forecast -f data.csv --horizon 30".green());

    println!("\n  # Enter interactive mode");
    println!("  {}", "chronos --interactive".green());

    println!("\nFor more help on a specific command, use:");
    println!("  {}", "chronos <command> --help".yellow());
    println!();
}

/// Show help for import command
fn show_import_help() {
    println!("\n{}", "IMPORT - Data Import and Preprocessing".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Import time series data from various formats and perform preprocessing.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos import [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "--format <FORMAT>".green(), "File format (csv, json, excel, parquet)");
    println!("  {:<25} {}", "-t, --time-column".green(), "Column name/index for timestamps");
    println!("  {:<25} {}", "-v, --value-columns".green(), "Column names for values (comma-separated)");
    println!("  {:<25} {}", "--missing <METHOD>".green(), "Handle missing values (drop, interpolate, forward, backward)");
    println!("  {:<25} {}", "--resample <FREQ>".green(), "Resample to regular frequency");
    println!("  {:<25} {}", "--validate".green(), "Validate data quality after import");
    println!("  {:<25} {}", "-o, --output <FILE>".green(), "Output preprocessed data");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Import CSV with automatic detection");
    println!("  {}", "chronos import -f sales.csv".green());

    println!("\n  # Import with specific columns");
    println!("  {}", "chronos import -f data.csv -t date -v revenue,quantity".green());

    println!("\n  # Import and handle missing values");
    println!("  {}", "chronos import -f data.csv --missing interpolate".green());

    println!("\n  # Import, resample, and validate");
    println!("  {}", "chronos import -f hourly.csv --resample 1D --validate".green());

    println!("\n  # Import Excel file");
    println!("  {}", "chronos import -f data.xlsx --format excel".green());
    println!();
}

/// Show help for stats command
fn show_stats_help() {
    println!("\n{}", "STATS - Statistical Analysis".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Perform comprehensive statistical analysis on time series data.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos stats [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --column <NAME>".green(), "Column to analyze");
    println!("  {:<25} {}", "--normality".green(), "Include normality tests");
    println!("  {:<25} {}", "--stationarity".green(), "Include stationarity tests");
    println!("  {:<25} {}", "--autocorr <LAGS>".green(), "Compute autocorrelation");
    println!("  {:<25} {}", "--changepoints".green(), "Detect change points");
    println!("  {:<25} {}", "-o, --output <FILE>".green(), "Output results to file");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Basic statistical analysis");
    println!("  {}", "chronos stats -f data.csv".green());

    println!("\n  # Full statistical analysis");
    println!("  {}", "chronos stats -f data.csv --normality --stationarity --changepoints".green());

    println!("\n  # Autocorrelation analysis");
    println!("  {}", "chronos stats -f data.csv --autocorr 50".green());

    println!("\n  # Export results to JSON");
    println!("  {}", "chronos stats -f data.csv -o stats.json --format json".green());
    println!();
}

/// Show help for trend command
fn show_trend_help() {
    println!("\n{}", "TREND - Trend Analysis and Decomposition".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Analyze trends, perform decomposition, and detrend time series data.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos trend [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --column <NAME>".green(), "Column to analyze");
    println!("  {:<25} {}", "-m, --method <METHOD>".green(), "Analysis method (detect, decompose, detrend, all)");
    println!("  {:<25} {}", "--decomposition".green(), "Method (classical, stl, x11)");
    println!("  {:<25} {}", "--detrending".green(), "Method (linear, polynomial, moving_average, hp_filter)");
    println!("  {:<25} {}", "--period <N>".green(), "Seasonal period");
    println!("  {:<25} {}", "--export-components".green(), "Export decomposed components");
    println!("  {:<25} {}", "-o, --output <FILE>".green(), "Output results");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Detect trend");
    println!("  {}", "chronos trend -f data.csv --method detect".green());

    println!("\n  # STL decomposition");
    println!("  {}", "chronos trend -f data.csv --method decompose --decomposition stl".green());

    println!("\n  # Linear detrending");
    println!("  {}", "chronos trend -f data.csv --method detrend --detrending linear".green());

    println!("\n  # Full analysis with export");
    println!("  {}", "chronos trend -f data.csv --method all --export-components".green());
    println!();
}

/// Show help for seasonal command
fn show_seasonal_help() {
    println!("\n{}", "SEASONAL - Seasonality Analysis".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Detect and analyze seasonal patterns in time series data.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos seasonal [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --column <NAME>".green(), "Column to analyze");
    println!("  {:<25} {}", "-m, --method <METHOD>".green(), "Method (detect, strength, adjust, all)");
    println!("  {:<25} {}", "--max-period <N>".green(), "Maximum period to test");
    println!("  {:<25} {}", "--min-period <N>".green(), "Minimum period to test");
    println!("  {:<25} {}", "--periods <P1,P2>".green(), "Specific periods to test");
    println!("  {:<25} {}", "--adjustment <METHOD>".green(), "Adjustment method (x13, stl, moving_average)");
    println!("  {:<25} {}", "--export-adjusted".green(), "Export seasonally adjusted series");
    println!("  {:<25} {}", "--calendar-effects".green(), "Analyze calendar effects");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Detect seasonality");
    println!("  {}", "chronos seasonal -f data.csv --method detect".green());

    println!("\n  # Test specific periods");
    println!("  {}", "chronos seasonal -f data.csv --periods 7,30,365".green());

    println!("\n  # Seasonal adjustment");
    println!("  {}", "chronos seasonal -f data.csv --method adjust --adjustment stl".green());

    println!("\n  # Full analysis with calendar effects");
    println!("  {}", "chronos seasonal -f data.csv --method all --calendar-effects".green());
    println!();
}

/// Show help for anomaly command
fn show_anomaly_help() {
    println!("\n{}", "ANOMALY - Anomaly Detection".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Detect anomalies and outliers in time series data.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos anomaly [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --column <NAME>".green(), "Column to analyze");
    println!("  {:<25} {}", "-m, --method <METHOD>".green(), "Method (zscore, iqr, isolation_forest, all)");
    println!("  {:<25} {}", "-t, --threshold <VAL>".green(), "Detection threshold");
    println!("  {:<25} {}", "-w, --window <SIZE>".green(), "Window size for contextual detection");
    println!("  {:<25} {}", "--mark".green(), "Mark anomalies in output");
    println!("  {:<25} {}", "--export-scores".green(), "Export anomaly scores");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Z-score based detection");
    println!("  {}", "chronos anomaly -f data.csv --method zscore --threshold 3".green());

    println!("\n  # IQR method");
    println!("  {}", "chronos anomaly -f data.csv --method iqr".green());

    println!("\n  # Contextual anomalies");
    println!("  {}", "chronos anomaly -f data.csv --window 30 --mark".green());

    println!("\n  # All methods with scores");
    println!("  {}", "chronos anomaly -f data.csv --method all --export-scores".green());
    println!();
}

/// Show help for forecast command
fn show_forecast_help() {
    println!("\n{}", "FORECAST - Time Series Forecasting".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Generate forecasts and predictions for time series data.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos forecast [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --column <NAME>".green(), "Column to forecast");
    println!("  {:<25} {}", "-m, --method <METHOD>".green(), "Method (arima, exponential, prophet, ensemble)");
    println!("  {:<25} {}", "-h, --horizon <N>".green(), "Forecast horizon");
    println!("  {:<25} {}", "--confidence <LEVEL>".green(), "Confidence level (0-1)");
    println!("  {:<25} {}", "--backtest".green(), "Perform backtesting");
    println!("  {:<25} {}", "--backtest-windows".green(), "Number of backtest windows");
    println!("  {:<25} {}", "--export-forecast".green(), "Export forecasts");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Simple forecast");
    println!("  {}", "chronos forecast -f data.csv --horizon 30".green());

    println!("\n  # ARIMA with confidence intervals");
    println!("  {}", "chronos forecast -f data.csv --method arima --confidence 0.95".green());

    println!("\n  # Forecast with backtesting");
    println!("  {}", "chronos forecast -f data.csv --backtest --backtest-windows 5".green());

    println!("\n  # Ensemble forecast");
    println!("  {}", "chronos forecast -f data.csv --method ensemble --export-forecast".green());
    println!();
}

/// Show help for correlate command
fn show_correlate_help() {
    println!("\n{}", "CORRELATE - Correlation Analysis".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Analyze correlations between time series variables.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos correlate [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --columns <COLS>".green(), "Columns to correlate");
    println!("  {:<25} {}", "-m, --method <METHOD>".green(), "Method (pearson, spearman, kendall)");
    println!("  {:<25} {}", "--lagged".green(), "Include lagged correlations");
    println!("  {:<25} {}", "--max-lag <N>".green(), "Maximum lag");
    println!("  {:<25} {}", "--alpha <LEVEL>".green(), "Significance level");
    println!("  {:<25} {}", "--heatmap".green(), "Generate correlation heatmap");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Basic correlation");
    println!("  {}", "chronos correlate -f data.csv".green());

    println!("\n  # Specific columns with Spearman");
    println!("  {}", "chronos correlate -f data.csv -c col1,col2 --method spearman".green());

    println!("\n  # Lagged correlations");
    println!("  {}", "chronos correlate -f data.csv --lagged --max-lag 10".green());

    println!("\n  # With heatmap visualization");
    println!("  {}", "chronos correlate -f data.csv --heatmap".green());
    println!();
}

/// Show help for plot command
fn show_plot_help() {
    println!("\n{}", "PLOT - Data Visualization".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Create various visualizations for time series data.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos plot [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --columns <COLS>".green(), "Columns to plot");
    println!("  {:<25} {}", "-t, --type <TYPE>".green(), "Plot type (line, scatter, histogram, box, heatmap)");
    println!("  {:<25} {}", "--title <TITLE>".green(), "Plot title");
    println!("  {:<25} {}", "--xlabel <LABEL>".green(), "X-axis label");
    println!("  {:<25} {}", "--ylabel <LABEL>".green(), "Y-axis label");
    println!("  {:<25} {}", "--size <W,H>".green(), "Figure size");
    println!("  {:<25} {}", "--dpi <VALUE>".green(), "DPI for output");
    println!("  {:<25} {}", "--grid".green(), "Add grid");
    println!("  {:<25} {}", "--interactive".green(), "Interactive plot");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Simple line plot");
    println!("  {}", "chronos plot -f data.csv".green());

    println!("\n  # Multiple series");
    println!("  {}", "chronos plot -f data.csv -c col1,col2,col3".green());

    println!("\n  # Histogram with customization");
    println!("  {}", "chronos plot -f data.csv --type histogram --title \"Distribution\"".green());

    println!("\n  # Interactive plot");
    println!("  {}", "chronos plot -f data.csv --interactive".green());
    println!();
}

/// Show help for report command
fn show_report_help() {
    println!("\n{}", "REPORT - Comprehensive Reports".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "DESCRIPTION:".yellow().bold());
    println!("  Generate comprehensive analysis reports with insights and recommendations.");

    println!("\n{}", "USAGE:".yellow().bold());
    println!("  chronos report [OPTIONS]");

    println!("\n{}", "OPTIONS:".yellow().bold());
    println!("  {:<25} {}", "-f, --file <FILE>".green(), "Input file path");
    println!("  {:<25} {}", "-c, --column <NAME>".green(), "Column to analyze");
    println!("  {:<25} {}", "-t, --template".green(), "Template (executive, technical, comprehensive)");
    println!("  {:<25} {}", "--sections <LIST>".green(), "Sections to include");
    println!("  {:<25} {}", "--title <TITLE>".green(), "Report title");
    println!("  {:<25} {}", "--author <NAME>".green(), "Report author");
    println!("  {:<25} {}", "--comprehensive".green(), "Include all analyses");
    println!("  {:<25} {}", "--max-insights <N>".green(), "Maximum insights");
    println!("  {:<25} {}", "-o, --output <FILE>".green(), "Output file");

    println!("\n{}", "EXAMPLES:".yellow().bold());

    println!("\n  # Executive summary");
    println!("  {}", "chronos report -f data.csv --template executive".green());

    println!("\n  # Technical report");
    println!("  {}", "chronos report -f data.csv --template technical --comprehensive".green());

    println!("\n  # Custom sections");
    println!("  {}", "chronos report -f data.csv --sections stats,trend,seasonal".green());

    println!("\n  # PDF report with metadata");
    println!("  {}", "chronos report -f data.csv --title \"Q4 Analysis\" --author \"Data Team\" -o report.pdf".green());
    println!();
}

/// Show tutorial for beginners
pub fn show_tutorial() {
    println!("\n{}", "Chronos Tutorial - Getting Started".cyan().bold());
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "STEP 1: Import Your Data".yellow().bold());
    println!("  First, import your time series data:");
    println!("  {}", "chronos import -f your_data.csv --validate".green());

    println!("\n{}", "STEP 2: Explore with Statistics".yellow().bold());
    println!("  Get basic statistics about your data:");
    println!("  {}", "chronos stats -f your_data.csv".green());

    println!("\n{}", "STEP 3: Analyze Patterns".yellow().bold());
    println!("  Look for trends and seasonality:");
    println!("  {}", "chronos trend -f your_data.csv --method detect".green());
    println!("  {}", "chronos seasonal -f your_data.csv --method detect".green());

    println!("\n{}", "STEP 4: Check for Anomalies".yellow().bold());
    println!("  Identify unusual patterns:");
    println!("  {}", "chronos anomaly -f your_data.csv".green());

    println!("\n{}", "STEP 5: Create Visualizations".yellow().bold());
    println!("  Visualize your data:");
    println!("  {}", "chronos plot -f your_data.csv --type line".green());

    println!("\n{}", "STEP 6: Generate Forecasts".yellow().bold());
    println!("  Predict future values:");
    println!("  {}", "chronos forecast -f your_data.csv --horizon 30".green());

    println!("\n{}", "STEP 7: Create a Report".yellow().bold());
    println!("  Generate a comprehensive report:");
    println!("  {}", "chronos report -f your_data.csv --comprehensive".green());

    println!("\n{}", "INTERACTIVE MODE:".yellow().bold());
    println!("  For exploration, use interactive mode:");
    println!("  {}", "chronos --interactive".green());
    println!("  Then load your file and explore:");
    println!("  {}", "chronos> load your_data.csv".green());
    println!("  {}", "chronos> stats".green());
    println!("  {}", "chronos> plot".green());

    println!("\n{}", "For more details on any command:".yellow());
    println!("  {}", "chronos <command> --help".green());
    println!();
}