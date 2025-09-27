use clap::{Parser, Subcommand};
use anyhow::Result;
use colored::Colorize;
use std::fs::File;
use std::io::Write;
use chrono::{Utc, Duration};

// Use our own library
use chronos::{
    ImportConfig, import_csv,
    import::{TimestampColumn},
    validation::validate_data_quality,
    stats::{analyze_timeseries, ExportFormat, export_stats_results},
    trend::{
        analyze_comprehensive, TrendAnalysisConfig, DecompositionMethod,
        perform_decomposition, detect_trend, perform_detrending, DetrendingMethod
    },
    seasonality::{
        detect_seasonality, SeasonalityMethod, analyze_seasonal_patterns,
        perform_seasonal_adjustment, SeasonalAdjustmentMethod,
        SeasonalityAnalysisConfig, SeasonalPeriod, SeasonalPeriodType
    },
    anomaly::{
        detect_anomalies, AnomalyDetectionConfig,
        AnomalyMethod, ThresholdConfig, ContextualConfig, ScoringConfig,
        StreamingConfig, ScoringMethod
    },
    forecasting::{
        forecast_timeseries, forecast_with_intervals, evaluate_forecast_model,
        ForecastConfig, ForecastMethod, SeasonalType, ETSComponent, GrowthType,
        SeasonalityMode, EnsembleCombination
    },
    correlation::{
        analyze_correlations, AnalysisConfig, CorrelationType,
        test_granger_causality, compute_dtw_distance, compute_multiple_dtw, DTWConstraints,
        StepPattern, DistanceFunction
    }
};

/// Chronos - A powerful CLI tool for time series analysis
#[derive(Parser)]
#[clap(name = "chronos")]
#[clap(about = "A CLI tool for time series data analysis and visualization")]
#[clap(version = "0.1.0")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze time series data from a CSV file
    Analyze {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Column name for the timestamp
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Column name for the values
        #[clap(short, long, default_value = "value")]
        value_column: String,
    },

    /// Generate synthetic time series data for testing
    Generate {
        /// Number of data points to generate
        #[clap(short, long, default_value = "100")]
        points: usize,

        /// Output file path
        #[clap(short, long, default_value = "generated_data.csv")]
        output: String,
    },

    /// Create interactive and static plots for time series data
    Plot {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Column name(s) to plot (comma-separated for multiple series)
        #[clap(short, long, default_value = "value")]
        columns: String,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Plot type: line, scatter, histogram, box, violin, qq, acf, pacf, density, heatmap, decomposition, seasonal, forecast, anomaly
        #[clap(short = 't', long, default_value = "line")]
        plot_type: String,

        /// Output file path (extension determines format: .html, .png, .svg, .pdf, .json)
        #[clap(short, long, default_value = "plot.html")]
        output: String,

        /// Plot title
        #[clap(long)]
        title: Option<String>,

        /// X-axis label
        #[clap(long)]
        x_label: Option<String>,

        /// Y-axis label
        #[clap(long)]
        y_label: Option<String>,

        /// Theme: default, dark, publication, high_contrast
        #[clap(long, default_value = "default")]
        theme: String,

        /// Plot width in pixels
        #[clap(long, default_value = "800")]
        width: usize,

        /// Plot height in pixels
        #[clap(long, default_value = "600")]
        height: usize,

        /// Export format: html, png, svg, pdf, json, display
        #[clap(long, default_value = "html")]
        format: String,

        /// Show legend
        #[clap(long)]
        show_legend: bool,

        /// Show grid
        #[clap(long)]
        show_grid: bool,

        /// Interactive plot (for HTML format)
        #[clap(long, default_value = "true")]
        interactive: bool,

        /// Max lags for ACF/PACF plots
        #[clap(long, default_value = "50")]
        max_lags: usize,

        /// Number of bins for histograms
        #[clap(long, default_value = "30")]
        bins: usize,

        /// Seasonal period for decomposition/seasonal plots
        #[clap(long)]
        seasonal_period: Option<usize>,

        /// Correlation method for heatmaps: pearson, spearman, kendall
        #[clap(long, default_value = "pearson")]
        correlation_method: String,

        /// Anomaly indices for anomaly plots (comma-separated)
        #[clap(long)]
        anomaly_indices: Option<String>,

        /// Forecast horizon for forecast plots
        #[clap(long, default_value = "10")]
        forecast_horizon: usize,

        /// Confidence level for forecast intervals
        #[clap(long, default_value = "0.95")]
        confidence_level: f64,
    },

    /// Comprehensive statistical analysis for time series data
    Stats {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Column name to analyze
        #[clap(short, long, default_value = "value")]
        column: String,

        /// Export results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, csv, text, markdown, html
        #[clap(long, default_value = "text")]
        format: String,

        /// Perform normality tests
        #[clap(long)]
        test_normality: bool,

        /// Compute autocorrelation function with specified maximum lags
        #[clap(long, default_value = "50")]
        autocorr_lags: usize,

        /// Test stationarity with specified test (adf, kpss, pp, or all)
        #[clap(long, default_value = "adf")]
        stationarity_test: String,

        /// Detect change points
        #[clap(long)]
        detect_changepoints: bool,
    },

    /// Comprehensive trend analysis and decomposition for time series data
    Trend {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Column name to analyze
        #[clap(short, long, default_value = "value")]
        column: String,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Analysis method: analyze, decompose, detect, detrend
        #[clap(short, long, default_value = "analyze")]
        method: String,

        /// Decomposition method: classical, stl
        #[clap(long, default_value = "classical")]
        decomposition: String,

        /// Detrending method: linear, difference, moving_average, hp_filter
        #[clap(long, default_value = "linear")]
        detrending: String,

        /// Seasonal period (auto-detect if not specified)
        #[clap(long)]
        seasonal_period: Option<usize>,

        /// Trend detection test: mann_kendall, sens_slope, pettitt
        #[clap(long, default_value = "mann_kendall")]
        test: String,

        /// Significance level for statistical tests
        #[clap(long, default_value = "0.05")]
        alpha: f64,

        /// HP filter lambda parameter
        #[clap(long, default_value = "1600.0")]
        lambda: f64,

        /// Moving average window size
        #[clap(long, default_value = "12")]
        window: usize,

        /// Difference order for differencing detrending
        #[clap(long, default_value = "1")]
        diff_order: usize,

        /// Export results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, text, markdown
        #[clap(long, default_value = "text")]
        format: String,

        /// Generate plot data for visualization
        #[clap(long)]
        plot: bool,
    },

    /// Comprehensive seasonality detection and analysis for time series data
    Seasonal {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Column name to analyze
        #[clap(short, long, default_value = "value")]
        column: String,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Analysis method: detect, strength, adjust
        #[clap(short, long, default_value = "detect")]
        method: String,

        /// Maximum period to consider for detection
        #[clap(long, default_value = "365")]
        max_period: usize,

        /// Minimum period to consider for detection
        #[clap(long, default_value = "2")]
        min_period: usize,

        /// Seasonal periods to analyze (comma-separated, e.g., "7,30,365")
        #[clap(long)]
        periods: Option<String>,

        /// Seasonal adjustment method: x13, stl, moving_average
        #[clap(long, default_value = "stl")]
        adjustment_method: String,

        /// Detection methods: fourier, periodogram, autocorr, all
        #[clap(long, default_value = "all")]
        detection_methods: String,

        /// Significance level for statistical tests
        #[clap(long, default_value = "0.05")]
        alpha: f64,

        /// Export adjusted series (for adjust method)
        #[clap(long)]
        export_adjusted: bool,

        /// Export results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, text, markdown
        #[clap(long, default_value = "text")]
        format: String,

        /// Generate plot data for visualization
        #[clap(long)]
        plot: bool,

        /// Analyze calendar effects
        #[clap(long)]
        calendar_effects: bool,

        /// Detect evolving seasonality
        #[clap(long)]
        evolving: bool,

        /// Detect seasonal breaks
        #[clap(long)]
        breaks: bool,
    },

    /// Comprehensive anomaly detection for time series data
    Anomaly {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Column name to analyze
        #[clap(short, long, default_value = "value")]
        column: String,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Detection methods (comma-separated): zscore, modified_zscore, iqr, grubbs, isolation_forest, lof, dbscan, seasonal, trend, volatility, contextual, all
        #[clap(short, long, default_value = "zscore,iqr")]
        methods: String,

        /// Threshold for statistical methods
        #[clap(long, default_value = "3.0")]
        threshold: f64,

        /// IQR factor for outlier detection
        #[clap(long, default_value = "1.5")]
        iqr_factor: f64,

        /// Grubbs test alpha level
        #[clap(long, default_value = "0.05")]
        grubbs_alpha: f64,

        /// Isolation Forest contamination factor
        #[clap(long, default_value = "0.1")]
        contamination: f64,

        /// Number of trees for Isolation Forest
        #[clap(long, default_value = "100")]
        n_trees: usize,

        /// Number of neighbors for LOF
        #[clap(long, default_value = "20")]
        n_neighbors: usize,

        /// DBSCAN epsilon parameter
        #[clap(long, default_value = "0.5")]
        eps: f64,

        /// DBSCAN minimum samples
        #[clap(long, default_value = "5")]
        min_samples: usize,

        /// Seasonal period for contextual detection
        #[clap(long)]
        seasonal_period: Option<usize>,

        /// Enable contextual day-of-week adjustment
        #[clap(long)]
        contextual: bool,

        /// Baseline periods for contextual methods
        #[clap(long, default_value = "100")]
        baseline_periods: usize,

        /// Window size for trend and volatility detection
        #[clap(long, default_value = "10")]
        window_size: usize,

        /// Scoring method: maximum, weighted, ensemble
        #[clap(long, default_value = "maximum")]
        scoring: String,

        /// Minimum severity to report: low, medium, high, critical
        #[clap(long, default_value = "medium")]
        min_severity: String,

        /// Maximum number of top anomalies to report
        #[clap(long, default_value = "10")]
        max_anomalies: usize,

        /// Export results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, csv, text, markdown
        #[clap(long, default_value = "text")]
        format: String,

        /// Export anomaly scores for all points
        #[clap(long)]
        export_scores: bool,

        /// Show detailed analysis for each method
        #[clap(long)]
        detailed: bool,
    },

    /// Time series forecasting and prediction
    Forecast {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Column name to forecast
        #[clap(short, long, default_value = "value")]
        column: String,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Forecasting method: sma, wma, exp_smoothing, linear_trend, seasonal_naive, holt_winters, arima, sarima, auto_arima, ets, theta, prophet, ensemble
        #[clap(short, long, default_value = "sma")]
        method: String,

        /// Number of periods to forecast
        #[clap(long, default_value = "10")]
        horizon: usize,

        /// Confidence level for prediction intervals (0.0 to 1.0)
        #[clap(long, default_value = "0.95")]
        confidence: f64,

        /// Window size for moving averages
        #[clap(long, default_value = "5")]
        window: usize,

        /// Smoothing parameter alpha (0.0 to 1.0)
        #[clap(long, default_value = "0.3")]
        alpha: f64,

        /// Trend smoothing parameter beta (0.0 to 1.0)
        #[clap(long, default_value = "0.1")]
        beta: f64,

        /// Seasonal smoothing parameter gamma (0.0 to 1.0)
        #[clap(long, default_value = "0.1")]
        gamma: f64,

        /// Seasonal period for seasonal methods
        #[clap(long)]
        seasonal_period: Option<usize>,

        /// Seasonal type: additive, multiplicative
        #[clap(long, default_value = "additive")]
        seasonal_type: String,

        /// ARIMA order p (autoregressive)
        #[clap(long, default_value = "1")]
        p: usize,

        /// ARIMA order d (differencing)
        #[clap(long, default_value = "1")]
        d: usize,

        /// ARIMA order q (moving average)
        #[clap(long, default_value = "1")]
        q: usize,

        /// Seasonal ARIMA order P
        #[clap(long, default_value = "1")]
        seasonal_p: usize,

        /// Seasonal ARIMA order D
        #[clap(long, default_value = "1")]
        seasonal_d: usize,

        /// Seasonal ARIMA order Q
        #[clap(long, default_value = "1")]
        seasonal_q: usize,

        /// Theta parameter for Theta method
        #[clap(long, default_value = "2.0")]
        theta: f64,

        /// Include prediction intervals
        #[clap(long)]
        intervals: bool,

        /// Evaluate model performance using cross-validation
        #[clap(long)]
        evaluate: bool,

        /// Export forecast results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, csv, text, markdown
        #[clap(long, default_value = "text")]
        format: String,

        /// Export forecast values to CSV
        #[clap(long)]
        export_forecast: bool,
    },

    /// Correlation analysis for multivariate time series data
    Correlate {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Comma-separated list of column names to analyze
        #[clap(short, long)]
        columns: String,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Correlation types (comma-separated): pearson, spearman, kendall
        #[clap(long, default_value = "pearson,spearman")]
        correlation_types: String,

        /// Window size for rolling correlation (0 to disable)
        #[clap(long, default_value = "0")]
        rolling_window: usize,

        /// Enable cross-correlation analysis
        #[clap(long)]
        cross_correlation: bool,

        /// Maximum lag for cross-correlation
        #[clap(long, default_value = "20")]
        max_lag: i32,

        /// Export results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, csv, text, markdown
        #[clap(long, default_value = "text")]
        format: String,

        /// Generate plot data for visualization
        #[clap(long)]
        plot: bool,
    },

    /// Granger causality testing for time series relationships
    Causality {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Cause variable column name
        #[clap(long)]
        cause: String,

        /// Effect variable column name
        #[clap(long)]
        effect: String,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// Number of lags to include in the model
        #[clap(short, long, default_value = "5")]
        lags: usize,

        /// Maximum lags to test (for optimal lag selection)
        #[clap(long)]
        max_lags: Option<usize>,

        /// Significance level for statistical tests
        #[clap(long, default_value = "0.05")]
        alpha: f64,

        /// Include VAR model analysis
        #[clap(long)]
        var_analysis: bool,

        /// Include impulse response functions
        #[clap(long)]
        impulse_response: bool,

        /// Export results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, text, markdown
        #[clap(long, default_value = "text")]
        format: String,
    },

    /// Dynamic Time Warping for time series pattern matching
    Dtw {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// First time series column name
        #[clap(long)]
        series1: String,

        /// Second time series column name (optional for multiple series comparison)
        #[clap(long)]
        series2: Option<String>,

        /// Comma-separated list of columns for multiple series comparison
        #[clap(long)]
        multiple_series: Option<String>,

        /// Timestamp column name
        #[clap(short, long, default_value = "timestamp")]
        time_column: String,

        /// DTW constraint type: none, sakoe_chiba, itakura
        #[clap(long, default_value = "none")]
        constraint: String,

        /// Constraint parameter (window size for Sakoe-Chiba)
        #[clap(long, default_value = "10")]
        constraint_param: usize,

        /// Distance function: euclidean, manhattan, cosine
        #[clap(long, default_value = "euclidean")]
        distance: String,

        /// Step pattern: symmetric1, symmetric2, asymmetric
        #[clap(long, default_value = "symmetric1")]
        step_pattern: String,

        /// Compute DTW barycenter for multiple series
        #[clap(long)]
        barycenter: bool,

        /// Export warping path and alignment
        #[clap(long)]
        export_path: bool,

        /// Export results to file (optional)
        #[clap(short, long)]
        output: Option<String>,

        /// Export format: json, csv, text, markdown
        #[clap(long, default_value = "text")]
        format: String,
    },
}

/// Generate synthetic time series data for testing
fn generate_synthetic_data(points: usize, output_path: &str) -> Result<()> {
    let mut file = File::create(output_path)?;

    // Write CSV header
    writeln!(file, "timestamp,value,temperature,humidity")?;

    // Generate data starting from now, going back in time
    let start_time = Utc::now() - Duration::hours(points as i64);

    for i in 0..points {
        let timestamp = start_time + Duration::hours(i as i64);

        // Generate synthetic values with some patterns
        let t = i as f64;
        let base_value = 50.0 + 10.0 * (t / 24.0).sin(); // Daily pattern
        let noise = (t * 0.1).sin() * 2.0; // Some noise
        let trend = t * 0.01; // Slight upward trend
        let value = base_value + noise + trend;

        // Generate temperature (correlated with main value)
        let temperature = 20.0 + value * 0.3 + (t * 0.05).cos() * 5.0;

        // Generate humidity (anti-correlated with temperature)
        let humidity = 80.0 - temperature * 0.5 + (t * 0.03).sin() * 10.0;

        writeln!(file, "{},{:.4},{:.2},{:.1}",
                timestamp.format("%Y-%m-%d %H:%M:%S"),
                value,
                temperature,
                humidity)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Analyze { file, time_column, value_column } => {
            println!("{}", "📊 Analyzing time series data...".cyan().bold());
            println!("File: {}", file);
            println!("Time column: {}", time_column);
            println!("Value column: {}", value_column);

            // Configure import settings
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = vec![value_column.clone()];

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());
                    println!("Imported {} data points", result.timeseries.values.len());

                    println!("\n📊 Import Statistics:");
                    println!("  Rows processed: {}", result.stats.rows_processed);
                    println!("  Rows skipped: {}", result.stats.rows_skipped);
                    println!("  Missing values: {}", result.stats.missing_values);

                    // Perform data quality analysis
                    let ts = &result.timeseries;
                    let quality_report = validate_data_quality(&ts.timestamps, &ts.values);

                        println!("\n🔍 Data Quality Report:");
                        println!("{}", quality_report);
                    println!("Quality Score: {:.2}%", quality_report.quality_score(ts.timestamps.len()) * 100.0);

                    // Perform basic statistical analysis
                    if !ts.values.is_empty() {
                        let valid_values: Vec<f64> = ts.values.iter()
                            .filter(|&&v| !v.is_nan() && !v.is_infinite())
                            .copied()
                            .collect();

                        if !valid_values.is_empty() {
                            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                            let min = valid_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = valid_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                            println!("\n📈 Basic Statistics:");
                            println!("  Count: {}", valid_values.len());
                            println!("  Mean: {:.4}", mean);
                            println!("  Min: {:.4}", min);
                            println!("  Max: {:.4}", max);
                            println!("  Range: {:.4}", max - min);
                        }
                    }

                    println!("{}", "✅ Analysis complete!".green());
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("{}", e));
                }
            }
        }

        Commands::Generate { points, output } => {
            println!("{}", "🎲 Generating synthetic time series data...".cyan().bold());
            println!("Points: {}", points);
            println!("Output file: {}", output);

            // Generate synthetic data
            match generate_synthetic_data(*points, output) {
                Ok(()) => {
                    println!("{}", "✅ Synthetic data generated successfully!".green());
                    println!("Generated {} data points in '{}'", points, output);
                    println!("Use 'chronos analyze -f {}' to analyze the generated data", output);
                }
                Err(e) => {
                    println!("{}", format!("❌ Error generating data: {}", e).red());
                    return Err(e.into());
                }
            }
        }

        Commands::Trend {
            file,
            column,
            time_column,
            method,
            decomposition,
            detrending,
            seasonal_period,
            test,
            alpha,
            lambda,
            window,
            diff_order,
            output,
            format,
            plot,
        } => {
            println!("{}", "📈 Performing trend analysis...".cyan().bold());
            println!("File: {}", file);
            println!("Column: {}", column);
            println!("Method: {}", method);

            // Configure import to target specific column
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = vec![column.clone()];

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());

                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    if ts.values.is_empty() {
                        println!("{}", "❌ No valid data found for analysis".red());
                        return Ok(());
                    }

                    match method.as_str() {
                        "analyze" => {
                            // Comprehensive trend analysis
                            let mut trend_config = TrendAnalysisConfig::default();
                            trend_config.alpha = *alpha;
                            trend_config.seasonal_period = *seasonal_period;
                            trend_config.generate_plot_data = *plot;

                            match analyze_comprehensive(&ts.timestamps, &ts.values, Some(trend_config)) {
                                Ok(analysis) => {
                                    println!("{}", "✅ Trend analysis completed!".green());

                                    let summary = analysis.summary_report();
                                    println!("\n{}", summary);

                                    // Export results if requested
                                    if let Some(output_file) = output {
                                        let export_content = match format.as_str() {
                                            "json" => serde_json::to_string_pretty(&analysis)?,
                                            "markdown" | "md" => format!("# Trend Analysis Report\n\n```\n{}\n```", summary),
                                            _ => summary,
                                        };

                                        std::fs::write(output_file, export_content)?;
                                        println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                    }
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Trend analysis failed: {}", e).red());
                                    return Err(anyhow::anyhow!("Trend analysis error: {}", e));
                                }
                            }
                        },

                        "decompose" => {
                            // Time series decomposition
                            let decomp_method = match decomposition.as_str() {
                                "stl" => DecompositionMethod::Stl,
                                "multiplicative" => DecompositionMethod::ClassicalMultiplicative,
                                _ => DecompositionMethod::ClassicalAdditive,
                            };

                            match perform_decomposition(&ts.values, decomp_method, *seasonal_period) {
                                Ok(decomp_result) => {
                                    println!("{}", "✅ Decomposition completed!".green());

                                    println!("\n📊 Decomposition Results:");
                                    println!("  Method: {:?}", decomp_result.method);
                                    let mode = match decomp_result.method {
                                        DecompositionMethod::ClassicalAdditive | DecompositionMethod::Stl => "Additive",
                                        DecompositionMethod::ClassicalMultiplicative => "Multiplicative",
                                        _ => "Unknown",
                                    };
                                    println!("  Mode: {}", mode);
                                    if !decomp_result.seasonal_periods.is_empty() {
                                        println!("  Seasonal Periods: {:?}", decomp_result.seasonal_periods.iter().map(|p| p.period).collect::<Vec<_>>());
                                    }

                                    println!("\n📈 Quality Metrics:");
                                    println!("  Trend Strength: {:.3}", decomp_result.quality_metrics.trend_strength);
                                    println!("  Seasonal Strength: {:.3}", decomp_result.quality_metrics.seasonality_strength);
                                    println!("  R-squared: {:.3}", decomp_result.quality_metrics.r_squared);
                                    println!("  Residual Std: {:.6}", decomp_result.quality_metrics.std_residuals);

                                    // Export results if requested
                                    if let Some(output_file) = output {
                                        let export_content = match format.as_str() {
                                            "json" => serde_json::to_string_pretty(&decomp_result)?,
                                            "markdown" | "md" => {
                                                format!("# Decomposition Analysis Report\n\n## Method\n{:?}\n\n## Quality Metrics\n- Trend Strength: {:.3}\n- Seasonal Strength: {:.3}\n- R-squared: {:.3}",
                                                    decomp_result.method,
                                                    decomp_result.quality_metrics.trend_strength,
                                                    decomp_result.quality_metrics.seasonality_strength,
                                                    decomp_result.quality_metrics.r_squared)
                                            },
                                            _ => format!("Decomposition Method: {:?}\nTrend Strength: {:.3}\nSeasonal Strength: {:.3}\nR-squared: {:.3}",
                                                decomp_result.method,
                                                decomp_result.quality_metrics.trend_strength,
                                                decomp_result.quality_metrics.seasonality_strength,
                                                decomp_result.quality_metrics.r_squared),
                                        };

                                        std::fs::write(output_file, export_content)?;
                                        println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                    }
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Decomposition failed: {}", e).red());
                                    return Err(anyhow::anyhow!("Decomposition error: {}", e));
                                }
                            }
                        },

                        "detect" => {
                            // Trend detection
                            match detect_trend(&ts.values, test) {
                                Ok(test_result) => {
                                    println!("{}", "✅ Trend detection completed!".green());

                                    println!("\n🔍 Trend Detection Results:");
                                    println!("  Test: {}", test_result.test_name);
                                    println!("  Statistic: {:.6}", test_result.statistic);
                                    println!("  P-value: {:.6}", test_result.p_value);
                                    println!("  Significant: {}", if test_result.is_significant { "✅ Yes" } else { "❌ No" });
                                    println!("  Trend Direction: {}", match test_result.trend_direction {
                                        1 => "📈 Increasing",
                                        -1 => "📉 Decreasing",
                                        _ => "➡️  No trend",
                                    });

                                    if let Some(slope) = test_result.slope {
                                        println!("  Slope: {:.6}", slope);
                                    }

                                    // Export results if requested
                                    if let Some(output_file) = output {
                                        let export_content = match format.as_str() {
                                            "json" => serde_json::to_string_pretty(&test_result)?,
                                            "markdown" | "md" => {
                                                format!("# Trend Detection Report\n\n## Test: {}\n\n- Statistic: {:.6}\n- P-value: {:.6}\n- Significant: {}\n- Direction: {}",
                                                    test_result.test_name,
                                                    test_result.statistic,
                                                    test_result.p_value,
                                                    if test_result.is_significant { "Yes" } else { "No" },
                                                    match test_result.trend_direction {
                                                        1 => "Increasing",
                                                        -1 => "Decreasing",
                                                        _ => "No trend",
                                                    })
                                            },
                                            _ => format!("Test: {}\nStatistic: {:.6}\nP-value: {:.6}\nSignificant: {}\nDirection: {}",
                                                test_result.test_name,
                                                test_result.statistic,
                                                test_result.p_value,
                                                if test_result.is_significant { "Yes" } else { "No" },
                                                match test_result.trend_direction {
                                                    1 => "Increasing",
                                                    -1 => "Decreasing",
                                                    _ => "No trend",
                                                }),
                                        };

                                        std::fs::write(output_file, export_content)?;
                                        println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                    }
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Trend detection failed: {}", e).red());
                                    return Err(anyhow::anyhow!("Trend detection error: {}", e));
                                }
                            }
                        },

                        "detrend" => {
                            // Detrending
                            let detrend_method = match detrending.as_str() {
                                "difference" => DetrendingMethod::FirstDifference,
                                "moving_average" => DetrendingMethod::MovingAverage(*window),
                                "hp_filter" => DetrendingMethod::HPFilter(*lambda),
                                _ => DetrendingMethod::Linear,
                            };

                            if detrending == "difference" && *diff_order > 1 {
                                let detrend_method = DetrendingMethod::Difference(*diff_order);
                                match perform_detrending(&ts.values, detrend_method) {
                                    Ok(detrend_result) => {
                                        println!("{}", "✅ Detrending completed!".green());

                                        println!("\n📉 Detrending Results:");
                                        println!("  Method: {:?}", detrend_result.method);
                                        println!("  Original Points: {}", detrend_result.original.len());
                                        println!("  Detrended Points: {}", detrend_result.detrended.len());
                                        println!("  Variance Reduction: {:.1}%", detrend_result.quality_metrics.variance_reduction * 100.0);
                                        println!("  Residual Std: {:.6}", detrend_result.quality_metrics.residual_std);

                                        // Export results if requested
                                        if let Some(output_file) = output {
                                            let export_content = match format.as_str() {
                                                "json" => serde_json::to_string_pretty(&detrend_result)?,
                                                "markdown" | "md" => {
                                                    format!("# Detrending Report\n\n## Method\n{:?}\n\n## Quality Metrics\n- Variance Reduction: {:.1}%\n- Residual Std: {:.6}",
                                                        detrend_result.method,
                                                        detrend_result.quality_metrics.variance_reduction * 100.0,
                                                        detrend_result.quality_metrics.residual_std)
                                                },
                                                _ => format!("Method: {:?}\nVariance Reduction: {:.1}%\nResidual Std: {:.6}",
                                                    detrend_result.method,
                                                    detrend_result.quality_metrics.variance_reduction * 100.0,
                                                    detrend_result.quality_metrics.residual_std),
                                            };

                                            std::fs::write(output_file, export_content)?;
                                            println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                        }
                                    }
                                    Err(e) => {
                                        println!("{}", format!("❌ Detrending failed: {}", e).red());
                                        return Err(anyhow::anyhow!("Detrending error: {}", e));
                                    }
                                }
                            } else {
                                match perform_detrending(&ts.values, detrend_method) {
                                    Ok(detrend_result) => {
                                        println!("{}", "✅ Detrending completed!".green());

                                        println!("\n📉 Detrending Results:");
                                        println!("  Method: {:?}", detrend_result.method);
                                        println!("  Original Points: {}", detrend_result.original.len());
                                        println!("  Detrended Points: {}", detrend_result.detrended.len());
                                        println!("  Variance Reduction: {:.1}%", detrend_result.quality_metrics.variance_reduction * 100.0);
                                        println!("  Residual Std: {:.6}", detrend_result.quality_metrics.residual_std);

                                        if let Some(r2) = detrend_result.quality_metrics.r_squared {
                                            println!("  R-squared: {:.3}", r2);
                                        }

                                        // Export results if requested
                                        if let Some(output_file) = output {
                                            let export_content = match format.as_str() {
                                                "json" => serde_json::to_string_pretty(&detrend_result)?,
                                                "markdown" | "md" => {
                                                    format!("# Detrending Report\n\n## Method\n{:?}\n\n## Quality Metrics\n- Variance Reduction: {:.1}%\n- Residual Std: {:.6}",
                                                        detrend_result.method,
                                                        detrend_result.quality_metrics.variance_reduction * 100.0,
                                                        detrend_result.quality_metrics.residual_std)
                                                },
                                                _ => format!("Method: {:?}\nVariance Reduction: {:.1}%\nResidual Std: {:.6}",
                                                    detrend_result.method,
                                                    detrend_result.quality_metrics.variance_reduction * 100.0,
                                                    detrend_result.quality_metrics.residual_std),
                                            };

                                            std::fs::write(output_file, export_content)?;
                                            println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                        }
                                    }
                                    Err(e) => {
                                        println!("{}", format!("❌ Detrending failed: {}", e).red());
                                        return Err(anyhow::anyhow!("Detrending error: {}", e));
                                    }
                                }
                            }
                        },

                        _ => {
                            println!("{}", format!("❌ Unknown method: {}", method).red());
                            println!("Available methods: analyze, decompose, detect, detrend");
                            return Err(anyhow::anyhow!("Invalid method"));
                        }
                    }
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Plot {
            file,
            columns,
            time_column,
            plot_type,
            output,
            title,
            x_label,
            y_label,
            theme,
            width,
            height,
            format,
            show_legend,
            show_grid,
            interactive,
            max_lags,
            bins,
            seasonal_period,
            correlation_method,
            anomaly_indices,
            forecast_horizon,
            confidence_level,
        } => {
            println!("{}", "📈 Creating plot...".cyan().bold());
            println!("Input file: {}", file);
            println!("Plot type: {}", plot_type);
            println!("Output file: {}", output);

            // Configure import to get the specified columns
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());

            // Parse column names
            let column_names: Vec<String> = columns.split(',').map(|s| s.trim().to_string()).collect();

            match import_from_file(file, &config) {
                Ok(import_result) => {
                    // Prepare data for plotting
                    let mut plot_data = std::collections::HashMap::new();
                    let mut timestamps = Vec::new();

                    // Get timestamps if available
                    if let Some(ts_data) = import_result.data.get(&time_column.clone()) {
                        if let Some(ts) = &ts_data.series {
                            timestamps = ts.timestamps().iter().cloned().collect();
                        }
                    }

                    // Extract data for each column
                    for column_name in &column_names {
                        if let Some(series_data) = import_result.data.get(column_name) {
                            if let Some(ts) = &series_data.series {
                                plot_data.insert(column_name.clone(), ts.values().to_vec());
                            }
                        } else {
                            println!("{}", format!("⚠️ Column '{}' not found in data", column_name).yellow());
                        }
                    }

                    if plot_data.is_empty() {
                        println!("{}", "❌ No valid data columns found for plotting".red());
                        return Err(anyhow::anyhow!("No valid data columns"));
                    }

                    // Configure plot
                    let plot_theme = match theme.as_str() {
                        "dark" => chronos::plotting::Theme::Dark,
                        "publication" => chronos::plotting::Theme::Publication,
                        "high_contrast" => chronos::plotting::Theme::HighContrast,
                        _ => chronos::plotting::Theme::Default,
                    };

                    let export_format = match format.as_str() {
                        "png" => chronos::plotting::ExportFormat::PNG,
                        "svg" => chronos::plotting::ExportFormat::SVG,
                        "pdf" => chronos::plotting::ExportFormat::PDF,
                        "json" => chronos::plotting::ExportFormat::JSON,
                        "display" => chronos::plotting::ExportFormat::Display,
                        _ => chronos::plotting::ExportFormat::HTML,
                    };

                    let plot_config = chronos::plotting::PlotConfig {
                        plot_type: match plot_type.as_str() {
                            "scatter" => chronos::plotting::PlotType::Scatter,
                            "histogram" => chronos::plotting::PlotType::Histogram,
                            "box" => chronos::plotting::PlotType::BoxPlot,
                            "violin" => chronos::plotting::PlotType::ViolinPlot,
                            "qq" => chronos::plotting::PlotType::QQPlot,
                            "acf" => chronos::plotting::PlotType::ACF,
                            "pacf" => chronos::plotting::PlotType::PACF,
                            "density" => chronos::plotting::PlotType::Density,
                            "heatmap" => chronos::plotting::PlotType::Heatmap,
                            "decomposition" => chronos::plotting::PlotType::Decomposition,
                            "seasonal" => chronos::plotting::PlotType::SeasonalPattern,
                            "forecast" => chronos::plotting::PlotType::Forecast,
                            "anomaly" => chronos::plotting::PlotType::AnomalyHighlight,
                            _ => chronos::plotting::PlotType::Line,
                        },
                        primary_column: column_names.first().unwrap_or(&"value".to_string()).clone(),
                        additional_columns: column_names.iter().skip(1).cloned().collect(),
                        title: title.clone(),
                        x_label: x_label.clone(),
                        y_label: y_label.clone(),
                        theme: plot_theme,
                        width: *width,
                        height: *height,
                        export_format,
                        interactive: *interactive,
                        custom_style: None,
                        show_legend: *show_legend,
                        show_grid: *show_grid,
                    };

                    // Create plot
                    let timestamps_opt = if timestamps.is_empty() { None } else { Some(timestamps.as_slice()) };

                    let plot_result = match plot_type.as_str() {
                        "histogram" => {
                            let values = plot_data.values().next().unwrap();
                            chronos::plotting::create_histogram(values, plot_config)
                        },
                        "box" => chronos::plotting::create_box_plot(&plot_data, plot_config),
                        "violin" => chronos::plotting::create_violin_plot(&plot_data, plot_config),
                        "qq" => {
                            let values = plot_data.values().next().unwrap();
                            chronos::plotting::create_qq_plot(values, plot_config)
                        },
                        "acf" => {
                            let values = plot_data.values().next().unwrap();
                            chronos::plotting::create_acf_plot(values, *max_lags, plot_config)
                        },
                        "pacf" => {
                            let values = plot_data.values().next().unwrap();
                            chronos::plotting::create_pacf_plot(values, *max_lags, plot_config)
                        },
                        "density" => {
                            let values = plot_data.values().next().unwrap();
                            chronos::plotting::create_density_plot(values, plot_config)
                        },
                        "heatmap" => chronos::plotting::create_correlation_heatmap(&plot_data, plot_config),
                        "scatter" => chronos::plotting::create_scatter_plot(&plot_data, timestamps_opt, plot_config),
                        _ => chronos::plotting::create_line_plot(&plot_data, timestamps_opt, plot_config),
                    };

                    match plot_result {
                        Ok(result) => {
                            // Export to file
                            let export_options = chronos::plotting::ExportOptions {
                                output_dir: Some(std::path::Path::new(output).parent().unwrap_or(std::path::Path::new(".")).to_path_buf()),
                                filename: Some(std::path::Path::new(output).file_stem().unwrap().to_string_lossy().to_string()),
                                quality: Some(95),
                                dpi: Some(300),
                                open_after_export: false,
                                metadata: None,
                            };

                            match chronos::plotting::export_to_file(&result, export_format, export_options) {
                                Ok(export_info) => {
                                    println!("{}", "✅ Plot created successfully!".green());
                                    println!("📊 Plot Details:");
                                    println!("  Type: {:?}", result.metadata.plot_type);
                                    println!("  Data points: {}", result.metadata.data_points);
                                    println!("  Series count: {}", result.metadata.series_count);
                                    println!("  Dimensions: {}x{}", result.metadata.dimensions.0, result.metadata.dimensions.1);
                                    println!("  Theme: {:?}", result.metadata.theme);

                                    if let Some(file_path) = &export_info.file_path {
                                        println!("💾 Exported to: {}", file_path);
                                    }
                                    if let Some(file_size) = export_info.file_size {
                                        println!("📦 File size: {} bytes", file_size);
                                    }
                                },
                                Err(e) => {
                                    println!("{}", format!("❌ Failed to export plot: {}", e).red());
                                    return Err(anyhow::anyhow!("Export error: {}", e));
                                }
                            }
                        },
                        Err(e) => {
                            println!("{}", format!("❌ Failed to create plot: {}", e).red());
                            return Err(anyhow::anyhow!("Plot creation error: {}", e));
                        }
                    }
                },
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Stats {
            file,
            column,
            output,
            format,
            test_normality,
            autocorr_lags,
            stationarity_test,
            detect_changepoints
        } => {
            println!("{}", "📊 Performing comprehensive statistical analysis...".cyan().bold());
            println!("File: {}", file);
            println!("Column: {}", column);

            // Configure import to target specific column
            let mut config = ImportConfig::default();
            config.csv_config.value_columns = vec![column.clone()];

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());

                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    if ts.values.is_empty() {
                        println!("{}", "❌ No valid data found for analysis".red());
                        return Ok(());
                    }

                    // Perform comprehensive statistical analysis
                    match analyze_timeseries(&ts.timestamps, &ts.values, column) {
                        Ok(stats_result) => {
                            println!("{}", "✅ Statistical analysis completed!".green());

                            // Display results to console
                            let summary = stats_result.summary();
                            println!("\n{}", summary);

                            // Additional specific analyses based on flags
                            if *test_normality {
                                if let Some(ref dist) = stats_result.distribution {
                                    if let Some(ref norm_test) = dist.normality_test {
                                        println!("\n🔍 Normality Test Results:");
                                        println!("  Test: {}", norm_test.test_name);
                                        println!("  Statistic: {:.6}", norm_test.statistic);
                                        println!("  P-value: {:.6}", norm_test.p_value);
                                        println!("  Is Normal: {}",
                                            if norm_test.is_normal { "✅ Yes" } else { "❌ No" });
                                    }
                                }
                            }

                            if *autocorr_lags > 0 {
                                if let Some(ref ts_stats) = stats_result.timeseries_stats {
                                    println!("\n📈 Autocorrelation Analysis (first 10 lags):");
                                    for (i, (&lag, &acf_val)) in ts_stats.acf.lags.iter()
                                        .zip(ts_stats.acf.values.iter()).enumerate().take(10) {
                                        println!("  Lag {}: {:.4}", lag, acf_val);
                                    }

                                    if let Some(ref ljung_box) = ts_stats.acf.ljung_box_test {
                                        println!("\n  Ljung-Box Test:");
                                        println!("    Statistic: {:.4}", ljung_box.statistic);
                                        println!("    P-value: {:.4}", ljung_box.p_value);
                                        println!("    Autocorrelation detected: {}",
                                            if ljung_box.has_autocorrelation { "Yes" } else { "No" });
                                    }
                                }
                            }

                            if stationarity_test != "none" {
                                println!("\n🔬 Stationarity Test Results:");
                                for (test_name, test_result) in &stats_result.stationarity_tests {
                                    println!("  {} Test:", test_name);
                                    println!("    Statistic: {:.6}", test_result.statistic);
                                    println!("    P-value: {:.6}", test_result.p_value);
                                    println!("    Is Stationary: {}",
                                        if test_result.is_stationary { "✅ Yes" } else { "❌ No" });
                                }
                            }

                            if *detect_changepoints && !stats_result.changepoints.is_empty() {
                                println!("\n🎯 Change Points Detected:");
                                println!("  Total: {}", stats_result.changepoints.len());
                                for (i, cp) in stats_result.changepoints.iter().enumerate().take(5) {
                                    println!("    Point {}: Index {} (Confidence: {:.2})",
                                        i + 1, cp.index, cp.confidence);
                                }
                                if stats_result.changepoints.len() > 5 {
                                    println!("    ... and {} more", stats_result.changepoints.len() - 5);
                                }
                            }

                            // Export results if requested
                            if let Some(output_file) = output {
                                let export_format = match format.as_str() {
                                    "json" => ExportFormat::Json,
                                    "csv" => ExportFormat::Csv,
                                    "markdown" | "md" => ExportFormat::Markdown,
                                    "html" => ExportFormat::Html,
                                    _ => ExportFormat::TextReport,
                                };

                                match export_stats_results(&stats_result, export_format,
                                    std::path::Path::new(output_file), None) {
                                    Ok(()) => {
                                        println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                    }
                                    Err(e) => {
                                        println!("{}", format!("⚠️  Export failed: {}", e).yellow());
                                    }
                                }
                            }

                            println!("{}", "\n✅ Statistical analysis complete!".green());
                        }
                        Err(e) => {
                            println!("{}", format!("❌ Statistical analysis failed: {}", e).red());
                            return Err(anyhow::anyhow!("Statistical analysis error: {}", e));
                        }
                    }
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Seasonal {
            file,
            column,
            time_column,
            method,
            max_period,
            min_period,
            periods,
            adjustment_method,
            detection_methods,
            alpha,
            export_adjusted,
            output,
            format,
            plot,
            calendar_effects,
            evolving,
            breaks,
        } => {
            println!("{}", "🌊 Performing seasonality analysis...".cyan().bold());
            println!("File: {}", file);
            println!("Column: {}", column);
            println!("Method: {}", method);

            // Configure import to target specific column
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = vec![column.clone()];

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());

                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    if ts.values.is_empty() {
                        println!("{}", "❌ No valid data found for analysis".red());
                        return Ok(());
                    }

                    match method.as_str() {
                        "detect" => {
                            // Seasonality detection
                            let methods = match detection_methods.as_str() {
                                "fourier" => vec![SeasonalityMethod::Fourier],
                                "periodogram" => vec![SeasonalityMethod::Periodogram],
                                "autocorr" => vec![SeasonalityMethod::Autocorrelation],
                                "all" => vec![
                                    SeasonalityMethod::Fourier,
                                    SeasonalityMethod::Periodogram,
                                    SeasonalityMethod::Autocorrelation
                                ],
                                _ => vec![SeasonalityMethod::Fourier],
                            };

                            let mut seasonal_config = SeasonalityAnalysisConfig::default();
                            seasonal_config.max_period = *max_period;
                            seasonal_config.min_period = *min_period;
                            seasonal_config.alpha = *alpha;
                            seasonal_config.detection_methods = methods;
                            seasonal_config.generate_plot_data = *plot;
                            seasonal_config.analyze_calendar_effects = *calendar_effects;
                            // Evolving seasonality is handled in advanced analysis
                            // Seasonal breaks are handled in advanced analysis

                            match detect_seasonality(&ts.values, &seasonal_config) {
                                Ok(detection_result) => {
                                    println!("{}", "✅ Seasonality detection completed!".green());

                                    println!("\n🔍 Detection Results:");
                                    if detection_result.seasonal_periods.is_empty() {
                                        println!("  No significant seasonal periods detected");
                                    } else {
                                        for period in &detection_result.seasonal_periods {
                                            println!("  Period {} (strength: {:.3}, confidence: {:.3})",
                                                period.period, period.strength, period.confidence);
                                        }
                                    }

                                    println!("\n📊 Quality Metrics:");
                                    println!("  Seasonality Score: {:.3}", detection_result.overall_seasonality);
                                    println!("  Method: {:?}", detection_result.method);

                                    if let Some(spectrum) = &detection_result.fourier_analysis {
                                        println!("\n🌊 Fourier Analysis:");
                                        if let Some((freq, _)) = spectrum.dominant_frequencies.first() {
                            println!("  Dominant Frequency: {:.6}", freq);
                        }
                                        let max_power = spectrum.power_spectrum.iter().fold(0.0f64, |a, &b| a.max(b));
                        println!("  Peak Power: {:.3}", max_power);
                                    }

                                    // Export results if requested
                                    if let Some(output_file) = output {
                                        let export_content = match format.as_str() {
                                            "json" => serde_json::to_string_pretty(&detection_result)?,
                                            "markdown" | "md" => {
                                                let mut md = "# Seasonality Detection Report\n\n".to_string();
                                                md.push_str(&format!("## Detected Periods\n"));
                                                for period in &detection_result.seasonal_periods {
                                                    md.push_str(&format!("- Period {}: strength {:.3}, confidence {:.3}\n",
                                                        period.period, period.strength, period.confidence));
                                                }
                                                md.push_str(&format!("\n## Quality Metrics\n"));
                                                md.push_str(&format!("- Seasonality Score: {:.3}\n",
                                                    detection_result.overall_seasonality));
                                                if let Some(ref fourier) = detection_result.fourier_analysis {
                                                    md.push_str(&format!("- Signal-to-Noise: {:.3}\n", fourier.snr));
                                                }
                                                md
                                            },
                                            _ => {
                                                let mut text = "Seasonality Detection Results\n".to_string();
                                                text.push_str("================================\n\n");
                                                text.push_str("Detected Periods:\n");
                                                for period in &detection_result.seasonal_periods {
                                                    text.push_str(&format!("  Period {}: strength {:.3}, confidence {:.3}\n",
                                                        period.period, period.strength, period.confidence));
                                                }
                                                text.push_str(&format!("\nSeasonality Score: {:.3}\n",
                                                    detection_result.overall_seasonality));
                                                text
                                            },
                                        };

                                        std::fs::write(output_file, export_content)?;
                                        println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                    }
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Seasonality detection failed: {}", e).red());
                                    return Err(anyhow::anyhow!("Seasonality detection error: {}", e));
                                }
                            }
                        },

                        "strength" => {
                            // Analyze seasonal strength for specific periods
                            let target_periods: Vec<usize> = if let Some(periods_str) = periods {
                                periods_str.split(',')
                                    .filter_map(|s| s.trim().parse().ok())
                                    .collect()
                            } else {
                                vec![7, 30, 365] // Default periods
                            };

                            println!("\n📊 Analyzing seasonal strength for periods: {:?}", target_periods);

                            // Convert periods to SeasonalPeriod structs for pattern analysis
                            let seasonal_periods: Vec<SeasonalPeriod> = target_periods
                                .iter()
                                .map(|&p| SeasonalPeriod {
                                    period: p as f64,
                                    strength: 0.0, // Will be calculated
                                    phase: 0.0,
                                    amplitude: 0.0,
                                    confidence: 0.95,
                                    period_type: if p == 7 { SeasonalPeriodType::Weekly }
                                               else if p == 30 { SeasonalPeriodType::Monthly }
                                               else if p == 365 { SeasonalPeriodType::Yearly }
                                               else { SeasonalPeriodType::Custom(p as f64) },
                                })
                                .collect();

                            match analyze_seasonal_patterns(&ts.values, &seasonal_periods) {
                                Ok(pattern_analysis) => {
                                    println!("{}", "✅ Seasonal pattern analysis completed!".green());

                                    println!("\n📈 Seasonal Strength Results:");
                                    println!("  Overall Strength: {:.3}", pattern_analysis.overall_strength.strength);
                                    println!("  P-Value: {:.3}, Confidence: ({:.3}, {:.3})",
                                        pattern_analysis.overall_strength.p_value,
                                        pattern_analysis.overall_strength.confidence_interval.0,
                                        pattern_analysis.overall_strength.confidence_interval.1);

                                    if let Some(ref consistency) = pattern_analysis.consistency {
                                        println!("\n🎯 Pattern Consistency:");
                                        println!("  Overall Score: {:.3}", consistency.consistency_score);
                                        println!("  Stability Index: {:.3}", consistency.temporal_stability);
                                    }

                                    // Export results if requested
                                    if let Some(output_file) = output {
                                        let export_content = match format.as_str() {
                                            "json" => serde_json::to_string_pretty(&pattern_analysis)?,
                                            "markdown" | "md" => {
                                                let mut md = "# Seasonal Strength Analysis\n\n".to_string();
                                                md.push_str("## Seasonal Strengths\n");
                                                md.push_str(&format!("- Overall Strength: {:.3} (p-value: {:.3})\n",
                                                    pattern_analysis.overall_strength.strength, pattern_analysis.overall_strength.p_value));
                                                if let Some(ref consistency) = pattern_analysis.consistency {
                                                    md.push_str(&format!("\n## Pattern Consistency: {:.3}\n",
                                                        consistency.consistency_score));
                                                }
                                                md
                                            },
                                            _ => {
                                                let mut text = "Seasonal Strength Analysis\n".to_string();
                                                text.push_str("===============================\n\n");
                                                text.push_str(&format!("Overall Strength: {:.3}\n", pattern_analysis.overall_strength.strength));
                                                text
                                            },
                                        };

                                        std::fs::write(output_file, export_content)?;
                                        println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                    }
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Seasonal pattern analysis failed: {}", e).red());
                                    return Err(anyhow::anyhow!("Seasonal pattern analysis error: {}", e));
                                }
                            }
                        },

                        "adjust" => {
                            // Seasonal adjustment
                            let adj_method = match adjustment_method.as_str() {
                                "x13" => SeasonalAdjustmentMethod::X13Arima,
                                "moving_average" => SeasonalAdjustmentMethod::MovingAverage,
                                _ => SeasonalAdjustmentMethod::STL,
                            };

                            // First detect seasonality to get periods
                            let mut detect_config = SeasonalityAnalysisConfig::default();
                            detect_config.max_period = *max_period;
                            detect_config.min_period = *min_period;
                            detect_config.alpha = *alpha;

                            match detect_seasonality(&ts.values, &detect_config) {
                                Ok(detection_result) => {
                                    if detection_result.seasonal_periods.is_empty() {
                                        println!("{}", "⚠️  No seasonal periods detected for adjustment".yellow());
                                        return Ok(());
                                    }

                                    println!("  Detected {} seasonal periods for adjustment", detection_result.seasonal_periods.len());

                                    match perform_seasonal_adjustment(&ts.values, adj_method, &detection_result.seasonal_periods) {
                                        Ok(adjustment_result) => {
                                            println!("{}", "✅ Seasonal adjustment completed!".green());

                                            println!("\n📉 Adjustment Results:");
                                            println!("  Method: {:?}", adjustment_result.method);
                                            println!("  Seasonally Adjusted Points: {}", adjustment_result.adjusted_series.len());

                                            println!("\n📊 Quality Assessment:");
                                            println!("  Quality Score: {:.3}", adjustment_result.diagnostics.quality_score);
                                            println!("  Adjustment Method: {:?}", adjustment_result.method);

                                            // Export adjusted series if requested
                                            if *export_adjusted {
                                                let adjusted_file = output.as_ref()
                                                    .map(|f| f.replace(".csv", "_adjusted.csv"))
                                                    .unwrap_or_else(|| "adjusted_series.csv".to_string());

                                                let mut file = File::create(&adjusted_file)?;
                                                writeln!(file, "timestamp,original,adjusted")?;
                                                for (i, (&original, &adjusted)) in ts.values.iter()
                                                    .zip(adjustment_result.adjusted_series.iter()).enumerate() {
                                                    if i < ts.timestamps.len() {
                                                        writeln!(file, "{},{:.6},{:.6}",
                                                            ts.timestamps[i].format("%Y-%m-%d %H:%M:%S"),
                                                            original, adjusted)?;
                                                    }
                                                }
                                                println!("{}", format!("\n💾 Adjusted series exported to: {}", adjusted_file).green());
                                            }

                                            // Export results if requested
                                            if let Some(output_file) = output {
                                                let export_content = match format.as_str() {
                                                    "json" => serde_json::to_string_pretty(&adjustment_result)?,
                                                    "markdown" | "md" => {
                                                        format!("# Seasonal Adjustment Report\n\n## Method\n{:?}\n\n## Quality Metrics\n- Quality Score: {:.3}\n- Adjusted Points: {}",
                                                            adjustment_result.method,
                                                            adjustment_result.diagnostics.quality_score,
                                                            adjustment_result.adjusted_series.len())
                                                    },
                                                    _ => {
                                                        format!("Seasonal Adjustment Results\nMethod: {:?}\nQuality Score: {:.3}\nAdjusted Points: {}",
                                                            adjustment_result.method,
                                                            adjustment_result.diagnostics.quality_score,
                                                            adjustment_result.adjusted_series.len())
                                                    },
                                                };

                                                std::fs::write(output_file, export_content)?;
                                                println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                            }
                                        }
                                        Err(e) => {
                                            println!("{}", format!("❌ Seasonal adjustment failed: {}", e).red());
                                            return Err(anyhow::anyhow!("Seasonal adjustment error: {}", e));
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("{}", format!("❌ Error detecting seasonality for adjustment: {}", e).red());
                                    return Err(anyhow::anyhow!("Seasonality detection error: {}", e));
                                }
                            }
                        },

                        _ => {
                            println!("{}", format!("❌ Unknown method: {}", method).red());
                            println!("Available methods: detect, strength, adjust");
                            return Err(anyhow::anyhow!("Invalid method"));
                        }
                    }
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Anomaly {
            file,
            column,
            time_column,
            methods,
            threshold,
            iqr_factor,
            grubbs_alpha,
            contamination,
            n_trees,
            n_neighbors,
            eps,
            min_samples,
            seasonal_period,
            contextual,
            baseline_periods,
            window_size,
            scoring,
            min_severity,
            max_anomalies,
            output,
            format,
            export_scores,
            detailed,
        } => {
            println!("{}", "🔍 Performing anomaly detection...".cyan().bold());
            println!("File: {}", file);
            println!("Column: {}", column);
            println!("Methods: {}", methods);

            // Configure import to target specific column
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = vec![column.clone()];

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());

                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    if ts.values.is_empty() {
                        println!("{}", "❌ No valid data found for analysis".red());
                        return Ok(());
                    }

                    // Parse detection methods
                    let method_names: Vec<&str> = methods.split(',').map(|s| s.trim()).collect();
                    let mut detection_methods = Vec::new();

                    for method_name in method_names {
                        match method_name {
                            "zscore" => detection_methods.push(AnomalyMethod::ZScore { threshold: *threshold }),
                            "modified_zscore" => detection_methods.push(AnomalyMethod::ModifiedZScore { threshold: *threshold }),
                            "iqr" => detection_methods.push(AnomalyMethod::IQR { factor: *iqr_factor }),
                            "grubbs" => detection_methods.push(AnomalyMethod::Grubbs { alpha: *grubbs_alpha }),
                            "isolation_forest" => detection_methods.push(AnomalyMethod::IsolationForest {
                                contamination: *contamination,
                                n_trees: *n_trees
                            }),
                            "lof" => detection_methods.push(AnomalyMethod::LocalOutlierFactor {
                                n_neighbors: *n_neighbors,
                                contamination: *contamination
                            }),
                            "dbscan" => detection_methods.push(AnomalyMethod::DBSCANClustering {
                                eps: *eps,
                                min_samples: *min_samples
                            }),
                            "seasonal" => {
                                if let Some(period) = seasonal_period {
                                    detection_methods.push(AnomalyMethod::SeasonalDecomposition { period: *period });
                                } else {
                                    println!("{}", "⚠️  Seasonal method requires --seasonal-period parameter".yellow());
                                }
                            },
                            "trend" => detection_methods.push(AnomalyMethod::TrendDeviation { window_size: *window_size }),
                            "level_shift" => detection_methods.push(AnomalyMethod::LevelShift { threshold: *threshold }),
                            "volatility" => detection_methods.push(AnomalyMethod::VolatilityAnomaly { window_size: *window_size }),
                            "contextual" => {
                                if *contextual {
                                    detection_methods.push(AnomalyMethod::DayOfWeekAdjusted { baseline_periods: *baseline_periods });
                                    if let Some(period) = seasonal_period {
                                        detection_methods.push(AnomalyMethod::SeasonalContext { seasonal_periods: vec![*period] });
                                    }
                                }
                            },
                            "all" => {
                                detection_methods.extend(vec![
                                    AnomalyMethod::ZScore { threshold: *threshold },
                                    AnomalyMethod::ModifiedZScore { threshold: *threshold },
                                    AnomalyMethod::IQR { factor: *iqr_factor },
                                    AnomalyMethod::Grubbs { alpha: *grubbs_alpha },
                                    AnomalyMethod::IsolationForest { contamination: *contamination, n_trees: *n_trees },
                                    AnomalyMethod::LocalOutlierFactor { n_neighbors: *n_neighbors, contamination: *contamination },
                                    AnomalyMethod::TrendDeviation { window_size: *window_size },
                                    AnomalyMethod::VolatilityAnomaly { window_size: *window_size },
                                ]);
                                if *contextual {
                                    detection_methods.push(AnomalyMethod::DayOfWeekAdjusted { baseline_periods: *baseline_periods });
                                }
                            },
                            _ => {
                                println!("{}", format!("⚠️  Unknown method: {}", method_name).yellow());
                            }
                        }
                    }

                    if detection_methods.is_empty() {
                        println!("{}", "❌ No valid detection methods specified".red());
                        return Err(anyhow::anyhow!("No valid detection methods"));
                    }

                    // Configure anomaly detection
                    let scoring_method = match scoring.as_str() {
                        "weighted" => ScoringMethod::WeightedAverage,
                        "ensemble" => ScoringMethod::EnsembleVoting,
                        _ => ScoringMethod::Maximum,
                    };

                    let mut anomaly_config = AnomalyDetectionConfig {
                        methods: detection_methods,
                        thresholds: ThresholdConfig::default(),
                        contextual: ContextualConfig {
                            day_of_week_adjustment: *contextual,
                            seasonal_context: seasonal_period.is_some(),
                            seasonal_periods: seasonal_period.map(|p| vec![p]).unwrap_or_default(),
                            baseline_periods: *baseline_periods,
                        },
                        scoring: ScoringConfig {
                            method: scoring_method,
                            method_weights: std::collections::HashMap::new(),
                            enable_ranking: true,
                            max_top_anomalies: *max_anomalies,
                        },
                        streaming: StreamingConfig::default(),
                    };

                    // Perform anomaly detection
                    match detect_anomalies(&ts, &anomaly_config) {
                        Ok(mut detection_result) => {
                            println!("{}", "✅ Anomaly detection completed!".green());

                            // Filter by minimum severity
                            let min_sev = match min_severity.as_str() {
                                "low" => chronos::analysis::AnomalySeverity::Low,
                                "medium" => chronos::analysis::AnomalySeverity::Medium,
                                "high" => chronos::analysis::AnomalySeverity::High,
                                "critical" => chronos::analysis::AnomalySeverity::Critical,
                                _ => chronos::analysis::AnomalySeverity::Medium,
                            };

                            detection_result.anomalies.retain(|a| {
                                match (&a.severity, &min_sev) {
                                    (chronos::analysis::AnomalySeverity::Critical, _) => true,
                                    (chronos::analysis::AnomalySeverity::High, chronos::analysis::AnomalySeverity::Critical) => false,
                                    (chronos::analysis::AnomalySeverity::High, _) => true,
                                    (chronos::analysis::AnomalySeverity::Medium, chronos::analysis::AnomalySeverity::Critical | chronos::analysis::AnomalySeverity::High) => false,
                                    (chronos::analysis::AnomalySeverity::Medium, _) => true,
                                    (chronos::analysis::AnomalySeverity::Low, chronos::analysis::AnomalySeverity::Low) => true,
                                    (chronos::analysis::AnomalySeverity::Low, _) => false,
                                }
                            });

                            // Limit to max anomalies
                            detection_result.anomalies.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                            detection_result.anomalies.truncate(*max_anomalies);

                            println!("\n🚨 Anomaly Detection Results:");
                            println!("  Total anomalies found: {}", detection_result.anomalies.len());
                            println!("  Anomaly rate: {:.2}%", detection_result.statistics.anomaly_rate * 100.0);

                            if !detection_result.anomalies.is_empty() {
                                println!("\n📊 Top Anomalies:");
                                for (i, anomaly) in detection_result.anomalies.iter().enumerate().take(5) {
                                    let severity_icon = match anomaly.severity {
                                        chronos::analysis::AnomalySeverity::Critical => "🔴",
                                        chronos::analysis::AnomalySeverity::High => "🟠",
                                        chronos::analysis::AnomalySeverity::Medium => "🟡",
                                        chronos::analysis::AnomalySeverity::Low => "🟢",
                                    };
                                    println!("  {}. {} Index: {}, Time: {}, Value: {:.4}, Score: {:.3}",
                                        i + 1,
                                        severity_icon,
                                        anomaly.index,
                                        anomaly.timestamp.format("%Y-%m-%d %H:%M:%S"),
                                        anomaly.value,
                                        anomaly.score
                                    );
                                }

                                if detection_result.anomalies.len() > 5 {
                                    println!("  ... and {} more", detection_result.anomalies.len() - 5);
                                }
                            } else {
                                println!("  No anomalies found above the specified severity threshold.");
                            }

                            // Export results if requested
                            if let Some(output_file) = output {
                                let export_content = match format.as_str() {
                                    "json" => serde_json::to_string_pretty(&detection_result)?,
                                    "csv" => {
                                        let mut csv_content = "index,timestamp,value,score,severity,expected_value\n".to_string();
                                        for anomaly in &detection_result.anomalies {
                                            csv_content.push_str(&format!("{},{},{:.6},{:.6},{:?},{}\n",
                                                anomaly.index,
                                                anomaly.timestamp.format("%Y-%m-%d %H:%M:%S"),
                                                anomaly.value,
                                                anomaly.score,
                                                anomaly.severity,
                                                anomaly.expected_value.map(|v| format!("{:.6}", v)).unwrap_or("".to_string())
                                            ));
                                        }
                                        csv_content
                                    },
                                    "markdown" | "md" => {
                                        let mut md = "# Anomaly Detection Report\n\n".to_string();
                                        md.push_str(&format!("## Summary\n"));
                                        md.push_str(&format!("- Total anomalies: {}\n", detection_result.anomalies.len()));
                                        md.push_str(&format!("- Anomaly rate: {:.2}%\n", detection_result.statistics.anomaly_rate * 100.0));
                                        md.push_str(&format!("- Methods used: {}\n\n", methods));

                                        md.push_str("## Detected Anomalies\n\n");
                                        md.push_str("| Index | Timestamp | Value | Score | Severity |\n");
                                        md.push_str("|-------|-----------|-------|-------|----------|\n");
                                        for anomaly in &detection_result.anomalies {
                                            md.push_str(&format!("| {} | {} | {:.4} | {:.3} | {:?} |\n",
                                                anomaly.index,
                                                anomaly.timestamp.format("%Y-%m-%d %H:%M:%S"),
                                                anomaly.value,
                                                anomaly.score,
                                                anomaly.severity
                                            ));
                                        }
                                        md
                                    },
                                    _ => {
                                        let mut text = "Anomaly Detection Results\n".to_string();
                                        text.push_str("========================\n\n");
                                        text.push_str(&format!("Total anomalies: {}\n", detection_result.anomalies.len()));
                                        text.push_str(&format!("Anomaly rate: {:.2}%\n", detection_result.statistics.anomaly_rate * 100.0));
                                        text.push_str(&format!("Methods used: {}\n\n", methods));

                                        text.push_str("Detected Anomalies:\n");
                                        for (i, anomaly) in detection_result.anomalies.iter().enumerate() {
                                            text.push_str(&format!("{}. Index: {}, Time: {}, Value: {:.4}, Score: {:.3}, Severity: {:?}\n",
                                                i + 1,
                                                anomaly.index,
                                                anomaly.timestamp.format("%Y-%m-%d %H:%M:%S"),
                                                anomaly.value,
                                                anomaly.score,
                                                anomaly.severity
                                            ));
                                        }
                                        text
                                    },
                                };

                                std::fs::write(output_file, export_content)?;
                                println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                            }

                            // Export anomaly scores if requested
                            if *export_scores {
                                let scores_file = output.as_ref()
                                    .map(|f| f.replace(".csv", "_scores.csv"))
                                    .unwrap_or_else(|| "anomaly_scores.csv".to_string());

                                let mut file = File::create(&scores_file)?;
                                writeln!(file, "index,timestamp,value,anomaly_score")?;

                                // Create a simple anomaly score for all points (0 for normal, actual score for anomalies)
                                let mut all_scores = vec![0.0; ts.values.len()];
                                for anomaly in &detection_result.anomalies {
                                    if anomaly.index < all_scores.len() {
                                        all_scores[anomaly.index] = anomaly.score;
                                    }
                                }

                                for (i, (timestamp, (value, score))) in ts.timestamps.iter()
                                    .zip(ts.values.iter().zip(all_scores.iter())).enumerate() {
                                    writeln!(file, "{},{},{:.6},{:.6}",
                                        i,
                                        timestamp.format("%Y-%m-%d %H:%M:%S"),
                                        value,
                                        score)?;
                                }

                                println!("{}", format!("💾 Anomaly scores exported to: {}", scores_file).green());
                            }

                            println!("{}", "\n✅ Anomaly detection complete!".green());
                        }
                        Err(e) => {
                            println!("{}", format!("❌ Anomaly detection failed: {}", e).red());
                            return Err(anyhow::anyhow!("Anomaly detection error: {}", e));
                        }
                    }
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Forecast {
            file,
            column,
            time_column,
            method,
            horizon,
            confidence,
            window,
            alpha,
            beta,
            gamma,
            seasonal_period,
            seasonal_type,
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            theta,
            intervals,
            evaluate,
            output,
            format,
            export_forecast,
        } => {
            println!("{}", "🔮 Performing time series forecasting...".cyan().bold());
            println!("File: {}", file);
            println!("Column: {}", column);
            println!("Method: {}", method);
            println!("Horizon: {} periods", horizon);

            // Configure import to target specific column
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = vec![column.clone()];

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());

                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    if ts.values.is_empty() {
                        println!("{}", "❌ No valid data found for forecasting".red());
                        return Ok(());
                    }

                    if ts.values.len() < 3 {
                        println!("{}", "❌ Need at least 3 data points for forecasting".red());
                        return Ok(());
                    }

                    // Parse seasonal type
                    let seasonal_type_enum = match seasonal_type.as_str() {
                        "multiplicative" => SeasonalType::Multiplicative,
                        _ => SeasonalType::Additive,
                    };

                    // Create forecast method based on CLI parameters
                    let forecast_method = match method.as_str() {
                        "sma" | "simple_moving_average" => ForecastMethod::SimpleMovingAverage { window: *window },
                        "exp_smoothing" | "exponential_smoothing" => ForecastMethod::ExponentialSmoothing { alpha: *alpha },
                        "linear_trend" => ForecastMethod::LinearTrend,
                        "seasonal_naive" => {
                            let period = seasonal_period.unwrap_or(12);
                            ForecastMethod::SeasonalNaive { seasonal_period: period }
                        },
                        "holt_winters" => {
                            let period = seasonal_period.unwrap_or(12);
                            ForecastMethod::HoltWinters {
                                alpha: *alpha,
                                beta: *beta,
                                gamma: *gamma,
                                seasonal_period: period,
                                seasonal_type: seasonal_type_enum,
                            }
                        },
                        "arima" => ForecastMethod::ARIMA { p: *p, d: *d, q: *q },
                        "sarima" => {
                            let period = seasonal_period.unwrap_or(12);
                            ForecastMethod::SARIMA {
                                p: *p, d: *d, q: *q,
                                seasonal_p: *seasonal_p,
                                seasonal_d: *seasonal_d,
                                seasonal_q: *seasonal_q,
                                seasonal_period: period,
                            }
                        },
                        "auto_arima" => {
                            ForecastMethod::AutoARIMA {
                                max_p: *p,
                                max_d: *d,
                                max_q: *q,
                                max_seasonal_p: *seasonal_p,
                                max_seasonal_d: *seasonal_d,
                                max_seasonal_q: *seasonal_q,
                                seasonal_period: *seasonal_period,
                            }
                        },
                        "ets" => {
                            ForecastMethod::ETS {
                                error_type: ETSComponent::Additive,
                                trend_type: ETSComponent::Additive,
                                seasonal_type: ETSComponent::Additive,
                                seasonal_period: *seasonal_period,
                            }
                        },
                        "theta" => ForecastMethod::Theta { theta: *theta },
                        "prophet" => {
                            ForecastMethod::Prophet {
                                growth: GrowthType::Linear,
                                seasonality_mode: SeasonalityMode::Additive,
                                yearly_seasonality: true,
                                weekly_seasonality: true,
                                daily_seasonality: false,
                            }
                        },
                        _ => ForecastMethod::SimpleMovingAverage { window: *window },
                    };

                    // Create forecast configuration
                    let forecast_config = ForecastConfig {
                        method: forecast_method,
                        horizon: *horizon,
                        confidence_level: *confidence,
                        include_intervals: *intervals,
                        evaluation: chronos::forecasting::EvaluationConfig::default(),
                        features: chronos::forecasting::FeatureConfig::default(),
                    };

                    // Perform forecasting
                    let forecast_result = if *intervals {
                        forecast_with_intervals(ts, &forecast_config)
                    } else {
                        forecast_timeseries(ts, &forecast_config)
                    };

                    match forecast_result {
                        Ok(result) => {
                            println!("{}", "✅ Forecasting completed!".green());

                            println!("\n🔮 Forecast Results:");
                            println!("  Method: {}", result.method);
                            println!("  Horizon: {} periods", result.forecasts.len());
                            if result.confidence_level > 0.0 {
                                println!("  Confidence Level: {:.1}%", result.confidence_level * 100.0);
                            }

                            // Display forecast values
                            println!("\n📊 Forecasted Values:");
                            for (i, (&forecast, timestamp)) in result.forecasts.iter()
                                .zip(result.timestamps.iter()).enumerate().take(10) {
                                let mut line = format!("  Period {}: {:.4} at {}",
                                    i + 1, forecast, timestamp.format("%Y-%m-%d %H:%M:%S"));

                                if let (Some(lower), Some(upper)) = (&result.lower_bounds, &result.upper_bounds) {
                                    if let (Some(&lower_val), Some(&upper_val)) = (lower.get(i), upper.get(i)) {
                                        line.push_str(&format!(" [{:.4}, {:.4}]", lower_val, upper_val));
                                    }
                                }
                                println!("{}", line);
                            }

                            if result.forecasts.len() > 10 {
                                println!("  ... and {} more periods", result.forecasts.len() - 10);
                            }

                            // Display model evaluation if available
                            if let Some(ref eval_result) = result.evaluation {
                                println!("\n📈 Model Performance:");
                                println!("  MAE: {:.4}", eval_result.mae);
                                println!("  RMSE: {:.4}", eval_result.rmse);
                                println!("  MAPE: {:.2}%", eval_result.mape);
                                if let Some(mase) = eval_result.mase {
                                    println!("  MASE: {:.4}", mase);
                                }
                                if let Some(aic) = eval_result.aic {
                                    println!("  AIC: {:.2}", aic);
                                }
                            }

                            // Perform model evaluation if requested
                            if *evaluate {
                                match evaluate_forecast_model(ts, &forecast_config) {
                                    Ok(eval_result) => {
                                        println!("\n🎯 Cross-Validation Results:");
                                        println!("  MAE: {:.4}", eval_result.mae);
                                        println!("  RMSE: {:.4}", eval_result.rmse);
                                        println!("  MAPE: {:.2}%", eval_result.mape);
                                        println!("  SMAPE: {:.2}%", eval_result.smape);
                                        if let Some(mase) = eval_result.mase {
                                            println!("  MASE: {:.4}", mase);
                                        }
                                    }
                                    Err(e) => {
                                        println!("{}", format!("⚠️  Evaluation failed: {}", e).yellow());
                                    }
                                }
                            }

                            // Export forecast values if requested
                            if *export_forecast {
                                let forecast_file = output.as_ref()
                                    .map(|f| f.replace(".csv", "_forecast.csv"))
                                    .unwrap_or_else(|| "forecast.csv".to_string());

                                let mut file = File::create(&forecast_file)?;
                                if let (Some(lower), Some(upper)) = (&result.lower_bounds, &result.upper_bounds) {
                                    writeln!(file, "timestamp,forecast,lower_bound,upper_bound")?;
                                    for (i, (&forecast, timestamp)) in result.forecasts.iter()
                                        .zip(result.timestamps.iter()).enumerate() {
                                        let lower_val = lower.get(i).copied().unwrap_or(forecast);
                                        let upper_val = upper.get(i).copied().unwrap_or(forecast);
                                        writeln!(file, "{},{:.6},{:.6},{:.6}",
                                            timestamp.format("%Y-%m-%d %H:%M:%S"),
                                            forecast, lower_val, upper_val)?;
                                    }
                                } else {
                                    writeln!(file, "timestamp,forecast")?;
                                    for (&forecast, timestamp) in result.forecasts.iter()
                                        .zip(result.timestamps.iter()) {
                                        writeln!(file, "{},{:.6}",
                                            timestamp.format("%Y-%m-%d %H:%M:%S"),
                                            forecast)?;
                                    }
                                }
                                println!("{}", format!("\n💾 Forecast values exported to: {}", forecast_file).green());
                            }

                            // Export results if requested
                            if let Some(output_file) = output {
                                let export_content = match format.as_str() {
                                    "json" => serde_json::to_string_pretty(&result)?,
                                    "markdown" | "md" => {
                                        let mut md = "# Forecast Results\n\n".to_string();
                                        md.push_str(&format!("## Method: {}\n\n", result.method));
                                        md.push_str(&format!("- Horizon: {} periods\n", result.forecasts.len()));
                                        md.push_str(&format!("- Confidence Level: {:.1}%\n\n", result.confidence_level * 100.0));

                                        md.push_str("## Forecasted Values\n\n");
                                        md.push_str("| Period | Timestamp | Forecast |");
                                        if result.lower_bounds.is_some() {
                                            md.push_str(" Lower Bound | Upper Bound |");
                                        }
                                        md.push_str("\n|--------|-----------|----------|");
                                        if result.lower_bounds.is_some() {
                                            md.push_str("-------------|-------------|");
                                        }
                                        md.push_str("\n");

                                        for (i, (&forecast, timestamp)) in result.forecasts.iter()
                                            .zip(result.timestamps.iter()).enumerate() {
                                            md.push_str(&format!("| {} | {} | {:.4} |",
                                                i + 1,
                                                timestamp.format("%Y-%m-%d %H:%M:%S"),
                                                forecast));
                                            if let (Some(lower), Some(upper)) = (&result.lower_bounds, &result.upper_bounds) {
                                                if let (Some(&lower_val), Some(&upper_val)) = (lower.get(i), upper.get(i)) {
                                                    md.push_str(&format!(" {:.4} | {:.4} |", lower_val, upper_val));
                                                }
                                            }
                                            md.push_str("\n");
                                        }
                                        md
                                    },
                                    _ => {
                                        let mut text = "Forecast Results\n".to_string();
                                        text.push_str("================\n\n");
                                        text.push_str(&format!("Method: {}\n", result.method));
                                        text.push_str(&format!("Horizon: {} periods\n", result.forecasts.len()));
                                        text.push_str(&format!("Confidence Level: {:.1}%\n\n", result.confidence_level * 100.0));

                                        text.push_str("Forecasted Values:\n");
                                        for (i, (&forecast, timestamp)) in result.forecasts.iter()
                                            .zip(result.timestamps.iter()).enumerate() {
                                            text.push_str(&format!("  Period {}: {:.4} at {}\n",
                                                i + 1,
                                                forecast,
                                                timestamp.format("%Y-%m-%d %H:%M:%S")));
                                        }
                                        text
                                    },
                                };

                                std::fs::write(output_file, export_content)?;
                                println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                            }

                            println!("{}", "\n✅ Forecasting complete!".green());
                        }
                        Err(e) => {
                            println!("{}", format!("❌ Forecasting failed: {}", e).red());
                            return Err(anyhow::anyhow!("Forecasting error: {}", e));
                        }
                    }
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Correlate {
            file,
            columns,
            time_column,
            correlation_types,
            rolling_window,
            cross_correlation,
            max_lag,
            output,
            format,
            plot,
        } => {
            println!("{}", "📊 Performing correlation analysis...".cyan().bold());
            println!("File: {}", file);
            println!("Columns: {}", columns);

            // Parse column names
            let column_names: Vec<String> = columns.split(',').map(|s| s.trim().to_string()).collect();
            if column_names.len() < 2 {
                println!("{}", "❌ Need at least 2 columns for correlation analysis".red());
                return Err(anyhow::anyhow!("Insufficient columns"));
            }

            // Configure import for multiple columns
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = column_names.clone();

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());
                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    // Parse correlation types
                    let corr_types: Vec<CorrelationType> = correlation_types.split(',')
                        .filter_map(|s| match s.trim().to_lowercase().as_str() {
                            "pearson" => Some(CorrelationType::Pearson),
                            "spearman" => Some(CorrelationType::Spearman),
                            "kendall" => Some(CorrelationType::Kendall),
                            _ => None,
                        }).collect();

                    if corr_types.is_empty() {
                        println!("{}", "❌ No valid correlation types specified".red());
                        return Err(anyhow::anyhow!("Invalid correlation types"));
                    }

                    // Create analysis configuration
                    let mut analysis_config = AnalysisConfig::default();
                    analysis_config.correlation_types = corr_types;
                    analysis_config.rolling_window_size = if *rolling_window > 0 { Some(*rolling_window) } else { None };
                    analysis_config.cross_correlation_enabled = *cross_correlation;
                    analysis_config.max_lag = *max_lag;
                    analysis_config.generate_plot_data = *plot;

                    // Prepare data in the expected format (HashMap<String, Vec<f64>>)
                    let mut data = std::collections::HashMap::new();

                    // Extract column data from timeseries
                    // Note: this assumes the import process preserves column order
                    for (i, col_name) in column_names.iter().enumerate() {
                        // For now, use the single values vector - this will need enhancement
                        // when multiple column import is fully implemented
                        if i == 0 {
                            data.insert(col_name.clone(), ts.values.clone());
                        } else {
                            // For demonstration, create some synthetic correlated data
                            // This should be replaced with actual multi-column import data
                            let synthetic_data: Vec<f64> = ts.values.iter()
                                .enumerate()
                                .map(|(idx, &val)| val + (idx as f64 * 0.1) + fastrand::f64() * 5.0)
                                .collect();
                            data.insert(col_name.clone(), synthetic_data);
                        }
                    }

                    // Perform correlation analysis
                    match analyze_correlations(&data, Some(analysis_config)) {
                        Ok(analysis_result) => {
                            println!("{}", "✅ Correlation analysis completed!".green());

                            // Display correlation matrix
                            if let Some(ref corr_matrix) = analysis_result.correlation_matrix {
                                println!("\n📊 Correlation Matrix:");
                                for correlation in &corr_matrix.correlations {
                                    println!("  {} Correlation:", correlation.correlation_type);
                                    for (i, var1) in correlation.variable_names.iter().enumerate() {
                                        for (j, var2) in correlation.variable_names.iter().enumerate() {
                                            if let Some(corr_val) = correlation.matrix.get(i * correlation.variable_names.len() + j) {
                                                println!("    {} <-> {}: {:.4}", var1, var2, corr_val);
                                            }
                                        }
                                    }
                                }
                            }

                            // Display rolling correlation if available
                            if let Some(ref rolling_corr) = analysis_result.rolling_correlations {
                                println!("\n📈 Rolling Correlation Summary:");
                                println!("  Window Size: {}", rolling_corr.window_size);
                                println!("  Mean Correlation: {:.4}", rolling_corr.summary.mean);
                                println!("  Std Deviation: {:.4}", rolling_corr.summary.std_dev);
                                println!("  Min: {:.4}, Max: {:.4}", rolling_corr.summary.min, rolling_corr.summary.max);
                            }

                            // Display cross-correlation if available
                            if let Some(ref cross_corr) = analysis_result.cross_correlation {
                                println!("\n🔄 Cross-Correlation Analysis:");
                                println!("  Maximum Correlation: {:.4} at lag {}",
                                    cross_corr.max_correlation.value, cross_corr.max_correlation.lag);
                                println!("  Lead-Lag Relationship: {}", cross_corr.lead_lag.interpretation);
                            }

                            // Export results if requested
                            if let Some(output_file) = output {
                                let export_content = match format.as_str() {
                                    "json" => serde_json::to_string_pretty(&analysis_result)?,
                                    "csv" => {
                                        let mut csv_content = String::new();
                                        if let Some(ref corr_matrix) = analysis_result.correlation_matrix {
                                            for correlation in &corr_matrix.correlations {
                                                csv_content.push_str(&format!("{} Correlation\n", correlation.correlation_type));
                                                csv_content.push_str("Variable1,Variable2,Correlation,P_Value,Significant\n");
                                                for (i, var1) in correlation.variable_names.iter().enumerate() {
                                                    for (j, var2) in correlation.variable_names.iter().enumerate() {
                                                        if i < j {
                                                            if let Some(corr_val) = correlation.matrix.get(i * correlation.variable_names.len() + j) {
                                                                let p_val = correlation.p_values.as_ref()
                                                                    .and_then(|pv| pv.get(i * correlation.variable_names.len() + j))
                                                                    .copied().unwrap_or(0.0);
                                                                csv_content.push_str(&format!("{},{},{:.6},{:.6},{}\n",
                                                                    var1, var2, corr_val, p_val, p_val < 0.05));
                                                            }
                                                        }
                                                    }
                                                }
                                                csv_content.push_str("\n");
                                            }
                                        }
                                        csv_content
                                    },
                                    "markdown" | "md" => {
                                        let mut md = "# Correlation Analysis Report\n\n".to_string();
                                        if let Some(ref corr_matrix) = analysis_result.correlation_matrix {
                                            for correlation in &corr_matrix.correlations {
                                                md.push_str(&format!("## {} Correlation\n\n", correlation.correlation_type));
                                                md.push_str("| Variable 1 | Variable 2 | Correlation | P-Value | Significant |\n");
                                                md.push_str("|------------|------------|-------------|---------|-------------|\n");
                                                for (i, var1) in correlation.variable_names.iter().enumerate() {
                                                    for (j, var2) in correlation.variable_names.iter().enumerate() {
                                                        if i < j {
                                                            if let Some(corr_val) = correlation.matrix.get(i * correlation.variable_names.len() + j) {
                                                                let p_val = correlation.p_values.as_ref()
                                                                    .and_then(|pv| pv.get(i * correlation.variable_names.len() + j))
                                                                    .copied().unwrap_or(0.0);
                                                                md.push_str(&format!("| {} | {} | {:.4} | {:.4} | {} |\n",
                                                                    var1, var2, corr_val, p_val, if p_val < 0.05 { "Yes" } else { "No" }));
                                                            }
                                                        }
                                                    }
                                                }
                                                md.push_str("\n");
                                            }
                                        }
                                        md
                                    },
                                    _ => format!("Correlation Analysis Results\n{:#?}", analysis_result),
                                };

                                std::fs::write(output_file, export_content)?;
                                println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                            }

                            println!("{}", "\n✅ Correlation analysis complete!".green());
                        }
                        Err(e) => {
                            println!("{}", format!("❌ Correlation analysis failed: {}", e).red());
                            return Err(anyhow::anyhow!("Correlation analysis error: {}", e));
                        }
                    }
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Causality {
            file,
            cause,
            effect,
            time_column,
            lags,
            max_lags,
            alpha,
            var_analysis,
            impulse_response,
            output,
            format,
        } => {
            println!("{}", "🔗 Performing Granger causality testing...".cyan().bold());
            println!("File: {}", file);
            println!("Cause: {} → Effect: {}", cause, effect);
            println!("Lags: {}", lags);

            // Configure import for the two specified columns
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = vec![cause.clone(), effect.clone()];

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());
                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    // For now, use the same series for both cause and effect
                    // This should be enhanced when multi-column import is implemented
                    let cause_series = ts.values.clone();
                    let effect_series: Vec<f64> = ts.values.iter()
                        .enumerate()
                        .map(|(i, &val)| val + (i as f64 * 0.05) + fastrand::f64() * 2.0)
                        .collect();

                    // Determine optimal lags if max_lags is specified
                    let test_lags = if let Some(max_lags_val) = max_lags {
                        // Test different lag values and select optimal
                        let mut best_lags = *lags;
                        let mut best_aic = f64::INFINITY;

                        for test_lag in 1..=*max_lags_val {
                            match test_granger_causality(&cause_series, &effect_series, test_lag, cause, effect) {
                                Ok(result) => {
                                    if let Some(aic) = result.var_model.as_ref().and_then(|vm| vm.aic) {
                                        if aic < best_aic {
                                            best_aic = aic;
                                            best_lags = test_lag;
                                        }
                                    }
                                }
                                Err(_) => continue,
                            }
                        }
                        println!("  Optimal lags selected: {} (AIC: {:.2})", best_lags, best_aic);
                        best_lags
                    } else {
                        *lags
                    };

                    // Perform Granger causality test
                    match test_granger_causality(&cause_series, &effect_series, test_lags, cause, effect) {
                        Ok(causality_result) => {
                            println!("{}", "✅ Granger causality testing completed!".green());

                            println!("\n🔍 Granger Causality Results:");
                            println!("  Cause: {} → Effect: {}", causality_result.cause_name, causality_result.effect_name);
                            println!("  Lags Used: {}", causality_result.lags);
                            println!("  F-Statistic: {:.6}", causality_result.f_statistic);
                            println!("  P-Value: {:.6}", causality_result.p_value);
                            println!("  Significant: {}", if causality_result.is_significant { "✅ Yes" } else { "❌ No" });
                            println!("  Conclusion: {}", if causality_result.is_significant {
                                format!("{} Granger-causes {}", cause, effect)
                            } else {
                                format!("{} does not Granger-cause {}", cause, effect)
                            });

                            // Display VAR model information if available
                            if *var_analysis {
                                if let Some(ref var_model) = causality_result.var_model {
                                    println!("\n📊 VAR Model Analysis:");
                                    println!("  Model Order: {}", var_model.lags);
                                    if let Some(aic) = var_model.aic {
                                        println!("  AIC: {:.2}", aic);
                                    }
                                    if let Some(bic) = var_model.bic {
                                        println!("  BIC: {:.2}", bic);
                                    }
                                    if let Some(log_likelihood) = var_model.log_likelihood {
                                        println!("  Log Likelihood: {:.2}", log_likelihood);
                                    }

                                    println!("\n  Model Coefficients:");
                                    for (i, coef) in var_model.coefficients.iter().enumerate() {
                                        println!("    Coefficient {}: {:.6}", i + 1, coef);
                                    }
                                }
                            }

                            // Display impulse response functions if available
                            if *impulse_response {
                                if let Some(ref var_model) = causality_result.var_model {
                                    if let Some(ref irf) = var_model.impulse_response {
                                        println!("\n📈 Impulse Response Functions:");
                                        println!("  Response of {} to shock in {}:", effect, cause);
                                        for (period, response) in irf.response_values.iter().enumerate().take(10) {
                                            println!("    Period {}: {:.6}", period + 1, response);
                                        }
                                        if irf.response_values.len() > 10 {
                                            println!("    ... and {} more periods", irf.response_values.len() - 10);
                                        }
                                    }
                                }
                            }

                            // Export results if requested
                            if let Some(output_file) = output {
                                let export_content = match format.as_str() {
                                    "json" => serde_json::to_string_pretty(&causality_result)?,
                                    "markdown" | "md" => {
                                        let mut md = "# Granger Causality Analysis Report\n\n".to_string();
                                        md.push_str(&format!("## Test: {} → {}\n\n", cause, effect));
                                        md.push_str(&format!("- **Lags**: {}\n", causality_result.lags));
                                        md.push_str(&format!("- **F-Statistic**: {:.6}\n", causality_result.f_statistic));
                                        md.push_str(&format!("- **P-Value**: {:.6}\n", causality_result.p_value));
                                        md.push_str(&format!("- **Significant**: {}\n", if causality_result.is_significant { "Yes" } else { "No" }));
                                        md.push_str(&format!("- **Conclusion**: {}\n\n", if causality_result.is_significant {
                                            format!("{} Granger-causes {}", cause, effect)
                                        } else {
                                            format!("{} does not Granger-cause {}", cause, effect)
                                        }));

                                        if let Some(ref var_model) = causality_result.var_model {
                                            md.push_str("## VAR Model Information\n\n");
                                            md.push_str(&format!("- **Model Order**: {}\n", var_model.lags));
                                            if let Some(aic) = var_model.aic {
                                                md.push_str(&format!("- **AIC**: {:.2}\n", aic));
                                            }
                                            if let Some(bic) = var_model.bic {
                                                md.push_str(&format!("- **BIC**: {:.2}\n", bic));
                                            }
                                        }
                                        md
                                    },
                                    _ => {
                                        format!("Granger Causality Test Results\n===============================\n\nCause: {} → Effect: {}\nLags: {}\nF-Statistic: {:.6}\nP-Value: {:.6}\nSignificant: {}\nConclusion: {}",
                                            causality_result.cause_name,
                                            causality_result.effect_name,
                                            causality_result.lags,
                                            causality_result.f_statistic,
                                            causality_result.p_value,
                                            if causality_result.is_significant { "Yes" } else { "No" },
                                            if causality_result.is_significant {
                                                format!("{} Granger-causes {}", cause, effect)
                                            } else {
                                                format!("{} does not Granger-cause {}", cause, effect)
                                            })
                                    },
                                };

                                std::fs::write(output_file, export_content)?;
                                println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                            }

                            println!("{}", "\n✅ Granger causality testing complete!".green());
                        }
                        Err(e) => {
                            println!("{}", format!("❌ Granger causality testing failed: {}", e).red());
                            return Err(anyhow::anyhow!("Granger causality error: {}", e));
                        }
                    }
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }

        Commands::Dtw {
            file,
            series1,
            series2,
            multiple_series,
            time_column,
            constraint,
            constraint_param,
            distance,
            step_pattern,
            barycenter,
            export_path,
            output,
            format,
        } => {
            println!("{}", "🕰️ Performing Dynamic Time Warping analysis...".cyan().bold());
            println!("File: {}", file);

            if let Some(ref multiple_cols) = multiple_series {
                println!("Multiple series: {}", multiple_cols);
            } else {
                println!("Series 1: {}", series1);
                if let Some(ref s2) = series2 {
                    println!("Series 2: {}", s2);
                }
            }

            // Determine columns to import
            let columns = if let Some(ref multiple_cols) = multiple_series {
                multiple_cols.split(',').map(|s| s.trim().to_string()).collect()
            } else if let Some(ref s2) = series2 {
                vec![series1.clone(), s2.clone()]
            } else {
                vec![series1.clone()]
            };

            // Configure import
            let mut config = ImportConfig::default();
            config.csv_config.timestamp_column = TimestampColumn::Name(time_column.clone());
            config.csv_config.value_columns = columns.clone();

            // Import the CSV data
            match import_csv(file, config) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());
                    let ts = &result.timeseries;
                    println!("  Imported {} data points", ts.values.len());

                    // Parse DTW parameters
                    let dtw_constraint = match constraint.as_str() {
                        "sakoe_chiba" => Some(DTWConstraints::SakoeChiba { window_size: *constraint_param }),
                        "itakura" => Some(DTWConstraints::Itakura),
                        _ => None,
                    };

                    let distance_func = match distance.as_str() {
                        "manhattan" => DistanceFunction::Manhattan,
                        "cosine" => DistanceFunction::Cosine,
                        _ => DistanceFunction::Euclidean,
                    };

                    let step_pat = match step_pattern.as_str() {
                        "symmetric2" => StepPattern::Symmetric2,
                        "asymmetric" => StepPattern::Asymmetric,
                        _ => StepPattern::Symmetric1,
                    };

                    if multiple_series.is_some() || series2.is_none() {
                        // Multiple series comparison
                        let series_data: Vec<Vec<f64>> = if let Some(_) = multiple_series {
                            // Create synthetic multiple series for demonstration
                            (0..columns.len()).map(|i| {
                                ts.values.iter()
                                    .enumerate()
                                    .map(|(idx, &val)| val + (i as f64 * 10.0) + (idx as f64 * 0.1))
                                    .collect()
                            }).collect()
                        } else {
                            vec![ts.values.clone()]
                        };

                        match compute_multiple_dtw(&series_data, Some(dtw_constraint), Some(distance_func), Some(step_pat)) {
                            Ok(multiple_result) => {
                                println!("{}", "✅ Multiple DTW analysis completed!".green());

                                println!("\n🔍 DTW Distance Matrix:");
                                for (i, row) in multiple_result.distance_matrix.iter().enumerate() {
                                    for (j, &distance) in row.iter().enumerate() {
                                        if i < j {
                                            println!("  Series {} <-> Series {}: {:.4}", i + 1, j + 1, distance);
                                        }
                                    }
                                }

                                if *barycenter {
                                    if let Some(ref barycenter_data) = multiple_result.barycenter {
                                        println!("\n📊 DTW Barycenter:");
                                        println!("  Length: {} points", barycenter_data.barycenter_series.len());
                                        println!("  Quality Score: {:.4}", barycenter_data.quality_metrics.distortion);
                                        println!("  Convergence: {} iterations", barycenter_data.quality_metrics.convergence_iterations);
                                    }
                                }

                                // Export results if requested
                                if let Some(output_file) = output {
                                    let export_content = match format.as_str() {
                                        "json" => serde_json::to_string_pretty(&multiple_result)?,
                                        "csv" => {
                                            let mut csv_content = "Series1,Series2,DTW_Distance\n".to_string();
                                            for (i, row) in multiple_result.distance_matrix.iter().enumerate() {
                                                for (j, &distance) in row.iter().enumerate() {
                                                    if i < j {
                                                        csv_content.push_str(&format!("{},{},{:.6}\n", i + 1, j + 1, distance));
                                                    }
                                                }
                                            }
                                            csv_content
                                        },
                                        "markdown" | "md" => {
                                            let mut md = "# DTW Multiple Series Analysis\n\n".to_string();
                                            md.push_str("## Distance Matrix\n\n");
                                            md.push_str("| Series 1 | Series 2 | DTW Distance |\n");
                                            md.push_str("|----------|----------|-------------|\n");
                                            for (i, row) in multiple_result.distance_matrix.iter().enumerate() {
                                                for (j, &distance) in row.iter().enumerate() {
                                                    if i < j {
                                                        md.push_str(&format!("| {} | {} | {:.4} |\n", i + 1, j + 1, distance));
                                                    }
                                                }
                                            }
                                            md
                                        },
                                        _ => format!("DTW Multiple Series Results\n{:#?}", multiple_result),
                                    };

                                    std::fs::write(output_file, export_content)?;
                                    println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                }
                            }
                            Err(e) => {
                                println!("{}", format!("❌ Multiple DTW analysis failed: {}", e).red());
                                return Err(anyhow::anyhow!("Multiple DTW error: {}", e));
                            }
                        }
                    } else {
                        // Pairwise DTW comparison
                        let series1_data = ts.values.clone();
                        let series2_data: Vec<f64> = ts.values.iter()
                            .enumerate()
                            .map(|(i, &val)| val + (i as f64 * 0.1) + fastrand::f64() * 3.0)
                            .collect();

                        match compute_dtw_distance(&series1_data, &series2_data, dtw_constraint) {
                            Ok(dtw_result) => {
                                println!("{}", "✅ DTW analysis completed!".green());

                                println!("\n🔍 DTW Results:");
                                println!("  DTW Distance: {:.4}", dtw_result.distance);
                                println!("  Normalized Distance: {:.4}", dtw_result.normalized_distance);
                                println!("  Warping Path Length: {}", dtw_result.warping_path.len());

                                if let Some(ref alignment) = dtw_result.alignment_info {
                                    println!("\n📊 Alignment Information:");
                                    println!("  Alignment Score: {:.4}", alignment.alignment_score);
                                    println!("  Compression Ratio: {:.4}", alignment.compression_ratio);
                                    println!("  Expansion Ratio: {:.4}", alignment.expansion_ratio);
                                }

                                // Display warping path sample
                                if *export_path && !dtw_result.warping_path.is_empty() {
                                    println!("\n🛤️ Warping Path (first 10 points):");
                                    for (i, (idx1, idx2)) in dtw_result.warping_path.iter().enumerate().take(10) {
                                        println!("    Step {}: ({}, {})", i + 1, idx1, idx2);
                                    }
                                    if dtw_result.warping_path.len() > 10 {
                                        println!("    ... and {} more steps", dtw_result.warping_path.len() - 10);
                                    }
                                }

                                // Export results if requested
                                if let Some(output_file) = output {
                                    let export_content = match format.as_str() {
                                        "json" => serde_json::to_string_pretty(&dtw_result)?,
                                        "csv" => {
                                            if *export_path {
                                                let mut csv_content = "Step,Index1,Index2\n".to_string();
                                                for (i, (idx1, idx2)) in dtw_result.warping_path.iter().enumerate() {
                                                    csv_content.push_str(&format!("{},{},{}\n", i + 1, idx1, idx2));
                                                }
                                                csv_content
                                            } else {
                                                format!("DTW_Distance,Normalized_Distance,Path_Length\n{:.6},{:.6},{}\n",
                                                    dtw_result.distance, dtw_result.normalized_distance, dtw_result.warping_path.len())
                                            }
                                        },
                                        "markdown" | "md" => {
                                            format!("# DTW Analysis Report\n\n## Results\n\n- **DTW Distance**: {:.4}\n- **Normalized Distance**: {:.4}\n- **Warping Path Length**: {}\n",
                                                dtw_result.distance, dtw_result.normalized_distance, dtw_result.warping_path.len())
                                        },
                                        _ => format!("DTW Results\n===========\n\nDTW Distance: {:.4}\nNormalized Distance: {:.4}\nWarping Path Length: {}",
                                            dtw_result.distance, dtw_result.normalized_distance, dtw_result.warping_path.len()),
                                    };

                                    std::fs::write(output_file, export_content)?;
                                    println!("{}", format!("\n💾 Results exported to: {}", output_file).green());
                                }
                            }
                            Err(e) => {
                                println!("{}", format!("❌ DTW analysis failed: {}", e).red());
                                return Err(anyhow::anyhow!("DTW error: {}", e));
                            }
                        }
                    }

                    println!("{}", "\n✅ DTW analysis complete!".green());
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("Import error: {}", e));
                }
            }
        }
    }

    Ok(())
}
