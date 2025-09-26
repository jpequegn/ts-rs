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

    /// Visualize time series data as charts
    Visualize {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,

        /// Output image file path
        #[clap(short, long, default_value = "chart.png")]
        output: String,
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

        Commands::Visualize { file, output } => {
            println!("{}", "📈 Creating visualization...".cyan().bold());
            println!("Input file: {}", file);
            println!("Output file: {}", output);

            // TODO: Implement visualization logic
            println!("{}", "✅ Chart created! (placeholder)".green());
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
    }

    Ok(())
}
