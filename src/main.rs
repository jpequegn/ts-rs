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
    stats::{analyze_timeseries, ExportFormat, export_stats_results}
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
    }

    Ok(())
}
