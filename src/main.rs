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
    validation::validate_data_quality
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

    /// Calculate statistical measures for time series data
    Stats {
        /// Path to the CSV file containing time series data
        #[clap(short, long)]
        file: String,
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

        Commands::Stats { file } => {
            println!("{}", "📋 Calculating statistics...".cyan().bold());
            println!("File: {}", file);

            // Import the CSV data with auto-detection
            match import_csv(file, ImportConfig::default()) {
                Ok(result) => {
                    println!("{}", "✅ Data imported successfully!".green());

                    println!("\n📊 Import Statistics:");
                    println!("  Rows processed: {}", result.stats.rows_processed);
                    println!("  Rows skipped: {}", result.stats.rows_skipped);
                    println!("  Missing values: {}", result.stats.missing_values);

                    // Calculate detailed statistics for the time series
                    let ts = &result.timeseries;

                    println!("\n📈 Time Series Statistics:");
                    println!("  Total data points: {}", ts.values.len());

                    if !ts.values.is_empty() {
                        let valid_values: Vec<f64> = ts.values.iter()
                                .filter(|&&v| !v.is_nan() && !v.is_infinite())
                                .copied()
                                .collect();

                        let nan_count = ts.values.iter().filter(|&&v| v.is_nan()).count();
                        let inf_count = ts.values.iter().filter(|&&v| v.is_infinite()).count();

                        println!("  Valid data points: {}", valid_values.len());
                        println!("  NaN values: {}", nan_count);
                        println!("  Infinite values: {}", inf_count);

                        if !valid_values.is_empty() {
                            // Basic statistics
                            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                            let min = valid_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = valid_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                            // Calculate variance and standard deviation
                            let variance = valid_values.iter()
                                .map(|&x| (x - mean).powi(2))
                                .sum::<f64>() / valid_values.len() as f64;
                            let std_dev = variance.sqrt();

                            // Calculate percentiles
                            let mut sorted_values = valid_values.clone();
                            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let median = if sorted_values.len() % 2 == 0 {
                                let mid = sorted_values.len() / 2;
                                (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
                            } else {
                                sorted_values[sorted_values.len() / 2]
                            };

                            println!("\n  Descriptive Statistics:");
                            println!("    Mean: {:.6}", mean);
                            println!("    Median: {:.6}", median);
                            println!("    Standard Deviation: {:.6}", std_dev);
                            println!("    Variance: {:.6}", variance);
                            println!("    Minimum: {:.6}", min);
                            println!("    Maximum: {:.6}", max);
                            println!("    Range: {:.6}", max - min);
                        }
                    }

                    // Data quality analysis
                    let quality_report = validate_data_quality(&ts.timestamps, &ts.values);
                    println!("\n  Data Quality:");
                    println!("    Quality Score: {:.1}%", quality_report.quality_score(ts.timestamps.len()) * 100.0);
                    if quality_report.has_issues() {
                        if quality_report.nan_count > 0 {
                            println!("    ⚠️  {} NaN values found", quality_report.nan_count);
                        }
                        if quality_report.infinite_count > 0 {
                            println!("    ⚠️  {} infinite values found", quality_report.infinite_count);
                        }
                        if quality_report.duplicate_timestamps > 0 {
                            println!("    ⚠️  {} duplicate timestamps found", quality_report.duplicate_timestamps);
                        }
                        if !quality_report.gaps.is_empty() {
                            println!("    ⚠️  {} time gaps detected", quality_report.gaps.len());
                        }
                        if quality_report.potential_outliers > 0 {
                            println!("    ⚠️  {} potential outliers detected", quality_report.potential_outliers);
                        }
                    } else {
                        println!("    ✅ No data quality issues detected");
                    }

                    println!("{}", "\n✅ Statistics calculated!".green());
                }
                Err(e) => {
                    println!("{}", format!("❌ Error importing data: {}", e).red());
                    return Err(anyhow::anyhow!("{}", e));
                }
            }
        }
    }

    Ok(())
}
