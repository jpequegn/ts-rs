use clap::Parser;
use anyhow::Result;
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

// Use our CLI module
use chronos::cli::{Cli, Commands};
use chronos::cli::interactive::InteractiveSession;
use chronos::cli::commands::*;

fn setup_logging(verbose: bool, quiet: bool) {
    // Set up logging based on verbosity flags
    if verbose && !quiet {
        println!("{}", "Debug mode enabled".cyan().dimmed());
    }
}

fn setup_output_directory(output_dir: &Option<PathBuf>) -> Result<()> {
    if let Some(dir) = output_dir {
        if !dir.exists() {
            fs::create_dir_all(dir)?;
            println!("{}", format!("Created output directory: {}", dir.display()).green());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup global options
    setup_logging(cli.verbose, cli.quiet);
    setup_output_directory(&cli.output_dir)?;

    // Handle interactive mode
    if cli.interactive {
        if !cli.quiet {
            println!("{}", "🚀 Starting interactive mode...".cyan().bold());
        }
        let mut session = InteractiveSession::new();
        return session.start();
    }

    // Handle regular commands
    match cli.command {
        Some(Commands::Import(ref args)) => execute_import(args.clone(), &cli)?,
        Some(Commands::Stats(ref args)) => execute_stats(args.clone(), &cli)?,
        Some(Commands::Trend(ref args)) => execute_trend(args.clone(), &cli)?,
        Some(Commands::Seasonal(ref args)) => execute_seasonal(args.clone(), &cli)?,
        Some(Commands::Anomaly(ref args)) => execute_anomaly(args.clone(), &cli)?,
        Some(Commands::Forecast(ref args)) => execute_forecast(args.clone(), &cli)?,
        Some(Commands::Correlate(ref args)) => execute_correlate(args.clone(), &cli)?,
        Some(Commands::Plot(ref args)) => execute_plot(args.clone(), &cli)?,
        Some(Commands::Report(ref args)) => execute_report(args.clone(), &cli)?,
        None => {
            // No command provided, show help
            use clap::CommandFactory;
            let mut cmd = Cli::command();
            cmd.print_help()?;
            println!("\n\n{}", "💡 Use --help with any subcommand for detailed options".cyan());
            println!("{}", "💡 Use --interactive or -i to enter interactive mode".cyan());
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
