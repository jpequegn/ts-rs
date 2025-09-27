//! CLI Tests
//!
//! Comprehensive tests for the CLI functionality including command parsing,
//! execution, and interactive mode.

use super::*;
use super::commands::*;
use super::interactive::*;
use anyhow::Result;
use std::env;
use std::fs;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_parsing() {
        // Test basic CLI parsing
        let cli = Cli::try_parse_from(&["chronos", "--help"]);
        assert!(cli.is_err()); // --help causes clap to exit with help message
    }

    #[test]
    fn test_import_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "import",
            "--file", "test.csv",
            "--format", "csv",
            "--time-column", "timestamp",
            "--value-columns", "value",
            "--missing", "interpolate",
            "--validate"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Import(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.format, Some(ImportFormat::Csv));
            assert_eq!(cmd.time_column, Some("timestamp".to_string()));
            assert_eq!(cmd.value_columns, Some("value".to_string()));
            assert_eq!(cmd.missing, "interpolate");
            assert!(cmd.validate);
        } else {
            panic!("Expected Import command");
        }
    }

    #[test]
    fn test_stats_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "stats",
            "--file", "test.csv",
            "--column", "value",
            "--normality",
            "--stationarity",
            "--autocorr", "50",
            "--changepoints"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Stats(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.column, Some("value".to_string()));
            assert!(cmd.normality);
            assert!(cmd.stationarity);
            assert_eq!(cmd.autocorr, Some(50));
            assert!(cmd.changepoints);
        } else {
            panic!("Expected Stats command");
        }
    }

    #[test]
    fn test_trend_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "trend",
            "--file", "test.csv",
            "--column", "value",
            "--method", "decompose",
            "--decomposition", "stl",
            "--period", "12",
            "--export-components"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Trend(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.column, "value");
            assert_eq!(cmd.method, "decompose");
            assert_eq!(cmd.decomposition, "stl");
            assert_eq!(cmd.period, Some(12));
            assert!(cmd.export_components);
        } else {
            panic!("Expected Trend command");
        }
    }

    #[test]
    fn test_seasonal_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "seasonal",
            "--file", "test.csv",
            "--column", "value",
            "--method", "detect",
            "--max-period", "365",
            "--min-period", "2",
            "--periods", "7,30,365",
            "--calendar-effects"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Seasonal(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.column, "value");
            assert_eq!(cmd.method, "detect");
            assert_eq!(cmd.max_period, 365);
            assert_eq!(cmd.min_period, 2);
            assert_eq!(cmd.periods, Some("7,30,365".to_string()));
            assert!(cmd.calendar_effects);
        } else {
            panic!("Expected Seasonal command");
        }
    }

    #[test]
    fn test_anomaly_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "anomaly",
            "--file", "test.csv",
            "--column", "value",
            "--method", "zscore",
            "--threshold", "3.0",
            "--window", "30",
            "--mark",
            "--export-scores"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Anomaly(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.column, "value");
            assert_eq!(cmd.method, "zscore");
            assert_eq!(cmd.threshold, Some(3.0));
            assert_eq!(cmd.window, Some(30));
            assert!(cmd.mark);
            assert!(cmd.export_scores);
        } else {
            panic!("Expected Anomaly command");
        }
    }

    #[test]
    fn test_forecast_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "forecast",
            "--file", "test.csv",
            "--column", "value",
            "--method", "arima",
            "--horizon", "30",
            "--confidence", "0.95",
            "--backtest",
            "--backtest-windows", "10",
            "--export-forecast"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Forecast(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.column, "value");
            assert_eq!(cmd.method, "arima");
            assert_eq!(cmd.horizon, 30);
            assert_eq!(cmd.confidence, 0.95);
            assert!(cmd.backtest);
            assert_eq!(cmd.backtest_windows, 10);
            assert!(cmd.export_forecast);
        } else {
            panic!("Expected Forecast command");
        }
    }

    #[test]
    fn test_correlate_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "correlate",
            "--file", "test.csv",
            "--columns", "value1,value2,value3",
            "--method", "pearson",
            "--lagged",
            "--max-lag", "20",
            "--alpha", "0.01",
            "--heatmap"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Correlate(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.columns, Some("value1,value2,value3".to_string()));
            assert_eq!(cmd.method, "pearson");
            assert!(cmd.lagged);
            assert_eq!(cmd.max_lag, 20);
            assert_eq!(cmd.alpha, 0.01);
            assert!(cmd.heatmap);
        } else {
            panic!("Expected Correlate command");
        }
    }

    #[test]
    fn test_plot_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "plot",
            "--file", "test.csv",
            "--columns", "value",
            "--plot-type", "line",
            "--title", "Test Plot",
            "--xlabel", "Time",
            "--ylabel", "Value",
            "--size", "10,8",
            "--dpi", "150",
            "--grid",
            "--interactive"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Plot(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.columns, Some("value".to_string()));
            assert_eq!(cmd.plot_type, "line");
            assert_eq!(cmd.title, Some("Test Plot".to_string()));
            assert_eq!(cmd.xlabel, Some("Time".to_string()));
            assert_eq!(cmd.ylabel, Some("Value".to_string()));
            assert_eq!(cmd.size, "10,8");
            assert_eq!(cmd.dpi, 150);
            assert!(cmd.grid);
            assert!(cmd.interactive);
        } else {
            panic!("Expected Plot command");
        }
    }

    #[test]
    fn test_report_command_parsing() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "report",
            "--file", "test.csv",
            "--column", "value",
            "--template", "comprehensive",
            "--sections", "stats,trends,seasonality",
            "--title", "Test Report",
            "--author", "Test Author",
            "--comprehensive",
            "--max-insights", "15"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        if let Some(Commands::Report(cmd)) = cli.command {
            assert_eq!(cmd.file.to_str().unwrap(), "test.csv");
            assert_eq!(cmd.column, Some("value".to_string()));
            assert_eq!(cmd.template, "comprehensive");
            assert_eq!(cmd.sections, Some("stats,trends,seasonality".to_string()));
            assert_eq!(cmd.title, Some("Test Report".to_string()));
            assert_eq!(cmd.author, Some("Test Author".to_string()));
            assert!(cmd.comprehensive);
            assert_eq!(cmd.max_insights, 15);
        } else {
            panic!("Expected Report command");
        }
    }

    #[test]
    fn test_global_options() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "--config", "config.toml",
            "--verbose",
            "--output-dir", "/tmp/output",
            "--format", "json",
            "stats",
            "--file", "test.csv"
        ]);

        assert!(cli.is_ok());
        let cli = cli.unwrap();

        assert_eq!(cli.config, Some(PathBuf::from("config.toml")));
        assert!(cli.verbose);
        assert_eq!(cli.output_dir, Some(PathBuf::from("/tmp/output")));
        assert!(matches!(cli.format, OutputFormat::Json));
    }

    #[test]
    fn test_quiet_and_verbose_conflict() {
        let cli = Cli::try_parse_from(&[
            "chronos",
            "--verbose",
            "--quiet",
            "stats",
            "--file", "test.csv"
        ]);

        // Should fail due to conflicting flags
        assert!(cli.is_err());
    }

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Text.extension(), "txt");
        assert_eq!(OutputFormat::Json.extension(), "json");
        assert_eq!(OutputFormat::Csv.extension(), "csv");
        assert_eq!(OutputFormat::Markdown.extension(), "md");
        assert_eq!(OutputFormat::Html.extension(), "html");
        assert_eq!(OutputFormat::Pdf.extension(), "pdf");
    }

    #[test]
    fn test_create_output_path() {
        let base_path = PathBuf::from("test");
        let format = OutputFormat::Json;
        let result = create_output_path(&base_path, &format);
        assert_eq!(result, PathBuf::from("test.json"));

        // Test with existing extension
        let base_path = PathBuf::from("test.csv");
        let format = OutputFormat::Json;
        let result = create_output_path(&base_path, &format);
        assert_eq!(result, PathBuf::from("test.csv")); // Should keep existing extension
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[test]
    fn test_command_execution() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let test_file = temp_dir.path().join("test.csv");

        // Create a simple test CSV file
        fs::write(&test_file, "timestamp,value\n2023-01-01,1.0\n2023-01-02,2.0\n")?;

        // Test import command execution
        let import_cmd = ImportCommand {
            file: test_file.clone(),
            format: Some(ImportFormat::Csv),
            time_column: Some("timestamp".to_string()),
            value_columns: Some("value".to_string()),
            missing: "interpolate".to_string(),
            resample: None,
            output: None,
            validate: false,
        };

        let cli = Cli {
            config: None,
            verbose: false,
            quiet: true, // Quiet mode to avoid output during tests
            output_dir: None,
            format: OutputFormat::Text,
            interactive: false,
            command: None,
        };

        // Test that command execution doesn't panic
        let result = execute_import(import_cmd, &cli);
        assert!(result.is_ok());

        // Test stats command execution
        let stats_cmd = StatsCommand {
            file: test_file.clone(),
            column: Some("value".to_string()),
            normality: false,
            stationarity: false,
            autocorr: None,
            changepoints: false,
            output: None,
        };

        let result = execute_stats(stats_cmd, &cli);
        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn test_export_data() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let output_path = temp_dir.path().join("test_output");

        let test_data = "test,data\n1,2\n3,4\n";
        let format = OutputFormat::Csv;

        let result = export_data(test_data, &output_path, &format);
        assert!(result.is_ok());

        // Check that file was created with correct extension
        let expected_path = temp_dir.path().join("test_output.csv");
        assert!(expected_path.exists());

        let content = fs::read_to_string(&expected_path)?;
        assert_eq!(content, test_data);

        Ok(())
    }

    // Note: output directory creation is tested in main.rs integration tests
}

#[cfg(test)]
mod interactive_tests {
    use super::*;

    #[test]
    fn test_interactive_session_creation() {
        // Test that we can create a session without panicking
        let _session = InteractiveSession::new();
        // Note: Fields are private, so we can only test creation
    }

    #[test]
    fn test_session_context_creation() {
        // Test that we can create a session context
        let context = SessionContext::default();
        assert!(context.files.is_empty());
        assert!(context.results.is_empty());
        assert!(context.variables.is_empty());
    }
}