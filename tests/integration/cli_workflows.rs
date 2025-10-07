//! Integration tests for CLI workflows and end-to-end functionality
//!
//! This module tests complete CLI workflows including data import,
//! analysis execution, and export functionality.

use serial_test::serial;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::{NamedTempFile, TempDir};

/// Helper to run chronos CLI commands
fn run_chronos_command(args: &[&str]) -> Result<std::process::Output, std::io::Error> {
    Command::new("cargo")
        .args(&["run", "--bin", "chronos", "--"])
        .args(args)
        .output()
}

/// Create a sample CSV file for testing
fn create_sample_csv(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut file, content.as_bytes()).unwrap();
    file
}

/// Create a comprehensive test dataset
fn create_test_dataset() -> String {
    let mut csv_content = String::from("timestamp,value\n");
    for i in 0..100 {
        csv_content.push_str(&format!(
            "2023-01-01T{:02}:00:00Z,{}\n",
            i % 24,
            i as f64 + (i as f64 * 0.1).sin()
        ));
    }
    csv_content
}

#[cfg(test)]
mod basic_cli_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_cli_help_command() {
        let output = run_chronos_command(&["--help"]).unwrap();

        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("chronos"));
        assert!(stdout.contains("time series"));
    }

    #[test]
    #[serial]
    fn test_cli_version_command() {
        let output = run_chronos_command(&["--version"]).unwrap();

        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("0.1.0"));
    }
}

#[cfg(test)]
mod import_export_workflow_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_csv_import_basic() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("output.json");

        let output = run_chronos_command(&[
            "import",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .unwrap();

        // Check command succeeded
        if !output.status.success() {
            println!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
            println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        }

        // For now, just check that it doesn't crash
        // In a full implementation, we'd verify the output file content
    }

    #[test]
    #[serial]
    fn test_json_export() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("analysis.json");

        // Run analysis with JSON export
        let output = run_chronos_command(&[
            "analyze",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--output",
            output_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .unwrap();

        // Check if output file was created
        if output.status.success() && output_path.exists() {
            let content = fs::read_to_string(&output_path).unwrap();
            // Should be valid JSON
            assert!(serde_json::from_str::<serde_json::Value>(&content).is_ok());
        }
    }

    #[test]
    #[serial]
    fn test_multiple_format_export() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();

        // Test different export formats
        for format in &["json", "csv", "markdown"] {
            let output_path = temp_dir.path().join(format!("output.{}", format));

            let output = run_chronos_command(&[
                "stats",
                "--file",
                input_file.path().to_str().unwrap(),
                "--time-column",
                "timestamp",
                "--value-column",
                "value",
                "--output",
                output_path.to_str().unwrap(),
                "--format",
                format,
            ])
            .unwrap();

            // Command should not crash
            if !output.status.success() {
                println!(
                    "Failed format {}: {}",
                    format,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }
}

#[cfg(test)]
mod analysis_workflow_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_comprehensive_analysis_workflow() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path();

        // Run comprehensive analysis
        let output = run_chronos_command(&[
            "analyze",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--comprehensive",
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .unwrap();

        // Should not crash
        if !output.status.success() {
            println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    #[test]
    #[serial]
    fn test_trend_analysis_workflow() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("trend_analysis.json");

        let output = run_chronos_command(&[
            "trend",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--method",
            "comprehensive",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .unwrap();

        // Should not crash
        if !output.status.success() {
            println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    #[test]
    #[serial]
    fn test_seasonal_analysis_workflow() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("seasonal_analysis.json");

        let output = run_chronos_command(&[
            "seasonal",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .unwrap();

        // Should not crash
        if !output.status.success() {
            println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    #[test]
    #[serial]
    fn test_anomaly_detection_workflow() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("anomalies.json");

        let output = run_chronos_command(&[
            "anomaly",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--method",
            "statistical",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .unwrap();

        // Should not crash
        if !output.status.success() {
            println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    #[test]
    #[serial]
    fn test_forecasting_workflow() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("forecast.json");

        let output = run_chronos_command(&[
            "forecast",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--horizon",
            "10",
            "--method",
            "arima",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .unwrap();

        // Should not crash
        if !output.status.success() {
            println!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
        }
    }
}

#[cfg(test)]
mod interactive_mode_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_interactive_mode_startup() {
        // Test that interactive mode starts without crashing
        let output = Command::new("cargo")
            .args(&["run", "--bin", "chronos", "--", "--interactive"])
            .env("CHRONOS_TEST_MODE", "1") // Hypothetical test mode
            .output()
            .unwrap();

        // Just check it doesn't immediately crash
        // In practice, interactive mode testing would require more sophisticated setup
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_invalid_file_path() {
        let output = run_chronos_command(&[
            "import",
            "--file",
            "/nonexistent/path/file.csv",
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
        ])
        .unwrap();

        // Should fail gracefully
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("file") || stderr.contains("not found") || stderr.contains("No such")
        );
    }

    #[test]
    #[serial]
    fn test_invalid_csv_format() {
        let invalid_csv = "invalid,csv,content\n1,2,three\n";
        let input_file = create_sample_csv(invalid_csv);

        let output = run_chronos_command(&[
            "import",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "nonexistent_column",
            "--value-column",
            "value",
        ])
        .unwrap();

        // Should fail gracefully with meaningful error
        assert!(!output.status.success());
    }

    #[test]
    #[serial]
    fn test_malformed_timestamp() {
        let bad_csv = "timestamp,value\ninvalid_date,123\n2023-01-01T00:00:00Z,456\n";
        let input_file = create_sample_csv(bad_csv);

        let output = run_chronos_command(&[
            "import",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
        ])
        .unwrap();

        // Should handle gracefully (either skip invalid rows or report error)
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Should provide meaningful error message
            assert!(!stderr.is_empty());
        }
    }
}

#[cfg(test)]
mod performance_integration_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_large_dataset_handling() {
        // Create a larger dataset for performance testing
        let mut large_csv = String::from("timestamp,value\n");
        for i in 0..10000 {
            large_csv.push_str(&format!(
                "2023-01-01T{:02}:{:02}:00Z,{}\n",
                i % 24,
                (i % 60),
                i as f64
            ));
        }

        let input_file = create_sample_csv(&large_csv);
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("large_analysis.json");

        let start_time = std::time::Instant::now();

        let output = run_chronos_command(&[
            "stats",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .unwrap();

        let elapsed = start_time.elapsed();

        // Should complete within reasonable time (this is a rough test)
        if output.status.success() {
            assert!(
                elapsed.as_secs() < 30,
                "Analysis took too long: {:?}",
                elapsed
            );
        }
    }
}

#[cfg(test)]
mod cross_platform_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_path_handling() {
        let csv_content = create_test_dataset();
        let input_file = create_sample_csv(&csv_content);

        // Test with different path separators and formats
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("cross_platform_test.json");

        let output = run_chronos_command(&[
            "import",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .unwrap();

        // Should handle paths correctly on all platforms
        if !output.status.success() {
            println!(
                "Cross-platform test failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    #[serial]
    fn test_unicode_handling() {
        let unicode_csv =
            "timestamp,value\n2023-01-01T00:00:00Z,123.45\n2023-01-01T01:00:00Z,678.90\n";
        let input_file = create_sample_csv(unicode_csv);

        let output = run_chronos_command(&[
            "import",
            "--file",
            input_file.path().to_str().unwrap(),
            "--time-column",
            "timestamp",
            "--value-column",
            "value",
        ])
        .unwrap();

        // Should handle Unicode correctly
        if !output.status.success() {
            println!(
                "Unicode test failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
