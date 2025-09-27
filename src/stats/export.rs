//! Export functionality for statistical analysis results
//!
//! Provides various export formats including JSON, CSV, and text reports
//! for statistical analysis results from the time series analysis engine.

use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::StatisticalAnalysisResult;

/// Supported export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format with full detail
    Json,
    /// CSV format (tabular data)
    Csv,
    /// Human-readable text report
    TextReport,
    /// Markdown format report
    Markdown,
    /// HTML report
    Html,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Include detailed metadata
    pub include_metadata: bool,
    /// Include confidence intervals
    pub include_confidence_intervals: bool,
    /// Include raw statistical values
    pub include_raw_values: bool,
    /// Decimal precision for numeric values
    pub decimal_precision: usize,
    /// Include plots data (for visualization)
    pub include_plot_data: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        ExportConfig {
            include_metadata: true,
            include_confidence_intervals: true,
            include_raw_values: false,
            decimal_precision: 4,
            include_plot_data: true,
        }
    }
}

/// Export statistical analysis results to various formats
///
/// # Arguments
/// * `results` - The statistical analysis results to export
/// * `format` - The desired export format
/// * `output_path` - Path to the output file
/// * `config` - Export configuration options
///
/// # Returns
/// * `Result<(), Box<dyn std::error::Error>>` - Success or error
pub fn export_stats_results(
    results: &StatisticalAnalysisResult,
    format: ExportFormat,
    output_path: &Path,
    config: Option<ExportConfig>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = config.unwrap_or_default();

    match format {
        ExportFormat::Json => export_json(results, output_path, &config),
        ExportFormat::Csv => export_csv(results, output_path, &config),
        ExportFormat::TextReport => export_text_report(results, output_path, &config),
        ExportFormat::Markdown => export_markdown(results, output_path, &config),
        ExportFormat::Html => export_html(results, output_path, &config),
    }
}

/// Export results as JSON
fn export_json(
    results: &StatisticalAnalysisResult,
    output_path: &Path,
    _config: &ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(results)?;
    let mut file = File::create(output_path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Export results as CSV
fn export_csv(
    results: &StatisticalAnalysisResult,
    output_path: &Path,
    config: &ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;
    let precision = config.decimal_precision;

    // Write header
    writeln!(file, "Statistic,Value,Category")?;

    // Basic metadata
    writeln!(file, "Column,{},Metadata", results.metadata.column_name)?;
    writeln!(file, "Sample_Count,{},Metadata", results.metadata.n_samples)?;
    writeln!(file, "Missing_Count,{},Metadata", results.metadata.n_missing)?;

    // Descriptive statistics
    if let Some(ref desc) = results.descriptive {
        writeln!(file, "Mean,{:.precision$},Descriptive", desc.mean, precision = precision)?;
        writeln!(file, "Median,{:.precision$},Descriptive", desc.median, precision = precision)?;
        writeln!(file, "Std_Dev,{:.precision$},Descriptive", desc.std_dev, precision = precision)?;
        writeln!(file, "Min,{:.precision$},Descriptive", desc.min, precision = precision)?;
        writeln!(file, "Max,{:.precision$},Descriptive", desc.max, precision = precision)?;
        writeln!(file, "Range,{:.precision$},Descriptive", desc.range, precision = precision)?;
        writeln!(file, "Q25,{:.precision$},Descriptive", desc.quantiles.q25, precision = precision)?;
        writeln!(file, "Q75,{:.precision$},Descriptive", desc.quantiles.q75, precision = precision)?;
        writeln!(file, "IQR,{:.precision$},Descriptive", desc.quantiles.iqr, precision = precision)?;
        writeln!(file, "Skewness,{:.precision$},Descriptive", desc.skewness, precision = precision)?;
        writeln!(file, "Kurtosis,{:.precision$},Descriptive", desc.kurtosis, precision = precision)?;
    }

    // Distribution statistics
    if let Some(ref dist) = results.distribution {
        writeln!(file, "Distribution_Skewness,{:.precision$},Distribution", dist.skewness, precision = precision)?;
        writeln!(file, "Distribution_Kurtosis,{:.precision$},Distribution", dist.kurtosis, precision = precision)?;

        if let Some(ref norm) = dist.normality_test {
            writeln!(file, "Normality_Test,{},Distribution", norm.test_name)?;
            writeln!(file, "Normality_Statistic,{:.precision$},Distribution", norm.statistic, precision = precision)?;
            writeln!(file, "Normality_P_Value,{:.precision$},Distribution", norm.p_value, precision = precision)?;
            writeln!(file, "Is_Normal,{},Distribution", norm.is_normal)?;
        }
    }

    // Stationarity tests
    for (test_name, test) in &results.stationarity_tests {
        writeln!(file, "{}_Statistic,{:.precision$},Stationarity", test_name, test.statistic, precision = precision)?;
        writeln!(file, "{}_P_Value,{:.precision$},Stationarity", test_name, test.p_value, precision = precision)?;
        writeln!(file, "{}_Is_Stationary,{},Stationarity", test_name, test.is_stationary)?;
    }

    // Change points
    for (i, cp) in results.changepoints.iter().enumerate() {
        writeln!(file, "ChangePoint_{}_Index,{},ChangePoint", i + 1, cp.index)?;
        writeln!(file, "ChangePoint_{}_Confidence,{:.precision$},ChangePoint", i + 1, cp.confidence, precision = precision)?;
        writeln!(file, "ChangePoint_{}_Type,{:?},ChangePoint", i + 1, cp.change_type)?;
    }

    // Time series statistics (if requested)
    if config.include_raw_values {
        if let Some(ref ts_stats) = results.timeseries_stats {
            // Export ACF values
            for (i, &acf_val) in ts_stats.acf.values.iter().enumerate().take(10) {
                writeln!(file, "ACF_Lag_{},{:.precision$},TimeSeries", i, acf_val, precision = precision)?;
            }

            // Export PACF values
            for (i, &pacf_val) in ts_stats.pacf.values.iter().enumerate().take(10) {
                writeln!(file, "PACF_Lag_{},{:.precision$},TimeSeries", i + 1, pacf_val, precision = precision)?;
            }
        }
    }

    Ok(())
}

/// Export results as text report
fn export_text_report(
    results: &StatisticalAnalysisResult,
    output_path: &Path,
    config: &ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;
    let precision = config.decimal_precision;

    // Use the built-in summary method and enhance it
    let basic_summary = results.summary();
    writeln!(file, "{}", basic_summary)?;

    // Add detailed sections if requested
    if config.include_metadata {
        writeln!(file, "\n=== Analysis Metadata ===")?;
        writeln!(file, "Analysis Duration: {} ms", results.metadata.duration_ms)?;
        writeln!(file, "Timestamp: {}", results.metadata.timestamp.format("%Y-%m-%d %H:%M:%S UTC"))?;
    }

    if let Some(ref desc) = results.descriptive {
        writeln!(file, "\n=== Detailed Descriptive Statistics ===")?;
        writeln!(file, "Coefficient of Variation: {:.precision$}", desc.coefficient_of_variation, precision = precision)?;

        writeln!(file, "\nPercentiles:")?;
        writeln!(file, "  1st percentile: {:.precision$}", desc.quantiles.p01, precision = precision)?;
        writeln!(file, "  5th percentile: {:.precision$}", desc.quantiles.p05, precision = precision)?;
        writeln!(file, " 10th percentile: {:.precision$}", desc.quantiles.p10, precision = precision)?;
        writeln!(file, " 90th percentile: {:.precision$}", desc.quantiles.p90, precision = precision)?;
        writeln!(file, " 95th percentile: {:.precision$}", desc.quantiles.p95, precision = precision)?;
        writeln!(file, " 99th percentile: {:.precision$}", desc.quantiles.p99, precision = precision)?;

        if let Some(mode) = desc.mode {
            writeln!(file, "Mode: {:.precision$}", mode, precision = precision)?;
        }
    }

    if let Some(ref dist) = results.distribution {
        writeln!(file, "\n=== Distribution Analysis ===")?;

        writeln!(file, "Histogram Bins: {}", dist.histogram.bin_centers.len())?;
        writeln!(file, "Bin Width: {:.precision$}", dist.histogram.bin_width, precision = precision)?;

        if config.include_raw_values {
            writeln!(file, "\nHistogram Data:")?;
            for (i, (&center, &count)) in dist.histogram.bin_centers.iter()
                .zip(dist.histogram.counts.iter()).enumerate() {
                writeln!(file, "  Bin {}: [{:.precision$}] = {} observations",
                    i + 1, center, count, precision = precision)?;
            }
        }
    }

    if config.include_confidence_intervals {
        if let Some(ref ts_stats) = results.timeseries_stats {
            writeln!(file, "\n=== Time Series Analysis ===")?;

            writeln!(file, "Autocorrelation Function (first 10 lags):")?;
            for (_i, (&lag, (&acf, &(lower, upper)))) in ts_stats.acf.lags.iter()
                .zip(ts_stats.acf.values.iter().zip(ts_stats.acf.confidence_intervals.iter()))
                .enumerate().take(10) {
                writeln!(file, "  Lag {}: {:.precision$} [CI: {:.precision$}, {:.precision$}]",
                    lag, acf, lower, upper, precision = precision)?;
            }

            writeln!(file, "\nPartial Autocorrelation Function (first 5 lags):")?;
            for (_i, (&lag, (&pacf, &(lower, upper)))) in ts_stats.pacf.lags.iter()
                .zip(ts_stats.pacf.values.iter().zip(ts_stats.pacf.confidence_intervals.iter()))
                .enumerate().take(5) {
                writeln!(file, "  Lag {}: {:.precision$} [CI: {:.precision$}, {:.precision$}]",
                    lag, pacf, lower, upper, precision = precision)?;
            }

            if let Some(ref ljung_box) = ts_stats.acf.ljung_box_test {
                writeln!(file, "\nLjung-Box Test:")?;
                writeln!(file, "  Test Statistic: {:.precision$}", ljung_box.statistic, precision = precision)?;
                writeln!(file, "  P-value: {:.precision$}", ljung_box.p_value, precision = precision)?;
                writeln!(file, "  Has Autocorrelation: {}", ljung_box.has_autocorrelation)?;
            }
        }
    }

    writeln!(file, "\n=== End of Report ===")?;
    writeln!(file, "Generated by Chronos Time Series Analysis Engine")?;

    Ok(())
}

/// Export results as Markdown
fn export_markdown(
    results: &StatisticalAnalysisResult,
    output_path: &Path,
    config: &ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;
    let precision = config.decimal_precision;

    writeln!(file, "# Statistical Analysis Report")?;
    writeln!(file)?;
    writeln!(file, "**Column:** {}", results.metadata.column_name)?;
    writeln!(file, "**Samples:** {} (Missing: {})", results.metadata.n_samples, results.metadata.n_missing)?;
    writeln!(file, "**Analysis Date:** {}", results.metadata.timestamp.format("%Y-%m-%d %H:%M:%S UTC"))?;
    writeln!(file)?;

    // Descriptive Statistics
    if let Some(ref desc) = results.descriptive {
        writeln!(file, "## Descriptive Statistics")?;
        writeln!(file)?;
        writeln!(file, "| Statistic | Value |")?;
        writeln!(file, "|-----------|-------|")?;
        writeln!(file, "| Count | {} |", desc.count)?;
        writeln!(file, "| Mean | {:.precision$} |", desc.mean, precision = precision)?;
        writeln!(file, "| Median | {:.precision$} |", desc.median, precision = precision)?;
        writeln!(file, "| Std Dev | {:.precision$} |", desc.std_dev, precision = precision)?;
        writeln!(file, "| Min | {:.precision$} |", desc.min, precision = precision)?;
        writeln!(file, "| Max | {:.precision$} |", desc.max, precision = precision)?;
        writeln!(file, "| Range | {:.precision$} |", desc.range, precision = precision)?;
        writeln!(file, "| Q25 | {:.precision$} |", desc.quantiles.q25, precision = precision)?;
        writeln!(file, "| Q75 | {:.precision$} |", desc.quantiles.q75, precision = precision)?;
        writeln!(file, "| IQR | {:.precision$} |", desc.quantiles.iqr, precision = precision)?;
        writeln!(file, "| Skewness | {:.precision$} |", desc.skewness, precision = precision)?;
        writeln!(file, "| Kurtosis | {:.precision$} |", desc.kurtosis, precision = precision)?;
        writeln!(file)?;
    }

    // Distribution Analysis
    if let Some(ref dist) = results.distribution {
        writeln!(file, "## Distribution Analysis")?;
        writeln!(file)?;
        writeln!(file, "**Skewness:** {:.precision$}", dist.skewness, precision = precision)?;
        writeln!(file, "**Kurtosis:** {:.precision$}", dist.kurtosis, precision = precision)?;
        writeln!(file)?;

        if let Some(ref norm) = dist.normality_test {
            writeln!(file, "### Normality Test ({})", norm.test_name)?;
            writeln!(file, "- **Test Statistic:** {:.precision$}", norm.statistic, precision = precision)?;
            writeln!(file, "- **P-value:** {:.precision$}", norm.p_value, precision = precision)?;
            writeln!(file, "- **Is Normal:** {}", if norm.is_normal { "✅ Yes" } else { "❌ No" })?;
            writeln!(file)?;
        }
    }

    // Stationarity Tests
    if !results.stationarity_tests.is_empty() {
        writeln!(file, "## Stationarity Tests")?;
        writeln!(file)?;
        writeln!(file, "| Test | Statistic | P-value | Is Stationary |")?;
        writeln!(file, "|------|-----------|---------|---------------|")?;

        for (name, test) in &results.stationarity_tests {
            writeln!(file, "| {} | {:.precision$} | {:.precision$} | {} |",
                name,
                test.statistic,
                test.p_value,
                if test.is_stationary { "✅ Yes" } else { "❌ No" },
                precision = precision
            )?;
        }
        writeln!(file)?;
    }

    // Change Points
    if !results.changepoints.is_empty() {
        writeln!(file, "## Change Points")?;
        writeln!(file)?;
        writeln!(file, "**Total Change Points Detected:** {}", results.changepoints.len())?;
        writeln!(file)?;

        writeln!(file, "| Index | Type | Confidence | Test Statistic |")?;
        writeln!(file, "|-------|------|------------|----------------|")?;

        for cp in &results.changepoints {
            writeln!(file, "| {} | {:?} | {:.1}% | {:.precision$} |",
                cp.index,
                cp.change_type,
                cp.confidence * 100.0,
                cp.test_statistic,
                precision = precision
            )?;
        }
        writeln!(file)?;
    }

    writeln!(file, "---")?;
    writeln!(file, "*Generated by Chronos Time Series Analysis Engine*")?;

    Ok(())
}

/// Export results as HTML
fn export_html(
    results: &StatisticalAnalysisResult,
    output_path: &Path,
    config: &ExportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;
    let precision = config.decimal_precision;

    writeln!(file, "<!DOCTYPE html>")?;
    writeln!(file, "<html lang='en'>")?;
    writeln!(file, "<head>")?;
    writeln!(file, "    <meta charset='UTF-8'>")?;
    writeln!(file, "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")?;
    writeln!(file, "    <title>Statistical Analysis Report - {}</title>", results.metadata.column_name)?;
    writeln!(file, "    <style>")?;
    writeln!(file, "        body {{ font-family: Arial, sans-serif; margin: 2rem; }}")?;
    writeln!(file, "        h1, h2, h3 {{ color: #2c3e50; }}")?;
    writeln!(file, "        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}")?;
    writeln!(file, "        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}")?;
    writeln!(file, "        th {{ background-color: #f2f2f2; }}")?;
    writeln!(file, "        .metadata {{ background-color: #f8f9fa; padding: 1rem; border-radius: 5px; }}")?;
    writeln!(file, "        .statistic {{ margin: 1rem 0; }}")?;
    writeln!(file, "        .positive {{ color: #27ae60; }}")?;
    writeln!(file, "        .negative {{ color: #e74c3c; }}")?;
    writeln!(file, "    </style>")?;
    writeln!(file, "</head>")?;
    writeln!(file, "<body>")?;

    writeln!(file, "    <h1>Statistical Analysis Report</h1>")?;

    writeln!(file, "    <div class='metadata'>")?;
    writeln!(file, "        <h3>Analysis Metadata</h3>")?;
    writeln!(file, "        <p><strong>Column:</strong> {}</p>", results.metadata.column_name)?;
    writeln!(file, "        <p><strong>Samples:</strong> {} (Missing: {})</p>",
        results.metadata.n_samples, results.metadata.n_missing)?;
    writeln!(file, "        <p><strong>Analysis Date:</strong> {}</p>",
        results.metadata.timestamp.format("%Y-%m-%d %H:%M:%S UTC"))?;
    writeln!(file, "        <p><strong>Duration:</strong> {} ms</p>", results.metadata.duration_ms)?;
    writeln!(file, "    </div>")?;

    // Descriptive Statistics
    if let Some(ref desc) = results.descriptive {
        writeln!(file, "    <h2>Descriptive Statistics</h2>")?;
        writeln!(file, "    <table>")?;
        writeln!(file, "        <thead><tr><th>Statistic</th><th>Value</th></tr></thead>")?;
        writeln!(file, "        <tbody>")?;
        writeln!(file, "            <tr><td>Count</td><td>{}</td></tr>", desc.count)?;
        writeln!(file, "            <tr><td>Mean</td><td>{:.precision$}</td></tr>", desc.mean, precision = precision)?;
        writeln!(file, "            <tr><td>Median</td><td>{:.precision$}</td></tr>", desc.median, precision = precision)?;
        writeln!(file, "            <tr><td>Standard Deviation</td><td>{:.precision$}</td></tr>", desc.std_dev, precision = precision)?;
        writeln!(file, "            <tr><td>Minimum</td><td>{:.precision$}</td></tr>", desc.min, precision = precision)?;
        writeln!(file, "            <tr><td>Maximum</td><td>{:.precision$}</td></tr>", desc.max, precision = precision)?;
        writeln!(file, "            <tr><td>Range</td><td>{:.precision$}</td></tr>", desc.range, precision = precision)?;
        writeln!(file, "            <tr><td>25th Percentile</td><td>{:.precision$}</td></tr>", desc.quantiles.q25, precision = precision)?;
        writeln!(file, "            <tr><td>75th Percentile</td><td>{:.precision$}</td></tr>", desc.quantiles.q75, precision = precision)?;
        writeln!(file, "            <tr><td>IQR</td><td>{:.precision$}</td></tr>", desc.quantiles.iqr, precision = precision)?;
        writeln!(file, "            <tr><td>Skewness</td><td>{:.precision$}</td></tr>", desc.skewness, precision = precision)?;
        writeln!(file, "            <tr><td>Kurtosis</td><td>{:.precision$}</td></tr>", desc.kurtosis, precision = precision)?;
        writeln!(file, "        </tbody>")?;
        writeln!(file, "    </table>")?;
    }

    // Distribution Analysis
    if let Some(ref dist) = results.distribution {
        writeln!(file, "    <h2>Distribution Analysis</h2>")?;

        if let Some(ref norm) = dist.normality_test {
            writeln!(file, "    <h3>{} Test</h3>", norm.test_name)?;
            writeln!(file, "    <p><strong>Test Statistic:</strong> {:.precision$}</p>", norm.statistic, precision = precision)?;
            writeln!(file, "    <p><strong>P-value:</strong> {:.precision$}</p>", norm.p_value, precision = precision)?;
            writeln!(file, "    <p><strong>Is Normal:</strong> <span class='{}'>{}</span></p>",
                if norm.is_normal { "positive" } else { "negative" },
                if norm.is_normal { "Yes ✅" } else { "No ❌" }
            )?;
        }
    }

    // Stationarity Tests
    if !results.stationarity_tests.is_empty() {
        writeln!(file, "    <h2>Stationarity Tests</h2>")?;
        writeln!(file, "    <table>")?;
        writeln!(file, "        <thead><tr><th>Test</th><th>Statistic</th><th>P-value</th><th>Is Stationary</th></tr></thead>")?;
        writeln!(file, "        <tbody>")?;

        for (name, test) in &results.stationarity_tests {
            writeln!(file, "            <tr>")?;
            writeln!(file, "                <td>{}</td>", name)?;
            writeln!(file, "                <td>{:.precision$}</td>", test.statistic, precision = precision)?;
            writeln!(file, "                <td>{:.precision$}</td>", test.p_value, precision = precision)?;
            writeln!(file, "                <td><span class='{}'>{}</span></td>",
                if test.is_stationary { "positive" } else { "negative" },
                if test.is_stationary { "Yes ✅" } else { "No ❌" }
            )?;
            writeln!(file, "            </tr>")?;
        }
        writeln!(file, "        </tbody>")?;
        writeln!(file, "    </table>")?;
    }

    // Change Points
    if !results.changepoints.is_empty() {
        writeln!(file, "    <h2>Change Points</h2>")?;
        writeln!(file, "    <p><strong>Total Change Points Detected:</strong> {}</p>", results.changepoints.len())?;

        writeln!(file, "    <table>")?;
        writeln!(file, "        <thead><tr><th>Index</th><th>Type</th><th>Confidence</th><th>Test Statistic</th></tr></thead>")?;
        writeln!(file, "        <tbody>")?;

        for cp in &results.changepoints {
            writeln!(file, "            <tr>")?;
            writeln!(file, "                <td>{}</td>", cp.index)?;
            writeln!(file, "                <td>{:?}</td>", cp.change_type)?;
            writeln!(file, "                <td>{:.1}%</td>", cp.confidence * 100.0)?;
            writeln!(file, "                <td>{:.precision$}</td>", cp.test_statistic, precision = precision)?;
            writeln!(file, "            </tr>")?;
        }
        writeln!(file, "        </tbody>")?;
        writeln!(file, "    </table>")?;
    }

    writeln!(file, "    <hr>")?;
    writeln!(file, "    <footer>")?;
    writeln!(file, "        <p><em>Generated by Chronos Time Series Analysis Engine</em></p>")?;
    writeln!(file, "    </footer>")?;
    writeln!(file, "</body>")?;
    writeln!(file, "</html>")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use crate::stats::{StatisticalAnalysisResult, AnalysisMetadata};

    fn create_sample_results() -> StatisticalAnalysisResult {
        let mut result = StatisticalAnalysisResult::new("test_column".to_string(), 100);
        result.metadata = AnalysisMetadata {
            n_samples: 100,
            n_missing: 5,
            timestamp: chrono::Utc::now(),
            column_name: "test_column".to_string(),
            duration_ms: 150,
        };
        result
    }

    #[test]
    fn test_json_export() {
        let results = create_sample_results();
        let temp_file = NamedTempFile::new().unwrap();

        let result = export_json(&results, temp_file.path(), &ExportConfig::default());
        assert!(result.is_ok());

        // Verify file was created and contains JSON
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("test_column"));
        assert!(content.contains("n_samples"));
    }

    #[test]
    fn test_csv_export() {
        let results = create_sample_results();
        let temp_file = NamedTempFile::new().unwrap();

        let result = export_csv(&results, temp_file.path(), &ExportConfig::default());
        assert!(result.is_ok());

        // Verify CSV format
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("Statistic,Value,Category"));
        assert!(content.contains("Column,test_column,Metadata"));
    }

    #[test]
    fn test_text_report_export() {
        let results = create_sample_results();
        let temp_file = NamedTempFile::new().unwrap();

        let result = export_text_report(&results, temp_file.path(), &ExportConfig::default());
        assert!(result.is_ok());

        // Verify text format
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("Statistical Analysis Report"));
        assert!(content.contains("test_column"));
    }

    #[test]
    fn test_markdown_export() {
        let results = create_sample_results();
        let temp_file = NamedTempFile::new().unwrap();

        let result = export_markdown(&results, temp_file.path(), &ExportConfig::default());
        assert!(result.is_ok());

        // Verify markdown format
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("# Statistical Analysis Report"));
        assert!(content.contains("**Column:**"));
    }

    #[test]
    fn test_html_export() {
        let results = create_sample_results();
        let temp_file = NamedTempFile::new().unwrap();

        let result = export_html(&results, temp_file.path(), &ExportConfig::default());
        assert!(result.is_ok());

        // Verify HTML format
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("<title>"));
        assert!(content.contains("test_column"));
    }

    #[test]
    fn test_export_config() {
        let config = ExportConfig {
            decimal_precision: 2,
            include_metadata: false,
            include_confidence_intervals: false,
            include_raw_values: true,
            include_plot_data: false,
        };

        let results = create_sample_results();
        let temp_file = NamedTempFile::new().unwrap();

        let result = export_csv(&results, temp_file.path(), &config);
        assert!(result.is_ok());
    }
}