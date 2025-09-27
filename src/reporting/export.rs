//! # Report Export Functions
//!
//! Functions for exporting reports to various formats including Markdown, HTML, PDF, and JSON.

use crate::reporting::types::{ReportContent, ReportResult, ReportExportFormat};
use crate::Result;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use serde_json;

/// Export a report to a file
pub fn export_report(
    report: &ReportResult,
    output_path: &str,
    format: Option<ReportExportFormat>,
) -> Result<u64> {
    let export_format = format.unwrap_or_else(|| {
        // Determine format from file extension
        let path = Path::new(output_path);
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("md") => ReportExportFormat::Markdown,
            Some("html") => ReportExportFormat::HTML,
            Some("pdf") => ReportExportFormat::PDF,
            Some("json") => ReportExportFormat::JSON,
            Some("txt") => ReportExportFormat::Text,
            _ => ReportExportFormat::HTML, // Default to HTML
        }
    });

    match export_format {
        ReportExportFormat::Markdown => {
            if let ReportContent::Markdown(content) = &report.content {
                export_to_markdown(content, output_path)
            } else {
                Err("Report content is not in Markdown format".into())
            }
        },
        ReportExportFormat::HTML => {
            if let ReportContent::HTML(content) = &report.content {
                export_to_html(content, output_path)
            } else {
                Err("Report content is not in HTML format".into())
            }
        },
        ReportExportFormat::PDF => export_to_pdf(&report.content, output_path),
        ReportExportFormat::JSON => {
            if let ReportContent::JSON(content) = &report.content {
                export_to_json(content, output_path)
            } else {
                Err("Report content is not in JSON format".into())
            }
        },
        ReportExportFormat::Text => {
            if let ReportContent::Text(content) = &report.content {
                export_to_text(content, output_path)
            } else {
                Err("Report content is not in Text format".into())
            }
        },
    }
}

/// Export content to Markdown file
pub fn export_to_markdown(content: &str, output_path: &str) -> Result<u64> {
    // Ensure the directory exists
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    // Write the markdown content
    let mut file = File::create(output_path)?;
    file.write_all(content.as_bytes())?;

    Ok(content.len() as u64)
}

/// Export content to HTML file
pub fn export_to_html(content: &str, output_path: &str) -> Result<u64> {
    // Ensure the directory exists
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    // Write the HTML content
    let mut file = File::create(output_path)?;
    file.write_all(content.as_bytes())?;

    Ok(content.len() as u64)
}

/// Export content to PDF file
pub fn export_to_pdf(content: &ReportContent, output_path: &str) -> Result<u64> {
    // Ensure the directory exists
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    match content {
        ReportContent::PDF(pdf_data) => {
            let mut file = File::create(output_path)?;
            file.write_all(pdf_data)?;
            Ok(pdf_data.len() as u64)
        },
        ReportContent::HTML(html_content) => {
            // Convert HTML to PDF using a simple text-based approach
            // In a production system, this would use a proper HTML-to-PDF library
            let pdf_content = create_simple_pdf_from_html(html_content);
            let mut file = File::create(output_path)?;
            file.write_all(&pdf_content)?;
            Ok(pdf_content.len() as u64)
        },
        ReportContent::Markdown(md_content) => {
            // Convert Markdown to simple PDF
            let pdf_content = create_simple_pdf_from_text(md_content);
            let mut file = File::create(output_path)?;
            file.write_all(&pdf_content)?;
            Ok(pdf_content.len() as u64)
        },
        ReportContent::Text(text_content) => {
            let pdf_content = create_simple_pdf_from_text(text_content);
            let mut file = File::create(output_path)?;
            file.write_all(&pdf_content)?;
            Ok(pdf_content.len() as u64)
        },
        _ => Err("Unsupported content format for PDF export".into()),
    }
}

/// Export content to JSON file
pub fn export_to_json(content: &serde_json::Value, output_path: &str) -> Result<u64> {
    // Ensure the directory exists
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    // Write the JSON content with pretty formatting
    let json_string = serde_json::to_string_pretty(content)?;
    let mut file = File::create(output_path)?;
    file.write_all(json_string.as_bytes())?;

    Ok(json_string.len() as u64)
}

/// Export content to plain text file
pub fn export_to_text(content: &str, output_path: &str) -> Result<u64> {
    // Ensure the directory exists
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    // Write the text content
    let mut file = File::create(output_path)?;
    file.write_all(content.as_bytes())?;

    Ok(content.len() as u64)
}

/// Create a simple PDF from HTML content
/// Note: This is a simplified implementation. In production, use a proper HTML-to-PDF library
fn create_simple_pdf_from_html(html_content: &str) -> Vec<u8> {
    // Strip HTML tags and create a simple text-based PDF
    let text_content = strip_html_tags(html_content);
    create_simple_pdf_from_text(&text_content)
}

/// Create a simple PDF from plain text
/// Note: This creates a minimal PDF structure. For production use, consider using proper PDF libraries
fn create_simple_pdf_from_text(text_content: &str) -> Vec<u8> {
    // This is a very basic PDF structure - in production, use a proper PDF library like printpdf
    let pdf_header = b"%PDF-1.4\n";
    let catalog = b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n\n";
    let pages = b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n\n";

    // Escape special characters in text
    let escaped_text = text_content
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", "\\n");

    let content_length = escaped_text.len() + 50; // Approximate content length
    let page = format!(
        "3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n/Resources <<\n/Font <<\n/F1 <<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\n>>\n>>\n>>\nendobj\n\n"
    );

    let content_obj = format!(
        "4 0 obj\n<<\n/Length {}\n>>\nstream\nBT\n/F1 12 Tf\n50 750 Td\n15 TL\n({})*\nET\nendstream\nendobj\n\n",
        content_length,
        escaped_text.chars().take(1000).collect::<String>() // Limit content for simple PDF
    );

    let xref_offset = pdf_header.len() + catalog.len() + pages.len() + page.len() + content_obj.len();
    let xref = format!(
        "xref\n0 5\n0000000000 65535 f \n{:010} 00000 n \n{:010} 00000 n \n{:010} 00000 n \n{:010} 00000 n \n",
        pdf_header.len(),
        pdf_header.len() + catalog.len(),
        pdf_header.len() + catalog.len() + pages.len(),
        pdf_header.len() + catalog.len() + pages.len() + page.len()
    );

    let trailer = format!(
        "trailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n{}\n%%EOF\n",
        xref_offset
    );

    let mut pdf_content = Vec::new();
    pdf_content.extend_from_slice(pdf_header);
    pdf_content.extend_from_slice(catalog);
    pdf_content.extend_from_slice(pages);
    pdf_content.extend_from_slice(page.as_bytes());
    pdf_content.extend_from_slice(content_obj.as_bytes());
    pdf_content.extend_from_slice(xref.as_bytes());
    pdf_content.extend_from_slice(trailer.as_bytes());

    pdf_content
}

/// Strip HTML tags from content (simple implementation)
fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;

    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '<' {
            // Check for script or style tags
            if i + 6 < chars.len() && chars[i..i+7].iter().collect::<String>().to_lowercase() == "<script" {
                in_script = true;
            } else if i + 5 < chars.len() && chars[i..i+6].iter().collect::<String>().to_lowercase() == "<style" {
                in_style = true;
            } else if i + 8 < chars.len() && chars[i..i+9].iter().collect::<String>().to_lowercase() == "</script>" {
                in_script = false;
                i += 8; // Skip the closing tag
                continue;
            } else if i + 7 < chars.len() && chars[i..i+8].iter().collect::<String>().to_lowercase() == "</style>" {
                in_style = false;
                i += 7; // Skip the closing tag
                continue;
            }
            in_tag = true;
        } else if chars[i] == '>' && in_tag {
            in_tag = false;
            i += 1;
            continue;
        }

        if !in_tag && !in_script && !in_style {
            if chars[i] == '\n' || chars[i] == '\r' {
                result.push(' ');
            } else {
                result.push(chars[i]);
            }
        }

        i += 1;
    }

    // Clean up multiple spaces
    result.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Batch export multiple reports
pub fn batch_export_reports(
    reports: &[ReportResult],
    output_directory: &str,
    format: ReportExportFormat,
) -> Result<Vec<String>> {
    // Ensure output directory exists
    fs::create_dir_all(output_directory)?;

    let mut exported_files = Vec::new();

    for (i, report) in reports.iter().enumerate() {
        let filename = generate_report_filename(i, &format, &report.metadata.title);
        let output_path = Path::new(output_directory).join(&filename);
        let output_path_str = output_path.to_string_lossy().to_string();

        export_report(report, &output_path_str, Some(format.clone()))?;
        exported_files.push(output_path_str);
    }

    Ok(exported_files)
}

/// Generate a filename for a report
fn generate_report_filename(index: usize, format: &ReportExportFormat, title: &str) -> String {
    // Sanitize title for filename
    let sanitized_title = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>()
        .replace(' ', "_");

    let extension = match format {
        ReportExportFormat::Markdown => "md",
        ReportExportFormat::HTML => "html",
        ReportExportFormat::PDF => "pdf",
        ReportExportFormat::JSON => "json",
        ReportExportFormat::Text => "txt",
    };

    format!("{}_{:03}.{}", sanitized_title, index + 1, extension)
}

/// Get file extension for export format
pub fn get_file_extension(format: &ReportExportFormat) -> &'static str {
    match format {
        ReportExportFormat::Markdown => "md",
        ReportExportFormat::HTML => "html",
        ReportExportFormat::PDF => "pdf",
        ReportExportFormat::JSON => "json",
        ReportExportFormat::Text => "txt",
    }
}

/// Determine export format from file path
pub fn determine_format_from_path(path: &str) -> ReportExportFormat {
    let path = Path::new(path);
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("md") => ReportExportFormat::Markdown,
        Some("html") | Some("htm") => ReportExportFormat::HTML,
        Some("pdf") => ReportExportFormat::PDF,
        Some("json") => ReportExportFormat::JSON,
        Some("txt") => ReportExportFormat::Text,
        _ => ReportExportFormat::HTML, // Default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::reporting::types::{ReportResult, ReportContent, ReportMetadata, AnalysisData, DataSummary};

    fn create_test_report() -> ReportResult {
        ReportResult {
            content: ReportContent::HTML("<html><body><h1>Test Report</h1></body></html>".to_string()),
            analysis_data: AnalysisData {
                statistics: None,
                trend: None,
                seasonality: None,
                anomalies: None,
                forecasting: None,
                correlation: None,
                data_quality: None,
                plots: Vec::new(),
                data_summary: DataSummary {
                    n_points: 100,
                    date_range: None,
                    frequency: None,
                    columns: vec!["test".to_string()],
                    missing_percentage: 0.0,
                },
            },
            insights: Vec::new(),
            metadata: ReportMetadata::default(),
            export_info: None,
        }
    }

    #[test]
    fn test_export_to_html() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_report.html");

        let html_content = "<html><body><h1>Test</h1></body></html>";
        let result = export_to_html(html_content, output_path.to_str().unwrap());

        assert!(result.is_ok());
        assert!(output_path.exists());

        let written_content = std::fs::read_to_string(&output_path).unwrap();
        assert_eq!(written_content, html_content);
    }

    #[test]
    fn test_export_to_markdown() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_report.md");

        let markdown_content = "# Test Report\n\nThis is a test.";
        let result = export_to_markdown(markdown_content, output_path.to_str().unwrap());

        assert!(result.is_ok());
        assert!(output_path.exists());

        let written_content = std::fs::read_to_string(&output_path).unwrap();
        assert_eq!(written_content, markdown_content);
    }

    #[test]
    fn test_export_to_json() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_report.json");

        let json_content = serde_json::json!({
            "title": "Test Report",
            "data": [1, 2, 3]
        });

        let result = export_to_json(&json_content, output_path.to_str().unwrap());

        assert!(result.is_ok());
        assert!(output_path.exists());

        let written_content = std::fs::read_to_string(&output_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&written_content).unwrap();
        assert_eq!(parsed, json_content);
    }

    #[test]
    fn test_strip_html_tags() {
        let html = "<html><head><title>Test</title></head><body><h1>Header</h1><p>Paragraph</p></body></html>";
        let result = strip_html_tags(html);
        assert_eq!(result, "Test Header Paragraph");
    }

    #[test]
    fn test_generate_report_filename() {
        let filename = generate_report_filename(0, &ReportExportFormat::HTML, "Test Report");
        assert_eq!(filename, "Test_Report_001.html");

        let filename = generate_report_filename(5, &ReportExportFormat::PDF, "Executive Summary!");
        assert_eq!(filename, "Executive_Summary__006.pdf");
    }

    #[test]
    fn test_determine_format_from_path() {
        assert_eq!(determine_format_from_path("report.html"), ReportExportFormat::HTML);
        assert_eq!(determine_format_from_path("report.md"), ReportExportFormat::Markdown);
        assert_eq!(determine_format_from_path("report.pdf"), ReportExportFormat::PDF);
        assert_eq!(determine_format_from_path("report.json"), ReportExportFormat::JSON);
        assert_eq!(determine_format_from_path("report.txt"), ReportExportFormat::Text);
        assert_eq!(determine_format_from_path("report.xyz"), ReportExportFormat::HTML); // Default
    }

    #[test]
    fn test_batch_export() {
        let temp_dir = TempDir::new().unwrap();
        let reports = vec![create_test_report(), create_test_report()];

        let result = batch_export_reports(
            &reports,
            temp_dir.path().to_str().unwrap(),
            ReportExportFormat::HTML
        );

        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 2);

        for file in files {
            assert!(Path::new(&file).exists());
        }
    }
}