//! # Export Functions
//!
//! Functions for exporting plots to various formats including PNG, SVG, PDF, HTML, and JSON.

use crate::plotting::types::*;
use chrono::Utc;
use serde_json;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Export options for customizing output
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Output directory
    pub output_dir: Option<PathBuf>,

    /// Custom filename (without extension)
    pub filename: Option<String>,

    /// Image quality for raster formats (0-100)
    pub quality: Option<u32>,

    /// DPI for image exports
    pub dpi: Option<u32>,

    /// Whether to open the exported file after creation
    pub open_after_export: bool,

    /// Custom metadata to include
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            output_dir: None,
            filename: None,
            quality: Some(95),
            dpi: Some(300),
            open_after_export: false,
            metadata: None,
        }
    }
}

/// Export a plot to a file
pub fn export_to_file(
    plot_result: &PlotResult,
    format: ExportFormat,
    options: ExportOptions,
) -> Result<ExportInfo, Box<dyn std::error::Error>> {
    let output_dir = options.output_dir.clone().unwrap_or_else(|| PathBuf::from("."));

    // Ensure output directory exists
    fs::create_dir_all(&output_dir)?;

    // Generate filename
    let filename = generate_filename(&options.filename, &format);
    let file_path = output_dir.join(&filename);

    let file_size = match format {
        ExportFormat::HTML => export_html_file(plot_result, &file_path)?,
        ExportFormat::JSON => export_json_file(plot_result, &file_path)?,
        ExportFormat::PNG => export_png_file(plot_result, &file_path, &options)?,
        ExportFormat::SVG => export_svg_file(plot_result, &file_path, &options)?,
        ExportFormat::PDF => export_pdf_file(plot_result, &file_path, &options)?,
        ExportFormat::Display => return Err("Display format cannot be exported to file".into()),
    };

    let export_info = ExportInfo {
        format,
        file_path: Some(file_path.to_string_lossy().to_string()),
        file_size: Some(file_size),
        exported_at: Utc::now(),
    };

    // Open file if requested
    if options.open_after_export {
        open_file(&file_path)?;
    }

    Ok(export_info)
}

/// Export plot as HTML
pub fn export_to_html(
    plot_result: &PlotResult,
    file_path: Option<&Path>,
) -> Result<String, Box<dyn std::error::Error>> {
    match &plot_result.content {
        PlotContent::HTML(html) => {
            if let Some(path) = file_path {
                let mut file = File::create(path)?;
                file.write_all(html.as_bytes())?;
                Ok(format!("HTML exported to: {}", path.display()))
            } else {
                Ok(html.clone())
            }
        },
        _ => Err("Plot content is not HTML format".into()),
    }
}

/// Export plot as PNG image
pub fn export_to_png(
    plot_result: &PlotResult,
    file_path: &Path,
    options: &ExportOptions,
) -> Result<u64, Box<dyn std::error::Error>> {
    // For now, return placeholder implementation
    // In a full implementation, this would use plotly's image export capabilities
    // or convert HTML to PNG using a headless browser

    let placeholder_content = create_png_placeholder(&plot_result.metadata);
    let mut file = File::create(file_path)?;
    file.write_all(&placeholder_content)?;

    Ok(placeholder_content.len() as u64)
}

/// Export plot as SVG vector graphics
pub fn export_to_svg(
    plot_result: &PlotResult,
    file_path: &Path,
    options: &ExportOptions,
) -> Result<u64, Box<dyn std::error::Error>> {
    // Placeholder implementation
    // In a full implementation, this would extract SVG from plotly or convert HTML

    let svg_content = create_svg_placeholder(&plot_result.metadata);
    let mut file = File::create(file_path)?;
    file.write_all(svg_content.as_bytes())?;

    Ok(svg_content.len() as u64)
}

/// Export plot as PDF
pub fn export_to_pdf(
    plot_result: &PlotResult,
    file_path: &Path,
    options: &ExportOptions,
) -> Result<u64, Box<dyn std::error::Error>> {
    // Placeholder implementation
    // In a full implementation, this would use a PDF generation library
    // or convert HTML to PDF using a headless browser

    let pdf_content = create_pdf_placeholder(&plot_result.metadata);
    let mut file = File::create(file_path)?;
    file.write_all(&pdf_content)?;

    Ok(pdf_content.len() as u64)
}

/// Export HTML to file
fn export_html_file(
    plot_result: &PlotResult,
    file_path: &PathBuf,
) -> Result<u64, Box<dyn std::error::Error>> {
    match &plot_result.content {
        PlotContent::HTML(html) => {
            let enhanced_html = enhance_html_for_export(html, &plot_result.metadata);
            let mut file = File::create(file_path)?;
            file.write_all(enhanced_html.as_bytes())?;
            Ok(enhanced_html.len() as u64)
        },
        _ => Err("Plot content is not HTML format".into()),
    }
}

/// Export JSON to file
fn export_json_file(
    plot_result: &PlotResult,
    file_path: &PathBuf,
) -> Result<u64, Box<dyn std::error::Error>> {
    let json_content = match &plot_result.content {
        PlotContent::JSON(json) => serde_json::to_string_pretty(json)?,
        _ => {
            // Convert other formats to JSON representation
            serde_json::to_string_pretty(plot_result)?
        }
    };

    let mut file = File::create(file_path)?;
    file.write_all(json_content.as_bytes())?;
    Ok(json_content.len() as u64)
}

/// Export PNG to file
fn export_png_file(
    plot_result: &PlotResult,
    file_path: &PathBuf,
    options: &ExportOptions,
) -> Result<u64, Box<dyn std::error::Error>> {
    export_to_png(plot_result, file_path, options)
}

/// Export SVG to file
fn export_svg_file(
    plot_result: &PlotResult,
    file_path: &PathBuf,
    options: &ExportOptions,
) -> Result<u64, Box<dyn std::error::Error>> {
    export_to_svg(plot_result, file_path, options)
}

/// Export PDF to file
fn export_pdf_file(
    plot_result: &PlotResult,
    file_path: &PathBuf,
    options: &ExportOptions,
) -> Result<u64, Box<dyn std::error::Error>> {
    export_to_pdf(plot_result, file_path, options)
}

/// Generate filename based on options and format
fn generate_filename(custom_name: &Option<String>, format: &ExportFormat) -> String {
    let base_name = custom_name
        .as_ref()
        .cloned()
        .unwrap_or_else(|| format!("plot_{}", Utc::now().format("%Y%m%d_%H%M%S")));

    let extension = match format {
        ExportFormat::HTML => "html",
        ExportFormat::PNG => "png",
        ExportFormat::SVG => "svg",
        ExportFormat::PDF => "pdf",
        ExportFormat::JSON => "json",
        ExportFormat::Display => return base_name,
    };

    format!("{}.{}", base_name, extension)
}

/// Enhance HTML with metadata and styling for export
fn enhance_html_for_export(html: &str, metadata: &PlotMetadata) -> String {
    let header = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Time Series Plot - {}</title>
    <meta name="generator" content="ts-rs plotting system">
    <meta name="created" content="{}">
    <meta name="plot-type" content="{:?}">
    <meta name="data-points" content="{}">
    <meta name="series-count" content="{}">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ text-align: center; }}
        .metadata {{ margin-top: 20px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="plot-container">
"#,
        metadata.plot_type.to_string(),
        metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC"),
        metadata.plot_type,
        metadata.data_points,
        metadata.series_count
    );

    let footer = format!(
        r#"
    </div>
    <div class="metadata">
        <p>Generated by ts-rs plotting system</p>
        <p>Plot type: {:?} | Data points: {} | Series: {} | Created: {}</p>
    </div>
</body>
</html>"#,
        metadata.plot_type,
        metadata.data_points,
        metadata.series_count,
        metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC")
    );

    // Insert the original HTML content between header and footer
    if html.contains("<html>") {
        // HTML is already complete, just add metadata
        html.replace("</body>", &format!("{}</body>", footer.split("</body>").next().unwrap_or("")))
    } else {
        // HTML is just the plot div, wrap it completely
        format!("{}{}{}", header, html, footer)
    }
}

/// Create PNG placeholder content
fn create_png_placeholder(metadata: &PlotMetadata) -> Vec<u8> {
    // This is a minimal PNG header - in a real implementation,
    // you would use plotly's image export or a proper image library
    let mut png_data = Vec::new();

    // PNG signature
    png_data.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    // Add some placeholder data
    let info = format!("Plot: {:?}, {} data points", metadata.plot_type, metadata.data_points);
    png_data.extend_from_slice(info.as_bytes());

    png_data
}

/// Create SVG placeholder content
fn create_svg_placeholder(metadata: &PlotMetadata) -> String {
    let stroke_color = "#dee2e6";
    let fill_color_666 = "#666666";
    let fill_color_999 = "#999999";

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#f8f9fa" stroke="{}"/>
    <text x="50%" y="50%" text-anchor="middle" font-family="Arial" font-size="16">
        Plot: {:?}
    </text>
    <text x="50%" y="65%" text-anchor="middle" font-family="Arial" font-size="12" fill="{}">
        {} data points, {} series
    </text>
    <text x="50%" y="80%" text-anchor="middle" font-family="Arial" font-size="10" fill="{}">
        Generated by ts-rs plotting system
    </text>
</svg>"#,
        metadata.dimensions.0,
        metadata.dimensions.1,
        stroke_color,
        metadata.plot_type,
        fill_color_666,
        metadata.data_points,
        metadata.series_count,
        fill_color_999
    )
}

/// Create PDF placeholder content
fn create_pdf_placeholder(metadata: &PlotMetadata) -> Vec<u8> {
    // This is a minimal PDF structure - in a real implementation,
    // you would use a proper PDF library like printpdf
    let pdf_content = format!(
        "%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 100
>>
stream
BT
/F1 12 Tf
50 750 Td
(Plot: {:?}) Tj
0 -20 Td
({} data points, {} series) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000079 00000 n
0000000136 00000 n
0000000215 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
365
%%EOF",
        metadata.plot_type,
        metadata.data_points,
        metadata.series_count
    );

    pdf_content.into_bytes()
}

/// Open file with system default application
fn open_file(file_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(file_path)
            .spawn()?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(&["/C", "start", file_path.to_str().unwrap_or("")])
            .spawn()?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(file_path)
            .spawn()?;
    }

    Ok(())
}

/// Batch export multiple plots
pub fn batch_export(
    plots: &[PlotResult],
    formats: &[ExportFormat],
    options: ExportOptions,
) -> Result<Vec<ExportInfo>, Box<dyn std::error::Error>> {
    let mut export_results = Vec::new();

    for (i, plot) in plots.iter().enumerate() {
        for format in formats {
            let mut plot_options = options.clone();

            // Add index to filename for batch exports
            if let Some(ref base_name) = options.filename {
                plot_options.filename = Some(format!("{}_{}", base_name, i + 1));
            } else {
                plot_options.filename = Some(format!("plot_{}", i + 1));
            }

            let export_info = export_to_file(plot, format.clone(), plot_options)?;
            export_results.push(export_info);
        }
    }

    Ok(export_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_test_plot_result() -> PlotResult {
        PlotResult {
            content: PlotContent::HTML("<div>Test plot</div>".to_string()),
            metadata: PlotMetadata {
                plot_type: PlotType::Line,
                created_at: Utc::now(),
                data_points: 100,
                series_count: 2,
                dimensions: (800, 600),
                theme: Theme::Default,
            },
            export_info: None,
        }
    }

    #[test]
    fn test_generate_filename() {
        let filename = generate_filename(&Some("test_plot".to_string()), &ExportFormat::HTML);
        assert_eq!(filename, "test_plot.html");

        let filename = generate_filename(&None, &ExportFormat::PNG);
        assert!(filename.ends_with(".png"));
    }

    #[test]
    fn test_export_options_default() {
        let options = ExportOptions::default();
        assert_eq!(options.quality, Some(95));
        assert_eq!(options.dpi, Some(300));
        assert!(!options.open_after_export);
    }

    #[test]
    fn test_export_to_html() {
        let plot_result = create_test_plot_result();
        let result = export_to_html(&plot_result, None);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Test plot"));
    }

    #[test]
    fn test_enhance_html_for_export() {
        let plot_result = create_test_plot_result();
        let enhanced = enhance_html_for_export("<div>Test</div>", &plot_result.metadata);
        assert!(enhanced.contains("<!DOCTYPE html>"));
        assert!(enhanced.contains("Test"));
        assert!(enhanced.contains("ts-rs plotting system"));
    }

    #[test]
    fn test_create_svg_placeholder() {
        let plot_result = create_test_plot_result();
        let svg = create_svg_placeholder(&plot_result.metadata);
        assert!(svg.contains("<?xml version"));
        assert!(svg.contains("Plot: Line"));
        assert!(svg.contains("100 data points"));
    }

    #[test]
    fn test_export_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let plot_result = create_test_plot_result();

        let options = ExportOptions {
            output_dir: Some(temp_dir.path().to_path_buf()),
            filename: Some("test_export".to_string()),
            ..ExportOptions::default()
        };

        let export_info = export_to_file(&plot_result, ExportFormat::HTML, options).unwrap();

        assert!(export_info.file_path.is_some());
        assert!(export_info.file_size.is_some());

        let file_path = temp_dir.path().join("test_export.html");
        assert!(file_path.exists());
    }
}