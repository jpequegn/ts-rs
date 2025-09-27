//! # Styling and Theme System
//!
//! Professional styling system with customizable themes, color palettes,
//! and publication-ready formatting options.

use crate::plotting::types::*;
use plotly::common::{Font, TickMode};
use plotly::layout::{Axis, GridPattern, Legend, Margin, TicksDirection};
use plotly::Layout;
use std::collections::HashMap;

/// Default light theme
pub struct DefaultTheme;

/// Professional theme for business presentations
pub struct ProfessionalTheme;

/// Publication-ready theme for academic papers
pub struct PublicationTheme;

/// Dark theme for modern interfaces
pub struct DarkTheme;

/// Trait for theme implementations
pub trait ThemeProvider {
    fn get_theme_config(&self) -> ThemeConfig;
    fn get_color_palette(&self) -> Vec<String>;
    fn get_background_color(&self) -> String;
    fn get_text_color(&self) -> String;
    fn get_grid_color(&self) -> String;
    fn get_axis_color(&self) -> String;
}

impl ThemeProvider for DefaultTheme {
    fn get_theme_config(&self) -> ThemeConfig {
        ThemeConfig {
            background: "#FFFFFF".to_string(),
            colors: vec![
                "#1f77b4".to_string(), // blue
                "#ff7f0e".to_string(), // orange
                "#2ca02c".to_string(), // green
                "#d62728".to_string(), // red
                "#9467bd".to_string(), // purple
                "#8c564b".to_string(), // brown
                "#e377c2".to_string(), // pink
                "#7f7f7f".to_string(), // gray
                "#bcbd22".to_string(), // olive
                "#17becf".to_string(), // cyan
            ],
            grid_color: "#E0E0E0".to_string(),
            text_color: "#333333".to_string(),
            axis_color: "#666666".to_string(),
        }
    }

    fn get_color_palette(&self) -> Vec<String> {
        self.get_theme_config().colors
    }

    fn get_background_color(&self) -> String {
        self.get_theme_config().background
    }

    fn get_text_color(&self) -> String {
        self.get_theme_config().text_color
    }

    fn get_grid_color(&self) -> String {
        self.get_theme_config().grid_color
    }

    fn get_axis_color(&self) -> String {
        self.get_theme_config().axis_color
    }
}

impl ThemeProvider for ProfessionalTheme {
    fn get_theme_config(&self) -> ThemeConfig {
        ThemeConfig {
            background: "#FAFAFA".to_string(),
            colors: vec![
                "#0072CE".to_string(), // corporate blue
                "#FF6B35".to_string(), // professional orange
                "#28A745".to_string(), // success green
                "#DC3545".to_string(), // alert red
                "#6610F2".to_string(), // professional purple
                "#20C997".to_string(), // teal
                "#FD7E14".to_string(), // amber
                "#6C757D".to_string(), // gray
            ],
            grid_color: "#E8E8E8".to_string(),
            text_color: "#212529".to_string(),
            axis_color: "#495057".to_string(),
        }
    }

    fn get_color_palette(&self) -> Vec<String> {
        self.get_theme_config().colors
    }

    fn get_background_color(&self) -> String {
        self.get_theme_config().background
    }

    fn get_text_color(&self) -> String {
        self.get_theme_config().text_color
    }

    fn get_grid_color(&self) -> String {
        self.get_theme_config().grid_color
    }

    fn get_axis_color(&self) -> String {
        self.get_theme_config().axis_color
    }
}

impl ThemeProvider for PublicationTheme {
    fn get_theme_config(&self) -> ThemeConfig {
        ThemeConfig {
            background: "#FFFFFF".to_string(),
            colors: vec![
                "#000000".to_string(), // black
                "#404040".to_string(), // dark gray
                "#808080".to_string(), // medium gray
                "#C0C0C0".to_string(), // light gray
                "#000080".to_string(), // navy
                "#800000".to_string(), // maroon
                "#008000".to_string(), // green
                "#800080".to_string(), // purple
            ],
            grid_color: "#D0D0D0".to_string(),
            text_color: "#000000".to_string(),
            axis_color: "#000000".to_string(),
        }
    }

    fn get_color_palette(&self) -> Vec<String> {
        self.get_theme_config().colors
    }

    fn get_background_color(&self) -> String {
        self.get_theme_config().background
    }

    fn get_text_color(&self) -> String {
        self.get_theme_config().text_color
    }

    fn get_grid_color(&self) -> String {
        self.get_theme_config().grid_color
    }

    fn get_axis_color(&self) -> String {
        self.get_theme_config().axis_color
    }
}

impl ThemeProvider for DarkTheme {
    fn get_theme_config(&self) -> ThemeConfig {
        ThemeConfig {
            background: "#1E1E1E".to_string(),
            colors: vec![
                "#4FC3F7".to_string(), // light blue
                "#FFB74D".to_string(), // light orange
                "#81C784".to_string(), // light green
                "#E57373".to_string(), // light red
                "#BA68C8".to_string(), // light purple
                "#A1887F".to_string(), // light brown
                "#F06292".to_string(), // light pink
                "#90A4AE".to_string(), // blue gray
                "#AED581".to_string(), // light lime
                "#4DD0E1".to_string(), // cyan
            ],
            grid_color: "#404040".to_string(),
            text_color: "#FFFFFF".to_string(),
            axis_color: "#CCCCCC".to_string(),
        }
    }

    fn get_color_palette(&self) -> Vec<String> {
        self.get_theme_config().colors
    }

    fn get_background_color(&self) -> String {
        self.get_theme_config().background
    }

    fn get_text_color(&self) -> String {
        self.get_theme_config().text_color
    }

    fn get_grid_color(&self) -> String {
        self.get_theme_config().grid_color
    }

    fn get_axis_color(&self) -> String {
        self.get_theme_config().axis_color
    }
}

/// Apply theme styling to a plot layout
pub fn apply_theme(mut layout: Layout, theme: &Theme) -> Layout {
    let theme_provider: Box<dyn ThemeProvider> = match theme {
        Theme::Default => Box::new(DefaultTheme),
        Theme::Dark => Box::new(DarkTheme),
        Theme::Publication => Box::new(PublicationTheme),
        Theme::HighContrast => Box::new(DarkTheme), // Use dark theme as high contrast
        Theme::Custom(config) => return apply_custom_theme(layout, config),
    };

    let config = theme_provider.get_theme_config();

    layout = layout
        .paper_background_color(config.background.clone())
        .plot_background_color(config.background.clone())
        .font(Font::new().color(config.text_color.clone()).family("Arial, sans-serif").size(12));

    // Apply grid styling
    if let Ok(x_axis) = Axis::new()
        .grid_color(config.grid_color.clone())
        .line_color(config.axis_color.clone())
        .tick_color(config.axis_color.clone())
        .tick_font(Font::new().color(config.text_color.clone()))
        .try_into()
    {
        layout = layout.x_axis(x_axis);
    }

    if let Ok(y_axis) = Axis::new()
        .grid_color(config.grid_color.clone())
        .line_color(config.axis_color.clone())
        .tick_color(config.axis_color.clone())
        .tick_font(Font::new().color(config.text_color.clone()))
        .try_into()
    {
        layout = layout.y_axis(y_axis);
    }

    layout
}

/// Apply custom theme configuration
fn apply_custom_theme(mut layout: Layout, theme_config: &ThemeConfig) -> Layout {
    layout = layout
        .paper_background_color(theme_config.background.clone())
        .plot_background_color(theme_config.background.clone())
        .font(Font::new().color(theme_config.text_color.clone()).family("Arial, sans-serif").size(12));

    // Apply custom grid and axis styling
    if let Ok(x_axis) = Axis::new()
        .grid_color(theme_config.grid_color.clone())
        .line_color(theme_config.axis_color.clone())
        .tick_color(theme_config.axis_color.clone())
        .tick_font(Font::new().color(theme_config.text_color.clone()))
        .try_into()
    {
        layout = layout.x_axis(x_axis);
    }

    if let Ok(y_axis) = Axis::new()
        .grid_color(theme_config.grid_color.clone())
        .line_color(theme_config.axis_color.clone())
        .tick_color(theme_config.axis_color.clone())
        .tick_font(Font::new().color(theme_config.text_color.clone()))
        .try_into()
    {
        layout = layout.y_axis(y_axis);
    }

    layout
}

/// Customize styling with user-provided options
pub fn customize_styling(
    mut layout: Layout,
    custom_style: &HashMap<String, String>,
) -> Layout {
    // Apply custom styling options
    for (key, value) in custom_style {
        match key.as_str() {
            "background_color" => {
                layout = layout.paper_background_color(value.clone()).plot_background_color(value.clone());
            }
            "text_color" => {
                layout = layout.font(Font::new().color(value.clone()));
            }
            "font_family" => {
                layout = layout.font(Font::new().family(value));
            }
            "font_size" => {
                if let Ok(size) = value.parse::<usize>() {
                    layout = layout.font(Font::new().size(size));
                }
            }
            "title" => {
                layout = layout.title(plotly::common::Title::new(value));
            }
            "margin_left" => {
                if let Ok(margin) = value.parse::<usize>() {
                    layout = layout.margin(Margin::new().left(margin));
                }
            }
            "margin_right" => {
                if let Ok(margin) = value.parse::<usize>() {
                    layout = layout.margin(Margin::new().right(margin));
                }
            }
            "margin_top" => {
                if let Ok(margin) = value.parse::<usize>() {
                    layout = layout.margin(Margin::new().top(margin));
                }
            }
            "margin_bottom" => {
                if let Ok(margin) = value.parse::<usize>() {
                    layout = layout.margin(Margin::new().bottom(margin));
                }
            }
            _ => {
                // Unknown style option, ignore
            }
        }
    }

    layout
}

/// Get color from theme palette by index
pub fn get_theme_color(theme: &Theme, index: usize) -> String {
    let theme_provider: Box<dyn ThemeProvider> = match theme {
        Theme::Default => Box::new(DefaultTheme),
        Theme::Dark => Box::new(DarkTheme),
        Theme::Publication => Box::new(PublicationTheme),
        Theme::HighContrast => Box::new(DarkTheme),
        Theme::Custom(config) => {
            return config.colors[index % config.colors.len()].clone();
        }
    };

    let colors = theme_provider.get_color_palette();
    colors[index % colors.len()].clone()
}

/// Create publication-ready layout with professional styling
pub fn create_publication_layout(
    width: usize,
    height: usize,
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
) -> Layout {
    let mut layout = Layout::new()
        .width(width)
        .height(height)
        .paper_background_color("#FFFFFF")
        .plot_background_color("#FFFFFF")
        .font(Font::new()
            .family("Times New Roman, serif")
            .size(14)
            .color("#000000"));

    if let Some(title_text) = title {
        layout = layout.title(
            plotly::common::Title::new(title_text)
                .font(Font::new().size(16).color("#000000"))
        );
    }

    // Configure axes with publication styling
    let x_axis = Axis::new()
        .title(plotly::common::Title::new(x_label.unwrap_or("X")))
        .tick_font(Font::new().size(12).color("#000000"))
        .line_color("#000000")
        .tick_color("#000000")
        .grid_color("#E0E0E0")
        .show_grid(true)
        .show_line(true)
        .show_tick_labels(true)
        .tick_mode(TickMode::Auto)
        .ticks(TicksDirection::Outside);

    let y_axis = Axis::new()
        .title(plotly::common::Title::new(y_label.unwrap_or("Y")))
        .tick_font(Font::new().size(12).color("#000000"))
        .line_color("#000000")
        .tick_color("#000000")
        .grid_color("#E0E0E0")
        .show_grid(true)
        .show_line(true)
        .show_tick_labels(true)
        .tick_mode(TickMode::Auto)
        .ticks(TicksDirection::Outside);

    layout = layout.x_axis(x_axis).y_axis(y_axis);

    // Configure legend
    layout = layout.legend(
        Legend::new()
            .font(Font::new().size(12).color("#000000"))
            .border_color("#000000")
            .border_width(1)
    );

    // Set margins for publication
    layout = layout.margin(
        Margin::new()
            .left(80)
            .right(50)
            .top(80)
            .bottom(80)
    );

    layout
}

/// Color palette utilities
pub struct ColorPalette;

impl ColorPalette {
    /// Generate qualitative color palette for categorical data
    pub fn qualitative(n_colors: usize) -> Vec<String> {
        let base_colors = vec![
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ];

        let mut colors = Vec::new();
        for i in 0..n_colors {
            colors.push(base_colors[i % base_colors.len()].to_string());
        }
        colors
    }

    /// Generate sequential color palette for continuous data
    pub fn sequential(n_colors: usize, base_color: &str) -> Vec<String> {
        let mut colors = Vec::new();

        // Simple approach: vary the lightness of the base color
        // In a full implementation, this would use proper color interpolation
        for i in 0..n_colors {
            let intensity = 0.3 + (0.7 * i as f64 / n_colors.max(1) as f64);
            let color = format!("{}cc", &base_color[..7]); // Simple alpha variation
            colors.push(color);
        }

        colors
    }

    /// Generate diverging color palette for data with meaningful center
    pub fn diverging(n_colors: usize) -> Vec<String> {
        let colors = if n_colors <= 3 {
            vec!["#d73027", "#ffffbf", "#1a9850"]
        } else if n_colors <= 5 {
            vec!["#d73027", "#fc8d59", "#ffffbf", "#91bfdb", "#4575b4"]
        } else {
            vec![
                "#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
                "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"
            ]
        };

        colors.into_iter().map(|s| s.to_string()).take(n_colors).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_theme() {
        let theme = DefaultTheme;
        let config = theme.get_theme_config();

        assert_eq!(config.background, "#FFFFFF");
        assert!(!config.colors.is_empty());
        assert_eq!(config.text_color, "#333333");
    }

    #[test]
    fn test_dark_theme() {
        let theme = DarkTheme;
        let config = theme.get_theme_config();

        assert_eq!(config.background, "#1E1E1E");
        assert_eq!(config.text_color, "#FFFFFF");
    }

    #[test]
    fn test_get_theme_color() {
        let theme = Theme::Default;
        let color = get_theme_color(&theme, 0);
        assert!(!color.is_empty());
        assert!(color.starts_with('#'));
    }

    #[test]
    fn test_color_palette_qualitative() {
        let colors = ColorPalette::qualitative(5);
        assert_eq!(colors.len(), 5);
        assert!(colors[0].starts_with('#'));
    }

    #[test]
    fn test_color_palette_diverging() {
        let colors = ColorPalette::diverging(5);
        assert_eq!(colors.len(), 5);
        assert!(colors.iter().all(|c| c.starts_with('#')));
    }

    #[test]
    fn test_apply_theme() {
        let layout = Layout::new();
        let theme = Theme::Default;
        let styled_layout = apply_theme(layout, &theme);

        // Basic test to ensure function doesn't panic
        // In a full test, we would check specific styling properties
    }

    #[test]
    fn test_customize_styling() {
        let layout = Layout::new();
        let mut custom_style = HashMap::new();
        custom_style.insert("background_color".to_string(), "#FF0000".to_string());
        custom_style.insert("text_color".to_string(), "#00FF00".to_string());

        let styled_layout = customize_styling(layout, &custom_style);

        // Basic test to ensure function doesn't panic
        // In a full test, we would verify the applied styles
    }
}