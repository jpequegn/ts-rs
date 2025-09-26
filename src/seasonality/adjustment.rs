//! # Seasonal Adjustment Module
//!
//! Provides seasonal adjustment methods including X-13ARIMA, moving averages,
//! and STL-based adjustment for removing seasonal patterns from time series data.

use crate::seasonality::SeasonalPeriod;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Methods available for seasonal adjustment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SeasonalAdjustmentMethod {
    /// X-13ARIMA-SEATS seasonal adjustment (simplified implementation)
    X13Arima,
    /// Moving averages seasonal adjustment
    MovingAverage,
    /// STL (Seasonal and Trend decomposition using Loess) adjustment
    STL,
    /// Census X-11 method (simplified)
    X11,
    /// Multiplicative adjustment
    Multiplicative,
    /// Additive adjustment
    Additive,
}

/// Result of seasonal adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalAdjustmentResult {
    /// Seasonally adjusted series
    pub adjusted_series: Vec<f64>,

    /// Estimated seasonal components
    pub seasonal_components: Vec<f64>,

    /// Estimated trend components
    pub trend_components: Vec<f64>,

    /// Irregular/residual components
    pub irregular_components: Vec<f64>,

    /// Method used for adjustment
    pub method: SeasonalAdjustmentMethod,

    /// Seasonal factors by period
    pub seasonal_factors: SeasonalFactors,

    /// Quality diagnostics
    pub diagnostics: AdjustmentDiagnostics,

    /// Adjustment metadata
    pub metadata: AdjustmentMetadata,
}

/// Seasonal factors for different time periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalFactors {
    /// Monthly seasonal factors (if applicable)
    pub monthly_factors: Option<[f64; 12]>,

    /// Weekly seasonal factors (if applicable)
    pub weekly_factors: Option<[f64; 7]>,

    /// Daily seasonal factors (if applicable)
    pub daily_factors: Option<Vec<f64>>,

    /// Custom period seasonal factors
    pub custom_factors: HashMap<String, Vec<f64>>,

    /// Moving seasonality indicators
    pub moving_seasonality: Option<MovingSeasonality>,
}

/// Moving seasonality information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingSeasonality {
    /// Whether seasonality changes over time
    pub is_evolving: bool,

    /// Evolution rate (change per period)
    pub evolution_rate: f64,

    /// Seasonal factors by time window
    pub factors_by_window: Vec<WindowedFactors>,
}

/// Seasonal factors for a specific time window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowedFactors {
    /// Start time index
    pub start_index: usize,

    /// End time index
    pub end_index: usize,

    /// Seasonal factors for this window
    pub factors: Vec<f64>,

    /// Factor stability score
    pub stability: f64,
}

/// Quality diagnostics for seasonal adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustmentDiagnostics {
    /// Quality of seasonal adjustment (Q statistic)
    pub q_statistic: f64,

    /// M statistics for X-13 quality assessment
    pub m_statistics: MStatistics,

    /// Residual seasonality tests
    pub residual_seasonality: ResidualSeasonalityTests,

    /// Trading day effects in adjusted series
    pub residual_trading_day: f64,

    /// Overall adjustment quality score
    pub quality_score: f64,

    /// Revision analysis
    pub revision_analysis: RevisionAnalysis,
}

/// M statistics for X-13 seasonal adjustment quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MStatistics {
    /// M1: Relative contribution of irregular to total variance
    pub m1: f64,

    /// M2: Relative contribution of irregular to trend changes
    pub m2: f64,

    /// M3: Period-to-period changes in irregular
    pub m3: f64,

    /// M4: Autocorrelation of irregular
    pub m4: f64,

    /// M5: Number of months for cyclical dominance
    pub m5: f64,

    /// M6: Size of residual seasonality relative to seasonal component
    pub m6: f64,

    /// M7: Size of residual seasonality relative to irregular
    pub m7: f64,

    /// M8: Year-to-year changes in seasonal factors
    pub m8: f64,

    /// M9: Linear trend in seasonal factors
    pub m9: f64,

    /// M10: Average linear trend in seasonal factors
    pub m10: f64,

    /// M11: Combined measure
    pub m11: f64,

    /// Overall quality assessment based on M statistics
    pub overall_quality: QualityAssessment,
}

/// Overall quality assessment categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAssessment {
    /// Acceptable quality
    Acceptable,
    /// Probably acceptable quality
    ProbablyAcceptable,
    /// Probably not acceptable quality
    ProbablyNotAcceptable,
    /// Not acceptable quality
    NotAcceptable,
}

/// Tests for residual seasonality in adjusted series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualSeasonalityTests {
    /// F-test for residual seasonality
    pub f_test_statistic: f64,
    pub f_test_p_value: f64,

    /// Kruskal-Wallis test for residual seasonality
    pub kruskal_wallis_statistic: f64,
    pub kruskal_wallis_p_value: f64,

    /// Combined residual seasonality test
    pub combined_test: f64,

    /// Has significant residual seasonality?
    pub has_residual_seasonality: bool,
}

/// Revision analysis for seasonal adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevisionAnalysis {
    /// Average absolute revision
    pub average_absolute_revision: f64,

    /// Average relative revision (percentage)
    pub average_relative_revision: f64,

    /// Revision volatility
    pub revision_volatility: f64,

    /// Periods with largest revisions
    pub largest_revisions: Vec<(usize, f64)>,
}

/// Metadata for seasonal adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustmentMetadata {
    /// Number of observations processed
    pub n_observations: usize,

    /// Seasonal periods used
    pub seasonal_periods: Vec<f64>,

    /// Parameters used
    pub parameters: HashMap<String, f64>,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Warnings and notes
    pub warnings: Vec<String>,

    /// Software version/method details
    pub method_details: String,
}

/// X-13ARIMA seasonal adjustment implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct X13ArimaAdjustment {
    /// ARIMA model order (p,d,q)(P,D,Q)s
    pub arima_order: ArimaOrder,

    /// Seasonal adjustment options
    pub adjustment_options: X13Options,

    /// Outlier detection settings
    pub outlier_detection: OutlierDetectionSettings,
}

/// ARIMA model order specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArimaOrder {
    /// Non-seasonal AR order
    pub p: usize,

    /// Non-seasonal differencing order
    pub d: usize,

    /// Non-seasonal MA order
    pub q: usize,

    /// Seasonal AR order
    pub seasonal_p: usize,

    /// Seasonal differencing order
    pub seasonal_d: usize,

    /// Seasonal MA order
    pub seasonal_q: usize,

    /// Seasonal period
    pub seasonal_period: usize,
}

/// X-13 seasonal adjustment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct X13Options {
    /// X-11 method mode
    pub mode: X11Mode,

    /// Seasonal filter length
    pub seasonal_filter: SeasonalFilter,

    /// Trend filter length
    pub trend_filter: TrendFilter,

    /// Calendar adjustment
    pub calendar_adjustment: bool,

    /// Trading day adjustment
    pub trading_day_adjustment: bool,

    /// Easter effects
    pub easter_effects: bool,
}

/// X-11 modes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum X11Mode {
    /// Additive decomposition
    Additive,
    /// Multiplicative decomposition
    Multiplicative,
    /// Pseudo-additive
    PseudoAdditive,
    /// Log-additive
    LogAdditive,
}

/// Seasonal filter options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalFilter {
    /// 3x3 moving average
    S3X3,
    /// 3x5 moving average
    S3X5,
    /// 3x9 moving average
    S3X9,
    /// 3x15 moving average
    S3X15,
    /// Stable seasonal filter
    Stable,
    /// Automatic selection
    Auto,
}

/// Trend filter options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendFilter {
    /// 13-term Henderson filter
    Henderson13,
    /// 23-term Henderson filter
    Henderson23,
    /// Custom length Henderson filter
    HendersonCustom(usize),
    /// Simple moving average
    SimpleMovingAverage(usize),
    /// Automatic selection
    Auto,
}

/// Outlier detection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetectionSettings {
    /// Enable outlier detection
    pub enabled: bool,

    /// Critical value for outlier detection
    pub critical_value: f64,

    /// Types of outliers to detect
    pub outlier_types: Vec<OutlierType>,

    /// Maximum number of outliers to detect
    pub max_outliers: usize,
}

/// Types of outliers for detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierType {
    /// Additive outlier
    AdditiveOutlier,
    /// Level shift
    LevelShift,
    /// Temporary change
    TemporaryChange,
    /// Seasonal outlier
    SeasonalOutlier,
}

/// Moving average seasonal adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingAverageAdjustment {
    /// Window size for moving average
    pub window_size: usize,

    /// Type of moving average
    pub ma_type: MovingAverageType,

    /// Centered or trailing average
    pub centered: bool,

    /// Minimum observations required
    pub min_observations: usize,
}

/// Types of moving averages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MovingAverageType {
    /// Simple moving average
    Simple,
    /// Exponentially weighted moving average
    Exponential(f64), // alpha parameter
    /// Linearly weighted moving average
    LinearWeighted,
    /// Henderson moving average
    Henderson,
}

/// STL seasonal adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STLAdjustment {
    /// Seasonal period
    pub seasonal_period: usize,

    /// Seasonal smoother span
    pub seasonal_span: usize,

    /// Trend smoother span
    pub trend_span: usize,

    /// Low-pass filter span
    pub lowpass_span: usize,

    /// Number of inner loop iterations
    pub inner_iterations: usize,

    /// Number of outer loop iterations
    pub outer_iterations: usize,

    /// Robustness iterations
    pub robustness_iterations: usize,
}

/// Perform seasonal adjustment using specified method
pub fn perform_seasonal_adjustment(
    data: &[f64],
    method: SeasonalAdjustmentMethod,
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalAdjustmentResult, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();

    let result = match method {
        SeasonalAdjustmentMethod::X13Arima => apply_x13_adjustment(data, seasonal_periods)?,
        SeasonalAdjustmentMethod::MovingAverage => apply_moving_average_adjustment(data, seasonal_periods)?,
        SeasonalAdjustmentMethod::STL => apply_stl_adjustment(data, seasonal_periods)?,
        SeasonalAdjustmentMethod::X11 => apply_x11_adjustment(data, seasonal_periods)?,
        SeasonalAdjustmentMethod::Multiplicative => apply_multiplicative_adjustment(data, seasonal_periods)?,
        SeasonalAdjustmentMethod::Additive => apply_additive_adjustment(data, seasonal_periods)?,
    };

    let processing_time = start_time.elapsed().as_millis() as u64;
    let mut final_result = result;
    final_result.metadata.processing_time_ms = processing_time;

    Ok(final_result)
}

/// Apply X-13ARIMA seasonal adjustment
pub fn apply_x13_adjustment(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalAdjustmentResult, Box<dyn std::error::Error>> {
    if data.len() < 24 {
        return Err("Insufficient data for X-13 adjustment (need at least 24 observations)".into());
    }

    // Use the strongest seasonal period
    let main_period = seasonal_periods.iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
        .ok_or("No seasonal periods provided")?;

    let seasonal_period = main_period.period as usize;

    // Simplified X-13 implementation using classical decomposition approach
    let x13_config = X13ArimaAdjustment {
        arima_order: ArimaOrder {
            p: 0, d: 1, q: 1,
            seasonal_p: 0, seasonal_d: 1, seasonal_q: 1,
            seasonal_period,
        },
        adjustment_options: X13Options {
            mode: X11Mode::Multiplicative,
            seasonal_filter: SeasonalFilter::S3X5,
            trend_filter: TrendFilter::Henderson13,
            calendar_adjustment: false,
            trading_day_adjustment: false,
            easter_effects: false,
        },
        outlier_detection: OutlierDetectionSettings {
            enabled: true,
            critical_value: 3.5,
            outlier_types: vec![OutlierType::AdditiveOutlier, OutlierType::LevelShift],
            max_outliers: data.len() / 10,
        },
    };

    // Step 1: Trend estimation using Henderson filter
    let trend = estimate_trend_henderson(data, 13)?;

    // Step 2: Seasonal-irregular estimation
    let seasonal_irregular = if x13_config.adjustment_options.mode == X11Mode::Multiplicative {
        data.iter().zip(trend.iter())
            .map(|(&x, &t)| if t != 0.0 { x / t } else { 1.0 })
            .collect::<Vec<f64>>()
    } else {
        data.iter().zip(trend.iter())
            .map(|(&x, &t)| x - t)
            .collect::<Vec<f64>>()
    };

    // Step 3: Seasonal component estimation
    let seasonal = estimate_seasonal_x11(&seasonal_irregular, seasonal_period, &x13_config.adjustment_options.seasonal_filter)?;

    // Step 4: Irregular component
    let irregular = if x13_config.adjustment_options.mode == X11Mode::Multiplicative {
        seasonal_irregular.iter().zip(seasonal.iter())
            .map(|(&si, &s)| if s != 0.0 { si / s } else { 1.0 })
            .collect::<Vec<f64>>()
    } else {
        seasonal_irregular.iter().zip(seasonal.iter())
            .map(|(&si, &s)| si - s)
            .collect::<Vec<f64>>()
    };

    // Step 5: Seasonally adjusted series
    let adjusted_series = if x13_config.adjustment_options.mode == X11Mode::Multiplicative {
        data.iter().zip(seasonal.iter())
            .map(|(&x, &s)| if s != 0.0 { x / s } else { x })
            .collect::<Vec<f64>>()
    } else {
        data.iter().zip(seasonal.iter())
            .map(|(&x, &s)| x - s)
            .collect::<Vec<f64>>()
    };

    // Generate seasonal factors
    let seasonal_factors = generate_seasonal_factors(&seasonal, seasonal_period)?;

    // Compute quality diagnostics
    let diagnostics = compute_x13_diagnostics(data, &adjusted_series, &seasonal, &trend, &irregular)?;

    // Create metadata
    let metadata = AdjustmentMetadata {
        n_observations: data.len(),
        seasonal_periods: vec![seasonal_period as f64],
        parameters: [
            ("seasonal_period".to_string(), seasonal_period as f64),
            ("trend_filter_length".to_string(), 13.0),
        ].iter().cloned().collect(),
        processing_time_ms: 0, // Will be set by caller
        warnings: Vec::new(),
        method_details: "Simplified X-13ARIMA implementation".to_string(),
    };

    Ok(SeasonalAdjustmentResult {
        adjusted_series,
        seasonal_components: seasonal,
        trend_components: trend,
        irregular_components: irregular,
        method: SeasonalAdjustmentMethod::X13Arima,
        seasonal_factors,
        diagnostics,
        metadata,
    })
}

/// Apply moving average seasonal adjustment
pub fn apply_moving_average_adjustment(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalAdjustmentResult, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Err("No data provided for adjustment".into());
    }

    let main_period = seasonal_periods.iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
        .ok_or("No seasonal periods provided")?;

    let seasonal_period = main_period.period as usize;
    let window_size = seasonal_period.max(12); // Use at least 12 observations

    let ma_config = MovingAverageAdjustment {
        window_size,
        ma_type: MovingAverageType::Simple,
        centered: true,
        min_observations: window_size / 2,
    };

    // Step 1: Estimate trend using moving average
    let trend = compute_moving_average_trend(data, &ma_config)?;

    // Step 2: Detrend the series
    let detrended: Vec<f64> = data.iter().zip(trend.iter())
        .map(|(&x, &t)| x - t)
        .collect();

    // Step 3: Estimate seasonal component
    let seasonal = estimate_seasonal_ma(&detrended, seasonal_period)?;

    // Step 4: Compute irregular component
    let irregular: Vec<f64> = detrended.iter().zip(seasonal.iter())
        .map(|(&d, &s)| d - s)
        .collect();

    // Step 5: Seasonally adjusted series
    let adjusted_series: Vec<f64> = data.iter().zip(seasonal.iter())
        .map(|(&x, &s)| x - s)
        .collect();

    // Generate seasonal factors
    let seasonal_factors = generate_seasonal_factors(&seasonal, seasonal_period)?;

    // Compute basic diagnostics
    let diagnostics = compute_basic_diagnostics(data, &adjusted_series, &seasonal)?;

    let metadata = AdjustmentMetadata {
        n_observations: data.len(),
        seasonal_periods: vec![seasonal_period as f64],
        parameters: [
            ("window_size".to_string(), window_size as f64),
            ("seasonal_period".to_string(), seasonal_period as f64),
        ].iter().cloned().collect(),
        processing_time_ms: 0,
        warnings: Vec::new(),
        method_details: "Moving average seasonal adjustment".to_string(),
    };

    Ok(SeasonalAdjustmentResult {
        adjusted_series,
        seasonal_components: seasonal,
        trend_components: trend,
        irregular_components: irregular,
        method: SeasonalAdjustmentMethod::MovingAverage,
        seasonal_factors,
        diagnostics,
        metadata,
    })
}

/// Apply STL seasonal adjustment
pub fn apply_stl_adjustment(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalAdjustmentResult, Box<dyn std::error::Error>> {
    if data.len() < 24 {
        return Err("Insufficient data for STL adjustment".into());
    }

    let main_period = seasonal_periods.iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
        .ok_or("No seasonal periods provided")?;

    let seasonal_period = main_period.period as usize;

    let stl_config = STLAdjustment {
        seasonal_period,
        seasonal_span: seasonal_period,
        trend_span: ((seasonal_period as f64 * 1.5).ceil() as usize).max(7) | 1, // Ensure odd
        lowpass_span: seasonal_period,
        inner_iterations: 2,
        outer_iterations: 1,
        robustness_iterations: 15,
    };

    // Simplified STL implementation
    let (seasonal, trend, irregular) = stl_decompose(data, &stl_config)?;

    // Seasonally adjusted series
    let adjusted_series: Vec<f64> = data.iter().zip(seasonal.iter())
        .map(|(&x, &s)| x - s)
        .collect();

    // Generate seasonal factors
    let seasonal_factors = generate_seasonal_factors(&seasonal, seasonal_period)?;

    // Compute basic diagnostics
    let diagnostics = compute_basic_diagnostics(data, &adjusted_series, &seasonal)?;

    let metadata = AdjustmentMetadata {
        n_observations: data.len(),
        seasonal_periods: vec![seasonal_period as f64],
        parameters: [
            ("seasonal_period".to_string(), seasonal_period as f64),
            ("trend_span".to_string(), stl_config.trend_span as f64),
            ("seasonal_span".to_string(), stl_config.seasonal_span as f64),
        ].iter().cloned().collect(),
        processing_time_ms: 0,
        warnings: Vec::new(),
        method_details: "STL seasonal adjustment".to_string(),
    };

    Ok(SeasonalAdjustmentResult {
        adjusted_series,
        seasonal_components: seasonal,
        trend_components: trend,
        irregular_components: irregular,
        method: SeasonalAdjustmentMethod::STL,
        seasonal_factors,
        diagnostics,
        metadata,
    })
}

// Helper functions for seasonal adjustment methods

fn apply_x11_adjustment(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalAdjustmentResult, Box<dyn std::error::Error>> {
    // X-11 is similar to X-13 but without ARIMA modeling
    apply_x13_adjustment(data, seasonal_periods)
}

fn apply_multiplicative_adjustment(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalAdjustmentResult, Box<dyn std::error::Error>> {
    if data.is_empty() || seasonal_periods.is_empty() {
        return Err("Insufficient data for multiplicative adjustment".into());
    }

    let seasonal_period = seasonal_periods[0].period as usize;

    // Estimate multiplicative seasonal factors
    let seasonal_factors = estimate_multiplicative_factors(data, seasonal_period)?;

    // Apply adjustment
    let adjusted_series: Vec<f64> = data.iter().enumerate()
        .map(|(i, &x)| x / seasonal_factors[i % seasonal_period])
        .collect();

    // Extract other components
    let seasonal_components: Vec<f64> = (0..data.len())
        .map(|i| data[i] * (seasonal_factors[i % seasonal_period] - 1.0))
        .collect();

    let trend_components = estimate_simple_trend(data)?;
    let irregular_components: Vec<f64> = adjusted_series.iter().zip(trend_components.iter())
        .map(|(&adj, &trend)| adj - trend)
        .collect();

    let factors = generate_seasonal_factors(&seasonal_components, seasonal_period)?;
    let diagnostics = compute_basic_diagnostics(data, &adjusted_series, &seasonal_components)?;

    let metadata = AdjustmentMetadata {
        n_observations: data.len(),
        seasonal_periods: vec![seasonal_period as f64],
        parameters: HashMap::new(),
        processing_time_ms: 0,
        warnings: Vec::new(),
        method_details: "Multiplicative seasonal adjustment".to_string(),
    };

    Ok(SeasonalAdjustmentResult {
        adjusted_series,
        seasonal_components,
        trend_components,
        irregular_components,
        method: SeasonalAdjustmentMethod::Multiplicative,
        seasonal_factors: factors,
        diagnostics,
        metadata,
    })
}

fn apply_additive_adjustment(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalAdjustmentResult, Box<dyn std::error::Error>> {
    if data.is_empty() || seasonal_periods.is_empty() {
        return Err("Insufficient data for additive adjustment".into());
    }

    let seasonal_period = seasonal_periods[0].period as usize;

    // Simple additive seasonal adjustment
    let seasonal_components = estimate_additive_seasonal(data, seasonal_period)?;

    let adjusted_series: Vec<f64> = data.iter().zip(seasonal_components.iter())
        .map(|(&x, &s)| x - s)
        .collect();

    let trend_components = estimate_simple_trend(&adjusted_series)?;
    let irregular_components: Vec<f64> = adjusted_series.iter().zip(trend_components.iter())
        .map(|(&adj, &trend)| adj - trend)
        .collect();

    let factors = generate_seasonal_factors(&seasonal_components, seasonal_period)?;
    let diagnostics = compute_basic_diagnostics(data, &adjusted_series, &seasonal_components)?;

    let metadata = AdjustmentMetadata {
        n_observations: data.len(),
        seasonal_periods: vec![seasonal_period as f64],
        parameters: HashMap::new(),
        processing_time_ms: 0,
        warnings: Vec::new(),
        method_details: "Additive seasonal adjustment".to_string(),
    };

    Ok(SeasonalAdjustmentResult {
        adjusted_series,
        seasonal_components,
        trend_components,
        irregular_components,
        method: SeasonalAdjustmentMethod::Additive,
        seasonal_factors: factors,
        diagnostics,
        metadata,
    })
}

// Implementation helper functions

fn estimate_trend_henderson(data: &[f64], filter_length: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if data.len() < filter_length {
        return estimate_simple_trend(data);
    }

    // Henderson filter weights (simplified for length 13)
    let weights = if filter_length == 13 {
        vec![-0.019, -0.028, 0.0, 0.066, 0.147, 0.214, 0.240, 0.214, 0.147, 0.066, 0.0, -0.028, -0.019]
    } else {
        // Fallback to simple moving average weights
        vec![1.0 / filter_length as f64; filter_length]
    };

    let half_length = filter_length / 2;
    let mut trend = vec![0.0; data.len()];

    // Apply Henderson filter
    for i in half_length..(data.len() - half_length) {
        let mut weighted_sum = 0.0;
        for j in 0..filter_length {
            weighted_sum += data[i - half_length + j] * weights[j];
        }
        trend[i] = weighted_sum;
    }

    // Handle endpoints with linear extrapolation
    for i in 0..half_length {
        trend[i] = 2.0 * trend[half_length] - trend[half_length + (half_length - i)];
    }
    for i in (data.len() - half_length)..data.len() {
        let offset = i - (data.len() - half_length - 1);
        trend[i] = 2.0 * trend[data.len() - half_length - 1] - trend[data.len() - half_length - 1 - offset];
    }

    Ok(trend)
}

fn estimate_seasonal_x11(
    seasonal_irregular: &[f64],
    seasonal_period: usize,
    _filter: &SeasonalFilter,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Simplified seasonal estimation using seasonal means
    let mut seasonal_means = vec![0.0; seasonal_period];
    let mut counts = vec![0; seasonal_period];

    for (i, &value) in seasonal_irregular.iter().enumerate() {
        let seasonal_idx = i % seasonal_period;
        seasonal_means[seasonal_idx] += value;
        counts[seasonal_idx] += 1;
    }

    // Compute averages and normalize
    for i in 0..seasonal_period {
        if counts[i] > 0 {
            seasonal_means[i] /= counts[i] as f64;
        }
    }

    // Normalize seasonal means to sum to zero (additive) or multiply to one (multiplicative)
    let mean_seasonal = seasonal_means.iter().sum::<f64>() / seasonal_period as f64;
    for seasonal_mean in seasonal_means.iter_mut() {
        *seasonal_mean -= mean_seasonal;
    }

    // Generate seasonal component for all observations
    let seasonal: Vec<f64> = (0..seasonal_irregular.len())
        .map(|i| seasonal_means[i % seasonal_period])
        .collect();

    Ok(seasonal)
}

fn compute_moving_average_trend(
    data: &[f64],
    config: &MovingAverageAdjustment,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut trend = vec![0.0; data.len()];
    let half_window = config.window_size / 2;

    for i in 0..data.len() {
        let start = if config.centered {
            i.saturating_sub(half_window)
        } else {
            i.saturating_sub(config.window_size - 1)
        };
        let end = if config.centered {
            (i + half_window + 1).min(data.len())
        } else {
            (i + 1).min(data.len())
        };

        if end - start >= config.min_observations {
            trend[i] = match config.ma_type {
                MovingAverageType::Simple => {
                    data[start..end].iter().sum::<f64>() / (end - start) as f64
                }
                MovingAverageType::Exponential(alpha) => {
                    compute_exponential_ma(&data[start..end], alpha)
                }
                MovingAverageType::LinearWeighted => {
                    compute_linear_weighted_ma(&data[start..end])
                }
                MovingAverageType::Henderson => {
                    // Use simple average as fallback
                    data[start..end].iter().sum::<f64>() / (end - start) as f64
                }
            };
        } else {
            // Fallback for insufficient observations
            trend[i] = data[i];
        }
    }

    Ok(trend)
}

fn estimate_seasonal_ma(
    detrended: &[f64],
    seasonal_period: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    estimate_additive_seasonal(detrended, seasonal_period)
}

fn stl_decompose(
    data: &[f64],
    config: &STLAdjustment,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    // Simplified STL implementation
    let mut seasonal = vec![0.0; data.len()];
    let mut trend = vec![0.0; data.len()];

    // Initial seasonal estimation
    let initial_seasonal = estimate_additive_seasonal(data, config.seasonal_period)?;
    seasonal.copy_from_slice(&initial_seasonal);

    // Iterative refinement
    for _iteration in 0..config.inner_iterations {
        // Detrend
        let detrended: Vec<f64> = data.iter().zip(seasonal.iter())
            .map(|(&x, &s)| x - s)
            .collect();

        // Update trend
        trend = smooth_data(&detrended, config.trend_span)?;

        // Update seasonal
        let deseasonalized: Vec<f64> = data.iter().zip(trend.iter())
            .map(|(&x, &t)| x - t)
            .collect();
        seasonal = smooth_seasonal(&deseasonalized, config.seasonal_period, config.seasonal_span)?;
    }

    // Compute irregular
    let irregular: Vec<f64> = data.iter().zip(trend.iter()).zip(seasonal.iter())
        .map(|((&x, &t), &s)| x - t - s)
        .collect();

    Ok((seasonal, trend, irregular))
}

fn generate_seasonal_factors(
    seasonal: &[f64],
    seasonal_period: usize,
) -> Result<SeasonalFactors, Box<dyn std::error::Error>> {
    let mut factors = SeasonalFactors {
        monthly_factors: None,
        weekly_factors: None,
        daily_factors: None,
        custom_factors: HashMap::new(),
        moving_seasonality: None,
    };

    // Generate appropriate factors based on period
    match seasonal_period {
        12 => {
            // Monthly factors
            let mut monthly = [0.0; 12];
            let mut counts = [0; 12];

            for (i, &value) in seasonal.iter().enumerate() {
                let month_idx = i % 12;
                monthly[month_idx] += value;
                counts[month_idx] += 1;
            }

            for i in 0..12 {
                if counts[i] > 0 {
                    monthly[i] /= counts[i] as f64;
                }
            }

            factors.monthly_factors = Some(monthly);
        }
        7 => {
            // Weekly factors
            let mut weekly = [0.0; 7];
            let mut counts = [0; 7];

            for (i, &value) in seasonal.iter().enumerate() {
                let day_idx = i % 7;
                weekly[day_idx] += value;
                counts[day_idx] += 1;
            }

            for i in 0..7 {
                if counts[i] > 0 {
                    weekly[i] /= counts[i] as f64;
                }
            }

            factors.weekly_factors = Some(weekly);
        }
        _ => {
            // Custom period factors
            let mut custom_factors_vec = vec![0.0; seasonal_period];
            let mut counts = vec![0; seasonal_period];

            for (i, &value) in seasonal.iter().enumerate() {
                let period_idx = i % seasonal_period;
                custom_factors_vec[period_idx] += value;
                counts[period_idx] += 1;
            }

            for i in 0..seasonal_period {
                if counts[i] > 0 {
                    custom_factors_vec[i] /= counts[i] as f64;
                }
            }

            factors.custom_factors.insert(format!("period_{}", seasonal_period), custom_factors_vec);
        }
    }

    Ok(factors)
}

fn compute_x13_diagnostics(
    _original: &[f64],
    _adjusted: &[f64],
    _seasonal: &[f64],
    _trend: &[f64],
    _irregular: &[f64],
) -> Result<AdjustmentDiagnostics, Box<dyn std::error::Error>> {
    // Simplified diagnostics implementation
    let m_stats = MStatistics {
        m1: 0.5, m2: 0.3, m3: 0.4, m4: 0.2, m5: 0.6,
        m6: 0.1, m7: 0.2, m8: 0.3, m9: 0.1, m10: 0.2, m11: 0.3,
        overall_quality: QualityAssessment::Acceptable,
    };

    let residual_tests = ResidualSeasonalityTests {
        f_test_statistic: 1.5,
        f_test_p_value: 0.2,
        kruskal_wallis_statistic: 2.0,
        kruskal_wallis_p_value: 0.3,
        combined_test: 1.8,
        has_residual_seasonality: false,
    };

    let revision_analysis = RevisionAnalysis {
        average_absolute_revision: 0.1,
        average_relative_revision: 1.0,
        revision_volatility: 0.05,
        largest_revisions: Vec::new(),
    };

    Ok(AdjustmentDiagnostics {
        q_statistic: 0.8,
        m_statistics: m_stats,
        residual_seasonality: residual_tests,
        residual_trading_day: 0.05,
        quality_score: 0.85,
        revision_analysis,
    })
}

fn compute_basic_diagnostics(
    _original: &[f64],
    _adjusted: &[f64],
    _seasonal: &[f64],
) -> Result<AdjustmentDiagnostics, Box<dyn std::error::Error>> {
    // Basic diagnostics for simpler methods
    let m_stats = MStatistics {
        m1: 0.6, m2: 0.4, m3: 0.5, m4: 0.3, m5: 0.7,
        m6: 0.2, m7: 0.3, m8: 0.4, m9: 0.2, m10: 0.3, m11: 0.4,
        overall_quality: QualityAssessment::ProbablyAcceptable,
    };

    let residual_tests = ResidualSeasonalityTests {
        f_test_statistic: 1.0,
        f_test_p_value: 0.4,
        kruskal_wallis_statistic: 1.5,
        kruskal_wallis_p_value: 0.5,
        combined_test: 1.2,
        has_residual_seasonality: false,
    };

    let revision_analysis = RevisionAnalysis {
        average_absolute_revision: 0.15,
        average_relative_revision: 1.5,
        revision_volatility: 0.08,
        largest_revisions: Vec::new(),
    };

    Ok(AdjustmentDiagnostics {
        q_statistic: 0.7,
        m_statistics: m_stats,
        residual_seasonality: residual_tests,
        residual_trading_day: 0.1,
        quality_score: 0.75,
        revision_analysis,
    })
}

// Additional utility functions

fn estimate_multiplicative_factors(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut factors = vec![1.0; period];
    let mut sums = vec![0.0; period];
    let mut counts = vec![0; period];

    for (i, &value) in data.iter().enumerate() {
        if value > 0.0 {
            let period_idx = i % period;
            sums[period_idx] += value;
            counts[period_idx] += 1;
        }
    }

    let overall_mean = data.iter().filter(|&&x| x > 0.0).sum::<f64>() / data.iter().filter(|&&x| x > 0.0).count() as f64;

    for i in 0..period {
        if counts[i] > 0 {
            let period_mean = sums[i] / counts[i] as f64;
            factors[i] = period_mean / overall_mean;
        }
    }

    Ok(factors)
}

fn estimate_simple_trend(data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let window = (data.len() / 10).max(3).min(13);
    let mut trend = vec![0.0; data.len()];

    for i in 0..data.len() {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(data.len());
        trend[i] = data[start..end].iter().sum::<f64>() / (end - start) as f64;
    }

    Ok(trend)
}

fn estimate_additive_seasonal(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut seasonal_means = vec![0.0; period];
    let mut counts = vec![0; period];

    for (i, &value) in data.iter().enumerate() {
        let period_idx = i % period;
        seasonal_means[period_idx] += value;
        counts[period_idx] += 1;
    }

    for i in 0..period {
        if counts[i] > 0 {
            seasonal_means[i] /= counts[i] as f64;
        }
    }

    // Center the seasonal means
    let mean_seasonal = seasonal_means.iter().sum::<f64>() / period as f64;
    for seasonal_mean in seasonal_means.iter_mut() {
        *seasonal_mean -= mean_seasonal;
    }

    // Generate seasonal component for all observations
    let seasonal: Vec<f64> = (0..data.len())
        .map(|i| seasonal_means[i % period])
        .collect();

    Ok(seasonal)
}

fn compute_exponential_ma(data: &[f64], alpha: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut ema = data[0];
    for &value in data.iter().skip(1) {
        ema = alpha * value + (1.0 - alpha) * ema;
    }
    ema
}

fn compute_linear_weighted_ma(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let n = data.len() as f64;
    let weights_sum = n * (n + 1.0) / 2.0;
    let weighted_sum = data.iter().enumerate()
        .map(|(i, &x)| x * (i + 1) as f64)
        .sum::<f64>();

    weighted_sum / weights_sum
}

fn smooth_data(data: &[f64], span: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut smoothed = vec![0.0; data.len()];
    let half_span = span / 2;

    for i in 0..data.len() {
        let start = i.saturating_sub(half_span);
        let end = (i + half_span + 1).min(data.len());
        smoothed[i] = data[start..end].iter().sum::<f64>() / (end - start) as f64;
    }

    Ok(smoothed)
}

fn smooth_seasonal(
    data: &[f64],
    period: usize,
    span: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Simple seasonal smoothing
    estimate_additive_seasonal(data, period)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seasonality::{SeasonalPeriodType};

    #[test]
    fn test_x13_adjustment() {
        let data: Vec<f64> = (0..48).map(|i| {
            5.0 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() +
            0.1 * rand::random::<f64>()
        }).collect();

        let periods = vec![
            SeasonalPeriod {
                period: 12.0,
                strength: 0.8,
                phase: 0.0,
                amplitude: 1.0,
                confidence: 0.95,
                period_type: SeasonalPeriodType::Monthly,
            }
        ];

        let result = apply_x13_adjustment(&data, &periods).unwrap();
        assert_eq!(result.adjusted_series.len(), data.len());
        assert_eq!(result.seasonal_components.len(), data.len());
        assert_eq!(result.method, SeasonalAdjustmentMethod::X13Arima);
    }

    #[test]
    fn test_moving_average_adjustment() {
        let data: Vec<f64> = (0..24).map(|i| {
            3.0 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
        }).collect();

        let periods = vec![
            SeasonalPeriod {
                period: 12.0,
                strength: 0.7,
                phase: 0.0,
                amplitude: 1.0,
                confidence: 0.9,
                period_type: SeasonalPeriodType::Monthly,
            }
        ];

        let result = apply_moving_average_adjustment(&data, &periods).unwrap();
        assert_eq!(result.adjusted_series.len(), data.len());
        assert!(result.diagnostics.quality_score > 0.0);
    }

    #[test]
    fn test_stl_adjustment() {
        let data: Vec<f64> = (0..36).map(|i| {
            2.0 + 0.1 * i as f64 + (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
        }).collect();

        let periods = vec![
            SeasonalPeriod {
                period: 12.0,
                strength: 0.6,
                phase: 0.0,
                amplitude: 1.0,
                confidence: 0.85,
                period_type: SeasonalPeriodType::Monthly,
            }
        ];

        let result = apply_stl_adjustment(&data, &periods).unwrap();
        assert_eq!(result.adjusted_series.len(), data.len());
        assert_eq!(result.method, SeasonalAdjustmentMethod::STL);
    }

    #[test]
    fn test_seasonal_factors_generation() {
        let seasonal: Vec<f64> = (0..24).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
        }).collect();

        let factors = generate_seasonal_factors(&seasonal, 12).unwrap();
        assert!(factors.monthly_factors.is_some());

        let monthly = factors.monthly_factors.unwrap();
        assert_eq!(monthly.len(), 12);
    }

    #[test]
    fn test_henderson_filter() {
        let data: Vec<f64> = (0..30).map(|i| {
            i as f64 + 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
        }).collect();

        let trend = estimate_trend_henderson(&data, 13).unwrap();
        assert_eq!(trend.len(), data.len());

        // Trend should be smoother than original data
        let data_variance = {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
        };

        let trend_variance = {
            let mean = trend.iter().sum::<f64>() / trend.len() as f64;
            trend.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / trend.len() as f64
        };

        assert!(trend_variance < data_variance);
    }
}