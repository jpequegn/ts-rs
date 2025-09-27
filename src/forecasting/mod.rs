//! # Time Series Forecasting Module
//!
//! Comprehensive forecasting capabilities for time series data, including
//! classical methods, ARIMA models, advanced techniques, and ensemble methods.

use crate::{TimeSeries, Result};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

pub mod classical;
pub mod arima;
pub mod advanced;
pub mod evaluation;
pub mod ensemble;
pub mod features;

pub use classical::*;
pub use arima::*;
pub use advanced::*;
pub use evaluation::*;
pub use ensemble::*;
pub use features::*;

/// Comprehensive forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastConfig {
    /// Forecasting method to use
    pub method: ForecastMethod,

    /// Number of periods to forecast
    pub horizon: usize,

    /// Confidence level for prediction intervals (0.0 to 1.0)
    pub confidence_level: f64,

    /// Whether to include prediction intervals
    pub include_intervals: bool,

    /// Evaluation configuration
    pub evaluation: EvaluationConfig,

    /// Feature engineering settings
    pub features: FeatureConfig,
}

/// Available forecasting methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastMethod {
    /// Simple moving average
    SimpleMovingAverage { window: usize },

    /// Weighted moving average
    WeightedMovingAverage { weights: Vec<f64> },

    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64 },

    /// Linear trend extrapolation
    LinearTrend,

    /// Seasonal naive forecasting
    SeasonalNaive { seasonal_period: usize },

    /// Holt-Winters exponential smoothing
    HoltWinters {
        alpha: f64,      // Level smoothing
        beta: f64,       // Trend smoothing
        gamma: f64,      // Seasonal smoothing
        seasonal_period: usize,
        seasonal_type: SeasonalType,
    },

    /// ARIMA model
    ARIMA {
        p: usize,        // Autoregressive order
        d: usize,        // Differencing order
        q: usize,        // Moving average order
    },

    /// Seasonal ARIMA (SARIMA)
    SARIMA {
        p: usize, d: usize, q: usize,              // Non-seasonal parameters
        seasonal_p: usize, seasonal_d: usize, seasonal_q: usize,  // Seasonal parameters
        seasonal_period: usize,
    },

    /// Auto-ARIMA with automatic parameter selection
    AutoARIMA {
        max_p: usize,
        max_d: usize,
        max_q: usize,
        max_seasonal_p: usize,
        max_seasonal_d: usize,
        max_seasonal_q: usize,
        seasonal_period: Option<usize>,
    },

    /// Error, Trend, Seasonal (ETS) model
    ETS {
        error_type: ETSComponent,
        trend_type: ETSComponent,
        seasonal_type: ETSComponent,
        seasonal_period: Option<usize>,
    },

    /// Theta method
    Theta { theta: f64 },

    /// Prophet-like decomposable model
    Prophet {
        growth: GrowthType,
        seasonality_mode: SeasonalityMode,
        yearly_seasonality: bool,
        weekly_seasonality: bool,
        daily_seasonality: bool,
    },

    /// Ensemble of multiple methods
    Ensemble {
        methods: Vec<ForecastMethod>,
        combination: EnsembleCombination,
        weights: Option<Vec<f64>>,
    },
}

/// Seasonal pattern types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SeasonalType {
    Additive,
    Multiplicative,
}

/// ETS model component types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ETSComponent {
    None,
    Additive,
    Multiplicative,
    Damped,  // For trend component only
}

/// Prophet growth types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GrowthType {
    Linear,
    Logistic { capacity: f64 },
}

/// Prophet seasonality modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SeasonalityMode {
    Additive,
    Multiplicative,
}

/// Ensemble combination methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EnsembleCombination {
    /// Simple average
    Average,
    /// Weighted average
    Weighted,
    /// Median combination
    Median,
    /// Best performer based on validation
    BestModel,
    /// Optimal weights from linear regression
    OptimalWeights,
}

/// Forecast result containing predictions and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Forecasted values
    pub forecasts: Vec<f64>,

    /// Timestamps for forecast periods
    pub timestamps: Vec<DateTime<Utc>>,

    /// Lower bounds of prediction intervals
    pub lower_bounds: Option<Vec<f64>>,

    /// Upper bounds of prediction intervals
    pub upper_bounds: Option<Vec<f64>>,

    /// Confidence level used for intervals
    pub confidence_level: f64,

    /// Method used for forecasting
    pub method: String,

    /// Model parameters and metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// In-sample fitted values
    pub fitted_values: Option<Vec<f64>>,

    /// Residuals from model fitting
    pub residuals: Option<Vec<f64>>,

    /// Model evaluation metrics
    pub evaluation: Option<ModelEvaluation>,

    /// Feature importance (if applicable)
    pub feature_importance: Option<HashMap<String, f64>>,
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluation {
    /// Mean Absolute Error
    pub mae: f64,
    /// Root Mean Square Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f64,
    /// Mean Absolute Scaled Error
    pub mase: Option<f64>,
    /// Akaike Information Criterion
    pub aic: Option<f64>,
    /// Bayesian Information Criterion
    pub bic: Option<f64>,
    /// Log-likelihood
    pub log_likelihood: Option<f64>,
    /// R-squared
    pub r_squared: Option<f64>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Configuration for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Enable cross-validation
    pub cross_validation: bool,
    /// Number of folds for cross-validation
    pub cv_folds: usize,
    /// Minimum training window size
    pub min_train_size: usize,
    /// Whether to perform walk-forward validation
    pub walk_forward: bool,
    /// Metrics to compute
    pub metrics: Vec<EvaluationMetric>,
    /// Whether to compute residual analysis
    pub residual_analysis: bool,
}

/// Available evaluation metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvaluationMetric {
    MAE,
    RMSE,
    MAPE,
    SMAPE,
    MASE,
    AIC,
    BIC,
    LogLikelihood,
    RSquared,
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Include lag features
    pub lag_features: Option<LagConfig>,
    /// Include rolling statistics
    pub rolling_features: Option<RollingConfig>,
    /// Include calendar features
    pub calendar_features: Option<CalendarConfig>,
    /// Include trend features
    pub trend_features: bool,
    /// Include seasonal decomposition features
    pub seasonal_features: bool,
}

/// Configuration for lag features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagConfig {
    /// Maximum number of lags to include
    pub max_lags: usize,
    /// Specific lag values to include
    pub specific_lags: Option<Vec<usize>>,
    /// Whether to include seasonal lags
    pub seasonal_lags: bool,
    /// Seasonal period for seasonal lags
    pub seasonal_period: Option<usize>,
}

/// Configuration for rolling statistics features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingConfig {
    /// Window sizes for rolling statistics
    pub windows: Vec<usize>,
    /// Statistics to compute
    pub statistics: Vec<RollingStatistic>,
}

/// Available rolling statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RollingStatistic {
    Mean,
    Median,
    Std,
    Min,
    Max,
    Quantile(f64),
    Skewness,
    Kurtosis,
}

/// Configuration for calendar features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarConfig {
    /// Include day of week features
    pub day_of_week: bool,
    /// Include month features
    pub month: bool,
    /// Include quarter features
    pub quarter: bool,
    /// Include year features
    pub year: bool,
    /// Include day of year features
    pub day_of_year: bool,
    /// Include week of year features
    pub week_of_year: bool,
    /// Include hour features (for intraday data)
    pub hour: bool,
    /// Include holiday indicators
    pub holidays: bool,
    /// Holiday calendar to use
    pub holiday_calendar: Option<String>,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            method: ForecastMethod::SimpleMovingAverage { window: 5 },
            horizon: 10,
            confidence_level: 0.95,
            include_intervals: true,
            evaluation: EvaluationConfig::default(),
            features: FeatureConfig::default(),
        }
    }
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            cross_validation: false,
            cv_folds: 5,
            min_train_size: 50,
            walk_forward: false,
            metrics: vec![
                EvaluationMetric::MAE,
                EvaluationMetric::RMSE,
                EvaluationMetric::MAPE,
                EvaluationMetric::SMAPE,
            ],
            residual_analysis: true,
        }
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            lag_features: None,
            rolling_features: None,
            calendar_features: None,
            trend_features: false,
            seasonal_features: false,
        }
    }
}

/// Main forecasting function
pub fn forecast_timeseries(
    timeseries: &TimeSeries,
    config: &ForecastConfig,
) -> Result<ForecastResult> {
    match &config.method {
        ForecastMethod::SimpleMovingAverage { window } => {
            classical::simple_moving_average_forecast(timeseries, *window, config.horizon)
        }
        ForecastMethod::WeightedMovingAverage { weights } => {
            classical::weighted_moving_average_forecast(timeseries, weights, config.horizon)
        }
        ForecastMethod::ExponentialSmoothing { alpha } => {
            classical::exponential_smoothing_forecast(timeseries, *alpha, config.horizon)
        }
        ForecastMethod::LinearTrend => {
            classical::linear_trend_forecast(timeseries, config.horizon)
        }
        ForecastMethod::SeasonalNaive { seasonal_period } => {
            classical::seasonal_naive_forecast(timeseries, *seasonal_period, config.horizon)
        }
        ForecastMethod::HoltWinters {
            alpha, beta, gamma, seasonal_period, seasonal_type
        } => {
            classical::holt_winters_forecast(
                timeseries, *alpha, *beta, *gamma, *seasonal_period, *seasonal_type, config.horizon
            )
        }
        ForecastMethod::ARIMA { p, d, q } => {
            arima::arima_forecast(timeseries, *p, *d, *q, config.horizon)
        }
        ForecastMethod::SARIMA {
            p, d, q, seasonal_p, seasonal_d, seasonal_q, seasonal_period
        } => {
            arima::sarima_forecast(
                timeseries, *p, *d, *q, *seasonal_p, *seasonal_d, *seasonal_q,
                *seasonal_period, config.horizon
            )
        }
        ForecastMethod::AutoARIMA {
            max_p, max_d, max_q, max_seasonal_p, max_seasonal_d, max_seasonal_q, seasonal_period
        } => {
            arima::auto_arima_forecast(
                timeseries, *max_p, *max_d, *max_q, *max_seasonal_p, *max_seasonal_d,
                *max_seasonal_q, *seasonal_period, config.horizon
            )
        }
        ForecastMethod::ETS { error_type, trend_type, seasonal_type, seasonal_period } => {
            advanced::ets_forecast(
                timeseries, *error_type, *trend_type, *seasonal_type, *seasonal_period, config.horizon
            )
        }
        ForecastMethod::Theta { theta } => {
            advanced::theta_forecast(timeseries, *theta, config.horizon)
        }
        ForecastMethod::Prophet {
            growth, seasonality_mode, yearly_seasonality, weekly_seasonality, daily_seasonality
        } => {
            advanced::prophet_forecast(
                timeseries, *growth, *seasonality_mode, *yearly_seasonality,
                *weekly_seasonality, *daily_seasonality, config.horizon
            )
        }
        ForecastMethod::Ensemble { methods, combination, weights } => {
            ensemble::ensemble_forecast(timeseries, methods, *combination, weights.as_ref(), config.horizon)
        }
    }
}

/// Perform comprehensive forecast evaluation
pub fn evaluate_forecast_model(
    timeseries: &TimeSeries,
    config: &ForecastConfig,
) -> Result<ModelEvaluation> {
    evaluation::evaluate_model(timeseries, config)
}

/// Generate forecast with confidence intervals
pub fn forecast_with_intervals(
    timeseries: &TimeSeries,
    config: &ForecastConfig,
) -> Result<ForecastResult> {
    let mut result = forecast_timeseries(timeseries, config)?;

    if config.include_intervals {
        let intervals = evaluation::compute_prediction_intervals(
            timeseries,
            &result.forecasts,
            config.confidence_level
        )?;
        result.lower_bounds = Some(intervals.0);
        result.upper_bounds = Some(intervals.1);
    }

    Ok(result)
}