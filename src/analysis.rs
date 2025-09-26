//! Analysis result structures and functionality

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// General purpose analysis result container
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Type of analysis performed
    pub analysis_type: String,

    /// Analysis results as flexible key-value pairs
    pub results: HashMap<String, serde_json::Value>,

    /// Optional metadata about the analysis
    pub metadata: HashMap<String, String>,

    /// Timestamp when the analysis was performed
    pub timestamp: DateTime<Utc>,
}

impl AnalysisResult {
    /// Creates a new analysis result
    pub fn new(analysis_type: String) -> Self {
        AnalysisResult {
            analysis_type,
            results: HashMap::new(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    /// Creates a new analysis result with specific timestamp
    pub fn with_timestamp(analysis_type: String, timestamp: DateTime<Utc>) -> Self {
        AnalysisResult {
            analysis_type,
            results: HashMap::new(),
            metadata: HashMap::new(),
            timestamp,
        }
    }

    /// Adds a result value
    pub fn add_result<T: Serialize>(&mut self, key: String, value: T) -> Result<(), serde_json::Error> {
        let json_value = serde_json::to_value(value)?;
        self.results.insert(key, json_value);
        Ok(())
    }

    /// Gets a result value
    pub fn get_result<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>, serde_json::Error> {
        match self.results.get(key) {
            Some(value) => {
                let deserialized = serde_json::from_value(value.clone())?;
                Ok(Some(deserialized))
            }
            None => Ok(None),
        }
    }

    /// Adds metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Converts to a specific analysis result type
    pub fn try_into_trend_analysis(self) -> Result<TrendAnalysis, String> {
        if self.analysis_type != "trend" {
            return Err(format!("Expected trend analysis, got {}", self.analysis_type));
        }

        TrendAnalysis::from_analysis_result(self)
    }

    /// Converts to a specific analysis result type
    pub fn try_into_seasonal_analysis(self) -> Result<SeasonalAnalysis, String> {
        if self.analysis_type != "seasonal" {
            return Err(format!("Expected seasonal analysis, got {}", self.analysis_type));
        }

        SeasonalAnalysis::from_analysis_result(self)
    }

    /// Converts to a specific analysis result type
    pub fn try_into_anomaly_detection(self) -> Result<AnomalyDetection, String> {
        if self.analysis_type != "anomaly" {
            return Err(format!("Expected anomaly detection, got {}", self.analysis_type));
        }

        AnomalyDetection::from_analysis_result(self)
    }
}

/// Trend analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Direction of the trend
    pub direction: TrendDirection,

    /// Strength of the trend (0.0 to 1.0)
    pub strength: f64,

    /// Slope of the linear trend line
    pub slope: f64,

    /// Y-intercept of the linear trend line
    pub intercept: f64,

    /// R-squared value of the linear fit
    pub r_squared: f64,

    /// P-value for trend significance test
    pub p_value: Option<f64>,

    /// Confidence interval for the slope
    pub confidence_interval: Option<(f64, f64)>,

    /// Timestamp when analysis was performed
    pub timestamp: DateTime<Utc>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Strong upward trend
    StronglyIncreasing,
    /// Moderate upward trend
    Increasing,
    /// No significant trend
    Stable,
    /// Moderate downward trend
    Decreasing,
    /// Strong downward trend
    StronglyDecreasing,
    /// Inconclusive or volatile
    Inconclusive,
}

impl TrendAnalysis {
    /// Creates a new trend analysis result
    pub fn new(direction: TrendDirection, strength: f64, slope: f64, intercept: f64, r_squared: f64) -> Self {
        TrendAnalysis {
            direction,
            strength,
            slope,
            intercept,
            r_squared,
            p_value: None,
            confidence_interval: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Creates from AnalysisResult
    pub fn from_analysis_result(result: AnalysisResult) -> Result<Self, String> {
        let direction: TrendDirection = result.get_result("direction")
            .map_err(|e| e.to_string())?
            .ok_or("Missing direction")?;
        let strength: f64 = result.get_result("strength")
            .map_err(|e| e.to_string())?
            .ok_or("Missing strength")?;
        let slope: f64 = result.get_result("slope")
            .map_err(|e| e.to_string())?
            .ok_or("Missing slope")?;
        let intercept: f64 = result.get_result("intercept")
            .map_err(|e| e.to_string())?
            .ok_or("Missing intercept")?;
        let r_squared: f64 = result.get_result("r_squared")
            .map_err(|e| e.to_string())?
            .ok_or("Missing r_squared")?;

        Ok(TrendAnalysis {
            direction,
            strength,
            slope,
            intercept,
            r_squared,
            p_value: result.get_result("p_value").ok().flatten(),
            confidence_interval: result.get_result("confidence_interval").ok().flatten(),
            timestamp: result.timestamp,
            metadata: result.metadata,
        })
    }

    /// Converts to general AnalysisResult
    pub fn to_analysis_result(self) -> AnalysisResult {
        let mut result = AnalysisResult::with_timestamp("trend".to_string(), self.timestamp);
        let _ = result.add_result("direction".to_string(), self.direction);
        let _ = result.add_result("strength".to_string(), self.strength);
        let _ = result.add_result("slope".to_string(), self.slope);
        let _ = result.add_result("intercept".to_string(), self.intercept);
        let _ = result.add_result("r_squared".to_string(), self.r_squared);

        if let Some(p_value) = self.p_value {
            let _ = result.add_result("p_value".to_string(), p_value);
        }
        if let Some(ci) = self.confidence_interval {
            let _ = result.add_result("confidence_interval".to_string(), ci);
        }

        result.metadata = self.metadata;
        result
    }
}

/// Seasonal analysis results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalAnalysis {
    /// Whether significant seasonality was detected
    pub has_seasonality: bool,

    /// Dominant seasonal period (in time units)
    pub dominant_period: Option<f64>,

    /// Strength of seasonality (0.0 to 1.0)
    pub seasonality_strength: f64,

    /// Detected seasonal periods and their strengths
    pub periods: Vec<SeasonalPeriod>,

    /// Seasonal decomposition components if available
    pub decomposition: Option<SeasonalDecomposition>,

    /// Test statistics for seasonality tests
    pub test_statistics: HashMap<String, f64>,

    /// Timestamp when analysis was performed
    pub timestamp: DateTime<Utc>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalPeriod {
    /// Period length in time units
    pub period: f64,
    /// Strength of this seasonal pattern
    pub strength: f64,
    /// Confidence level
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalDecomposition {
    /// Trend component
    pub trend: Vec<f64>,
    /// Seasonal component
    pub seasonal: Vec<f64>,
    /// Residual/remainder component
    pub residual: Vec<f64>,
}

impl SeasonalAnalysis {
    /// Creates a new seasonal analysis result
    pub fn new(has_seasonality: bool, seasonality_strength: f64) -> Self {
        SeasonalAnalysis {
            has_seasonality,
            dominant_period: None,
            seasonality_strength,
            periods: Vec::new(),
            decomposition: None,
            test_statistics: HashMap::new(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Creates from AnalysisResult
    pub fn from_analysis_result(result: AnalysisResult) -> Result<Self, String> {
        let has_seasonality: bool = result.get_result("has_seasonality")
            .map_err(|e| e.to_string())?
            .ok_or("Missing has_seasonality")?;
        let seasonality_strength: f64 = result.get_result("seasonality_strength")
            .map_err(|e| e.to_string())?
            .ok_or("Missing seasonality_strength")?;

        Ok(SeasonalAnalysis {
            has_seasonality,
            dominant_period: result.get_result("dominant_period").ok().flatten(),
            seasonality_strength,
            periods: result.get_result("periods").ok().flatten().unwrap_or_default(),
            decomposition: result.get_result("decomposition").ok().flatten(),
            test_statistics: result.get_result("test_statistics").ok().flatten().unwrap_or_default(),
            timestamp: result.timestamp,
            metadata: result.metadata,
        })
    }

    /// Converts to general AnalysisResult
    pub fn to_analysis_result(self) -> AnalysisResult {
        let mut result = AnalysisResult::with_timestamp("seasonal".to_string(), self.timestamp);
        let _ = result.add_result("has_seasonality".to_string(), self.has_seasonality);
        let _ = result.add_result("seasonality_strength".to_string(), self.seasonality_strength);

        if let Some(period) = self.dominant_period {
            let _ = result.add_result("dominant_period".to_string(), period);
        }
        if !self.periods.is_empty() {
            let _ = result.add_result("periods".to_string(), self.periods);
        }
        if let Some(decomp) = self.decomposition {
            let _ = result.add_result("decomposition".to_string(), decomp);
        }
        if !self.test_statistics.is_empty() {
            let _ = result.add_result("test_statistics".to_string(), self.test_statistics);
        }

        result.metadata = self.metadata;
        result
    }
}

/// Anomaly detection results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Detected anomalies with their indices and scores
    pub anomalies: Vec<Anomaly>,

    /// Threshold used for anomaly detection
    pub threshold: f64,

    /// Method used for detection
    pub method: String,

    /// Overall anomaly statistics
    pub statistics: AnomalyStatistics,

    /// Timestamp when analysis was performed
    pub timestamp: DateTime<Utc>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anomaly {
    /// Index in the original time series
    pub index: usize,
    /// Timestamp of the anomaly
    pub timestamp: DateTime<Utc>,
    /// Original value
    pub value: f64,
    /// Anomaly score (higher = more anomalous)
    pub score: f64,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Expected value if available
    pub expected_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyStatistics {
    /// Total number of anomalies detected
    pub total_anomalies: usize,
    /// Percentage of data points that are anomalous
    pub anomaly_rate: f64,
    /// Mean anomaly score
    pub mean_score: f64,
    /// Maximum anomaly score
    pub max_score: f64,
    /// Distribution by severity
    pub severity_counts: HashMap<String, usize>,
}

impl AnomalyDetection {
    /// Creates a new anomaly detection result
    pub fn new(method: String, threshold: f64) -> Self {
        AnomalyDetection {
            anomalies: Vec::new(),
            threshold,
            method,
            statistics: AnomalyStatistics {
                total_anomalies: 0,
                anomaly_rate: 0.0,
                mean_score: 0.0,
                max_score: 0.0,
                severity_counts: HashMap::new(),
            },
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Creates from AnalysisResult
    pub fn from_analysis_result(result: AnalysisResult) -> Result<Self, String> {
        let threshold: f64 = result.get_result("threshold")
            .map_err(|e| e.to_string())?
            .ok_or("Missing threshold")?;
        let method: String = result.get_result("method")
            .map_err(|e| e.to_string())?
            .ok_or("Missing method")?;

        Ok(AnomalyDetection {
            anomalies: result.get_result("anomalies").ok().flatten().unwrap_or_default(),
            threshold,
            method,
            statistics: result.get_result("statistics").ok().flatten().unwrap_or_else(|| AnomalyStatistics {
                total_anomalies: 0,
                anomaly_rate: 0.0,
                mean_score: 0.0,
                max_score: 0.0,
                severity_counts: HashMap::new(),
            }),
            timestamp: result.timestamp,
            metadata: result.metadata,
        })
    }

    /// Converts to general AnalysisResult
    pub fn to_analysis_result(self) -> AnalysisResult {
        let mut result = AnalysisResult::with_timestamp("anomaly".to_string(), self.timestamp);
        let _ = result.add_result("threshold".to_string(), self.threshold);
        let _ = result.add_result("method".to_string(), self.method);
        let _ = result.add_result("anomalies".to_string(), self.anomalies);
        let _ = result.add_result("statistics".to_string(), self.statistics);

        result.metadata = self.metadata;
        result
    }

    /// Updates statistics based on current anomalies
    pub fn update_statistics(&mut self, total_points: usize) {
        self.statistics.total_anomalies = self.anomalies.len();
        self.statistics.anomaly_rate = if total_points > 0 {
            self.anomalies.len() as f64 / total_points as f64
        } else {
            0.0
        };

        if !self.anomalies.is_empty() {
            self.statistics.mean_score = self.anomalies.iter().map(|a| a.score).sum::<f64>() / self.anomalies.len() as f64;
            self.statistics.max_score = self.anomalies.iter().map(|a| a.score).fold(f64::NEG_INFINITY, f64::max);

            // Count severities
            let mut severity_counts = HashMap::new();
            for anomaly in &self.anomalies {
                let severity_str = match anomaly.severity {
                    AnomalySeverity::Low => "low",
                    AnomalySeverity::Medium => "medium",
                    AnomalySeverity::High => "high",
                    AnomalySeverity::Critical => "critical",
                };
                *severity_counts.entry(severity_str.to_string()).or_insert(0) += 1;
            }
            self.statistics.severity_counts = severity_counts;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_result_basic() {
        let mut result = AnalysisResult::new("test".to_string());

        result.add_result("value".to_string(), 42.5).unwrap();
        result.add_metadata("author".to_string(), "test".to_string());

        let value: f64 = result.get_result("value").unwrap().unwrap();
        assert_eq!(value, 42.5);

        assert_eq!(result.get_metadata("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_trend_analysis_conversion() {
        let trend = TrendAnalysis::new(
            TrendDirection::Increasing,
            0.8,
            1.5,
            0.0,
            0.95
        );

        let general = trend.clone().to_analysis_result();
        let converted_back = general.try_into_trend_analysis().unwrap();

        assert_eq!(converted_back.direction, trend.direction);
        assert_eq!(converted_back.strength, trend.strength);
    }

    #[test]
    fn test_anomaly_statistics() {
        let mut detection = AnomalyDetection::new("zscore".to_string(), 3.0);

        detection.anomalies.push(Anomaly {
            index: 0,
            timestamp: Utc::now(),
            value: 100.0,
            score: 4.5,
            severity: AnomalySeverity::High,
            expected_value: Some(10.0),
        });

        detection.update_statistics(100);

        assert_eq!(detection.statistics.total_anomalies, 1);
        assert_eq!(detection.statistics.anomaly_rate, 0.01);
        assert_eq!(detection.statistics.mean_score, 4.5);
    }
}