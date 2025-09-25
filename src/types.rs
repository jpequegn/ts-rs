//! Core types and enums for time series data structures

use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents different time frequencies for time series data
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frequency {
    /// Nanosecond frequency
    Nanosecond,
    /// Microsecond frequency
    Microsecond,
    /// Millisecond frequency
    Millisecond,
    /// Second frequency
    Second,
    /// Minute frequency
    Minute,
    /// Hour frequency
    Hour,
    /// Day frequency
    Day,
    /// Week frequency
    Week,
    /// Month frequency
    Month,
    /// Quarter frequency
    Quarter,
    /// Year frequency
    Year,
    /// Custom frequency with a specific duration
    Custom(Duration),
}

impl Frequency {
    /// Returns the duration equivalent of the frequency
    pub fn to_duration(&self) -> Option<Duration> {
        match self {
            Frequency::Nanosecond => Some(Duration::from_nanos(1)),
            Frequency::Microsecond => Some(Duration::from_micros(1)),
            Frequency::Millisecond => Some(Duration::from_millis(1)),
            Frequency::Second => Some(Duration::from_secs(1)),
            Frequency::Minute => Some(Duration::from_secs(60)),
            Frequency::Hour => Some(Duration::from_secs(3600)),
            Frequency::Day => Some(Duration::from_secs(86400)),
            Frequency::Week => Some(Duration::from_secs(604800)),
            Frequency::Custom(duration) => Some(*duration),
            // Month, Quarter, Year don't have fixed durations
            Frequency::Month | Frequency::Quarter | Frequency::Year => None,
        }
    }

    /// Returns a human-readable string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Frequency::Nanosecond => "nanosecond",
            Frequency::Microsecond => "microsecond",
            Frequency::Millisecond => "millisecond",
            Frequency::Second => "second",
            Frequency::Minute => "minute",
            Frequency::Hour => "hour",
            Frequency::Day => "day",
            Frequency::Week => "week",
            Frequency::Month => "month",
            Frequency::Quarter => "quarter",
            Frequency::Year => "year",
            Frequency::Custom(_) => "custom",
        }
    }

    /// Attempts to infer frequency from a series of timestamps
    pub fn infer_from_timestamps(timestamps: &[DateTime<Utc>]) -> Option<Self> {
        if timestamps.len() < 2 {
            return None;
        }

        let mut intervals = Vec::new();
        for i in 1..timestamps.len().min(10) {
            let interval = timestamps[i] - timestamps[i - 1];
            intervals.push(interval);
        }

        // Find the most common interval
        let avg_interval = intervals.iter().sum::<chrono::Duration>() / intervals.len() as i32;

        let seconds = avg_interval.num_seconds();
        match seconds {
            s if s <= 0 => {
                let millis = avg_interval.num_milliseconds();
                match millis {
                    m if m <= 0 => Some(Frequency::Microsecond),
                    1 => Some(Frequency::Millisecond),
                    _ => Some(Frequency::Custom(Duration::from_millis(millis as u64))),
                }
            }
            1 => Some(Frequency::Second),
            60 => Some(Frequency::Minute),
            3600 => Some(Frequency::Hour),
            86400 => Some(Frequency::Day),
            604800 => Some(Frequency::Week),
            _ => Some(Frequency::Custom(Duration::from_secs(seconds as u64))),
        }
    }
}

impl std::fmt::Display for Frequency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Frequency::Custom(duration) => write!(f, "custom({}s)", duration.as_secs_f64()),
            _ => write!(f, "{}", self.as_str()),
        }
    }
}

/// Policy for handling missing values in time series
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MissingValuePolicy {
    /// Use NaN for missing values
    NaN,
    /// Skip/ignore missing values
    Skip,
    /// Forward fill - use last known value
    ForwardFill,
    /// Backward fill - use next known value
    BackwardFill,
    /// Linear interpolation between known values
    LinearInterpolation,
    /// Use a specific default value
    Default(String), // Store as string for JSON serialization
    /// Raise an error when missing values are encountered
    Error,
}

impl MissingValuePolicy {
    /// Returns a human-readable description of the policy
    pub fn description(&self) -> &'static str {
        match self {
            MissingValuePolicy::NaN => "Use NaN for missing values",
            MissingValuePolicy::Skip => "Skip missing values",
            MissingValuePolicy::ForwardFill => "Forward fill with last known value",
            MissingValuePolicy::BackwardFill => "Backward fill with next known value",
            MissingValuePolicy::LinearInterpolation => "Linear interpolation between known values",
            MissingValuePolicy::Default(_) => "Use default value for missing values",
            MissingValuePolicy::Error => "Raise error on missing values",
        }
    }

    /// Creates a default value policy
    pub fn default_value<T: ToString>(value: T) -> Self {
        MissingValuePolicy::Default(value.to_string())
    }

    /// Gets the default value if this is a Default policy
    pub fn get_default_value(&self) -> Option<&str> {
        match self {
            MissingValuePolicy::Default(value) => Some(value),
            _ => None,
        }
    }
}

impl std::fmt::Display for MissingValuePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MissingValuePolicy::Default(value) => write!(f, "default({})", value),
            _ => write!(f, "{}", match self {
                MissingValuePolicy::NaN => "nan",
                MissingValuePolicy::Skip => "skip",
                MissingValuePolicy::ForwardFill => "ffill",
                MissingValuePolicy::BackwardFill => "bfill",
                MissingValuePolicy::LinearInterpolation => "linear",
                MissingValuePolicy::Error => "error",
                MissingValuePolicy::Default(_) => "default", // Already handled above
            }),
        }
    }
}

impl Default for MissingValuePolicy {
    fn default() -> Self {
        MissingValuePolicy::NaN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_frequency_to_duration() {
        assert_eq!(Frequency::Second.to_duration(), Some(Duration::from_secs(1)));
        assert_eq!(Frequency::Minute.to_duration(), Some(Duration::from_secs(60)));
        assert_eq!(Frequency::Hour.to_duration(), Some(Duration::from_secs(3600)));
        assert_eq!(Frequency::Day.to_duration(), Some(Duration::from_secs(86400)));

        // Variable duration frequencies should return None
        assert_eq!(Frequency::Month.to_duration(), None);
        assert_eq!(Frequency::Year.to_duration(), None);
    }

    #[test]
    fn test_frequency_display() {
        assert_eq!(format!("{}", Frequency::Second), "second");
        assert_eq!(format!("{}", Frequency::Custom(Duration::from_secs(30))), "custom(30s)");
    }

    #[test]
    fn test_frequency_inference() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 1, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 2, 0).unwrap(),
        ];

        let freq = Frequency::infer_from_timestamps(&timestamps);
        assert_eq!(freq, Some(Frequency::Minute));
    }

    #[test]
    fn test_missing_value_policy() {
        assert_eq!(MissingValuePolicy::default(), MissingValuePolicy::NaN);

        let policy = MissingValuePolicy::default_value(42.0);
        assert_eq!(policy.get_default_value(), Some("42"));

        assert_eq!(format!("{}", MissingValuePolicy::Skip), "skip");
        assert_eq!(format!("{}", MissingValuePolicy::default_value(1.5)), "default(1.5)");
    }
}