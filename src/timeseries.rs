//! TimeSeries data structure and related functionality

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::{Frequency, MissingValuePolicy};
use crate::validation::{validate_timestamps, validate_equal_lengths, handle_duplicate_timestamps};
use crate::{Result, TimeSeriesError};

/// Core time series data structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Name or identifier for this time series
    pub name: String,

    /// Timestamps in ascending order
    pub timestamps: Vec<DateTime<Utc>>,

    /// Corresponding values for each timestamp
    pub values: Vec<f64>,

    /// Optional metadata as key-value pairs
    pub metadata: HashMap<String, String>,

    /// Optional frequency information
    pub frequency: Option<Frequency>,

    /// Policy for handling missing values
    pub missing_value_policy: MissingValuePolicy,
}

impl TimeSeries {
    /// Creates a new TimeSeries with validation
    pub fn new(
        name: String,
        timestamps: Vec<DateTime<Utc>>,
        values: Vec<f64>,
    ) -> Result<Self> {
        Self::with_options(
            name,
            timestamps,
            values,
            HashMap::new(),
            None,
            MissingValuePolicy::default(),
        )
    }

    /// Creates a new TimeSeries with full options
    pub fn with_options(
        name: String,
        mut timestamps: Vec<DateTime<Utc>>,
        mut values: Vec<f64>,
        metadata: HashMap<String, String>,
        frequency: Option<Frequency>,
        missing_value_policy: MissingValuePolicy,
    ) -> Result<Self> {
        // Validate equal lengths
        validate_equal_lengths(&timestamps, &values)?;

        // Handle duplicates before sorting
        handle_duplicate_timestamps(&mut timestamps, &mut values)?;

        // Create the time series
        let mut ts = TimeSeries {
            name,
            timestamps,
            values,
            metadata,
            frequency,
            missing_value_policy,
        };

        // Validate and sort timestamps
        ts.sort_by_timestamp()?;
        ts.validate()?;

        Ok(ts)
    }

    /// Creates a new empty TimeSeries
    pub fn empty(name: String) -> Self {
        TimeSeries {
            name,
            timestamps: Vec::new(),
            values: Vec::new(),
            metadata: HashMap::new(),
            frequency: None,
            missing_value_policy: MissingValuePolicy::default(),
        }
    }

    /// Validates the time series data
    pub fn validate(&self) -> Result<()> {
        // Check equal lengths
        validate_equal_lengths(&self.timestamps, &self.values)?;

        // Check timestamps are sorted
        validate_timestamps(&self.timestamps)?;

        // Check for NaN values if policy doesn't allow them
        if matches!(self.missing_value_policy, MissingValuePolicy::Error) {
            for (i, value) in self.values.iter().enumerate() {
                if value.is_nan() {
                    return Err(Box::new(TimeSeriesError::missing_data(
                        format!("NaN value found at index {} but policy is Error", i)
                    )));
                }
            }
        }

        Ok(())
    }

    /// Sorts the time series by timestamp
    fn sort_by_timestamp(&mut self) -> Result<()> {
        if self.timestamps.is_empty() {
            return Ok(());
        }

        // Create index vector for sorting
        let mut indices: Vec<usize> = (0..self.timestamps.len()).collect();

        // Sort indices by timestamp
        indices.sort_by(|&a, &b| self.timestamps[a].cmp(&self.timestamps[b]));

        // Reorder both timestamps and values
        let sorted_timestamps: Vec<_> = indices.iter().map(|&i| self.timestamps[i]).collect();
        let sorted_values: Vec<_> = indices.iter().map(|&i| self.values[i]).collect();

        self.timestamps = sorted_timestamps;
        self.values = sorted_values;

        Ok(())
    }

    /// Returns the length of the time series
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Returns whether the time series is empty
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Adds a new data point to the time series
    pub fn push(&mut self, timestamp: DateTime<Utc>, value: f64) -> Result<()> {
        // Check if timestamp already exists
        if self.timestamps.contains(&timestamp) {
            return Err(Box::new(TimeSeriesError::data_inconsistency(
                format!("Timestamp {} already exists", timestamp)
            )));
        }

        // Find insertion point to maintain sorted order
        let insert_pos = self.timestamps
            .binary_search(&timestamp)
            .unwrap_or_else(|pos| pos);

        self.timestamps.insert(insert_pos, timestamp);
        self.values.insert(insert_pos, value);

        self.validate()?;
        Ok(())
    }

    /// Removes a data point at the specified index
    pub fn remove(&mut self, index: usize) -> Result<(DateTime<Utc>, f64)> {
        if index >= self.len() {
            return Err(Box::new(TimeSeriesError::validation(
                format!("Index {} out of bounds for length {}", index, self.len())
            )));
        }

        let timestamp = self.timestamps.remove(index);
        let value = self.values.remove(index);

        Ok((timestamp, value))
    }

    /// Gets a value at the specified index
    pub fn get(&self, index: usize) -> Option<(DateTime<Utc>, f64)> {
        if index < self.len() {
            Some((self.timestamps[index], self.values[index]))
        } else {
            None
        }
    }

    /// Finds a value by timestamp
    pub fn get_by_timestamp(&self, timestamp: DateTime<Utc>) -> Option<f64> {
        self.timestamps.iter().position(|&t| t == timestamp)
            .map(|index| self.values[index])
    }

    /// Gets a slice of the time series between two timestamps (inclusive)
    pub fn slice(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<TimeSeries> {
        let start_idx = self.timestamps.iter().position(|&t| t >= start).unwrap_or(self.len());
        let end_idx = self.timestamps.iter().rposition(|&t| t <= end).map(|i| i + 1).unwrap_or(0);

        if start_idx >= end_idx {
            return Ok(TimeSeries::empty(format!("{}_slice", self.name)));
        }

        TimeSeries::with_options(
            format!("{}_slice", self.name),
            self.timestamps[start_idx..end_idx].to_vec(),
            self.values[start_idx..end_idx].to_vec(),
            self.metadata.clone(),
            self.frequency.clone(),
            self.missing_value_policy.clone(),
        )
    }

    /// Infers frequency from the timestamps
    pub fn infer_frequency(&mut self) {
        if self.timestamps.len() >= 2 {
            self.frequency = Frequency::infer_from_timestamps(&self.timestamps);
        }
    }

    /// Gets basic statistics for the time series
    pub fn stats(&self) -> TimeSeriesStats {
        if self.values.is_empty() {
            return TimeSeriesStats::empty();
        }

        let valid_values: Vec<f64> = self.values.iter()
            .filter(|&&v| !v.is_nan())
            .copied()
            .collect();

        if valid_values.is_empty() {
            return TimeSeriesStats::empty();
        }

        let count = valid_values.len();
        let sum: f64 = valid_values.iter().sum();
        let mean = sum / count as f64;

        let mut sorted = valid_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[count - 1];

        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let variance = valid_values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        TimeSeriesStats {
            count,
            missing_count: self.values.len() - count,
            mean,
            median,
            min,
            max,
            std_dev,
            variance,
        }
    }

    /// Adds metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// Basic statistics for a time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    pub count: usize,
    pub missing_count: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub variance: f64,
}

impl TimeSeriesStats {
    fn empty() -> Self {
        TimeSeriesStats {
            count: 0,
            missing_count: 0,
            mean: f64::NAN,
            median: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
            std_dev: f64::NAN,
            variance: f64::NAN,
        }
    }
}

impl std::fmt::Display for TimeSeriesStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stats(count={}, mean={:.2}, std={:.2}, min={:.2}, max={:.2})",
               self.count, self.mean, self.std_dev, self.min, self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_timeseries_creation_basic() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        assert_eq!(ts.name, "test");
        assert_eq!(ts.len(), 3);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_timeseries_validation_equal_lengths() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
        ];
        let values = vec![1.0]; // Different length

        let result = TimeSeries::new("test".to_string(), timestamps, values);
        assert!(result.is_err());
    }

    #[test]
    fn test_timeseries_sorts_by_timestamp() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
        ];
        let values = vec![3.0, 1.0, 2.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        // Should be sorted by timestamp now
        assert_eq!(ts.values[0], 1.0); // First timestamp's value
        assert_eq!(ts.values[1], 2.0); // Second timestamp's value
        assert_eq!(ts.values[2], 3.0); // Third timestamp's value
    }

    #[test]
    fn test_timeseries_handles_duplicates() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(), // Duplicate
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 3.0, 2.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        // Should have only 2 unique timestamps
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.values[0], 2.0); // Average of 1.0 and 3.0
        assert_eq!(ts.values[1], 2.0);
    }

    #[test]
    fn test_timeseries_push() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        ];
        let values = vec![1.0];

        let mut ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        let new_timestamp = Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap();
        ts.push(new_timestamp, 2.0).unwrap();

        assert_eq!(ts.len(), 2);
        assert_eq!(ts.values[1], 2.0);

        // Test duplicate timestamp
        let result = ts.push(new_timestamp, 3.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_timeseries_remove() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0];

        let mut ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        let (_removed_ts, removed_val) = ts.remove(0).unwrap();
        assert_eq!(removed_val, 1.0);
        assert_eq!(ts.len(), 1);

        // Test out of bounds
        let result = ts.remove(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_timeseries_get_by_timestamp() {
        let timestamp = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps = vec![timestamp];
        let values = vec![42.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        let value = ts.get_by_timestamp(timestamp);
        assert_eq!(value, Some(42.0));

        let missing_value = ts.get_by_timestamp(Utc.with_ymd_and_hms(2023, 1, 2, 0, 0, 0).unwrap());
        assert_eq!(missing_value, None);
    }

    #[test]
    fn test_timeseries_slice() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 3, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        let start = Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap();

        let slice = ts.slice(start, end).unwrap();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice.values[0], 2.0);
        assert_eq!(slice.values[1], 3.0);
    }

    #[test]
    fn test_timeseries_stats() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();
        let stats = ts.stats();

        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, 2.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.median, 2.0);
    }

    #[test]
    fn test_timeseries_stats_with_nan() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, f64::NAN, 3.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();
        let stats = ts.stats();

        assert_eq!(stats.count, 2); // Only valid values
        assert_eq!(stats.missing_count, 1);
        assert_eq!(stats.mean, 2.0); // (1 + 3) / 2
    }

    #[test]
    fn test_timeseries_metadata() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        ];
        let values = vec![1.0];

        let mut ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        ts.add_metadata("source".to_string(), "sensor_1".to_string());
        assert_eq!(ts.get_metadata("source"), Some(&"sensor_1".to_string()));
        assert_eq!(ts.get_metadata("missing"), None);
    }

    #[test]
    fn test_timeseries_infer_frequency() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0];

        let mut ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();
        ts.infer_frequency();

        assert_eq!(ts.frequency, Some(Frequency::Hour));
    }

    #[test]
    fn test_timeseries_missing_value_policy_error() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        ];
        let values = vec![f64::NAN];

        let result = TimeSeries::with_options(
            "test".to_string(),
            timestamps,
            values,
            HashMap::new(),
            None,
            MissingValuePolicy::Error,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_timeseries_empty() {
        let ts = TimeSeries::empty("empty_test".to_string());
        assert_eq!(ts.name, "empty_test");
        assert_eq!(ts.len(), 0);
        assert!(ts.is_empty());

        let stats = ts.stats();
        assert_eq!(stats.count, 0);
        assert!(stats.mean.is_nan());
    }

    #[test]
    fn test_timeseries_serialization() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        ];
        let values = vec![42.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        // Test serialization to JSON
        let json = serde_json::to_string(&ts).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("42"));

        // Test deserialization from JSON
        let deserialized: TimeSeries = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, ts.name);
        assert_eq!(deserialized.values, ts.values);
    }
}