//! Validation functions for time series data

use chrono::{DateTime, Utc};
use crate::{Result, TimeSeriesError};

/// Validates that timestamps are in ascending order
pub fn validate_timestamps(timestamps: &[DateTime<Utc>]) -> Result<()> {
    if timestamps.len() <= 1 {
        return Ok(());
    }

    for i in 1..timestamps.len() {
        if timestamps[i] <= timestamps[i - 1] {
            return Err(Box::new(TimeSeriesError::validation(
                format!(
                    "Timestamps not in ascending order at index {}: {} <= {}",
                    i, timestamps[i], timestamps[i - 1]
                )
            )));
        }
    }

    Ok(())
}

/// Validates that timestamps and values have equal lengths
pub fn validate_equal_lengths(timestamps: &[DateTime<Utc>], values: &[f64]) -> Result<()> {
    if timestamps.len() != values.len() {
        return Err(Box::new(TimeSeriesError::validation(
            format!(
                "Timestamps and values must have equal lengths: {} vs {}",
                timestamps.len(),
                values.len()
            )
        )));
    }

    Ok(())
}

/// Handles duplicate timestamps by various strategies
pub fn handle_duplicate_timestamps(
    timestamps: &mut Vec<DateTime<Utc>>,
    values: &mut Vec<f64>,
) -> Result<()> {
    if timestamps.len() <= 1 {
        return Ok(());
    }

    // Create a vector of (timestamp, value, original_index) tuples
    let mut data: Vec<(DateTime<Utc>, f64, usize)> = timestamps
        .iter()
        .zip(values.iter())
        .enumerate()
        .map(|(i, (&ts, &val))| (ts, val, i))
        .collect();

    // Sort by timestamp
    data.sort_by(|a, b| a.0.cmp(&b.0));

    // Check for duplicates and handle them
    let mut deduplicated = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let current_timestamp = data[i].0;
        let mut duplicate_group = vec![data[i]];

        // Find all duplicates of this timestamp
        let mut j = i + 1;
        while j < data.len() && data[j].0 == current_timestamp {
            duplicate_group.push(data[j]);
            j += 1;
        }

        if duplicate_group.len() > 1 {
            // Handle duplicates by taking the average value
            let avg_value = duplicate_group.iter().map(|(_, v, _)| v).sum::<f64>() / duplicate_group.len() as f64;
            deduplicated.push((current_timestamp, avg_value));

            // Note: In a more sophisticated implementation, we could have different strategies
            // like taking first, last, min, max, etc.
        } else {
            deduplicated.push((current_timestamp, duplicate_group[0].1));
        }

        i = j;
    }

    // Update the original vectors
    *timestamps = deduplicated.iter().map(|(ts, _)| *ts).collect();
    *values = deduplicated.iter().map(|(_, val)| *val).collect();

    Ok(())
}

/// Detects gaps in time series based on expected frequency
pub fn detect_gaps(
    timestamps: &[DateTime<Utc>],
    expected_interval_seconds: Option<i64>,
) -> Vec<(DateTime<Utc>, DateTime<Utc>)> {
    if timestamps.len() <= 1 {
        return Vec::new();
    }

    let mut gaps = Vec::new();

    // If no expected interval, try to infer it from first few intervals
    let expected_interval = if let Some(interval) = expected_interval_seconds {
        chrono::Duration::seconds(interval)
    } else {
        // Take median of first 10 intervals as expected
        let mut intervals: Vec<chrono::Duration> = timestamps
            .windows(2)
            .take(10.min(timestamps.len() - 1))
            .map(|w| w[1] - w[0])
            .collect();

        if intervals.is_empty() {
            return gaps;
        }

        intervals.sort();
        intervals[intervals.len() / 2]
    };

    // Allow 50% tolerance
    let max_allowed = expected_interval + chrono::Duration::milliseconds(
        (expected_interval.num_milliseconds() as f64 * 0.5) as i64
    );

    for window in timestamps.windows(2) {
        let actual_interval = window[1] - window[0];
        if actual_interval > max_allowed {
            gaps.push((window[0], window[1]));
        }
    }

    gaps
}

/// Validates data quality and returns a report
pub fn validate_data_quality(timestamps: &[DateTime<Utc>], values: &[f64]) -> DataQualityReport {
    let mut report = DataQualityReport::new();

    // Count missing values (NaN or infinite)
    for &value in values {
        if value.is_nan() {
            report.nan_count += 1;
        } else if value.is_infinite() {
            report.infinite_count += 1;
        }
    }

    // Check for duplicate timestamps
    let mut sorted_ts = timestamps.to_vec();
    sorted_ts.sort();
    for window in sorted_ts.windows(2) {
        if window[0] == window[1] {
            report.duplicate_timestamps += 1;
        }
    }

    // Detect potential outliers using IQR method
    let mut valid_values: Vec<f64> = values.iter()
        .filter(|&&v| !v.is_nan() && !v.is_infinite())
        .copied()
        .collect();

    if valid_values.len() >= 4 {
        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1_idx = valid_values.len() / 4;
        let q3_idx = 3 * valid_values.len() / 4;
        let q1 = valid_values[q1_idx];
        let q3 = valid_values[q3_idx];
        let iqr = q3 - q1;
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        for &value in values {
            if !value.is_nan() && !value.is_infinite() && (value < lower_bound || value > upper_bound) {
                report.potential_outliers += 1;
            }
        }
    }

    // Detect gaps
    report.gaps = detect_gaps(timestamps, None);

    report
}

/// Report on data quality issues
#[derive(Debug, Clone)]
pub struct DataQualityReport {
    pub nan_count: usize,
    pub infinite_count: usize,
    pub duplicate_timestamps: usize,
    pub potential_outliers: usize,
    pub gaps: Vec<(DateTime<Utc>, DateTime<Utc>)>,
}

impl DataQualityReport {
    fn new() -> Self {
        DataQualityReport {
            nan_count: 0,
            infinite_count: 0,
            duplicate_timestamps: 0,
            potential_outliers: 0,
            gaps: Vec::new(),
        }
    }

    pub fn has_issues(&self) -> bool {
        self.nan_count > 0
            || self.infinite_count > 0
            || self.duplicate_timestamps > 0
            || !self.gaps.is_empty()
    }

    pub fn quality_score(&self, total_points: usize) -> f64 {
        if total_points == 0 {
            return 1.0;
        }

        let issues = self.nan_count + self.infinite_count + self.duplicate_timestamps + self.gaps.len();
        let quality = 1.0 - (issues as f64 / total_points as f64);
        quality.max(0.0)
    }
}

impl std::fmt::Display for DataQualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Data Quality Report:")?;
        writeln!(f, "  NaN values: {}", self.nan_count)?;
        writeln!(f, "  Infinite values: {}", self.infinite_count)?;
        writeln!(f, "  Duplicate timestamps: {}", self.duplicate_timestamps)?;
        writeln!(f, "  Potential outliers: {}", self.potential_outliers)?;
        writeln!(f, "  Time gaps: {}", self.gaps.len())?;

        if !self.gaps.is_empty() {
            writeln!(f, "  Gap details:")?;
            for (start, end) in &self.gaps {
                writeln!(f, "    {} -> {}", start, end)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_validate_timestamps_sorted() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
        ];
        assert!(validate_timestamps(&timestamps).is_ok());
    }

    #[test]
    fn test_validate_timestamps_unsorted() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        ];
        assert!(validate_timestamps(&timestamps).is_err());
    }

    #[test]
    fn test_validate_equal_lengths() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0];
        assert!(validate_equal_lengths(&timestamps, &values).is_ok());

        let values_wrong = vec![1.0];
        assert!(validate_equal_lengths(&timestamps, &values_wrong).is_err());
    }

    #[test]
    fn test_handle_duplicate_timestamps() {
        let mut timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
        ];
        let mut values = vec![1.0, 3.0, 2.0];

        handle_duplicate_timestamps(&mut timestamps, &mut values).unwrap();

        assert_eq!(timestamps.len(), 2);
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], 2.0); // Average of 1.0 and 3.0
    }

    #[test]
    fn test_detect_gaps() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 4, 0, 0).unwrap(), // 3-hour gap
        ];

        let gaps = detect_gaps(&timestamps, Some(3600)); // Expect 1-hour intervals
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].0, timestamps[1]);
        assert_eq!(gaps[0].1, timestamps[2]);
    }

    #[test]
    fn test_data_quality_report() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, f64::NAN, 100.0]; // One NaN, one potential outlier

        let report = validate_data_quality(&timestamps, &values);
        assert_eq!(report.nan_count, 1);
        assert!(report.quality_score(3) < 1.0);
    }
}