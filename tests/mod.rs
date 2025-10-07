//! Comprehensive Testing Suite for Chronos Time Series Analysis
//!
//! This module organizes the complete testing infrastructure as specified in GitHub issue #16.
//!
//! ## Testing Categories
//!
//! ### Unit Tests
//! - Mathematical function correctness
//! - Statistical algorithm accuracy
//! - Edge case handling
//! - Numerical stability
//!
//! ### Integration Tests
//! - End-to-end CLI workflows
//! - Data import/export functionality
//! - Cross-platform compatibility
//! - Error handling and recovery
//!
//! ### Property-Based Testing
//! - Statistical invariants verification
//! - Algorithm properties across input ranges
//! - Fuzzing for robustness
//!
//! ### Benchmark Suite
//! - Performance regression detection
//! - Memory usage monitoring
//! - Scalability testing
//!
//! ### Statistical Validation
//! - Comparison with published results
//! - Reference dataset validation
//! - Cross-package accuracy verification
//!
//! ### Test Data
//! - Synthetic data generation
//! - Real-world reference datasets
//! - Edge case datasets

pub mod data;
pub mod integration;
pub mod property_based;
pub mod statistical_validation;
pub mod unit;

// Re-export commonly used testing utilities
pub use data::synthetic_generators::*;

#[cfg(test)]
mod comprehensive_test_runner {
    //! Test runner for coordinating comprehensive testing workflows

    use super::*;

    /// Run all unit tests
    #[test]
    #[ignore] // Use cargo test -- --ignored to run
    fn run_all_unit_tests() {
        // This would coordinate running all unit test modules
        // In practice, cargo handles this automatically
        println!("Running comprehensive unit test suite...");
    }

    /// Run all integration tests
    #[test]
    #[ignore] // Use cargo test -- --ignored to run
    fn run_all_integration_tests() {
        // This would coordinate running all integration test modules
        println!("Running comprehensive integration test suite...");
    }

    /// Run performance benchmarks
    #[test]
    #[ignore] // Use cargo test -- --ignored to run
    fn run_performance_benchmarks() {
        // This would run the benchmark suite
        // cargo bench is the proper way to run benchmarks
        println!("Run 'cargo bench' to execute performance benchmarks");
    }

    /// Run statistical validation tests
    #[test]
    #[ignore] // Use cargo test -- --ignored to run
    fn run_statistical_validation() {
        println!("Running statistical validation against reference datasets...");
    }
}

/// Testing utilities and helpers
pub mod utils {
    use chrono::{DateTime, TimeZone, Utc};
    use chronos::TimeSeries;

    /// Create a simple test time series for quick testing
    pub fn create_simple_test_series(name: &str, size: usize) -> TimeSeries {
        let timestamps: Vec<DateTime<Utc>> = (0..size)
            .map(|i| Utc.timestamp_opt(1000000000 + i as i64 * 3600, 0).unwrap())
            .collect();
        let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

        TimeSeries::new(name.to_string(), timestamps, values).unwrap()
    }

    /// Assert that two floating point values are approximately equal
    pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
        assert!(
            (a - b).abs() < epsilon,
            "Values not approximately equal: {} vs {}",
            a,
            b
        );
    }

    /// Assert that a statistical test result is valid
    pub fn assert_valid_p_value(p_value: f64) {
        assert!(
            p_value >= 0.0 && p_value <= 1.0,
            "Invalid p-value: {}",
            p_value
        );
        assert!(p_value.is_finite(), "P-value must be finite: {}", p_value);
    }
}
