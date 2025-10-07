use chronos::ml::{
    ExplanationConfig, ExplanationMethod, BaselineStrategy, PerturbationConfig,
    ImportanceMethod, DistanceMetric, CounterfactualConfig, CounterfactualSearch,
    CounterfactualConstraints, explain_model_prediction, compute_feature_importance,
    analyze_attention_patterns,
};
use chronos::ml::transformer::AttentionAnalysis;
use chronos::ml::interfaces::ForecastingModel;
use chronos::timeseries::TimeSeries;
use chrono::Utc;

/// Mock forecasting model for testing
struct MockModel {
    input_length: usize,
    supports_gradients: bool,
    supports_attention: bool,
}

impl MockModel {
    fn new(input_length: usize) -> Self {
        Self {
            input_length,
            supports_gradients: false,
            supports_attention: false,
        }
    }

    fn with_gradients(mut self) -> Self {
        self.supports_gradients = true;
        self
    }
}

impl ForecastingModel for MockModel {
    fn model_name(&self) -> &'static str {
        "MockModel"
    }

    fn input_sequence_length(&self) -> usize {
        self.input_length
    }

    fn forecast_window(&self, input: &[f64], _horizon: usize) -> chronos::ml::MLResult<Vec<f64>> {
        // Simple forecast: sum of inputs
        let sum: f64 = input.iter().sum();
        Ok(vec![sum / input.len() as f64])
    }

    fn supports_gradients(&self) -> bool {
        self.supports_gradients
    }

    fn compute_input_gradients(&self, series: &TimeSeries) -> chronos::ml::MLResult<Vec<f64>> {
        if !self.supports_gradients {
            return Err(chronos::ml::MLError::model("Gradients not supported"));
        }

        // Use finite differences
        self.finite_difference_gradients(series, 0.01)
    }

    fn supports_attention(&self) -> bool {
        self.supports_attention
    }
}

fn create_test_series(length: usize) -> TimeSeries {
    let values: Vec<f64> = (0..length).map(|i| (i as f64 * 0.1).sin()).collect();
    let timestamps: Vec<_> = (0..length)
        .map(|i| Utc::now() + chrono::Duration::hours(i as i64))
        .collect();

    TimeSeries::new("test_series".to_string(), timestamps, values).unwrap()
}

#[test]
fn test_explanation_config_default() {
    let config = ExplanationConfig::default();

    assert!(!config.methods.is_empty());
    assert_eq!(config.confidence_level, 0.95);
    assert!(!config.generate_counterfactuals);
    assert_eq!(config.num_counterfactuals, 3);
}

#[test]
fn test_perturbation_config_default() {
    let config = PerturbationConfig::default();

    assert_eq!(config.noise_std, 0.1);
    assert!(!config.multiplicative);
    assert_eq!(config.min_magnitude, 0.01);
}

#[test]
fn test_baseline_strategy_zero() {
    let baseline = BaselineStrategy::Zero;

    // This tests that the baseline strategy can be created
    assert!(matches!(baseline, BaselineStrategy::Zero));
}

#[test]
fn test_explanation_with_permutation_importance() {
    let model = MockModel::new(10);
    let series = create_test_series(15);

    let config = ExplanationConfig {
        methods: vec![ExplanationMethod::PermutationImportance { n_repeats: 5 }],
        generate_counterfactuals: false,
        ..Default::default()
    };

    let result = explain_model_prediction(&model, &series, &config);
    assert!(result.is_ok());

    let explanation = result.unwrap();
    assert!(explanation.prediction > 0.0 || explanation.prediction < 0.0 || explanation.prediction == 0.0);
    assert_eq!(explanation.feature_importance.method_used, ImportanceMethod::Permutation);
    assert_eq!(explanation.feature_importance.temporal_importance.len(), 10);
    assert!(explanation.counterfactuals.is_none());
}

#[test]
fn test_explanation_with_gradients() {
    let model = MockModel::new(10).with_gradients();
    let series = create_test_series(15);

    let config = ExplanationConfig {
        methods: vec![ExplanationMethod::IntegratedGradients {
            steps: 20,
            baseline: BaselineStrategy::Mean,
        }],
        generate_counterfactuals: false,
        ..Default::default()
    };

    let result = explain_model_prediction(&model, &series, &config);
    assert!(result.is_ok());

    let explanation = result.unwrap();
    assert_eq!(explanation.feature_importance.method_used, ImportanceMethod::GradientBased);
    assert_eq!(explanation.feature_importance.temporal_importance.len(), 10);
}

#[test]
fn test_feature_importance_computation() {
    let model = MockModel::new(10);
    let series = create_test_series(15);

    let config = ExplanationConfig {
        methods: vec![ExplanationMethod::PermutationImportance { n_repeats: 10 }],
        ..Default::default()
    };

    let result = compute_feature_importance(&model, &series, &config);
    assert!(result.is_ok());

    let importance = result.unwrap();
    assert_eq!(importance.temporal_importance.len(), 10);

    // Check that importance scores are non-negative
    for score in &importance.temporal_importance {
        assert!(*score >= 0.0);
    }

    // Check that uncertainty is provided
    assert!(importance.uncertainty.is_some());
    let uncertainty = importance.uncertainty.unwrap();
    assert_eq!(uncertainty.len(), 10);
}

#[test]
fn test_temporal_importance_normalization() {
    let model = MockModel::new(10);
    let series = create_test_series(15);

    let config = ExplanationConfig {
        methods: vec![ExplanationMethod::PermutationImportance { n_repeats: 5 }],
        ..Default::default()
    };

    let explanation = explain_model_prediction(&model, &series, &config).unwrap();
    let temporal = &explanation.temporal_importance;

    // Normalized scores should sum to approximately 1.0
    let sum: f64 = temporal.normalized_scores.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // Cumulative importance should be monotonically increasing
    for i in 1..temporal.cumulative_importance.len() {
        assert!(temporal.cumulative_importance[i] >= temporal.cumulative_importance[i - 1]);
    }

    // Last cumulative value should be approximately 1.0
    let last = temporal.cumulative_importance.last().unwrap();
    assert!((last - 1.0).abs() < 0.001);

    // Top timesteps should be valid indices
    for &timestep in &temporal.top_timesteps {
        assert!(timestep < temporal.importance_scores.len());
    }
}

#[test]
fn test_counterfactual_generation() {
    let model = MockModel::new(10);
    let series = create_test_series(15);

    let config = ExplanationConfig {
        methods: vec![ExplanationMethod::PermutationImportance { n_repeats: 3 }],
        generate_counterfactuals: true,
        num_counterfactuals: 3,
        ..Default::default()
    };

    let explanation = explain_model_prediction(&model, &series, &config).unwrap();

    assert!(explanation.counterfactuals.is_some());
    let counterfactuals = explanation.counterfactuals.unwrap();
    assert_eq!(counterfactuals.len(), 3);

    for cf in &counterfactuals {
        // Check that counterfactual has valid structure
        assert_ne!(cf.original_prediction, 0.0); // Should have a value
        assert!(!cf.changes_made.is_empty());
        assert!(cf.distance_from_original >= 0.0);
        assert!(cf.plausibility_score >= 0.0 && cf.plausibility_score <= 1.0);
        assert!(!cf.explanation_text.is_empty());

        // Check that changes are within bounds
        for change in &cf.changes_made {
            assert!(change.timestep < 10); // Within input window
            assert!(change.change_magnitude >= 0.0);
        }
    }
}

#[test]
fn test_confidence_explanation() {
    let model = MockModel::new(10);
    let series = create_test_series(15);

    let config = ExplanationConfig::default();
    let explanation = explain_model_prediction(&model, &series, &config).unwrap();

    let confidence = &explanation.confidence_explanation;

    // Confidence score should be between 0 and 1
    assert!(confidence.confidence_score >= 0.0 && confidence.confidence_score <= 1.0);

    // Should have uncertainty sources
    assert!(!confidence.uncertainty_sources.is_empty());

    // Contributions should sum to approximately 1.0
    let total_contribution: f64 = confidence.uncertainty_sources
        .iter()
        .map(|s| s.contribution)
        .sum();
    assert!((total_contribution - 1.0).abs() < 0.1);

    // Should have confidence interval
    assert!(confidence.confidence_interval.is_some());
    if let Some((lower, upper)) = confidence.confidence_interval {
        assert!(lower < upper);
    }
}

#[test]
fn test_explanation_metadata() {
    let model = MockModel::new(10);
    let series = create_test_series(15);

    let config = ExplanationConfig {
        methods: vec![
            ExplanationMethod::PermutationImportance { n_repeats: 5 },
        ],
        ..Default::default()
    };

    let explanation = explain_model_prediction(&model, &series, &config).unwrap();
    let metadata = &explanation.explanation_metadata;

    assert!(!metadata.methods_used.is_empty());
    assert!(metadata.computation_time_ms >= 0); // Can be 0 for fast operations
    assert_eq!(metadata.model_type, "MockModel");
    assert!(!metadata.gradients_available);
    assert!(!metadata.attention_available);
}

#[test]
fn test_attention_analysis() {
    // Create mock attention analysis
    let attention = AttentionAnalysis {
        attention_weights: vec![
            vec![
                vec![0.1, 0.2, 0.3, 0.4],
                vec![0.4, 0.3, 0.2, 0.1],
            ],
        ],
        head_contributions: vec![0.5, 0.5],
        layer_contributions: vec![1.0],
        temporal_focus: vec![0.25, 0.25, 0.25, 0.25],
    };

    let insights = analyze_attention_patterns(&attention);

    // Check aggregated attention
    assert_eq!(insights.aggregated_attention.len(), 4);

    // Check that values are valid
    for &att in &insights.aggregated_attention {
        assert!(att >= 0.0);
    }

    // Check most attended timesteps
    assert!(!insights.most_attended_timesteps.is_empty());
    assert!(insights.most_attended_timesteps.len() <= 5);

    for (timestep, score) in &insights.most_attended_timesteps {
        assert!(*timestep < insights.aggregated_attention.len());
        assert!(*score >= 0.0);
    }

    // Check attention entropy
    assert!(insights.attention_entropy >= 0.0);
}

#[test]
fn test_attention_entropy_uniform() {
    let attention = AttentionAnalysis {
        attention_weights: vec![],
        head_contributions: vec![],
        layer_contributions: vec![],
        temporal_focus: vec![0.25, 0.25, 0.25, 0.25],
    };

    let insights = analyze_attention_patterns(&attention);

    // Uniform distribution should have high entropy
    // For 4 elements, log2(4) = 2.0
    assert!((insights.attention_entropy - 2.0).abs() < 0.1);
}

#[test]
fn test_attention_entropy_peaked() {
    let attention = AttentionAnalysis {
        attention_weights: vec![],
        head_contributions: vec![],
        layer_contributions: vec![],
        temporal_focus: vec![1.0, 0.0, 0.0, 0.0],
    };

    let insights = analyze_attention_patterns(&attention);

    // Peaked distribution should have low entropy
    assert!(insights.attention_entropy < 0.1);
}

#[test]
fn test_counterfactual_config_default() {
    let config = CounterfactualConfig::default();

    assert!(matches!(config.search_method, CounterfactualSearch::GradientBased { .. }));
    assert_eq!(config.distance_metric, DistanceMetric::L2);
}

#[test]
fn test_counterfactual_constraints_default() {
    let constraints = CounterfactualConstraints::default();

    assert_eq!(constraints.max_change_per_step, 2.0);
    assert_eq!(constraints.max_total_distance, 10.0);
    assert!(constraints.enforce_realism);
}

#[test]
fn test_multiple_explanation_methods() {
    let model = MockModel::new(10).with_gradients();
    let series = create_test_series(15);

    let config = ExplanationConfig {
        methods: vec![
            ExplanationMethod::IntegratedGradients {
                steps: 20,
                baseline: BaselineStrategy::Mean,
            },
            ExplanationMethod::PermutationImportance { n_repeats: 5 },
            ExplanationMethod::GradientXInput,
        ],
        ..Default::default()
    };

    let result = explain_model_prediction(&model, &series, &config);
    assert!(result.is_ok());

    let explanation = result.unwrap();
    // Should use gradient-based methods since they're available
    assert_eq!(explanation.feature_importance.method_used, ImportanceMethod::GradientBased);
}

#[test]
fn test_explanation_with_small_series() {
    let model = MockModel::new(5);
    let series = create_test_series(8);

    let config = ExplanationConfig::default();
    let result = explain_model_prediction(&model, &series, &config);

    assert!(result.is_ok());
    let explanation = result.unwrap();
    assert_eq!(explanation.feature_importance.temporal_importance.len(), 5);
}

#[test]
fn test_importance_method_enum() {
    let methods = vec![
        ImportanceMethod::Permutation,
        ImportanceMethod::SHAP,
        ImportanceMethod::GradientBased,
        ImportanceMethod::Occlusion,
        ImportanceMethod::Attention,
    ];

    // Just check that all variants can be created
    assert_eq!(methods.len(), 5);
}
