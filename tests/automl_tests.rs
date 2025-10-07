use chrono::{DateTime, TimeZone, Utc};
use chronos::ml::automl::*;
use chronos::timeseries::TimeSeries;
use std::collections::HashMap;

/// Create test time series
fn create_test_timeseries() -> TimeSeries {
    let timestamps: Vec<DateTime<Utc>> = (0..100)
        .map(|i| Utc.timestamp_opt(1609459200 + i * 3600, 0).unwrap())
        .collect();

    let values: Vec<f64> = (0..100)
        .map(|i| 10.0 + (i as f64 * 0.1) + (i as f64 / 10.0).sin() * 5.0)
        .collect();

    TimeSeries::new(timestamps, values, None, None).unwrap()
}

#[test]
fn test_automl_config_default() {
    let config = AutoMLConfig::default();

    assert_eq!(config.model_types.len(), 3);
    assert_eq!(config.performance_metric, PerformanceMetric::RMSE);
    assert_eq!(config.cross_validation.n_folds, 5);
}

#[test]
fn test_search_space_creation() {
    let mut search_space = SearchSpace {
        categorical_params: HashMap::new(),
        integer_params: HashMap::new(),
        float_params: HashMap::new(),
        conditional_params: vec![],
    };

    search_space
        .integer_params
        .insert("hidden_size".to_string(), (32, 256));
    search_space
        .float_params
        .insert("learning_rate".to_string(), (0.0001, 0.1));

    assert_eq!(search_space.integer_params.len(), 1);
    assert_eq!(search_space.float_params.len(), 1);
}

#[test]
fn test_random_search_basic() {
    use chronos::ml::automl::hyperparameter::RandomSearchOptimizer;

    let mut search_space = SearchSpace {
        categorical_params: HashMap::new(),
        integer_params: HashMap::new(),
        float_params: HashMap::new(),
        conditional_params: vec![],
    };

    search_space
        .integer_params
        .insert("hidden_size".to_string(), (32, 128));
    search_space
        .float_params
        .insert("learning_rate".to_string(), (0.001, 0.1));

    let mut optimizer = RandomSearchOptimizer::new(10);

    // Test sampling configurations
    for _ in 0..10 {
        let config = optimizer.sample_configuration(&search_space);
        assert!(config.params.len() > 0);
    }
}

#[test]
fn test_grid_search_basic() {
    use chronos::ml::automl::hyperparameter::GridSearchOptimizer;

    let mut search_space = SearchSpace {
        categorical_params: HashMap::new(),
        integer_params: HashMap::new(),
        float_params: HashMap::new(),
        conditional_params: vec![],
    };

    search_space
        .integer_params
        .insert("param1".to_string(), (1, 3));

    let optimizer = GridSearchOptimizer::new(2);
    let grid = optimizer.generate_grid(&search_space);

    assert!(grid.len() > 0);
}

#[test]
fn test_bayesian_optimizer_creation() {
    use chronos::ml::automl::hyperparameter::BayesianOptimizer;

    let optimizer = BayesianOptimizer::new(AcquisitionFunction::EI, 5);

    let search_space = SearchSpace {
        categorical_params: HashMap::new(),
        integer_params: vec![("param1".to_string(), (1, 10))].into_iter().collect(),
        float_params: vec![("param2".to_string(), (0.0, 1.0))]
            .into_iter()
            .collect(),
        conditional_params: vec![],
    };

    // Should not panic
    let _ = optimizer;
}

#[test]
fn test_hyperband_optimizer() {
    use chronos::ml::automl::hyperparameter::HyperBandOptimizer;

    let mut optimizer = HyperBandOptimizer::new(81, 3.0);

    // Should generate configuration
    let config = optimizer.get_next_configuration();
    assert!(config.is_some());

    if let Some((n, r)) = config {
        assert!(n > 0);
        assert!(r > 0);
    }
}

#[test]
fn test_early_stopping_manager() {
    use chronos::ml::automl::early_stopping::EarlyStoppingManager;

    let config = EarlyStoppingConfig {
        patience: 5,
        min_delta: 0.001,
        mode: EarlyStoppingMode::Minimize,
        warmup_epochs: 2,
    };

    let mut manager = EarlyStoppingManager::new(config);

    // Should not stop early
    assert!(!manager.should_stop(1.0, 0));
    assert!(!manager.should_stop(0.9, 1));
    assert!(!manager.should_stop(0.8, 2));

    // Simulate no improvement
    for i in 3..10 {
        manager.should_stop(0.8, i);
    }

    // Should stop after patience epochs
    assert!(manager.should_stop(0.8, 10));
}

#[test]
fn test_dataset_feature_extraction() {
    use chronos::ml::automl::selection::extract_dataset_features;

    let ts = create_test_timeseries();
    let features = extract_dataset_features(&ts);

    assert!(features.statistical.mean > 0.0);
    assert!(features.statistical.std >= 0.0);
    assert_eq!(features.temporal.n_observations, 100);
    assert!(features.complexity.entropy >= 0.0);
}

#[test]
fn test_model_selector_creation() {
    use chronos::ml::automl::selection::{create_default_candidates, ModelSelector};

    let candidates = create_default_candidates();
    assert_eq!(candidates.len(), 3); // LSTM, GRU, Transformer

    let criteria = SelectionCriteria {
        performance_weight: 0.5,
        time_weight: 0.3,
        complexity_weight: 0.2,
        interpretability_weight: 0.0,
    };

    let selector = ModelSelector::new(candidates, criteria);
    assert!(selector.candidate_models.len() > 0);
}

#[test]
fn test_model_ranking() {
    use chronos::ml::automl::selection::{create_default_candidates, ModelSelector};

    let ts = create_test_timeseries();

    let candidates = create_default_candidates();
    let criteria = SelectionCriteria {
        performance_weight: 0.5,
        time_weight: 0.3,
        complexity_weight: 0.2,
        interpretability_weight: 0.0,
    };

    let selector = ModelSelector::new(candidates, criteria);
    let ranked = selector.rank_models(&ts);

    assert_eq!(ranked.len(), 3);

    // Scores should be between 0 and 1
    for (_, score) in &ranked {
        assert!(*score >= 0.0 && *score <= 1.0);
    }
}

#[test]
fn test_meta_learner_creation() {
    use chronos::ml::automl::meta_learning::{MetaLearner, SimilarityMetric};

    let meta_learner = MetaLearner::new(SimilarityMetric::Euclidean);

    // Should be created successfully
    let _ = meta_learner;
}

#[test]
fn test_nas_random_search() {
    use chronos::ml::automl::nas::{NASStrategy, NeuralArchitectureSearch};

    let ts = create_test_timeseries();

    let search_space = ArchitectureSearchSpace {
        layer_types: vec![LayerType::LSTM, LayerType::GRU],
        layer_sizes: vec![16..64],
        connection_patterns: vec![ConnectionPattern::Sequential],
        activation_functions: vec![ActivationType::ReLU, ActivationType::Tanh],
        depth_range: 1..5,
        width_range: 16..64,
    };

    let constraints = ArchitectureConstraints {
        max_parameters: 1_000_000,
        max_memory_gb: 1.0,
        max_latency_ms: 100.0,
    };

    let mut nas = NeuralArchitectureSearch::new(search_space, NASStrategy::Random, constraints);

    // Should complete search
    let result = nas.search_architecture(&ts);
    assert!(result.is_ok());

    if let Ok(arch) = result {
        assert!(arch.architecture.layers.len() > 0);
        assert!(arch.n_parameters <= 1_000_000);
    }
}

#[test]
fn test_nsga2_optimizer_creation() {
    use chronos::ml::automl::multi_objective::NSGA2Optimizer;

    let optimizer = NSGA2Optimizer::new(10, 5);

    // Should not panic
    let _ = optimizer;
}

#[test]
fn test_multi_objective_config() {
    let config = MultiObjectiveConfig {
        objectives: vec![
            Objective::Accuracy {
                metric: AccuracyMetric::RMSE,
            },
            Objective::TrainingTime,
        ],
        optimization_method: MOOMethod::NSGA2 {
            population_size: 20,
            generations: 10,
        },
        preference_weights: None,
    };

    assert_eq!(config.objectives.len(), 2);
}

#[test]
fn test_parameter_value_types() {
    let int_param = ParameterValue::Integer(42);
    let float_param = ParameterValue::Float(0.01);
    let string_param = ParameterValue::String("adam".to_string());
    let bool_param = ParameterValue::Boolean(true);

    // Should create successfully
    let _ = int_param;
    let _ = float_param;
    let _ = string_param;
    let _ = bool_param;
}

#[test]
fn test_optimization_budget() {
    use std::time::Duration;

    let budget = OptimizationBudget {
        max_time: Duration::from_secs(3600),
        max_trials: 100,
        max_compute_resources: ComputeResources::default(),
        target_performance: Some(0.95),
    };

    assert_eq!(budget.max_trials, 100);
    assert_eq!(budget.target_performance, Some(0.95));
}
