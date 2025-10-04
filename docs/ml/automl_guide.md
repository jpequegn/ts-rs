# AutoML Guide

Comprehensive guide to automated machine learning for time series forecasting.

## Overview

The AutoML module provides automated hyperparameter optimization, neural architecture search, and intelligent model selection for time series forecasting models.

## Features

### 1. Hyperparameter Optimization

Automatically find optimal hyperparameters using various search strategies:

#### Random Search
```rust
use chronos::ml::automl::*;

let search_space = SearchSpace {
    integer_params: vec![
        ("hidden_size".to_string(), (32, 256)),
        ("num_layers".to_string(), (1, 4)),
    ].into_iter().collect(),
    float_params: vec![
        ("learning_rate".to_string(), (0.0001, 0.1)),
        ("dropout_rate".to_string(), (0.0, 0.5)),
    ].into_iter().collect(),
    ..Default::default()
};

let method = OptimizationMethod::RandomSearch { n_trials: 50 };
```

#### Grid Search
```rust
let method = OptimizationMethod::GridSearch { resolution: 5 };
```

#### Bayesian Optimization
```rust
let method = OptimizationMethod::BayesianOptimization {
    acquisition_function: AcquisitionFunction::EI,
    n_initial_points: 10,
};
```

#### HyperBand
```rust
let method = OptimizationMethod::HyperBand {
    max_resource: 81,
    eta: 3.0,
};
```

### 2. Neural Architecture Search (NAS)

Automatically discover optimal neural network architectures:

#### Random Architecture Search
```rust
let search_space = ArchitectureSearchSpace {
    layer_types: vec![LayerType::LSTM, LayerType::GRU, LayerType::Attention],
    depth_range: 1..10,
    width_range: 16..256,
    activation_functions: vec![ActivationType::ReLU, ActivationType::Tanh],
    ..Default::default()
};

let constraints = ArchitectureConstraints {
    max_parameters: 10_000_000,
    max_memory_gb: 2.0,
    max_latency_ms: 100.0,
};

let strategy = NASStrategy::Random;
let result = search_neural_architecture(&data, &constraints, strategy)?;
```

#### Evolutionary NAS
```rust
let strategy = NASStrategy::Evolutionary {
    population_size: 20,
    generations: 50,
};
```

#### Progressive NAS
```rust
let strategy = NASStrategy::Progressive {
    complexity_schedule: vec![
        ComplexityLevel { max_depth: 3, max_width: 64, n_trials: 20 },
        ComplexityLevel { max_depth: 6, max_width: 128, n_trials: 30 },
        ComplexityLevel { max_depth: 10, max_width: 256, n_trials: 50 },
    ],
};
```

### 3. Automated Model Selection

Rank and select models based on dataset characteristics:

```rust
use chronos::ml::automl::selection::*;

let candidates = create_default_candidates();

let criteria = SelectionCriteria {
    performance_weight: 0.5,
    time_weight: 0.3,
    complexity_weight: 0.2,
    interpretability_weight: 0.0,
};

let selector = ModelSelector::new(candidates, criteria);
let top_models = selector.select_top_k_models(&data, 3);
```

### 4. Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```rust
let config = MultiObjectiveConfig {
    objectives: vec![
        Objective::Accuracy { metric: AccuracyMetric::RMSE },
        Objective::TrainingTime,
        Objective::ModelSize,
    ],
    optimization_method: MOOMethod::NSGA2 {
        population_size: 50,
        generations: 100,
    },
    preference_weights: None,
};

let pareto_front = optimize_multi_objective(&search_space, &config)?;
```

### 5. Meta-Learning

Leverage historical performance data for model recommendations:

```rust
use chronos::ml::automl::meta_learning::*;

let mut meta_learner = MetaLearner::new(SimilarityMetric::Euclidean);

// Add historical results
meta_learner.add_performance_entry(
    dataset_features,
    ModelType::LSTM,
    hyperparameters,
    performance,
);

// Get recommendations for new dataset
let recommendations = meta_learner.recommend_models(&new_features, 5);
```

### 6. Early Stopping

Intelligent early stopping to save computational resources:

```rust
use chronos::ml::automl::early_stopping::*;

let config = EarlyStoppingConfig {
    patience: 10,
    min_delta: 0.001,
    mode: EarlyStoppingMode::Minimize,
    warmup_epochs: 5,
};

let mut manager = EarlyStoppingManager::new(config);

for epoch in 0..max_epochs {
    let performance = train_one_epoch();

    if manager.should_stop(performance, epoch) {
        println!("Early stopping at epoch {}", epoch);
        break;
    }
}
```

## Complete AutoML Pipeline

```rust
use chronos::ml::automl::*;
use chronos::timeseries::TimeSeries;

// Configure AutoML
let config = AutoMLConfig {
    optimization_budget: OptimizationBudget {
        max_time: Duration::from_secs(3600), // 1 hour
        max_trials: 100,
        max_compute_resources: ComputeResources::default(),
        target_performance: Some(0.95),
    },
    model_types: vec![ModelType::LSTM, ModelType::GRU, ModelType::Transformer],
    performance_metric: PerformanceMetric::RMSE,
    cross_validation: CVConfig {
        n_folds: 5,
        cv_type: CVType::TimeSeriesSplit,
        validation_split: 0.2,
    },
    early_stopping: EarlyStoppingConfig {
        patience: 10,
        min_delta: 0.001,
        mode: EarlyStoppingMode::Minimize,
        warmup_epochs: 5,
    },
    parallelization: ParallelConfig {
        parallel_trials: 4,
        use_gpu: true,
        num_workers: 4,
    },
    search_strategy: SearchStrategy::Bayesian,
};

// Run AutoML optimization
let optimized_model = auto_optimize_model(&data, &config)?;

println!("Best model: {:?}", optimized_model.model_type);
println!("Performance: {:?}", optimized_model.performance_metrics);
println!("Hyperparameters: {:?}", optimized_model.hyperparameters);
```

## Performance Considerations

### Resource Management

- Set appropriate compute resource limits
- Use parallelization for faster optimization
- Enable early stopping to prevent wasted computation
- Consider memory constraints for large models

### Optimization Strategy Selection

- **Random Search**: Good baseline, works well with many hyperparameters
- **Grid Search**: Exhaustive but expensive, best for few parameters
- **Bayesian Optimization**: Best for expensive evaluations (e.g., neural networks)
- **HyperBand**: Excellent for budget-constrained optimization

### NAS Considerations

- Start with Progressive NAS for gradual complexity increase
- Use Evolutionary NAS for discrete architecture spaces
- Set reasonable architecture constraints to prevent memory issues

## Best Practices

1. **Start Simple**: Begin with random search before moving to more complex methods
2. **Use Cross-Validation**: Always validate with time series cross-validation
3. **Monitor Resources**: Track memory and computation time
4. **Leverage Meta-Learning**: Build up historical performance database
5. **Multi-Objective When Needed**: Consider trade-offs between accuracy, speed, and size
6. **Enable Early Stopping**: Save computation on unpromising trials
7. **Parallel Trials**: Utilize multi-core CPUs and GPUs when available

## Integration with Existing Models

AutoML integrates seamlessly with existing ML models:

```rust
// Optimize LSTM hyperparameters
let lstm_search_space = SearchSpace {
    integer_params: vec![
        ("hidden_size".to_string(), (32, 256)),
        ("num_layers".to_string(), (1, 4)),
    ].into_iter().collect(),
    float_params: vec![
        ("dropout_rate".to_string(), (0.0, 0.5)),
        ("learning_rate".to_string(), (0.0001, 0.01)),
    ].into_iter().collect(),
    ..Default::default()
};

let method = OptimizationMethod::BayesianOptimization {
    acquisition_function: AcquisitionFunction::EI,
    n_initial_points: 10,
};

let best_params = optimize_hyperparameters(
    ModelType::LSTM,
    &data,
    &lstm_search_space,
    method,
)?;
```

## Troubleshooting

### Out of Memory

- Reduce `max_parameters` in architecture constraints
- Decrease population size for evolutionary algorithms
- Lower `parallel_trials` setting

### Slow Optimization

- Reduce `max_trials` or `max_time`
- Enable early stopping with aggressive settings
- Use HyperBand instead of Bayesian optimization
- Increase `parallel_trials` if resources allow

### Poor Results

- Expand search space to include better regions
- Increase optimization budget (more trials/time)
- Use meta-learning to warm-start optimization
- Try different optimization methods

## Advanced Features

### Custom Objective Functions

Define custom objectives for multi-objective optimization:

```rust
let custom_objective = Objective::Interpretability {
    method: InterpretabilityMetric::FeatureImportance,
};
```

### Conditional Hyperparameters

Define parameters that depend on other parameters:

```rust
let conditional = ConditionalParameter {
    name: "lstm_hidden_size".to_string(),
    condition: ParameterCondition::Equals {
        param: "model_type".to_string(),
        value: "lstm".to_string(),
    },
    values: ParameterValues::Integer(32, 256),
};
```

### Performance Caching

AutoML caches performance results to avoid redundant evaluations.

## See Also

- [ML Module Overview](README.md)
- [LSTM/GRU Guide](lstm_gru_guide.md)
- [Transformer Guide](transformer_guide.md)
- [Ensemble Methods Guide](../forecasting/ensemble.md)
