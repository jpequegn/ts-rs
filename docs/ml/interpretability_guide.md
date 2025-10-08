# Model Interpretability and Explanation Guide

## Overview

The interpretability module provides comprehensive tools for understanding and explaining machine learning model predictions in time series analysis. This guide covers feature importance analysis, attention visualization, counterfactual explanations, and confidence estimation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Feature Importance Analysis](#feature-importance-analysis)
3. [Gradient-Based Attribution](#gradient-based-attribution)
4. [Attention Visualization](#attention-visualization)
5. [Counterfactual Explanations](#counterfactual-explanations)
6. [Confidence Estimation](#confidence-estimation)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

## Quick Start

### Basic Model Explanation

```rust
use chronos::ml::{
    ExplanationConfig, explain_model_prediction,
    LSTMForecaster, LSTMConfig, create_lstm_forecaster,
};
use chronos::timeseries::TimeSeries;

// Create and train a model
let config = LSTMConfig::default();
let forecaster = create_lstm_forecaster(&config, &training_data)?;

// Generate explanation with default settings
let explanation_config = ExplanationConfig::default();
let explanation = explain_model_prediction(
    &forecaster,
    &test_series,
    &explanation_config,
)?;

// Access results
println!("Prediction: {}", explanation.prediction);
println!("Top important timesteps: {:?}",
    explanation.temporal_importance.top_timesteps);
```

### Custom Explanation Configuration

```rust
use chronos::ml::{
    ExplanationConfig, ExplanationMethod, BaselineStrategy,
    PerturbationConfig,
};

let config = ExplanationConfig {
    methods: vec![
        ExplanationMethod::IntegratedGradients {
            steps: 100,
            baseline: BaselineStrategy::Mean,
        },
        ExplanationMethod::PermutationImportance { n_repeats: 20 },
    ],
    baseline_strategy: BaselineStrategy::Mean,
    perturbation_config: PerturbationConfig {
        noise_std: 0.15,
        multiplicative: false,
        min_magnitude: 0.01,
    },
    confidence_level: 0.95,
    generate_counterfactuals: true,
    num_counterfactuals: 5,
};

let explanation = explain_model_prediction(&model, &series, &config)?;
```

## Feature Importance Analysis

### Permutation Importance

Measures importance by randomly permuting feature values and observing prediction changes:

```rust
use chronos::ml::{compute_feature_importance, ExplanationConfig, ExplanationMethod};

let config = ExplanationConfig {
    methods: vec![
        ExplanationMethod::PermutationImportance { n_repeats: 50 },
    ],
    ..Default::default()
};

let importance = compute_feature_importance(&model, &data, &config)?;

// Access temporal importance scores
for (timestep, score) in importance.temporal_importance.iter().enumerate() {
    println!("Timestep {}: importance = {:.4}", timestep, score);
}

// Normalized scores (sum to 1)
let temporal = compute_temporal_importance(&importance);
println!("Normalized importance: {:?}", temporal.normalized_scores);
```

### Interpreting Importance Scores

- **High scores**: Timesteps with significant impact on predictions
- **Low scores**: Timesteps with minimal influence
- **Normalized scores**: Relative importance (sum to 1.0)
- **Cumulative importance**: Shows which timesteps contribute most cumulatively

## Gradient-Based Attribution

For neural network models that support gradients:

### Integrated Gradients

Computes attributions by integrating gradients along the path from baseline to input:

```rust
use chronos::ml::{ExplanationMethod, BaselineStrategy};

let config = ExplanationConfig {
    methods: vec![
        ExplanationMethod::IntegratedGradients {
            steps: 50,  // Number of integration steps
            baseline: BaselineStrategy::Zero,  // Starting point
        },
    ],
    ..Default::default()
};

let importance = compute_feature_importance(&model, &data, &config)?;
```

### Baseline Strategies

Different baseline choices affect attribution:

```rust
// Zero baseline (default)
BaselineStrategy::Zero

// Mean of the input
BaselineStrategy::Mean

// Gaussian noise
BaselineStrategy::Gaussian { mean: 0.0, std: 1.0 }

// Historical average
BaselineStrategy::Historical { lookback_periods: 10 }

// Custom baseline
BaselineStrategy::Custom(vec![0.5, 0.5, 0.5])
```

### Gradient × Input Attribution

Simpler but faster gradient-based method:

```rust
let config = ExplanationConfig {
    methods: vec![ExplanationMethod::GradientXInput],
    ..Default::default()
};
```

## Attention Visualization

For Transformer models with attention mechanisms:

### Analyzing Attention Patterns

```rust
use chronos::ml::{analyze_attention_patterns, TransformerForecaster};

// Get attention analysis from transformer
let attention = transformer.attention_analysis(&input_series)?;

// Analyze patterns
let insights = analyze_attention_patterns(&attention);

// Most attended timesteps
for (timestep, score) in &insights.most_attended_timesteps {
    println!("Timestep {}: attention = {:.4}", timestep, score);
}

// Attention entropy (distribution concentration)
println!("Attention entropy: {:.4}", insights.attention_entropy);
// Low entropy = focused attention
// High entropy = diffuse attention
```

### Attention Insights

```rust
pub struct AttentionInsights {
    /// Aggregated attention across all heads
    pub aggregated_attention: Vec<f64>,

    /// Most attended timesteps (index, score)
    pub most_attended_timesteps: Vec<(usize, f64)>,

    /// Entropy of attention distribution
    pub attention_entropy: f64,
}
```

### Interpreting Attention

- **High attention scores**: Timesteps the model focuses on
- **Low entropy**: Model has strong preference for specific timesteps
- **High entropy**: Model distributes attention broadly
- **Head specialization**: Different heads may focus on different patterns

## Counterfactual Explanations

Generate alternative scenarios to understand model behavior:

### Basic Counterfactuals

```rust
use chronos::ml::{CounterfactualConfig, CounterfactualSearch, CounterfactualConstraints};

let cf_config = CounterfactualConfig {
    search_method: CounterfactualSearch::GradientBased {
        learning_rate: 0.01,
        max_iterations: 100,
    },
    constraints: CounterfactualConstraints {
        max_change_per_step: 2.0,
        max_total_distance: 10.0,
        enforce_realism: true,
    },
    distance_metric: DistanceMetric::L2,
    target_change: 5.0,  // Desired prediction change
};

// Generate counterfactuals
let config = ExplanationConfig {
    generate_counterfactuals: true,
    num_counterfactuals: 3,
    ..Default::default()
};

let explanation = explain_model_prediction(&model, &series, &config)?;

if let Some(counterfactuals) = &explanation.counterfactuals {
    for (i, cf) in counterfactuals.iter().enumerate() {
        println!("\nCounterfactual {}:", i + 1);
        println!("  Original prediction: {:.2}", cf.original_prediction);
        println!("  Counterfactual prediction: {:.2}", cf.counterfactual_prediction);
        println!("  Distance: {:.2}", cf.distance_from_original);
        println!("  Plausibility: {:.2}", cf.plausibility_score);
        println!("  Explanation: {}", cf.explanation_text);

        // Inspect changes
        println!("  Changes made:");
        for change in &cf.changes_made {
            println!("    Timestep {}: {:.2} → {:.2} (Δ = {:.2})",
                change.timestep,
                change.original_value,
                change.new_value,
                change.change_magnitude
            );
        }
    }
}
```

### Search Methods

Different approaches for finding counterfactuals:

```rust
// Gradient-based optimization (fast, requires gradients)
CounterfactualSearch::GradientBased {
    learning_rate: 0.01,
    max_iterations: 100,
}

// Random perturbation search (simple, no gradients needed)
CounterfactualSearch::RandomSearch {
    max_attempts: 1000,
}

// Genetic algorithm (robust, slower)
CounterfactualSearch::GeneticAlgorithm {
    population_size: 50,
    generations: 100,
}
```

### Interpreting Counterfactuals

- **Distance**: How much the input was changed
- **Plausibility**: How realistic the counterfactual is (0-1)
- **Changes**: Specific modifications that led to different prediction
- **Target achievement**: Whether desired prediction change was reached

## Confidence Estimation

Understand model uncertainty and confidence:

```rust
let explanation = explain_model_prediction(&model, &series, &config)?;

let confidence = &explanation.confidence_explanation;
println!("Confidence score: {:.2}", confidence.confidence_score);

// Confidence interval
if let Some((lower, upper)) = confidence.confidence_interval {
    println!("95% CI: [{:.2}, {:.2}]", lower, upper);
}

// Uncertainty sources
for source in &confidence.uncertainty_sources {
    println!("{:?}: contribution = {:.2} - {}",
        source.source_type,
        source.contribution,
        source.description
    );
}
```

### Uncertainty Types

```rust
pub enum UncertaintyType {
    /// Aleatoric (data noise) - irreducible uncertainty
    Aleatoric,

    /// Epistemic (model uncertainty) - reducible with more data/training
    Epistemic,

    /// Distributional shift - input differs from training distribution
    DistributionalShift,

    /// Extrapolation - predicting beyond training range
    Extrapolation,
}
```

## Best Practices

### Choosing Explanation Methods

**For Neural Networks (LSTM, GRU, Transformer)**:
- Use `IntegratedGradients` for accurate attributions
- Enable `AttentionWeights` for Transformers
- Consider `GradientXInput` for faster approximations

**For Any Model**:
- Use `PermutationImportance` for model-agnostic explanations
- Increase `n_repeats` for more stable estimates
- Use `Occlusion` to understand local feature interactions

### Performance Considerations

```rust
// Fast configuration (suitable for real-time)
let fast_config = ExplanationConfig {
    methods: vec![
        ExplanationMethod::GradientXInput,
        ExplanationMethod::PermutationImportance { n_repeats: 5 },
    ],
    generate_counterfactuals: false,
    ..Default::default()
};

// Thorough configuration (for analysis)
let thorough_config = ExplanationConfig {
    methods: vec![
        ExplanationMethod::IntegratedGradients {
            steps: 100,
            baseline: BaselineStrategy::Mean,
        },
        ExplanationMethod::PermutationImportance { n_repeats: 50 },
        ExplanationMethod::Occlusion {
            window_size: 3,
            stride: 1,
        },
    ],
    generate_counterfactuals: true,
    num_counterfactuals: 10,
    ..Default::default()
};
```

### Baseline Selection Guidelines

- **Zero baseline**: Good for models trained with zero-padding
- **Mean baseline**: Best for normalized data
- **Gaussian baseline**: Useful for exploring model robustness
- **Historical baseline**: Appropriate for stationary time series

### Interpreting Results

1. **Feature Importance**: Focus on top 20% of timesteps for most impact
2. **Temporal Patterns**: Look for clusters of important consecutive timesteps
3. **Attention**: Compare with expected patterns (e.g., seasonality)
4. **Counterfactuals**: Validate that changes are realistic and actionable
5. **Confidence**: Low confidence suggests need for more data or model improvement

## Common Patterns

### Debugging Model Behavior

```rust
// Check if model focuses on right features
let explanation = explain_model_prediction(&model, &data, &config)?;

let top_timesteps = &explanation.temporal_importance.top_timesteps;
println!("Model focuses on timesteps: {:?}", top_timesteps);

// Verify against domain knowledge
if top_timesteps.contains(&expected_important_timestep) {
    println!("✓ Model correctly identifies important patterns");
} else {
    println!("⚠ Model may be focusing on spurious correlations");
}
```

### Comparing Model Decisions

```rust
// Explain predictions for multiple examples
let examples = vec![example1, example2, example3];
let explanations: Vec<_> = examples.iter()
    .map(|ex| explain_model_prediction(&model, ex, &config))
    .collect::<Result<_, _>>()?;

// Compare importance patterns
for (i, explanation) in explanations.iter().enumerate() {
    println!("\nExample {}:", i + 1);
    println!("  Prediction: {:.2}", explanation.prediction);
    println!("  Top timesteps: {:?}",
        explanation.temporal_importance.top_timesteps);
}
```

### Validating Counterfactuals

```rust
if let Some(counterfactuals) = &explanation.counterfactuals {
    for cf in counterfactuals {
        // Check plausibility
        if cf.plausibility_score < 0.5 {
            println!("⚠ Low plausibility counterfactual detected");
            continue;
        }

        // Check if changes are actionable
        let max_change = cf.changes_made.iter()
            .map(|c| c.change_magnitude)
            .fold(0.0, f64::max);

        if max_change > acceptable_threshold {
            println!("⚠ Changes too large to be practical");
        } else {
            println!("✓ Actionable counterfactual found");
        }
    }
}
```

## API Reference

### Main Functions

#### `explain_model_prediction`

```rust
pub fn explain_model_prediction<M: ForecastingModel>(
    model: &M,
    time_series: &TimeSeries,
    config: &ExplanationConfig,
) -> MLResult<ModelExplanation>
```

Generates a complete explanation for a model's prediction.

**Parameters**:
- `model`: The forecasting model implementing `ForecastingModel` trait
- `time_series`: Input time series to explain
- `config`: Configuration specifying explanation methods

**Returns**: `ModelExplanation` containing importance, attention, counterfactuals, and confidence

#### `compute_feature_importance`

```rust
pub fn compute_feature_importance<M: ForecastingModel>(
    model: &M,
    data: &TimeSeries,
    config: &ExplanationConfig,
) -> MLResult<FeatureImportance>
```

Computes feature importance scores using configured methods.

#### `analyze_attention_patterns`

```rust
pub fn analyze_attention_patterns(
    attention: &AttentionAnalysis,
) -> AttentionInsights
```

Analyzes attention patterns from Transformer models.

### Core Types

#### `ModelExplanation`

Complete explanation including all analysis results.

```rust
pub struct ModelExplanation {
    pub prediction: f64,
    pub feature_importance: FeatureImportance,
    pub temporal_importance: TemporalImportance,
    pub attention_weights: Option<AttentionAnalysis>,
    pub counterfactuals: Option<Vec<CounterfactualExplanation>>,
    pub confidence_explanation: ConfidenceExplanation,
    pub explanation_metadata: ExplanationMetadata,
}
```

#### `ExplanationConfig`

Configuration for explanation generation.

```rust
pub struct ExplanationConfig {
    pub methods: Vec<ExplanationMethod>,
    pub baseline_strategy: BaselineStrategy,
    pub perturbation_config: PerturbationConfig,
    pub confidence_level: f64,
    pub generate_counterfactuals: bool,
    pub num_counterfactuals: usize,
}
```

#### `FeatureImportance`

Feature importance scores across temporal and feature dimensions.

```rust
pub struct FeatureImportance {
    pub temporal_importance: Vec<f64>,
    pub feature_importance: HashMap<String, f64>,
    pub interaction_effects: Option<InteractionMatrix>,
    pub uncertainty: Option<Vec<f64>>,
    pub method_used: ImportanceMethod,
}
```

## Troubleshooting

### Issue: Importance scores are all similar

**Solution**: Try different methods or increase perturbation magnitude:
```rust
perturbation_config: PerturbationConfig {
    noise_std: 0.3,  // Increase from 0.1
    ..Default::default()
}
```

### Issue: Counterfactuals have low plausibility

**Solution**: Adjust constraints to be more realistic:
```rust
constraints: CounterfactualConstraints {
    max_change_per_step: 1.0,  // Reduce from 2.0
    enforce_realism: true,
    ..Default::default()
}
```

### Issue: Gradient computation fails

**Solution**: Check if model supports gradients:
```rust
if !model.supports_gradients() {
    println!("Model doesn't support gradients, using permutation importance");
    // Use permutation-based methods instead
}
```

## Examples

See `examples/interpretability_demo.rs` for a complete working example demonstrating all features.

## References

- Integrated Gradients: Sundararajan et al., 2017
- SHAP: Lundberg & Lee, 2017
- LIME: Ribeiro et al., 2016
- Attention Analysis: Vaswani et al., 2017
