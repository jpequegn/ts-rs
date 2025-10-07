//! Hyperparameter Optimization Module
//!
//! Implements various hyperparameter optimization algorithms:
//! - Random Search
//! - Grid Search
//! - Bayesian Optimization with Gaussian Processes
//! - HyperBand for efficient resource allocation
//! - BOHB (Bayesian + HyperBand)

use super::types::*;
use crate::ml::MLResult;
use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;

// ================================================================================================
// Random Search Implementation
// ================================================================================================

/// Random search optimizer
pub struct RandomSearchOptimizer {
    n_trials: usize,
    rng: rand::rngs::ThreadRng,
}

impl RandomSearchOptimizer {
    pub fn new(n_trials: usize) -> Self {
        Self {
            n_trials,
            rng: rand::thread_rng(),
        }
    }

    /// Sample random configuration from search space
    pub fn sample_configuration(&mut self, search_space: &SearchSpace) -> ParameterConfiguration {
        let mut params = HashMap::new();

        // Sample categorical parameters
        for (name, values) in &search_space.categorical_params {
            if let Some(value) = values.choose(&mut self.rng) {
                params.insert(name.clone(), ParameterValue::String(value.clone()));
            }
        }

        // Sample integer parameters
        for (name, (min, max)) in &search_space.integer_params {
            let value = self.rng.gen_range(*min..=*max);
            params.insert(name.clone(), ParameterValue::Integer(value));
        }

        // Sample float parameters
        for (name, (min, max)) in &search_space.float_params {
            let value = self.rng.gen_range(*min..=*max);
            params.insert(name.clone(), ParameterValue::Float(value));
        }

        ParameterConfiguration { params }
    }

    /// Run random search optimization
    pub fn optimize<F>(
        &mut self,
        search_space: &SearchSpace,
        objective_fn: F,
    ) -> MLResult<(ParameterConfiguration, f64)>
    where
        F: Fn(&ParameterConfiguration) -> MLResult<f64>,
    {
        let mut best_config = None;
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..self.n_trials {
            let config = self.sample_configuration(search_space);
            let score = objective_fn(&config)?;

            if score > best_score {
                best_score = score;
                best_config = Some(config);
            }
        }

        best_config
            .map(|config| (config, best_score))
            .ok_or_else(|| crate::ml::MLError::training("No valid configuration found"))
    }
}

// ================================================================================================
// Grid Search Implementation
// ================================================================================================

/// Grid search optimizer
pub struct GridSearchOptimizer {
    resolution: usize,
}

impl GridSearchOptimizer {
    pub fn new(resolution: usize) -> Self {
        Self { resolution }
    }

    /// Generate grid points for a parameter range
    fn generate_grid_points(&self, min: f64, max: f64) -> Vec<f64> {
        (0..=self.resolution)
            .map(|i| {
                let fraction = i as f64 / self.resolution as f64;
                min + fraction * (max - min)
            })
            .collect()
    }

    /// Generate all configurations from grid
    pub fn generate_grid(&self, search_space: &SearchSpace) -> Vec<ParameterConfiguration> {
        let mut configs = vec![ParameterConfiguration {
            params: HashMap::new(),
        }];

        // Grid for categorical parameters
        for (name, values) in &search_space.categorical_params {
            let mut new_configs = Vec::new();
            for config in &configs {
                for value in values {
                    let mut new_config = config.clone();
                    new_config
                        .params
                        .insert(name.clone(), ParameterValue::String(value.clone()));
                    new_configs.push(new_config);
                }
            }
            configs = new_configs;
        }

        // Grid for integer parameters
        for (name, (min, max)) in &search_space.integer_params {
            let mut new_configs = Vec::new();
            let step = (*max - *min) / self.resolution as i32;
            let step = step.max(1);

            for config in &configs {
                for value in (*min..=*max).step_by(step as usize) {
                    let mut new_config = config.clone();
                    new_config
                        .params
                        .insert(name.clone(), ParameterValue::Integer(value));
                    new_configs.push(new_config);
                }
            }
            configs = new_configs;
        }

        // Grid for float parameters
        for (name, (min, max)) in &search_space.float_params {
            let mut new_configs = Vec::new();
            let grid_points = self.generate_grid_points(*min, *max);

            for config in &configs {
                for &value in &grid_points {
                    let mut new_config = config.clone();
                    new_config
                        .params
                        .insert(name.clone(), ParameterValue::Float(value));
                    new_configs.push(new_config);
                }
            }
            configs = new_configs;
        }

        configs
    }

    /// Run grid search optimization
    pub fn optimize<F>(
        &self,
        search_space: &SearchSpace,
        objective_fn: F,
    ) -> MLResult<(ParameterConfiguration, f64)>
    where
        F: Fn(&ParameterConfiguration) -> MLResult<f64>,
    {
        let configs = self.generate_grid(search_space);
        let mut best_config = None;
        let mut best_score = f64::NEG_INFINITY;

        for config in configs {
            let score = objective_fn(&config)?;
            if score > best_score {
                best_score = score;
                best_config = Some(config);
            }
        }

        best_config
            .map(|config| (config, best_score))
            .ok_or_else(|| crate::ml::MLError::training("No valid configuration found"))
    }
}

// ================================================================================================
// Bayesian Optimization Implementation
// ================================================================================================

/// Bayesian optimizer using Gaussian Process
pub struct BayesianOptimizer {
    /// Observed parameter configurations
    observations: Vec<(Vec<f64>, f64)>,

    /// Acquisition function to use
    acquisition_function: AcquisitionFunction,

    /// Number of random initial points
    n_initial_points: usize,

    /// Exploration-exploitation balance (for UCB)
    kappa: f64,

    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl BayesianOptimizer {
    pub fn new(acquisition_function: AcquisitionFunction, n_initial_points: usize) -> Self {
        Self {
            observations: Vec::new(),
            acquisition_function,
            n_initial_points,
            kappa: 2.0,
            rng: rand::thread_rng(),
        }
    }

    /// Suggest next trial based on acquisition function
    pub fn suggest_next_trial(
        &mut self,
        search_space: &SearchSpace,
    ) -> MLResult<ParameterConfiguration> {
        // Use random search for initial points
        if self.observations.len() < self.n_initial_points {
            let mut random_search = RandomSearchOptimizer::new(1);
            return Ok(random_search.sample_configuration(search_space));
        }

        // Use Bayesian optimization for subsequent points
        self.optimize_acquisition(search_space)
    }

    /// Update optimizer with new observation
    pub fn update_with_result(&mut self, params: Vec<f64>, performance: f64) {
        self.observations.push((params, performance));
    }

    /// Optimize acquisition function
    fn optimize_acquisition(&mut self, search_space: &SearchSpace) -> MLResult<ParameterConfiguration> {
        // Simplified: sample many random points and pick best according to acquisition
        let n_samples = 1000;
        let mut best_config = None;
        let mut best_acquisition = f64::NEG_INFINITY;

        let mut random_search = RandomSearchOptimizer::new(1);

        for _ in 0..n_samples {
            let config = random_search.sample_configuration(search_space);
            let params = self.config_to_vector(&config);
            let acquisition_value = self.compute_acquisition(&params);

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_config = Some(config);
            }
        }

        best_config.ok_or_else(|| crate::ml::MLError::training("Failed to optimize acquisition"))
    }

    /// Compute acquisition function value
    fn compute_acquisition(&self, params: &[f64]) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }

        let (mean, std) = self.predict_gp(params);

        match self.acquisition_function {
            AcquisitionFunction::UCB => mean + self.kappa * std,
            AcquisitionFunction::EI => self.expected_improvement(mean, std),
            AcquisitionFunction::PI => self.probability_of_improvement(mean, std),
        }
    }

    /// Predict mean and std using simple GP (RBF kernel)
    fn predict_gp(&self, params: &[f64]) -> (f64, f64) {
        let mut weights = Vec::new();
        let mut weighted_sum = 0.0;
        let gamma = 0.1; // RBF kernel parameter

        for (obs_params, obs_value) in &self.observations {
            let dist_sq = params
                .iter()
                .zip(obs_params.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();

            let weight = (-gamma * dist_sq).exp();
            weights.push(weight);
            weighted_sum += weight * obs_value;
        }

        let total_weight: f64 = weights.iter().sum();
        let mean = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Simple uncertainty estimate based on distance to observations
        let min_dist = params
            .iter()
            .zip(self.observations[0].0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let std = (1.0 + min_dist).min(10.0);

        (mean, std)
    }

    /// Expected Improvement acquisition
    fn expected_improvement(&self, mean: f64, std: f64) -> f64 {
        if std == 0.0 {
            return 0.0;
        }

        let best_y = self
            .observations
            .iter()
            .map(|(_, y)| *y)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let z = (mean - best_y) / std;
        let ei = std * (z * self.normal_cdf(z) + self.normal_pdf(z));

        ei.max(0.0)
    }

    /// Probability of Improvement acquisition
    fn probability_of_improvement(&self, mean: f64, std: f64) -> f64 {
        if std == 0.0 {
            return 0.0;
        }

        let best_y = self
            .observations
            .iter()
            .map(|(_, y)| *y)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let z = (mean - best_y) / std;
        self.normal_cdf(z)
    }

    /// Normal CDF (simplified approximation)
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
    }

    /// Normal PDF
    fn normal_pdf(&self, x: f64) -> f64 {
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        (-0.5 * x * x).exp() / sqrt_2pi
    }

    /// Convert parameter configuration to vector
    fn config_to_vector(&self, config: &ParameterConfiguration) -> Vec<f64> {
        // Simplified: just extract numeric values
        config
            .params
            .values()
            .filter_map(|v| match v {
                ParameterValue::Float(f) => Some(*f),
                ParameterValue::Integer(i) => Some(*i as f64),
                _ => None,
            })
            .collect()
    }
}

// ================================================================================================
// HyperBand Implementation
// ================================================================================================

/// HyperBand optimizer for efficient resource allocation
pub struct HyperBandOptimizer {
    max_resource: usize,
    eta: f64,
    brackets: Vec<BracketConfig>,
    current_bracket: usize,
}

/// Configuration for a single HyperBand bracket
#[derive(Debug, Clone)]
pub struct BracketConfig {
    pub s: usize,
    pub n: usize,
    pub r: usize,
}

impl HyperBandOptimizer {
    pub fn new(max_resource: usize, eta: f64) -> Self {
        let s_max = (max_resource as f64).log(eta).floor() as usize;
        let mut brackets = Vec::new();

        for s in (0..=s_max).rev() {
            let n = ((s_max + 1) as f64 / (s + 1) as f64 * eta.powi(s as i32)).ceil() as usize;
            let r = max_resource / eta.powi(s as i32) as usize;

            brackets.push(BracketConfig { s, n, r });
        }

        Self {
            max_resource,
            eta,
            brackets,
            current_bracket: 0,
        }
    }

    /// Get next configuration and resource allocation
    pub fn get_next_configuration(&mut self) -> Option<(usize, usize)> {
        if self.current_bracket >= self.brackets.len() {
            return None;
        }

        let bracket = &self.brackets[self.current_bracket];
        Some((bracket.n, bracket.r))
    }

    /// Decide whether to continue training a configuration
    pub fn should_continue_training(
        &self,
        performance: f64,
        current_resource: usize,
        all_performances: &[(usize, f64)],
    ) -> bool {
        // Keep top 1/eta performers
        let threshold_idx = (all_performances.len() as f64 / self.eta).ceil() as usize;

        let mut sorted_perfs: Vec<_> = all_performances.iter().collect();
        sorted_perfs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        sorted_perfs
            .iter()
            .take(threshold_idx)
            .any(|(_, p)| (*p - performance).abs() < 1e-10)
    }

    /// Advance to next bracket
    pub fn next_bracket(&mut self) {
        self.current_bracket += 1;
    }
}

// ================================================================================================
// BOHB Implementation (Bayesian + HyperBand)
// ================================================================================================

/// BOHB optimizer combining Bayesian optimization and HyperBand
pub struct BOHBOptimizer {
    bayesian: BayesianOptimizer,
    hyperband: HyperBandOptimizer,
    min_budget: f64,
    max_budget: f64,
}

impl BOHBOptimizer {
    pub fn new(min_budget: f64, max_budget: f64, acquisition_fn: AcquisitionFunction) -> Self {
        let max_resource = max_budget as usize;
        let eta = 3.0;

        Self {
            bayesian: BayesianOptimizer::new(acquisition_fn, 10),
            hyperband: HyperBandOptimizer::new(max_resource, eta),
            min_budget,
            max_budget,
        }
    }

    /// Get next configuration using Bayesian optimization
    pub fn suggest_next_trial(
        &mut self,
        search_space: &SearchSpace,
    ) -> MLResult<ParameterConfiguration> {
        self.bayesian.suggest_next_trial(search_space)
    }

    /// Update with training result
    pub fn update_with_result(&mut self, params: Vec<f64>, performance: f64) {
        self.bayesian.update_with_result(params, performance);
    }

    /// Get resource allocation from HyperBand
    pub fn get_resource_allocation(&mut self) -> Option<(usize, usize)> {
        self.hyperband.get_next_configuration()
    }
}

// ================================================================================================
// Public API
// ================================================================================================

/// Optimize hyperparameters using specified method
pub fn optimize_hyperparameters<F>(
    search_space: &SearchSpace,
    optimization_method: &OptimizationMethod,
    objective_fn: F,
) -> MLResult<(ParameterConfiguration, f64)>
where
    F: Fn(&ParameterConfiguration) -> MLResult<f64>,
{
    match optimization_method {
        OptimizationMethod::RandomSearch { n_trials } => {
            let mut optimizer = RandomSearchOptimizer::new(*n_trials);
            optimizer.optimize(search_space, objective_fn)
        }

        OptimizationMethod::GridSearch { resolution } => {
            let optimizer = GridSearchOptimizer::new(*resolution);
            optimizer.optimize(search_space, objective_fn)
        }

        OptimizationMethod::BayesianOptimization {
            acquisition_function,
            n_initial_points,
        } => {
            let mut optimizer = BayesianOptimizer::new(*acquisition_function, *n_initial_points);

            // Run optimization loop
            let mut best_config = None;
            let mut best_score = f64::NEG_INFINITY;

            for _ in 0..100 {
                // max iterations
                let config = optimizer.suggest_next_trial(search_space)?;
                let score = objective_fn(&config)?;

                let params = optimizer.config_to_vector(&config);
                optimizer.update_with_result(params, score);

                if score > best_score {
                    best_score = score;
                    best_config = Some(config);
                }
            }

            best_config
                .map(|c| (c, best_score))
                .ok_or_else(|| crate::ml::MLError::training("Bayesian optimization failed"))
        }

        _ => Err(crate::ml::MLError::invalid_input(
            "Optimization method not yet implemented",
        )),
    }
}
