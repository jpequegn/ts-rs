//! Neural Architecture Search Module
//!
//! Implements automated neural architecture discovery:
//! - Random architecture sampling
//! - Evolutionary NAS with genetic algorithms
//! - Progressive NAS with complexity scheduling
//! - Architecture performance estimation

use super::types::*;
use crate::timeseries::TimeSeries;
use crate::ml::MLResult;
use rand::Rng;
use rand::seq::SliceRandom;

// ================================================================================================
// Neural Architecture Search
// ================================================================================================

pub struct NeuralArchitectureSearch {
    pub search_space: ArchitectureSearchSpace,
    pub search_strategy: NASStrategy,
    pub constraints: ArchitectureConstraints,
}

impl NeuralArchitectureSearch {
    pub fn new(
        search_space: ArchitectureSearchSpace,
        search_strategy: NASStrategy,
        constraints: ArchitectureConstraints,
    ) -> Self {
        Self {
            search_space,
            search_strategy,
            constraints,
        }
    }

    /// Search for optimal architecture
    pub fn search_architecture(&mut self, data: &TimeSeries) -> MLResult<OptimalArchitecture> {
        match &self.search_strategy {
            NASStrategy::Random => self.random_search(data, 50),

            NASStrategy::Evolutionary {
                population_size,
                generations,
            } => self.evolutionary_search(data, *population_size, *generations),

            NASStrategy::Progressive { complexity_schedule } => {
                self.progressive_search(data, complexity_schedule)
            }

            _ => Err(crate::ml::MLError::invalid_input(
                "NAS strategy not yet implemented",
            )),
        }
    }

    /// Random architecture search
    fn random_search(&self, _data: &TimeSeries, n_trials: usize) -> MLResult<OptimalArchitecture> {
        let mut rng = rand::thread_rng();
        let mut best_arch = None;
        let mut best_performance = 0.0;

        for _ in 0..n_trials {
            let arch = self.sample_random_architecture(&mut rng)?;

            if !self.satisfies_constraints(&arch) {
                continue;
            }

            let performance = self.estimate_performance(&arch);

            if performance > best_performance {
                best_performance = performance;
                best_arch = Some(arch);
            }
        }

        best_arch
            .map(|arch| self.create_optimal_architecture(arch, best_performance))
            .ok_or_else(|| crate::ml::MLError::training("No valid architecture found"))
    }

    /// Evolutionary NAS with genetic algorithm
    fn evolutionary_search(
        &self,
        _data: &TimeSeries,
        population_size: usize,
        generations: usize,
    ) -> MLResult<OptimalArchitecture> {
        let mut rng = rand::thread_rng();

        // Initialize population
        let mut population: Vec<_> = (0..population_size)
            .filter_map(|_| self.sample_random_architecture(&mut rng).ok())
            .filter(|arch| self.satisfies_constraints(arch))
            .collect();

        if population.is_empty() {
            return Err(crate::ml::MLError::training("Failed to initialize population"));
        }

        for _gen in 0..generations {
            // Evaluate fitness
            let fitness: Vec<_> = population
                .iter()
                .map(|arch| self.estimate_performance(arch))
                .collect();

            // Selection - keep top 50%
            let mut indexed: Vec<_> = population
                .iter()
                .zip(fitness.iter())
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1 .1.partial_cmp(a.1 .1).unwrap());

            let survivors: Vec<_> = indexed
                .iter()
                .take(population_size / 2)
                .map(|(i, _)| population[*i].clone())
                .collect();

            // Crossover and mutation
            population = survivors.clone();

            while population.len() < population_size {
                // Select two parents
                let parent1 = survivors.choose(&mut rng).unwrap();
                let parent2 = survivors.choose(&mut rng).unwrap();

                // Crossover
                if let Ok(child) = self.crossover(parent1, parent2, &mut rng) {
                    // Mutation
                    if let Ok(mutated) = self.mutate(&child, &mut rng, 0.1) {
                        if self.satisfies_constraints(&mutated) {
                            population.push(mutated);
                        }
                    }
                }
            }
        }

        // Return best architecture from final population
        let best_idx = population
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                self.estimate_performance(a)
                    .partial_cmp(&self.estimate_performance(b))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .ok_or_else(|| crate::ml::MLError::training("Population is empty"))?;

        let best_arch = population[best_idx].clone();
        let best_performance = self.estimate_performance(&best_arch);

        Ok(self.create_optimal_architecture(best_arch, best_performance))
    }

    /// Progressive NAS with increasing complexity
    fn progressive_search(
        &self,
        data: &TimeSeries,
        complexity_schedule: &[ComplexityLevel],
    ) -> MLResult<OptimalArchitecture> {
        let mut best_arch = None;
        let mut best_performance = 0.0;

        for level in complexity_schedule {
            // Create constrained search space for this level
            let constrained_space = ArchitectureSearchSpace {
                depth_range: 1..level.max_depth,
                width_range: 16..level.max_width,
                ..self.search_space.clone()
            };

            // Search within this level
            let level_nas = NeuralArchitectureSearch {
                search_space: constrained_space,
                search_strategy: NASStrategy::Random,
                constraints: self.constraints.clone(),
            };

            if let Ok(arch) = level_nas.random_search(data, level.n_trials) {
                if arch.performance > best_performance {
                    best_performance = arch.performance;
                    best_arch = Some(arch);
                }
            }
        }

        best_arch.ok_or_else(|| crate::ml::MLError::training("Progressive search failed"))
    }

    /// Sample random architecture
    fn sample_random_architecture(
        &self,
        rng: &mut rand::rngs::ThreadRng,
    ) -> MLResult<ArchitectureDescription> {
        let depth = rng.gen_range(self.search_space.depth_range.clone());
        let mut layers = Vec::new();

        for _ in 0..depth {
            let layer_type = self
                .search_space
                .layer_types
                .choose(rng)
                .ok_or_else(|| crate::ml::MLError::invalid_input("No layer types available"))?;

            let size = rng.gen_range(self.search_space.width_range.clone());

            let activation = self
                .search_space
                .activation_functions
                .choose(rng)
                .ok_or_else(|| crate::ml::MLError::invalid_input("No activation functions available"))?;

            layers.push(LayerDescription {
                layer_type: *layer_type,
                size,
                activation: *activation,
            });
        }

        // Simple sequential connections
        let mut connections = Vec::new();
        for i in 0..depth.saturating_sub(1) {
            connections.push(ConnectionDescription {
                from_layer: i,
                to_layer: i + 1,
                pattern: ConnectionPattern::Sequential,
            });
        }

        Ok(ArchitectureDescription { layers, connections })
    }

    /// Crossover two architectures
    fn crossover(
        &self,
        parent1: &ArchitectureDescription,
        parent2: &ArchitectureDescription,
        rng: &mut rand::rngs::ThreadRng,
    ) -> MLResult<ArchitectureDescription> {
        let crossover_point = rng.gen_range(0..parent1.layers.len().min(parent2.layers.len()));

        let mut layers = Vec::new();
        layers.extend_from_slice(&parent1.layers[..crossover_point]);
        layers.extend_from_slice(&parent2.layers[crossover_point..]);

        // Rebuild connections
        let mut connections = Vec::new();
        for i in 0..layers.len().saturating_sub(1) {
            connections.push(ConnectionDescription {
                from_layer: i,
                to_layer: i + 1,
                pattern: ConnectionPattern::Sequential,
            });
        }

        Ok(ArchitectureDescription { layers, connections })
    }

    /// Mutate an architecture
    fn mutate(
        &self,
        arch: &ArchitectureDescription,
        rng: &mut rand::rngs::ThreadRng,
        mutation_rate: f64,
    ) -> MLResult<ArchitectureDescription> {
        let mut layers = arch.layers.clone();

        for layer in &mut layers {
            if rng.gen::<f64>() < mutation_rate {
                // Mutate layer size
                layer.size = rng.gen_range(self.search_space.width_range.clone());
            }

            if rng.gen::<f64>() < mutation_rate {
                // Mutate activation
                if let Some(new_activation) = self.search_space.activation_functions.choose(rng) {
                    layer.activation = *new_activation;
                }
            }
        }

        Ok(ArchitectureDescription {
            layers,
            connections: arch.connections.clone(),
        })
    }

    /// Check if architecture satisfies constraints
    fn satisfies_constraints(&self, arch: &ArchitectureDescription) -> bool {
        let n_params = self.estimate_parameters(arch);
        let memory_gb = self.estimate_memory(arch);

        n_params <= self.constraints.max_parameters
            && memory_gb <= self.constraints.max_memory_gb
    }

    /// Estimate number of parameters
    fn estimate_parameters(&self, arch: &ArchitectureDescription) -> usize {
        let mut total = 0;
        for i in 0..arch.layers.len().saturating_sub(1) {
            let input_size = arch.layers[i].size;
            let output_size = arch.layers[i + 1].size;
            total += input_size * output_size + output_size; // weights + biases
        }
        total
    }

    /// Estimate memory usage in GB
    fn estimate_memory(&self, arch: &ArchitectureDescription) -> f64 {
        let n_params = self.estimate_parameters(arch);
        // 4 bytes per parameter (float32) + overhead
        (n_params * 4) as f64 / 1e9 * 1.5
    }

    /// Estimate performance (simplified heuristic)
    fn estimate_performance(&self, arch: &ArchitectureDescription) -> f64 {
        // Simple heuristic: favor moderate depth and width
        let depth = arch.layers.len() as f64;
        let avg_width = arch.layers.iter().map(|l| l.size as f64).sum::<f64>() / depth;

        let depth_score = 1.0 - ((depth - 5.0).abs() / 10.0).min(1.0);
        let width_score = 1.0 - ((avg_width - 64.0).abs() / 128.0).min(1.0);

        (depth_score + width_score) / 2.0
    }

    /// Create optimal architecture result
    fn create_optimal_architecture(
        &self,
        architecture: ArchitectureDescription,
        performance: f64,
    ) -> OptimalArchitecture {
        // Calculate values before moving architecture
        let n_parameters = self.estimate_parameters(&architecture);
        let memory_gb = self.estimate_memory(&architecture);

        OptimalArchitecture {
            architecture,
            performance,
            n_parameters,
            memory_gb,
        }
    }
}

// ================================================================================================
// Public API
// ================================================================================================

/// Search for optimal neural architecture
pub fn search_neural_architecture(
    data: &TimeSeries,
    constraints: &ArchitectureConstraints,
    search_strategy: NASStrategy,
) -> MLResult<OptimalArchitecture> {
    let search_space = create_default_search_space();

    let mut nas = NeuralArchitectureSearch::new(search_space, search_strategy, constraints.clone());

    nas.search_architecture(data)
}

/// Create default architecture search space
fn create_default_search_space() -> ArchitectureSearchSpace {
    ArchitectureSearchSpace {
        layer_types: vec![LayerType::LSTM, LayerType::GRU, LayerType::FeedForward],
        layer_sizes: vec![16..256],
        connection_patterns: vec![ConnectionPattern::Sequential, ConnectionPattern::Residual],
        activation_functions: vec![ActivationType::ReLU, ActivationType::Tanh, ActivationType::GELU],
        depth_range: 1..10,
        width_range: 16..256,
    }
}
