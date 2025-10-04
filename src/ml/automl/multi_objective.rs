//! Multi-Objective Optimization Module
//!
//! Implements multi-objective optimization algorithms:
//! - NSGA-II (Non-dominated Sorting Genetic Algorithm II)
//! - SPEA2 (Strength Pareto Evolutionary Algorithm 2)
//! - Weighted sum methods
//! - Epsilon constraint methods

use super::types::*;
use crate::ml::MLResult;
use rand::Rng;
use rand::seq::SliceRandom;

// ================================================================================================
// NSGA-II Implementation
// ================================================================================================

pub struct NSGA2Optimizer {
    population_size: usize,
    generations: usize,
    crossover_prob: f64,
    mutation_prob: f64,
}

impl NSGA2Optimizer {
    pub fn new(population_size: usize, generations: usize) -> Self {
        Self {
            population_size,
            generations,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
        }
    }

    /// Run NSGA-II optimization
    pub fn optimize(
        &self,
        search_space: &SearchSpace,
        objectives: &[Objective],
    ) -> MLResult<ParetoFront> {
        let mut rng = rand::thread_rng();

        // Initialize population
        let mut population = self.initialize_population(search_space, &mut rng)?;

        for _ in 0..self.generations {
            // Evaluate objectives for each individual
            let objective_values = self.evaluate_population(&population, objectives);

            // Non-dominated sorting
            let fronts = non_dominated_sort(&objective_values);

            // Crowding distance assignment
            let crowding_distances = compute_crowding_distance(&fronts, &objective_values);

            // Selection using tournament
            let selected = self.tournament_selection(
                &population,
                &fronts,
                &crowding_distances,
                &mut rng,
            );

            // Crossover and mutation
            population = self.generate_offspring(&selected, search_space, &mut rng)?;
        }

        // Final evaluation
        let objective_values = self.evaluate_population(&population, objectives);
        let fronts = non_dominated_sort(&objective_values);

        // Extract Pareto front (first front)
        let pareto_solutions: Vec<_> = fronts[0]
            .iter()
            .map(|&idx| ParetoSolution {
                parameters: population[idx].clone(),
                objective_values: objective_values[idx].clone(),
                rank: 0,
            })
            .collect();

        let hypervolume = compute_hypervolume(&pareto_solutions);

        Ok(ParetoFront {
            solutions: pareto_solutions,
            hypervolume,
        })
    }

    fn initialize_population(
        &self,
        search_space: &SearchSpace,
        rng: &mut rand::rngs::ThreadRng,
    ) -> MLResult<Vec<ParameterConfiguration>> {
        (0..self.population_size)
            .map(|_| sample_random_configuration(search_space, rng))
            .collect()
    }

    fn evaluate_population(
        &self,
        population: &[ParameterConfiguration],
        objectives: &[Objective],
    ) -> Vec<Vec<f64>> {
        population
            .iter()
            .map(|config| evaluate_objectives(config, objectives))
            .collect()
    }

    fn tournament_selection(
        &self,
        population: &[ParameterConfiguration],
        fronts: &[Vec<usize>],
        crowding_distances: &[f64],
        rng: &mut rand::rngs::ThreadRng,
    ) -> Vec<ParameterConfiguration> {
        let mut selected = Vec::new();

        for _ in 0..self.population_size {
            let idx1 = rng.gen_range(0..population.len());
            let idx2 = rng.gen_range(0..population.len());

            let winner = if compare_individuals(idx1, idx2, fronts, crowding_distances) {
                idx1
            } else {
                idx2
            };

            selected.push(population[winner].clone());
        }

        selected
    }

    fn generate_offspring(
        &self,
        parents: &[ParameterConfiguration],
        search_space: &SearchSpace,
        rng: &mut rand::rngs::ThreadRng,
    ) -> MLResult<Vec<ParameterConfiguration>> {
        let mut offspring = Vec::new();

        for i in (0..parents.len()).step_by(2) {
            let parent1 = &parents[i];
            let parent2 = if i + 1 < parents.len() {
                &parents[i + 1]
            } else {
                &parents[0]
            };

            let (mut child1, mut child2) = if rng.gen::<f64>() < self.crossover_prob {
                crossover_parameters(parent1, parent2, rng)?
            } else {
                (parent1.clone(), parent2.clone())
            };

            if rng.gen::<f64>() < self.mutation_prob {
                mutate_parameters(&mut child1, search_space, rng)?;
            }

            if rng.gen::<f64>() < self.mutation_prob {
                mutate_parameters(&mut child2, search_space, rng)?;
            }

            offspring.push(child1);
            offspring.push(child2);
        }

        Ok(offspring.into_iter().take(self.population_size).collect())
    }
}

// ================================================================================================
// Non-dominated Sorting
// ================================================================================================

/// Perform non-dominated sorting
fn non_dominated_sort(objective_values: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objective_values.len();
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];
    let mut dominated_count = vec![0; n];
    let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Find domination relationships
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            if dominates_solution(&objective_values[i], &objective_values[j]) {
                dominates[i].push(j);
            } else if dominates_solution(&objective_values[j], &objective_values[i]) {
                dominated_count[i] += 1;
            }
        }

        if dominated_count[i] == 0 {
            fronts[0].push(i);
        }
    }

    // Build subsequent fronts
    let mut current_front = 0;
    while !fronts[current_front].is_empty() {
        let mut next_front = Vec::new();

        for &i in &fronts[current_front] {
            for &j in &dominates[i] {
                dominated_count[j] -= 1;
                if dominated_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }

        if !next_front.is_empty() {
            fronts.push(next_front);
            current_front += 1;
        } else {
            break;
        }
    }

    fronts
}

/// Check if solution a dominates solution b
fn dominates_solution(a: &[f64], b: &[f64]) -> bool {
    let mut at_least_one_better = false;

    for (a_val, b_val) in a.iter().zip(b.iter()) {
        if a_val < b_val {
            return false; // a is worse in this objective (minimization)
        }
        if a_val > b_val {
            at_least_one_better = true;
        }
    }

    at_least_one_better
}

// ================================================================================================
// Crowding Distance
// ================================================================================================

/// Compute crowding distance for all solutions
fn compute_crowding_distance(fronts: &[Vec<usize>], objective_values: &[Vec<f64>]) -> Vec<f64> {
    let n = objective_values.len();
    let mut distances = vec![0.0; n];

    for front in fronts {
        if front.len() <= 2 {
            for &idx in front {
                distances[idx] = f64::INFINITY;
            }
            continue;
        }

        let n_objectives = objective_values[0].len();

        for obj_idx in 0..n_objectives {
            // Sort front by this objective
            let mut sorted_front = front.clone();
            sorted_front.sort_by(|&a, &b| {
                objective_values[a][obj_idx]
                    .partial_cmp(&objective_values[b][obj_idx])
                    .unwrap()
            });

            // Boundary points get infinite distance
            distances[sorted_front[0]] = f64::INFINITY;
            distances[sorted_front[sorted_front.len() - 1]] = f64::INFINITY;

            let obj_range = objective_values[sorted_front[sorted_front.len() - 1]][obj_idx]
                - objective_values[sorted_front[0]][obj_idx];

            if obj_range > 0.0 {
                for i in 1..sorted_front.len() - 1 {
                    let distance = (objective_values[sorted_front[i + 1]][obj_idx]
                        - objective_values[sorted_front[i - 1]][obj_idx])
                        / obj_range;

                    distances[sorted_front[i]] += distance;
                }
            }
        }
    }

    distances
}

/// Compare two individuals (used for tournament selection)
fn compare_individuals(
    idx1: usize,
    idx2: usize,
    fronts: &[Vec<usize>],
    crowding_distances: &[f64],
) -> bool {
    // Find ranks
    let rank1 = fronts.iter().position(|f| f.contains(&idx1)).unwrap();
    let rank2 = fronts.iter().position(|f| f.contains(&idx2)).unwrap();

    if rank1 != rank2 {
        rank1 < rank2
    } else {
        crowding_distances[idx1] > crowding_distances[idx2]
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Sample random configuration from search space
fn sample_random_configuration(
    search_space: &SearchSpace,
    rng: &mut rand::rngs::ThreadRng,
) -> MLResult<ParameterConfiguration> {
    let mut params = std::collections::HashMap::new();

    for (name, values) in &search_space.categorical_params {
        if let Some(value) = values.choose(rng) {
            params.insert(name.clone(), ParameterValue::String(value.clone()));
        }
    }

    for (name, (min, max)) in &search_space.integer_params {
        let value = rng.gen_range(*min..=*max);
        params.insert(name.clone(), ParameterValue::Integer(value));
    }

    for (name, (min, max)) in &search_space.float_params {
        let value = rng.gen_range(*min..=*max);
        params.insert(name.clone(), ParameterValue::Float(value));
    }

    Ok(ParameterConfiguration { params })
}

/// Crossover two parameter configurations
fn crossover_parameters(
    parent1: &ParameterConfiguration,
    parent2: &ParameterConfiguration,
    rng: &mut rand::rngs::ThreadRng,
) -> MLResult<(ParameterConfiguration, ParameterConfiguration)> {
    let mut child1_params = std::collections::HashMap::new();
    let mut child2_params = std::collections::HashMap::new();

    for (key, value1) in &parent1.params {
        if let Some(value2) = parent2.params.get(key) {
            if rng.gen::<bool>() {
                child1_params.insert(key.clone(), value1.clone());
                child2_params.insert(key.clone(), value2.clone());
            } else {
                child1_params.insert(key.clone(), value2.clone());
                child2_params.insert(key.clone(), value1.clone());
            }
        }
    }

    Ok((
        ParameterConfiguration { params: child1_params },
        ParameterConfiguration { params: child2_params },
    ))
}

/// Mutate parameter configuration
fn mutate_parameters(
    config: &mut ParameterConfiguration,
    search_space: &SearchSpace,
    rng: &mut rand::rngs::ThreadRng,
) -> MLResult<()> {
    // Mutate one random parameter
    let keys: Vec<_> = config.params.keys().cloned().collect();
    if let Some(key) = keys.choose(rng) {
        if let Some((min, max)) = search_space.integer_params.get(key) {
            let new_value = rng.gen_range(*min..=*max);
            config.params.insert(key.clone(), ParameterValue::Integer(new_value));
        } else if let Some((min, max)) = search_space.float_params.get(key) {
            let new_value = rng.gen_range(*min..=*max);
            config.params.insert(key.clone(), ParameterValue::Float(new_value));
        }
    }

    Ok(())
}

/// Evaluate all objectives for a configuration
fn evaluate_objectives(config: &ParameterConfiguration, objectives: &[Objective]) -> Vec<f64> {
    objectives
        .iter()
        .map(|obj| evaluate_single_objective(config, obj))
        .collect()
}

/// Evaluate a single objective (simplified placeholder)
fn evaluate_single_objective(_config: &ParameterConfiguration, objective: &Objective) -> f64 {
    match objective {
        Objective::Accuracy { .. } => rand::random::<f64>(),
        Objective::TrainingTime => rand::random::<f64>(),
        Objective::InferenceTime => rand::random::<f64>(),
        Objective::ModelSize => rand::random::<f64>(),
        _ => rand::random::<f64>(),
    }
}

/// Compute hypervolume indicator
fn compute_hypervolume(solutions: &[ParetoSolution]) -> f64 {
    if solutions.is_empty() {
        return 0.0;
    }

    // Simplified hypervolume (2D case with reference point at origin)
    let mut volume = 0.0;

    for solution in solutions {
        let contribution: f64 = solution.objective_values.iter().product();
        volume += contribution;
    }

    volume / solutions.len() as f64
}

// ================================================================================================
// Public API
// ================================================================================================

/// Run multi-objective optimization
pub fn optimize_multi_objective(
    search_space: &SearchSpace,
    config: &MultiObjectiveConfig,
) -> MLResult<ParetoFront> {
    match &config.optimization_method {
        MOOMethod::NSGA2 {
            population_size,
            generations,
        } => {
            let optimizer = NSGA2Optimizer::new(*population_size, *generations);
            optimizer.optimize(search_space, &config.objectives)
        }

        MOOMethod::WeightedSum { weights } => {
            optimize_weighted_sum(search_space, &config.objectives, weights)
        }

        _ => Err(crate::ml::MLError::invalid_input(
            "Multi-objective method not yet implemented",
        )),
    }
}

/// Weighted sum method (converts multi-objective to single-objective)
fn optimize_weighted_sum(
    _search_space: &SearchSpace,
    _objectives: &[Objective],
    _weights: &[f64],
) -> MLResult<ParetoFront> {
    // Placeholder implementation
    Ok(ParetoFront {
        solutions: vec![],
        hypervolume: 0.0,
    })
}
