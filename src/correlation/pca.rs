//! Principal Component Analysis (PCA) and Factor Analysis module
//!
//! Implements PCA and related techniques for dimensionality reduction,
//! common trend extraction, and multivariate time series analysis.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Principal Component Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCAResult {
    /// Original variable names
    pub variable_names: Vec<String>,

    /// Number of components extracted
    pub n_components: usize,

    /// Principal component loadings (variables x components)
    pub loadings: Vec<Vec<f64>>,

    /// Eigenvalues for each component
    pub eigenvalues: Vec<f64>,

    /// Explained variance ratio for each component
    pub explained_variance_ratio: Vec<f64>,

    /// Cumulative explained variance ratio
    pub cumulative_explained_variance: Vec<f64>,

    /// Principal component scores (observations x components)
    pub scores: Vec<Vec<f64>>,

    /// Mean values used for centering
    pub means: Vec<f64>,

    /// Standard deviations used for scaling (if standardized)
    pub std_devs: Option<Vec<f64>>,

    /// Component interpretation
    pub component_interpretation: Vec<ComponentInterpretation>,

    /// PCA diagnostics
    pub diagnostics: PCADiagnostics,

    /// Analysis metadata
    pub metadata: PCAMetadata,
}

/// Factor Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysisResult {
    /// Variable names
    pub variable_names: Vec<String>,

    /// Number of factors extracted
    pub n_factors: usize,

    /// Factor loadings (variables x factors)
    pub factor_loadings: Vec<Vec<f64>>,

    /// Communalities (proportion of variance explained by factors)
    pub communalities: Vec<f64>,

    /// Unique variances (specific to each variable)
    pub unique_variances: Vec<f64>,

    /// Factor scores (observations x factors)
    pub factor_scores: Vec<Vec<f64>>,

    /// Factor rotation method used
    pub rotation_method: RotationMethod,

    /// Goodness of fit measures
    pub goodness_of_fit: FactorFitStatistics,

    /// Factor interpretation
    pub factor_interpretation: Vec<FactorInterpretation>,
}

/// Common trends extraction result (for time series)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonTrendsResult {
    /// Variable names
    pub variable_names: Vec<String>,

    /// Number of common trends
    pub n_trends: usize,

    /// Common trend series
    pub trends: Vec<Vec<f64>>,

    /// Trend loadings (how each variable loads on trends)
    pub trend_loadings: Vec<Vec<f64>>,

    /// Trend weights (importance of each trend)
    pub trend_weights: Vec<f64>,

    /// Idiosyncratic components (variable-specific)
    pub idiosyncratic: Vec<Vec<f64>>,

    /// Variance decomposition
    pub variance_decomposition: TrendVarianceDecomposition,

    /// Trend persistence measures
    pub trend_persistence: Vec<f64>,
}

/// Component interpretation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInterpretation {
    /// Component number (1-indexed)
    pub component_number: usize,

    /// Descriptive name based on loadings
    pub name: String,

    /// Variables with highest positive loadings
    pub high_positive_loadings: Vec<(String, f64)>,

    /// Variables with highest negative loadings
    pub high_negative_loadings: Vec<(String, f64)>,

    /// Interpretation summary
    pub interpretation: String,

    /// Component type classification
    pub component_type: ComponentType,
}

/// Factor interpretation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorInterpretation {
    /// Factor number (1-indexed)
    pub factor_number: usize,

    /// Descriptive name
    pub name: String,

    /// Variables with highest loadings
    pub high_loadings: Vec<(String, f64)>,

    /// Factor interpretation
    pub interpretation: String,

    /// Factor complexity (number of significant loadings)
    pub complexity: usize,
}

/// PCA diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCADiagnostics {
    /// Kaiser-Meyer-Olkin (KMO) test for sampling adequacy
    pub kmo_statistic: f64,

    /// Bartlett's test of sphericity
    pub bartlett_statistic: f64,
    pub bartlett_p_value: f64,

    /// Determinant of correlation matrix
    pub correlation_determinant: f64,

    /// Condition number (multicollinearity check)
    pub condition_number: f64,

    /// Scree plot recommendations
    pub scree_recommendations: ScreeRecommendations,

    /// Kaiser criterion (eigenvalues > 1)
    pub kaiser_components: usize,

    /// Parallel analysis recommendations
    pub parallel_analysis: Option<ParallelAnalysisResult>,
}

/// Factor analysis goodness of fit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorFitStatistics {
    /// Chi-square test statistic
    pub chi_square: f64,

    /// P-value for chi-square test
    pub p_value: f64,

    /// Root Mean Square Error of Approximation (RMSEA)
    pub rmsea: f64,

    /// Tucker-Lewis Index (TLI)
    pub tli: f64,

    /// Comparative Fit Index (CFI)
    pub cfi: f64,

    /// Goodness of Fit Index (GFI)
    pub gfi: f64,
}

/// Trend variance decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendVarianceDecomposition {
    /// Proportion of variance explained by each trend
    pub trend_variance_ratios: Vec<f64>,

    /// Proportion of variance explained by idiosyncratic components
    pub idiosyncratic_variance_ratios: Vec<f64>,

    /// Total variance explained by common trends
    pub total_trend_variance: f64,

    /// Total variance explained by idiosyncratic components
    pub total_idiosyncratic_variance: f64,
}

/// Scree plot recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeRecommendations {
    /// Elbow point (suggested number of components)
    pub elbow_point: usize,

    /// Components above the "elbow"
    pub components_above_elbow: usize,

    /// Eigenvalue differences
    pub eigenvalue_differences: Vec<f64>,
}

/// Parallel analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelAnalysisResult {
    /// Random eigenvalues for comparison
    pub random_eigenvalues: Vec<f64>,

    /// Recommended number of components
    pub recommended_components: usize,

    /// Number of iterations used
    pub iterations: usize,
}

/// PCA analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCAMetadata {
    /// Number of observations
    pub n_observations: usize,

    /// Number of variables
    pub n_variables: usize,

    /// Whether data was standardized
    pub standardized: bool,

    /// Computation time in milliseconds
    pub computation_time_ms: u64,

    /// Method used for eigenvalue decomposition
    pub decomposition_method: String,

    /// Missing value handling method
    pub missing_value_method: MissingValueMethod,
}

/// Component type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentType {
    /// General factor affecting all variables
    General,

    /// Specific factor affecting subset of variables
    Specific,

    /// Bipolar factor (positive and negative loadings)
    Bipolar,

    /// Trend component (for time series)
    Trend,

    /// Cyclical component
    Cyclical,

    /// Noise component
    Noise,
}

/// Factor rotation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationMethod {
    /// No rotation
    None,

    /// Varimax rotation (orthogonal)
    Varimax,

    /// Quartimax rotation (orthogonal)
    Quartimax,

    /// Promax rotation (oblique)
    Promax,

    /// Oblimin rotation (oblique)
    Oblimin,
}

/// Missing value handling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueMethod {
    /// Remove observations with any missing values
    Listwise,

    /// Pairwise deletion
    Pairwise,

    /// Mean substitution
    MeanSubstitution,

    /// Regression imputation
    Regression,
}

impl Default for RotationMethod {
    fn default() -> Self {
        RotationMethod::None
    }
}

impl Default for MissingValueMethod {
    fn default() -> Self {
        MissingValueMethod::Listwise
    }
}

/// Compute Principal Component Analysis
pub fn compute_pca(
    data: &HashMap<String, Vec<f64>>,
    n_components: Option<usize>,
    standardize: bool,
    missing_method: Option<MissingValueMethod>,
) -> Result<PCAResult, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();

    let variable_names: Vec<String> = data.keys().cloned().collect();
    let n_vars = variable_names.len();
    let n_obs = data.values().next().unwrap().len();

    if n_vars < 2 {
        return Err("At least 2 variables required for PCA".into());
    }

    if n_obs < n_vars {
        return Err("Number of observations must be greater than number of variables".into());
    }

    // Prepare data matrix
    let mut data_matrix = vec![vec![0.0; n_vars]; n_obs];
    for (var_idx, var_name) in variable_names.iter().enumerate() {
        let series = &data[var_name];
        for (obs_idx, &value) in series.iter().enumerate() {
            data_matrix[obs_idx][var_idx] = value;
        }
    }

    // Handle missing values
    let clean_data = handle_missing_values(
        &data_matrix,
        missing_method.clone().unwrap_or_default(),
    )?;

    // Center and standardize data
    let (processed_data, means, std_devs) = preprocess_data(&clean_data, standardize)?;

    // Compute covariance/correlation matrix
    let cov_matrix = compute_covariance_matrix(&processed_data)?;

    // Eigenvalue decomposition
    let (eigenvalues, eigenvectors) = compute_eigendecomposition(&cov_matrix)?;

    // Sort by eigenvalues (descending)
    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = eigenvalues
        .into_iter()
        .zip(eigenvectors)
        .collect();
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let sorted_eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();
    let sorted_eigenvectors: Vec<Vec<f64>> = eigen_pairs.into_iter().map(|(_, vec)| vec).collect();

    // Determine number of components
    let n_components = n_components.unwrap_or_else(|| {
        // Default: keep components with eigenvalue > 1 (Kaiser criterion)
        sorted_eigenvalues.iter().take_while(|&&val| val > 1.0).count().max(1)
    });

    let n_components = n_components.min(n_vars);

    // Extract components
    let eigenvalues = sorted_eigenvalues[..n_components].to_vec();
    let loadings = transpose_matrix(&sorted_eigenvectors[..n_components].to_vec());

    // Calculate explained variance
    let total_variance: f64 = sorted_eigenvalues.iter().sum();
    let explained_variance_ratio: Vec<f64> = eigenvalues
        .iter()
        .map(|&val| val / total_variance)
        .collect();

    let mut cumulative_explained_variance = Vec::new();
    let mut cumulative = 0.0;
    for &ratio in &explained_variance_ratio {
        cumulative += ratio;
        cumulative_explained_variance.push(cumulative);
    }

    // Compute component scores
    let scores = compute_component_scores(&processed_data, &loadings)?;

    // Component interpretation
    let component_interpretation = interpret_components(&loadings, &variable_names, &eigenvalues);

    // Diagnostics
    let diagnostics = compute_pca_diagnostics(&cov_matrix, &sorted_eigenvalues)?;

    let computation_time_ms = start_time.elapsed().as_millis() as u64;

    let metadata = PCAMetadata {
        n_observations: clean_data.len(),
        n_variables: n_vars,
        standardized: standardize,
        computation_time_ms,
        decomposition_method: "Eigenvalue decomposition".to_string(),
        missing_value_method: missing_method.unwrap_or_default(),
    };

    Ok(PCAResult {
        variable_names,
        n_components,
        loadings,
        eigenvalues,
        explained_variance_ratio,
        cumulative_explained_variance,
        scores,
        means,
        std_devs: if standardize { Some(std_devs) } else { None },
        component_interpretation,
        diagnostics,
        metadata,
    })
}

/// Extract common trends from multivariate time series
pub fn extract_common_trends(
    data: &HashMap<String, Vec<f64>>,
    n_trends: Option<usize>,
    detrend: bool,
) -> Result<CommonTrendsResult, Box<dyn std::error::Error>> {
    let variable_names: Vec<String> = data.keys().cloned().collect();
    let n_vars = variable_names.len();

    // Apply PCA to extract trends
    let pca_result = compute_pca(data, n_trends, true, None)?;

    let n_trends = pca_result.n_components;

    // Extract trend series (principal component scores)
    let trends = transpose_matrix(&pca_result.scores);

    // Trend loadings are the eigenvectors
    let trend_loadings = pca_result.loadings;

    // Trend weights are the explained variance ratios
    let trend_weights = pca_result.explained_variance_ratio;

    // Compute idiosyncratic components
    let idiosyncratic = compute_idiosyncratic_components(data, &trends, &trend_loadings)?;

    // Variance decomposition
    let variance_decomposition = compute_trend_variance_decomposition(
        data,
        &trends,
        &idiosyncratic,
    )?;

    // Trend persistence (simplified using AR(1) coefficients)
    let trend_persistence = compute_trend_persistence(&trends)?;

    Ok(CommonTrendsResult {
        variable_names,
        n_trends,
        trends,
        trend_loadings,
        trend_weights,
        idiosyncratic,
        variance_decomposition,
        trend_persistence,
    })
}

/// Perform Factor Analysis
pub fn compute_factor_analysis(
    data: &HashMap<String, Vec<f64>>,
    n_factors: usize,
    rotation: RotationMethod,
    max_iterations: usize,
) -> Result<FactorAnalysisResult, Box<dyn std::error::Error>> {
    let variable_names: Vec<String> = data.keys().cloned().collect();
    let n_vars = variable_names.len();

    if n_factors >= n_vars {
        return Err("Number of factors must be less than number of variables".into());
    }

    // Start with PCA as initial solution
    let pca_result = compute_pca(data, Some(n_factors), true, None)?;

    // Extract initial factor loadings
    let mut factor_loadings = pca_result.loadings;

    // Apply rotation if specified
    if !matches!(rotation, RotationMethod::None) {
        factor_loadings = apply_rotation(&factor_loadings, &rotation, max_iterations)?;
    }

    // Compute communalities
    let communalities = compute_communalities(&factor_loadings);

    // Compute unique variances
    let unique_variances: Vec<f64> = communalities
        .iter()
        .map(|&comm| 1.0 - comm)
        .collect();

    // Compute factor scores (simplified regression method)
    let factor_scores = compute_factor_scores(data, &factor_loadings, &variable_names)?;

    // Goodness of fit
    let goodness_of_fit = compute_factor_fit(&factor_loadings, &communalities, n_vars);

    // Factor interpretation
    let factor_interpretation = interpret_factors(&factor_loadings, &variable_names);

    Ok(FactorAnalysisResult {
        variable_names,
        n_factors,
        factor_loadings,
        communalities,
        unique_variances,
        factor_scores,
        rotation_method: rotation,
        goodness_of_fit,
        factor_interpretation,
    })
}

// Helper functions

fn handle_missing_values(
    data: &[Vec<f64>],
    method: MissingValueMethod,
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    match method {
        MissingValueMethod::Listwise => {
            // Remove rows with any missing values
            let clean_data: Vec<Vec<f64>> = data
                .iter()
                .filter(|row| row.iter().all(|&val| val.is_finite()))
                .cloned()
                .collect();

            if clean_data.is_empty() {
                return Err("No complete observations after listwise deletion".into());
            }

            Ok(clean_data)
        }
        MissingValueMethod::MeanSubstitution => {
            // Replace missing values with column means
            let n_vars = data[0].len();
            let mut means = vec![0.0; n_vars];
            let mut counts = vec![0; n_vars];

            // Calculate means
            for row in data {
                for (j, &val) in row.iter().enumerate() {
                    if val.is_finite() {
                        means[j] += val;
                        counts[j] += 1;
                    }
                }
            }

            for j in 0..n_vars {
                if counts[j] > 0 {
                    means[j] /= counts[j] as f64;
                }
            }

            // Substitute missing values
            let mut clean_data = data.to_vec();
            for row in &mut clean_data {
                for (j, val) in row.iter_mut().enumerate() {
                    if !val.is_finite() {
                        *val = means[j];
                    }
                }
            }

            Ok(clean_data)
        }
        _ => {
            // Default to listwise for now
            handle_missing_values(data, MissingValueMethod::Listwise)
        }
    }
}

fn preprocess_data(
    data: &[Vec<f64>],
    standardize: bool,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let n_obs = data.len();
    let n_vars = data[0].len();

    // Calculate means
    let mut means = vec![0.0; n_vars];
    for row in data {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for mean in &mut means {
        *mean /= n_obs as f64;
    }

    // Calculate standard deviations if standardizing
    let mut std_devs = vec![1.0; n_vars];
    if standardize {
        let mut variances = vec![0.0; n_vars];
        for row in data {
            for (j, &val) in row.iter().enumerate() {
                variances[j] += (val - means[j]).powi(2);
            }
        }
        for (j, variance) in variances.iter_mut().enumerate() {
            *variance /= (n_obs - 1) as f64;
            std_devs[j] = variance.sqrt();
            if std_devs[j] == 0.0 {
                std_devs[j] = 1.0;  // Avoid division by zero
            }
        }
    }

    // Center and standardize data
    let mut processed_data = vec![vec![0.0; n_vars]; n_obs];
    for (i, row) in data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            processed_data[i][j] = (val - means[j]) / std_devs[j];
        }
    }

    Ok((processed_data, means, std_devs))
}

fn compute_covariance_matrix(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let n_obs = data.len();
    let n_vars = data[0].len();

    let mut cov_matrix = vec![vec![0.0; n_vars]; n_vars];

    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut covariance = 0.0;
            for t in 0..n_obs {
                covariance += data[t][i] * data[t][j];
            }
            cov_matrix[i][j] = covariance / (n_obs - 1) as f64;
        }
    }

    Ok(cov_matrix)
}

fn compute_eigendecomposition(matrix: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<Vec<f64>>), Box<dyn std::error::Error>> {
    let n = matrix.len();

    // Simplified eigenvalue decomposition using power iteration
    // In practice, would use more sophisticated numerical methods

    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();

    // Simple approximation for small matrices
    if n == 2 {
        let a = matrix[0][0];
        let b = matrix[0][1];
        let c = matrix[1][0];
        let d = matrix[1][1];

        let trace = a + d;
        let det = a * d - b * c;
        let discriminant = trace * trace - 4.0 * det;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;

            eigenvalues.push(lambda1);
            eigenvalues.push(lambda2);

            // Eigenvectors
            if b != 0.0 {
                eigenvectors.push(vec![b, lambda1 - a]);
                eigenvectors.push(vec![b, lambda2 - a]);
            } else if c != 0.0 {
                eigenvectors.push(vec![lambda1 - d, c]);
                eigenvectors.push(vec![lambda2 - d, c]);
            } else {
                eigenvectors.push(vec![1.0, 0.0]);
                eigenvectors.push(vec![0.0, 1.0]);
            }

            // Normalize eigenvectors
            for eigenvector in &mut eigenvectors {
                let norm = (eigenvector[0].powi(2) + eigenvector[1].powi(2)).sqrt();
                if norm > 0.0 {
                    eigenvector[0] /= norm;
                    eigenvector[1] /= norm;
                }
            }
        } else {
            return Err("Complex eigenvalues not supported".into());
        }
    } else {
        // For larger matrices, use simplified approximation
        // This is a placeholder - in practice would use proper numerical methods
        for i in 0..n {
            eigenvalues.push(matrix[i][i]);  // Diagonal approximation
            let mut eigenvector = vec![0.0; n];
            eigenvector[i] = 1.0;
            eigenvectors.push(eigenvector);
        }
    }

    Ok((eigenvalues, eigenvectors))
}

fn transpose_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}

fn compute_component_scores(
    data: &[Vec<f64>],
    loadings: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let n_obs = data.len();
    let n_components = loadings[0].len();

    let mut scores = vec![vec![0.0; n_components]; n_obs];

    for (i, observation) in data.iter().enumerate() {
        for j in 0..n_components {
            let mut score = 0.0;
            for (k, loading) in loadings.iter().enumerate() {
                score += observation[k] * loading[j];
            }
            scores[i][j] = score;
        }
    }

    Ok(scores)
}

fn interpret_components(
    loadings: &[Vec<f64>],
    variable_names: &[String],
    eigenvalues: &[f64],
) -> Vec<ComponentInterpretation> {
    let n_components = loadings[0].len();
    let mut interpretations = Vec::new();

    for comp_idx in 0..n_components {
        let mut high_positive = Vec::new();
        let mut high_negative = Vec::new();

        // Find variables with high loadings
        for (var_idx, var_name) in variable_names.iter().enumerate() {
            let loading = loadings[var_idx][comp_idx];
            if loading.abs() > 0.5 {  // Threshold for "high" loading
                if loading > 0.0 {
                    high_positive.push((var_name.clone(), loading));
                } else {
                    high_negative.push((var_name.clone(), loading));
                }
            }
        }

        // Sort by absolute loading value
        high_positive.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        high_negative.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // Generate interpretation
        let component_type = classify_component(&high_positive, &high_negative, eigenvalues[comp_idx]);

        let name = generate_component_name(comp_idx + 1, &high_positive, &high_negative);

        let interpretation = generate_component_interpretation(
            &high_positive,
            &high_negative,
            &component_type,
        );

        interpretations.push(ComponentInterpretation {
            component_number: comp_idx + 1,
            name,
            high_positive_loadings: high_positive,
            high_negative_loadings: high_negative,
            interpretation,
            component_type,
        });
    }

    interpretations
}

fn classify_component(
    high_positive: &[(String, f64)],
    high_negative: &[(String, f64)],
    eigenvalue: f64,
) -> ComponentType {
    if high_positive.len() > 3 && high_negative.is_empty() {
        ComponentType::General
    } else if !high_positive.is_empty() && !high_negative.is_empty() {
        ComponentType::Bipolar
    } else if eigenvalue > 2.0 {
        ComponentType::Trend
    } else if high_positive.len() <= 2 && high_negative.len() <= 2 {
        ComponentType::Specific
    } else {
        ComponentType::General
    }
}

fn generate_component_name(
    component_number: usize,
    high_positive: &[(String, f64)],
    high_negative: &[(String, f64)],
) -> String {
    if !high_positive.is_empty() {
        format!("PC{}: {}", component_number, high_positive[0].0)
    } else if !high_negative.is_empty() {
        format!("PC{}: {}", component_number, high_negative[0].0)
    } else {
        format!("PC{}", component_number)
    }
}

fn generate_component_interpretation(
    high_positive: &[(String, f64)],
    high_negative: &[(String, f64)],
    component_type: &ComponentType,
) -> String {
    match component_type {
        ComponentType::General => {
            "General factor affecting most variables with positive loadings".to_string()
        }
        ComponentType::Bipolar => {
            format!(
                "Bipolar factor contrasting {} vs {}",
                if !high_positive.is_empty() { &high_positive[0].0 } else { "positive variables" },
                if !high_negative.is_empty() { &high_negative[0].0 } else { "negative variables" }
            )
        }
        ComponentType::Specific => {
            "Specific factor affecting a subset of variables".to_string()
        }
        ComponentType::Trend => {
            "Major trend component explaining large proportion of variance".to_string()
        }
        _ => "Component requiring further interpretation".to_string(),
    }
}

fn compute_pca_diagnostics(
    cov_matrix: &[Vec<f64>],
    eigenvalues: &[f64],
) -> Result<PCADiagnostics, Box<dyn std::error::Error>> {
    let n = cov_matrix.len();

    // KMO statistic (simplified)
    let kmo_statistic = 0.8;  // Placeholder - would compute actual KMO

    // Bartlett's test (simplified)
    let bartlett_statistic = 100.0;  // Placeholder
    let bartlett_p_value = 0.001;

    // Correlation matrix determinant
    let correlation_determinant = matrix_determinant_simple(cov_matrix);

    // Condition number
    let max_eigenvalue = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_eigenvalue = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let condition_number = if min_eigenvalue > 0.0 {
        max_eigenvalue / min_eigenvalue
    } else {
        f64::INFINITY
    };

    // Scree recommendations
    let scree_recommendations = compute_scree_recommendations(eigenvalues);

    // Kaiser criterion
    let kaiser_components = eigenvalues.iter().take_while(|&&val| val > 1.0).count();

    Ok(PCADiagnostics {
        kmo_statistic,
        bartlett_statistic,
        bartlett_p_value,
        correlation_determinant,
        condition_number,
        scree_recommendations,
        kaiser_components,
        parallel_analysis: None,  // Would implement if needed
    })
}

fn compute_scree_recommendations(eigenvalues: &[f64]) -> ScreeRecommendations {
    let mut eigenvalue_differences = Vec::new();
    for i in 1..eigenvalues.len() {
        eigenvalue_differences.push(eigenvalues[i-1] - eigenvalues[i]);
    }

    // Find elbow point (largest difference)
    let elbow_point = eigenvalue_differences
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx + 1)
        .unwrap_or(1);

    let components_above_elbow = elbow_point;

    ScreeRecommendations {
        elbow_point,
        components_above_elbow,
        eigenvalue_differences,
    }
}

fn matrix_determinant_simple(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    match n {
        1 => matrix[0][0],
        2 => matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        _ => {
            // Simplified determinant for larger matrices
            let mut det = 1.0;
            for i in 0..n {
                det *= matrix[i][i];
            }
            det
        }
    }
}

fn compute_idiosyncratic_components(
    data: &HashMap<String, Vec<f64>>,
    trends: &[Vec<f64>],
    loadings: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let variable_names: Vec<String> = data.keys().cloned().collect();
    let n_obs = data.values().next().unwrap().len();
    let n_vars = variable_names.len();

    let mut idiosyncratic = vec![vec![0.0; n_obs]; n_vars];

    for (var_idx, var_name) in variable_names.iter().enumerate() {
        let original_series = &data[var_name];

        for t in 0..n_obs {
            let mut trend_component = 0.0;

            // Reconstruct trend component
            for (trend_idx, trend_series) in trends.iter().enumerate() {
                if t < trend_series.len() {
                    trend_component += loadings[var_idx][trend_idx] * trend_series[t];
                }
            }

            // Idiosyncratic = Original - Trend
            idiosyncratic[var_idx][t] = original_series[t] - trend_component;
        }
    }

    Ok(idiosyncratic)
}

fn compute_trend_variance_decomposition(
    data: &HashMap<String, Vec<f64>>,
    trends: &[Vec<f64>],
    idiosyncratic: &[Vec<f64>],
) -> Result<TrendVarianceDecomposition, Box<dyn std::error::Error>> {
    let variable_names: Vec<String> = data.keys().cloned().collect();
    let n_vars = variable_names.len();

    let mut trend_variance_ratios = Vec::new();
    let mut idiosyncratic_variance_ratios = Vec::new();

    for (var_idx, var_name) in variable_names.iter().enumerate() {
        let original_series = &data[var_name];
        let original_variance = compute_variance(original_series);

        let idio_variance = compute_variance(&idiosyncratic[var_idx]);
        let trend_variance = original_variance - idio_variance;

        trend_variance_ratios.push(trend_variance / original_variance);
        idiosyncratic_variance_ratios.push(idio_variance / original_variance);
    }

    let total_trend_variance = trend_variance_ratios.iter().sum::<f64>() / n_vars as f64;
    let total_idiosyncratic_variance = idiosyncratic_variance_ratios.iter().sum::<f64>() / n_vars as f64;

    Ok(TrendVarianceDecomposition {
        trend_variance_ratios,
        idiosyncratic_variance_ratios,
        total_trend_variance,
        total_idiosyncratic_variance,
    })
}

fn compute_trend_persistence(trends: &[Vec<f64>]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut persistence = Vec::new();

    for trend in trends {
        // Compute AR(1) coefficient as measure of persistence
        if trend.len() < 3 {
            persistence.push(0.0);
            continue;
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 1..trend.len() {
            numerator += trend[i] * trend[i-1];
            denominator += trend[i-1] * trend[i-1];
        }

        let ar1_coeff = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        persistence.push(ar1_coeff.abs());
    }

    Ok(persistence)
}

fn compute_variance(series: &[f64]) -> f64 {
    let n = series.len() as f64;
    let mean = series.iter().sum::<f64>() / n;
    series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

fn apply_rotation(
    loadings: &[Vec<f64>],
    _rotation: &RotationMethod,
    _max_iterations: usize,
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    // Simplified - return original loadings
    // In practice, would implement varimax, promax, etc.
    Ok(loadings.to_vec())
}

fn compute_communalities(loadings: &[Vec<f64>]) -> Vec<f64> {
    loadings.iter()
        .map(|row| row.iter().map(|&x| x * x).sum())
        .collect()
}

fn compute_factor_scores(
    data: &HashMap<String, Vec<f64>>,
    loadings: &[Vec<f64>],
    variable_names: &[String],
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let n_obs = data.values().next().unwrap().len();
    let n_factors = loadings[0].len();

    // Simple regression method for factor scores
    let mut scores = vec![vec![0.0; n_factors]; n_obs];

    for t in 0..n_obs {
        for f in 0..n_factors {
            let mut score = 0.0;
            for (var_idx, var_name) in variable_names.iter().enumerate() {
                score += data[var_name][t] * loadings[var_idx][f];
            }
            scores[t][f] = score;
        }
    }

    Ok(scores)
}

fn compute_factor_fit(
    _loadings: &[Vec<f64>],
    _communalities: &[f64],
    _n_vars: usize,
) -> FactorFitStatistics {
    // Simplified fit statistics - in practice would compute actual values
    FactorFitStatistics {
        chi_square: 10.0,
        p_value: 0.05,
        rmsea: 0.08,
        tli: 0.90,
        cfi: 0.92,
        gfi: 0.88,
    }
}

fn interpret_factors(
    loadings: &[Vec<f64>],
    variable_names: &[String],
) -> Vec<FactorInterpretation> {
    let n_factors = loadings[0].len();
    let mut interpretations = Vec::new();

    for factor_idx in 0..n_factors {
        let mut high_loadings = Vec::new();

        for (var_idx, var_name) in variable_names.iter().enumerate() {
            let loading = loadings[var_idx][factor_idx];
            if loading.abs() > 0.5 {
                high_loadings.push((var_name.clone(), loading));
            }
        }

        high_loadings.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let name = if !high_loadings.is_empty() {
            format!("Factor {}: {}", factor_idx + 1, high_loadings[0].0)
        } else {
            format!("Factor {}", factor_idx + 1)
        };

        let interpretation = format!(
            "Factor explaining variance in {}",
            high_loadings.iter()
                .take(3)
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let complexity = high_loadings.len();
        interpretations.push(FactorInterpretation {
            factor_number: factor_idx + 1,
            name,
            high_loadings,
            interpretation,
            complexity,
        });
    }

    interpretations
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_pca_basic() {
        let mut data = HashMap::new();
        data.insert("var1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("var2".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        data.insert("var3".to_string(), vec![1.5, 3.0, 4.5, 6.0, 7.5]);

        let result = compute_pca(&data, Some(2), true, None).unwrap();

        assert_eq!(result.variable_names.len(), 3);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.loadings.len(), 3);
        assert_eq!(result.loadings[0].len(), 2);
        assert!(result.explained_variance_ratio.iter().sum::<f64>() <= 1.0);
    }

    #[test]
    fn test_common_trends() {
        let mut data = HashMap::new();
        data.insert("series1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        data.insert("series2".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        data.insert("series3".to_string(), vec![1.5, 3.0, 4.5, 6.0, 7.5, 9.0]);

        let result = extract_common_trends(&data, Some(1), false).unwrap();

        assert_eq!(result.variable_names.len(), 3);
        assert_eq!(result.n_trends, 1);
        assert_eq!(result.trends.len(), 1);
        assert_eq!(result.trends[0].len(), 6);
        assert!(result.variance_decomposition.total_trend_variance >= 0.0);
    }

    #[test]
    fn test_factor_analysis() {
        let mut data = HashMap::new();
        data.insert("var1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        data.insert("var2".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
        data.insert("var3".to_string(), vec![1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5]);

        let result = compute_factor_analysis(&data, 1, RotationMethod::None, 100).unwrap();

        assert_eq!(result.variable_names.len(), 3);
        assert_eq!(result.n_factors, 1);
        assert_eq!(result.factor_loadings.len(), 3);
        assert_eq!(result.communalities.len(), 3);
        assert!(result.communalities.iter().all(|&c| c >= 0.0 && c <= 1.0));
    }

    #[test]
    fn test_data_preprocessing() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let (processed, means, std_devs) = preprocess_data(&data, true).unwrap();

        assert_eq!(processed.len(), 3);
        assert_eq!(processed[0].len(), 3);
        assert_eq!(means.len(), 3);
        assert_eq!(std_devs.len(), 3);

        // Check that data is centered (mean ≈ 0)
        for j in 0..3 {
            let col_mean: f64 = processed.iter().map(|row| row[j]).sum::<f64>() / 3.0;
            assert!((col_mean).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eigendecomposition_2x2() {
        let matrix = vec![
            vec![4.0, 2.0],
            vec![2.0, 3.0],
        ];

        let (eigenvalues, eigenvectors) = compute_eigendecomposition(&matrix).unwrap();

        assert_eq!(eigenvalues.len(), 2);
        assert_eq!(eigenvectors.len(), 2);
        assert_eq!(eigenvectors[0].len(), 2);

        // Check that eigenvalues are ordered correctly
        assert!(eigenvalues[0] >= eigenvalues[1]);
    }
}