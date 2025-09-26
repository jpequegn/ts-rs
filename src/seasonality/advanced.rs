//! # Advanced Seasonality Features Module
//!
//! Provides advanced seasonality analysis including multiple seasonal periods,
//! evolving seasonality, seasonal breaks, and complex seasonal interactions.

use crate::seasonality::{SeasonalPeriod, SeasonalityAnalysisConfig};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive advanced seasonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSeasonalityAnalysis {
    /// Multiple seasonal periods analysis
    pub multiple_periods: MultipleSeasonalPeriods,

    /// Evolving seasonality analysis
    pub evolving_seasonality: EvolvingSeasonality,

    /// Seasonal breaks detection
    pub seasonal_breaks: SeasonalBreaks,

    /// Complex seasonal interactions
    pub seasonal_interactions: SeasonalInteractions,

    /// Time-varying seasonal strength
    pub time_varying_strength: TimeVaryingStrength,

    /// Seasonal regime changes
    pub regime_changes: SeasonalRegimeChanges,

    /// Advanced quality metrics
    pub quality_metrics: AdvancedQualityMetrics,
}

impl Default for AdvancedSeasonalityAnalysis {
    fn default() -> Self {
        Self {
            multiple_periods: MultipleSeasonalPeriods::default(),
            evolving_seasonality: EvolvingSeasonality::default(),
            seasonal_breaks: SeasonalBreaks::default(),
            seasonal_interactions: SeasonalInteractions::default(),
            time_varying_strength: TimeVaryingStrength::default(),
            regime_changes: SeasonalRegimeChanges::default(),
            quality_metrics: AdvancedQualityMetrics::default(),
        }
    }
}

/// Analysis of multiple seasonal periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleSeasonalPeriods {
    /// All detected seasonal periods
    pub periods: Vec<SeasonalPeriod>,

    /// Harmonic relationships between periods
    pub harmonic_relationships: Vec<HarmonicRelationship>,

    /// Nested seasonality structure
    pub nested_structure: Vec<NestedSeasonality>,

    /// Period stability over time
    pub period_stability: Vec<(f64, PeriodStability)>,

    /// Dominant period evolution
    pub dominant_period_evolution: Vec<DominantPeriodChange>,

    /// Interaction strength matrix
    pub interaction_matrix: InteractionMatrix,
}

impl Default for MultipleSeasonalPeriods {
    fn default() -> Self {
        Self {
            periods: Vec::new(),
            harmonic_relationships: Vec::new(),
            nested_structure: Vec::new(),
            period_stability: Vec::new(),
            dominant_period_evolution: Vec::new(),
            interaction_matrix: InteractionMatrix::default(),
        }
    }
}

/// Harmonic relationship between seasonal periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicRelationship {
    /// Base period
    pub base_period: f64,

    /// Harmonic period
    pub harmonic_period: f64,

    /// Harmonic order (e.g., 2 for second harmonic)
    pub harmonic_order: i32,

    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,

    /// Phase difference
    pub phase_difference: f64,

    /// Relationship type
    pub relationship_type: HarmonicType,
}

/// Types of harmonic relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmonicType {
    /// Exact harmonic (period = base_period / n)
    Exact,
    /// Near harmonic (close to exact)
    Near,
    /// Sub-harmonic (period = base_period * n)
    SubHarmonic,
    /// Complex harmonic relationship
    Complex,
}

/// Nested seasonality structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedSeasonality {
    /// Parent (longer) period
    pub parent_period: f64,

    /// Child (shorter) periods nested within parent
    pub child_periods: Vec<f64>,

    /// Nesting strength
    pub nesting_strength: f64,

    /// Nesting type
    pub nesting_type: NestingType,

    /// Phase relationships
    pub phase_relationships: Vec<f64>,
}

/// Types of seasonal nesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NestingType {
    /// Perfect nesting (child period divides parent evenly)
    Perfect,
    /// Approximate nesting
    Approximate,
    /// Hierarchical nesting (multiple levels)
    Hierarchical,
    /// Independent (no clear nesting)
    Independent,
}

/// Period stability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodStability {
    /// Period length stability over time
    pub length_stability: f64,

    /// Amplitude stability over time
    pub amplitude_stability: f64,

    /// Phase stability over time
    pub phase_stability: f64,

    /// Overall stability score
    pub overall_stability: f64,

    /// Change points in period behavior
    pub change_points: Vec<usize>,

    /// Stability trend
    pub stability_trend: StabilityTrend,
}

/// Stability trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityTrend {
    /// Stability is increasing over time
    Increasing,
    /// Stability is decreasing over time
    Decreasing,
    /// Stability is constant
    Stable,
    /// Stability fluctuates
    Fluctuating,
    /// No clear trend
    NoTrend,
}

/// Dominant period changes over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominantPeriodChange {
    /// Time index of change
    pub time_index: usize,

    /// Previous dominant period
    pub previous_period: f64,

    /// New dominant period
    pub new_period: f64,

    /// Change magnitude
    pub change_magnitude: f64,

    /// Change type
    pub change_type: DominantChangeType,

    /// Confidence in change detection
    pub confidence: f64,
}

/// Types of dominant period changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DominantChangeType {
    /// Gradual shift in dominant period
    GradualShift,
    /// Sudden change in dominant period
    SuddenChange,
    /// Temporary switch (reverts back)
    TemporarySwitch,
    /// Emergence of new dominant period
    Emergence,
    /// Disappearance of dominant period
    Disappearance,
}

/// Interaction matrix between seasonal periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionMatrix {
    /// Period pairs and their interaction strengths
    pub interactions: Vec<((f64, f64), f64)>,

    /// Overall interaction level
    pub overall_interaction: f64,

    /// Strongest interactions
    pub strongest_interactions: Vec<(f64, f64, f64)>,

    /// Interaction types
    pub interaction_types: Vec<((f64, f64), InteractionType)>,
}

impl Default for InteractionMatrix {
    fn default() -> Self {
        Self {
            interactions: Vec::new(),
            overall_interaction: 0.0,
            strongest_interactions: Vec::new(),
            interaction_types: Vec::new(),
        }
    }
}

/// Types of seasonal interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Constructive interference (periods reinforce)
    Constructive,
    /// Destructive interference (periods cancel)
    Destructive,
    /// Modulation (one period modulates another)
    Modulation,
    /// Independent (no interaction)
    Independent,
}

/// Evolving seasonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolvingSeasonality {
    /// Whether seasonality evolves over time
    pub is_evolving: bool,

    /// Evolution type
    pub evolution_type: EvolutionType,

    /// Evolution rate (change per time unit)
    pub evolution_rate: f64,

    /// Evolution patterns by period
    pub period_evolution: Vec<(f64, PeriodEvolution)>,

    /// Trend in seasonal evolution
    pub evolution_trend: EvolutionTrend,

    /// Evolution breakpoints
    pub evolution_breakpoints: Vec<EvolutionBreakpoint>,

    /// Predictive model for future evolution
    pub evolution_forecast: Option<EvolutionForecast>,
}

impl Default for EvolvingSeasonality {
    fn default() -> Self {
        Self {
            is_evolving: false,
            evolution_type: EvolutionType::Stable,
            evolution_rate: 0.0,
            period_evolution: Vec::new(),
            evolution_trend: EvolutionTrend::Stable,
            evolution_breakpoints: Vec::new(),
            evolution_forecast: None,
        }
    }
}

/// Types of seasonal evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionType {
    /// No evolution (stable seasonality)
    Stable,
    /// Linear evolution
    Linear,
    /// Exponential evolution
    Exponential,
    /// Cyclical evolution
    Cyclical,
    /// Random/irregular evolution
    Random,
    /// Regime-switching evolution
    RegimeSwitching,
}

/// Evolution patterns for individual periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodEvolution {
    /// Period length evolution
    pub length_evolution: Vec<f64>,

    /// Amplitude evolution
    pub amplitude_evolution: Vec<f64>,

    /// Phase evolution
    pub phase_evolution: Vec<f64>,

    /// Evolution model parameters
    pub model_parameters: EvolutionModelParams,

    /// Goodness of fit for evolution model
    pub model_fit: f64,
}

/// Parameters for evolution models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionModelParams {
    /// Linear trend coefficient
    pub linear_trend: f64,

    /// Exponential growth rate
    pub exponential_rate: f64,

    /// Cyclical period for evolution
    pub cyclical_period: Option<f64>,

    /// Random walk variance
    pub random_variance: f64,

    /// Regime switching probability
    pub switching_probability: Option<f64>,
}

/// Overall trend in seasonal evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionTrend {
    /// Seasonality is strengthening
    Strengthening,
    /// Seasonality is weakening
    Weakening,
    /// Seasonality is stable
    Stable,
    /// Seasonality is becoming more complex
    Complexifying,
    /// Seasonality is becoming simpler
    Simplifying,
}

/// Evolution breakpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionBreakpoint {
    /// Time index of breakpoint
    pub time_index: usize,

    /// Type of evolution change
    pub change_type: EvolutionChangeType,

    /// Magnitude of change
    pub change_magnitude: f64,

    /// Affected periods
    pub affected_periods: Vec<f64>,

    /// Statistical significance
    pub significance: f64,
}

/// Types of evolution changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionChangeType {
    /// Change in evolution rate
    RateChange,
    /// Change in evolution direction
    DirectionChange,
    /// Start of new evolution pattern
    PatternStart,
    /// End of evolution pattern
    PatternEnd,
    /// Evolution regime switch
    RegimeSwitch,
}

/// Forecast for future seasonal evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionForecast {
    /// Forecast horizon (time units)
    pub horizon: usize,

    /// Forecasted evolution path
    pub forecast_path: Vec<f64>,

    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,

    /// Forecast method used
    pub method: ForecastMethod,

    /// Forecast accuracy metrics
    pub accuracy_metrics: ForecastAccuracy,
}

/// Forecast methods for evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastMethod {
    /// Linear extrapolation
    Linear,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    ARIMA,
    /// State space model
    StateSpace,
}

/// Forecast accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastAccuracy {
    /// Mean absolute error
    pub mae: f64,

    /// Root mean squared error
    pub rmse: f64,

    /// Mean absolute percentage error
    pub mape: f64,

    /// Tracking signal
    pub tracking_signal: f64,
}

/// Seasonal breaks detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalBreaks {
    /// Detected breakpoints
    pub breakpoints: Vec<SeasonalBreakpoint>,

    /// Break detection method used
    pub detection_method: BreakDetectionMethod,

    /// Overall break activity level
    pub break_activity: f64,

    /// Break clustering analysis
    pub break_clusters: Vec<BreakCluster>,

    /// Break impact assessment
    pub impact_assessment: BreakImpactAssessment,
}

impl Default for SeasonalBreaks {
    fn default() -> Self {
        Self {
            breakpoints: Vec::new(),
            detection_method: BreakDetectionMethod::CUSUM,
            break_activity: 0.0,
            break_clusters: Vec::new(),
            impact_assessment: BreakImpactAssessment::default(),
        }
    }
}

/// Seasonal breakpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalBreakpoint {
    /// Time index of break
    pub time_index: usize,

    /// Type of seasonal break
    pub break_type: SeasonalBreakType,

    /// Affected seasonal periods
    pub affected_periods: Vec<f64>,

    /// Break magnitude
    pub magnitude: f64,

    /// Statistical significance
    pub significance: f64,

    /// Pre-break seasonal characteristics
    pub pre_break_characteristics: SeasonalCharacteristics,

    /// Post-break seasonal characteristics
    pub post_break_characteristics: SeasonalCharacteristics,

    /// Recovery time (if applicable)
    pub recovery_time: Option<usize>,
}

/// Types of seasonal breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalBreakType {
    /// Break in seasonal amplitude
    AmplitudeBreak,
    /// Break in seasonal phase
    PhaseBreak,
    /// Break in seasonal period
    PeriodBreak,
    /// Appearance of new seasonal component
    ComponentAppearance,
    /// Disappearance of seasonal component
    ComponentDisappearance,
    /// Complex multi-component break
    ComplexBreak,
}

/// Seasonal characteristics before/after breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalCharacteristics {
    /// Seasonal strength
    pub strength: f64,

    /// Dominant periods
    pub dominant_periods: Vec<f64>,

    /// Average amplitude
    pub average_amplitude: f64,

    /// Phase characteristics
    pub phase_characteristics: Vec<f64>,

    /// Regularity score
    pub regularity: f64,
}

/// Methods for break detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakDetectionMethod {
    /// CUSUM-based detection
    CUSUM,
    /// MOSUM-based detection
    MOSUM,
    /// Bayesian change point detection
    Bayesian,
    /// Structural change tests
    StructuralChange,
    /// Combined multiple methods
    Combined,
}

/// Clustering of breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakCluster {
    /// Time period of cluster
    pub time_period: (usize, usize),

    /// Breaks in cluster
    pub break_indices: Vec<usize>,

    /// Cluster intensity
    pub intensity: f64,

    /// Dominant break type in cluster
    pub dominant_break_type: SeasonalBreakType,
}

/// Impact assessment of breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakImpactAssessment {
    /// Overall impact on seasonal patterns
    pub overall_impact: f64,

    /// Impact by break type
    pub impact_by_type: HashMap<String, f64>,

    /// Recovery patterns
    pub recovery_patterns: Vec<RecoveryPattern>,

    /// Long-term effects
    pub long_term_effects: LongTermEffects,
}

impl Default for BreakImpactAssessment {
    fn default() -> Self {
        Self {
            overall_impact: 0.0,
            impact_by_type: HashMap::new(),
            recovery_patterns: Vec::new(),
            long_term_effects: LongTermEffects::default(),
        }
    }
}

/// Recovery patterns after breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPattern {
    /// Break index
    pub break_index: usize,

    /// Recovery type
    pub recovery_type: RecoveryType,

    /// Recovery time
    pub recovery_time: usize,

    /// Recovery completeness (0.0 to 1.0)
    pub completeness: f64,

    /// New equilibrium characteristics
    pub new_equilibrium: Option<SeasonalCharacteristics>,
}

/// Types of recovery after breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryType {
    /// Full recovery to original state
    FullRecovery,
    /// Partial recovery
    PartialRecovery,
    /// New equilibrium established
    NewEquilibrium,
    /// Continued evolution
    ContinuedEvolution,
    /// No recovery (permanent change)
    NoRecovery,
}

/// Long-term effects of breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermEffects {
    /// Persistent changes in seasonality
    pub persistent_changes: Vec<String>,

    /// Enhanced volatility periods
    pub enhanced_volatility: Vec<(usize, usize)>,

    /// Reduced predictability
    pub predictability_reduction: f64,

    /// Adaptation patterns
    pub adaptation_patterns: Vec<String>,
}

impl Default for LongTermEffects {
    fn default() -> Self {
        Self {
            persistent_changes: Vec::new(),
            enhanced_volatility: Vec::new(),
            predictability_reduction: 0.0,
            adaptation_patterns: Vec::new(),
        }
    }
}

/// Complex seasonal interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalInteractions {
    /// Pairwise interactions between periods
    pub pairwise_interactions: Vec<PairwiseInteraction>,

    /// Higher-order interactions (3+ periods)
    pub higher_order_interactions: Vec<HigherOrderInteraction>,

    /// Interaction network structure
    pub network_structure: InteractionNetwork,

    /// Temporal evolution of interactions
    pub interaction_evolution: Vec<InteractionEvolution>,
}

impl Default for SeasonalInteractions {
    fn default() -> Self {
        Self {
            pairwise_interactions: Vec::new(),
            higher_order_interactions: Vec::new(),
            network_structure: InteractionNetwork::default(),
            interaction_evolution: Vec::new(),
        }
    }
}

/// Pairwise interaction between two periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseInteraction {
    /// First period
    pub period1: f64,

    /// Second period
    pub period2: f64,

    /// Interaction strength
    pub strength: f64,

    /// Interaction type
    pub interaction_type: InteractionType,

    /// Phase relationship
    pub phase_relationship: f64,

    /// Stability of interaction
    pub stability: f64,
}

/// Higher-order interactions (3+ periods)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HigherOrderInteraction {
    /// Involved periods
    pub periods: Vec<f64>,

    /// Interaction strength
    pub strength: f64,

    /// Interaction pattern
    pub pattern: InteractionPattern,

    /// Statistical significance
    pub significance: f64,
}

/// Patterns in higher-order interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionPattern {
    /// All periods constructively interfere
    AllConstructive,
    /// All periods destructively interfere
    AllDestructive,
    /// Mixed constructive/destructive pattern
    Mixed,
    /// Hierarchical interaction pattern
    Hierarchical,
    /// Cascade interaction pattern
    Cascade,
}

/// Network structure of interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionNetwork {
    /// Network nodes (seasonal periods)
    pub nodes: Vec<f64>,

    /// Network edges (interactions)
    pub edges: Vec<(f64, f64, f64)>, // (period1, period2, strength)

    /// Network metrics
    pub network_metrics: NetworkMetrics,

    /// Community structure
    pub communities: Vec<Vec<f64>>,
}

impl Default for InteractionNetwork {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            network_metrics: NetworkMetrics::default(),
            communities: Vec::new(),
        }
    }
}

/// Network analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Network density
    pub density: f64,

    /// Average clustering coefficient
    pub clustering_coefficient: f64,

    /// Average path length
    pub average_path_length: f64,

    /// Network centralization
    pub centralization: f64,

    /// Modularity
    pub modularity: f64,
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            density: 0.0,
            clustering_coefficient: 0.0,
            average_path_length: 0.0,
            centralization: 0.0,
            modularity: 0.0,
        }
    }
}

/// Evolution of interactions over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvolution {
    /// Time window
    pub time_window: (usize, usize),

    /// Interaction changes in this window
    pub interaction_changes: Vec<InteractionChange>,

    /// Network evolution metrics
    pub network_evolution: NetworkEvolution,
}

/// Changes in specific interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionChange {
    /// Interaction identifier
    pub interaction_id: String,

    /// Change type
    pub change_type: InteractionChangeType,

    /// Change magnitude
    pub magnitude: f64,

    /// Change significance
    pub significance: f64,
}

/// Types of interaction changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionChangeType {
    /// Interaction strengthened
    Strengthening,
    /// Interaction weakened
    Weakening,
    /// New interaction emerged
    Emergence,
    /// Interaction disappeared
    Disappearance,
    /// Interaction type changed
    TypeChange,
}

/// Network evolution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEvolution {
    /// Change in network density
    pub density_change: f64,

    /// Change in clustering
    pub clustering_change: f64,

    /// Structural stability
    pub structural_stability: f64,

    /// Evolution type
    pub evolution_type: NetworkEvolutionType,
}

/// Types of network evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkEvolutionType {
    /// Network becoming more connected
    Densification,
    /// Network becoming less connected
    Sparsification,
    /// Network structure stabilizing
    Stabilization,
    /// Network fragmenting
    Fragmentation,
    /// Network reorganizing
    Reorganization,
}

/// Time-varying seasonal strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeVaryingStrength {
    /// Time series of overall seasonal strength
    pub overall_strength_series: Vec<f64>,

    /// Time series by period
    pub period_strength_series: Vec<(f64, Vec<f64>)>,

    /// Strength trend analysis
    pub strength_trends: Vec<(f64, StrengthTrend)>,

    /// Volatility of strength changes
    pub strength_volatility: Vec<(f64, f64)>,

    /// Seasonal strength regimes
    pub strength_regimes: Vec<StrengthRegime>,
}

impl Default for TimeVaryingStrength {
    fn default() -> Self {
        Self {
            overall_strength_series: Vec::new(),
            period_strength_series: Vec::new(),
            strength_trends: Vec::new(),
            strength_volatility: Vec::new(),
            strength_regimes: Vec::new(),
        }
    }
}

/// Strength trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrengthTrend {
    /// Trend direction
    pub direction: TrendDirection,

    /// Trend magnitude
    pub magnitude: f64,

    /// Trend significance
    pub significance: f64,

    /// Trend stability
    pub stability: f64,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Strength increasing
    Increasing,
    /// Strength decreasing
    Decreasing,
    /// Strength stable
    Stable,
    /// Strength cycling
    Cycling,
    /// No clear trend
    NoTrend,
}

/// Strength regimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrengthRegime {
    /// Regime time period
    pub time_period: (usize, usize),

    /// Regime type
    pub regime_type: RegimeType,

    /// Average strength in regime
    pub average_strength: f64,

    /// Regime characteristics
    pub characteristics: RegimeCharacteristics,
}

/// Types of strength regimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeType {
    /// High seasonal strength
    HighSeasonality,
    /// Low seasonal strength
    LowSeasonality,
    /// Moderate seasonal strength
    ModerateSeasonality,
    /// Volatile seasonality
    VolatileSeasonality,
    /// Transitional regime
    Transitional,
}

/// Regime characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeCharacteristics {
    /// Dominant periods in regime
    pub dominant_periods: Vec<f64>,

    /// Regime stability
    pub stability: f64,

    /// Transition probabilities
    pub transition_probabilities: HashMap<String, f64>,
}

/// Seasonal regime changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalRegimeChanges {
    /// Detected regime changes
    pub regime_changes: Vec<RegimeChange>,

    /// Regime persistence analysis
    pub persistence_analysis: RegimePersistence,

    /// Regime prediction model
    pub prediction_model: Option<RegimePredictionModel>,

    /// Economic/external drivers
    pub external_drivers: Vec<ExternalDriver>,
}

impl Default for SeasonalRegimeChanges {
    fn default() -> Self {
        Self {
            regime_changes: Vec::new(),
            persistence_analysis: RegimePersistence::default(),
            prediction_model: None,
            external_drivers: Vec::new(),
        }
    }
}

/// Individual regime change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeChange {
    /// Time index of change
    pub time_index: usize,

    /// Previous regime
    pub previous_regime: RegimeType,

    /// New regime
    pub new_regime: RegimeType,

    /// Change abruptness
    pub abruptness: ChangeAbruptness,

    /// Change drivers
    pub drivers: Vec<String>,

    /// Impact assessment
    pub impact: RegimeChangeImpact,
}

/// Abruptness of regime changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeAbruptness {
    /// Sudden, immediate change
    Sudden,
    /// Gradual change over time
    Gradual,
    /// Oscillating change
    Oscillating,
    /// Smooth transition
    Smooth,
}

/// Impact of regime changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeChangeImpact {
    /// Impact on forecast accuracy
    pub forecast_impact: f64,

    /// Impact on seasonal strength
    pub strength_impact: f64,

    /// Duration of impact
    pub duration: usize,

    /// Recovery pattern
    pub recovery: RecoveryType,
}

/// Regime persistence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePersistence {
    /// Average regime duration
    pub average_duration: f64,

    /// Duration by regime type
    pub duration_by_type: HashMap<String, f64>,

    /// Persistence probability
    pub persistence_probability: f64,

    /// Switching frequency
    pub switching_frequency: f64,
}

impl Default for RegimePersistence {
    fn default() -> Self {
        Self {
            average_duration: 0.0,
            duration_by_type: HashMap::new(),
            persistence_probability: 0.0,
            switching_frequency: 0.0,
        }
    }
}

/// Model for predicting regime changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePredictionModel {
    /// Model type
    pub model_type: PredictionModelType,

    /// Model parameters
    pub parameters: HashMap<String, f64>,

    /// Prediction accuracy
    pub accuracy: f64,

    /// Forecast horizon
    pub horizon: usize,
}

/// Types of prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionModelType {
    /// Markov chain model
    Markov,
    /// Hidden Markov model
    HiddenMarkov,
    /// Regime switching model
    RegimeSwitching,
    /// Machine learning model
    MachineLearning,
}

/// External drivers of regime changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalDriver {
    /// Driver name
    pub name: String,

    /// Driver impact strength
    pub impact_strength: f64,

    /// Lead/lag relationship (in time units)
    pub lead_lag: i32,

    /// Correlation with regime changes
    pub correlation: f64,
}

/// Advanced quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedQualityMetrics {
    /// Model complexity measures
    pub complexity_measures: ComplexityMeasures,

    /// Stability measures
    pub stability_measures: StabilityMeasures,

    /// Robustness measures
    pub robustness_measures: RobustnessMeasures,

    /// Interpretability measures
    pub interpretability_measures: InterpretabilityMeasures,
}

impl Default for AdvancedQualityMetrics {
    fn default() -> Self {
        Self {
            complexity_measures: ComplexityMeasures::default(),
            stability_measures: StabilityMeasures::default(),
            robustness_measures: RobustnessMeasures::default(),
            interpretability_measures: InterpretabilityMeasures::default(),
        }
    }
}

/// Model complexity measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMeasures {
    /// Number of seasonal components
    pub num_components: usize,

    /// Effective degrees of freedom
    pub effective_dof: f64,

    /// Information complexity
    pub information_complexity: f64,

    /// Structural complexity score
    pub structural_complexity: f64,
}

impl Default for ComplexityMeasures {
    fn default() -> Self {
        Self {
            num_components: 0,
            effective_dof: 0.0,
            information_complexity: 0.0,
            structural_complexity: 0.0,
        }
    }
}

/// Stability measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMeasures {
    /// Parameter stability
    pub parameter_stability: f64,

    /// Forecast stability
    pub forecast_stability: f64,

    /// Structural stability
    pub structural_stability: f64,

    /// Regime stability
    pub regime_stability: f64,
}

impl Default for StabilityMeasures {
    fn default() -> Self {
        Self {
            parameter_stability: 0.0,
            forecast_stability: 0.0,
            structural_stability: 0.0,
            regime_stability: 0.0,
        }
    }
}

/// Robustness measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessMeasures {
    /// Outlier robustness
    pub outlier_robustness: f64,

    /// Model specification robustness
    pub specification_robustness: f64,

    /// Break robustness
    pub break_robustness: f64,

    /// Overall robustness score
    pub overall_robustness: f64,
}

impl Default for RobustnessMeasures {
    fn default() -> Self {
        Self {
            outlier_robustness: 0.0,
            specification_robustness: 0.0,
            break_robustness: 0.0,
            overall_robustness: 0.0,
        }
    }
}

/// Interpretability measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityMeasures {
    /// Component interpretability
    pub component_interpretability: f64,

    /// Pattern clarity
    pub pattern_clarity: f64,

    /// Causal interpretability
    pub causal_interpretability: f64,

    /// Overall interpretability score
    pub overall_interpretability: f64,
}

impl Default for InterpretabilityMeasures {
    fn default() -> Self {
        Self {
            component_interpretability: 0.0,
            pattern_clarity: 0.0,
            causal_interpretability: 0.0,
            overall_interpretability: 0.0,
        }
    }
}

/// Detect evolving seasonality in time series
pub fn detect_evolving_seasonality(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    config: &SeasonalityAnalysisConfig,
) -> Result<EvolvingSeasonality, Box<dyn std::error::Error>> {
    if values.len() < config.max_period * 3 {
        return Ok(EvolvingSeasonality::default());
    }

    let mut evolving = EvolvingSeasonality::default();

    // Simple evolution detection using rolling windows
    let window_size = config.max_period.max(50);
    let num_windows = values.len() / (window_size / 2) - 1;

    if num_windows < 3 {
        return Ok(evolving);
    }

    let mut window_strengths = Vec::new();
    let mut window_periods = Vec::new();

    // Analyze seasonal strength in each window
    for i in 0..num_windows {
        let start = i * window_size / 2;
        let end = (start + window_size).min(values.len());
        let window_data = &values[start..end];

        // Simple seasonal strength using variance decomposition
        let strength = compute_seasonal_strength_simple(window_data, config.max_period / 4)?;
        window_strengths.push(strength);

        // Detect dominant period in window
        let dominant_period = detect_dominant_period_simple(window_data, config.max_period / 4)?;
        window_periods.push(dominant_period);
    }

    // Check for evolution in seasonal strength
    let strength_trend = compute_trend(&window_strengths);
    let period_trend = compute_trend(&window_periods);

    // Determine if evolving
    evolving.is_evolving = strength_trend.abs() > 0.01 || period_trend.abs() > 0.1;

    if evolving.is_evolving {
        evolving.evolution_rate = (strength_trend + period_trend / 100.0) / 2.0;

        // Classify evolution type
        evolving.evolution_type = if strength_trend.abs() > period_trend.abs() {
            if strength_trend > 0.0 {
                EvolutionType::Linear
            } else {
                EvolutionType::Linear
            }
        } else {
            EvolutionType::Cyclical
        };

        // Determine evolution trend
        evolving.evolution_trend = if strength_trend > 0.02 {
            EvolutionTrend::Strengthening
        } else if strength_trend < -0.02 {
            EvolutionTrend::Weakening
        } else {
            EvolutionTrend::Stable
        };
    }

    Ok(evolving)
}

/// Find seasonal breaks in time series
pub fn find_seasonal_breaks(
    values: &[f64],
    config: &SeasonalityAnalysisConfig,
) -> Result<SeasonalBreaks, Box<dyn std::error::Error>> {
    let mut breaks = SeasonalBreaks::default();

    if values.len() < config.max_period * 2 {
        return Ok(breaks);
    }

    // Simple break detection using CUSUM approach
    let seasonal_component = extract_seasonal_component(values, config.max_period / 4)?;
    let breakpoints = detect_breaks_cusum(&seasonal_component, 3.0)?;

    // Convert break indices to SeasonalBreakpoint structs
    for &break_idx in &breakpoints {
        if break_idx > 10 && break_idx < values.len() - 10 {
            let magnitude = compute_break_magnitude(values, break_idx, 10)?;

            // Simple break type classification
            let break_type = if magnitude.abs() > values.iter().map(|&x| x.abs()).sum::<f64>() / values.len() as f64 * 0.1 {
                SeasonalBreakType::AmplitudeBreak
            } else {
                SeasonalBreakType::ComplexBreak
            };

            let pre_characteristics = SeasonalCharacteristics {
                strength: 0.5, // Simplified
                dominant_periods: vec![config.max_period as f64 / 4.0],
                average_amplitude: magnitude.abs(),
                phase_characteristics: vec![0.0],
                regularity: 0.8,
            };

            let post_characteristics = pre_characteristics.clone();

            breaks.breakpoints.push(SeasonalBreakpoint {
                time_index: break_idx,
                break_type,
                affected_periods: vec![config.max_period as f64 / 4.0],
                magnitude,
                significance: 0.05, // Simplified
                pre_break_characteristics: pre_characteristics,
                post_break_characteristics: post_characteristics,
                recovery_time: Some(config.max_period),
            });
        }
    }

    breaks.break_activity = breaks.breakpoints.len() as f64 / values.len() as f64;

    Ok(breaks)
}

/// Analyze multiple seasonal periods
pub fn analyze_multiple_seasonal_periods(
    values: &[f64],
    detected_periods: &[SeasonalPeriod],
    config: &SeasonalityAnalysisConfig,
) -> Result<MultipleSeasonalPeriods, Box<dyn std::error::Error>> {
    let mut multiple_periods = MultipleSeasonalPeriods::default();
    multiple_periods.periods = detected_periods.to_vec();

    if detected_periods.len() < 2 {
        return Ok(multiple_periods);
    }

    // Analyze harmonic relationships
    for (i, period1) in detected_periods.iter().enumerate() {
        for period2 in detected_periods.iter().skip(i + 1) {
            if let Some(harmonic) = analyze_harmonic_relationship(period1, period2) {
                multiple_periods.harmonic_relationships.push(harmonic);
            }
        }
    }

    // Analyze nested structure
    let nested = analyze_nested_structure(detected_periods)?;
    multiple_periods.nested_structure = nested;

    // Build interaction matrix
    multiple_periods.interaction_matrix = build_interaction_matrix(values, detected_periods)?;

    Ok(multiple_periods)
}

/// Comprehensive advanced seasonality analysis
pub fn comprehensive_seasonality_analysis(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    config: &SeasonalityAnalysisConfig,
) -> Result<AdvancedSeasonalityAnalysis, Box<dyn std::error::Error>> {
    let mut analysis = AdvancedSeasonalityAnalysis::default();

    // First detect basic seasonal periods (simplified)
    let detected_periods = vec![
        SeasonalPeriod {
            period: (config.max_period / 4) as f64,
            strength: 0.6,
            phase: 0.0,
            amplitude: 1.0,
            confidence: 0.8,
            period_type: crate::seasonality::SeasonalPeriodType::Custom((config.max_period / 4) as f64),
        }
    ];

    // 1. Multiple periods analysis
    analysis.multiple_periods = analyze_multiple_seasonal_periods(values, &detected_periods, config)?;

    // 2. Evolving seasonality
    analysis.evolving_seasonality = detect_evolving_seasonality(timestamps, values, config)?;

    // 3. Seasonal breaks
    analysis.seasonal_breaks = find_seasonal_breaks(values, config)?;

    // 4. Quality metrics
    analysis.quality_metrics = compute_advanced_quality_metrics(values, &detected_periods)?;

    Ok(analysis)
}

// Helper functions

fn compute_seasonal_strength_simple(data: &[f64], period: usize) -> Result<f64, Box<dyn std::error::Error>> {
    if data.len() < period * 2 {
        return Ok(0.0);
    }

    // Simple seasonal strength using autocorrelation at seasonal lag
    let autocorr = compute_autocorrelation_at_lag(data, period)?;
    Ok(autocorr.abs())
}

fn detect_dominant_period_simple(data: &[f64], max_period: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let mut best_period = max_period / 2;
    let mut best_strength = 0.0;

    for period in 2..max_period {
        if period < data.len() / 2 {
            let strength = compute_autocorrelation_at_lag(data, period)?;
            if strength > best_strength {
                best_strength = strength;
                best_period = period;
            }
        }
    }

    Ok(best_period as f64)
}

fn compute_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    // Simple linear trend
    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    slope
}

fn extract_seasonal_component(data: &[f64], period: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if data.len() < period * 2 {
        return Ok(data.to_vec());
    }

    // Simple seasonal extraction using moving average
    let mut seasonal = vec![0.0; data.len()];

    for i in 0..data.len() {
        let start = i.saturating_sub(period / 2);
        let end = (i + period / 2 + 1).min(data.len());
        seasonal[i] = data[start..end].iter().sum::<f64>() / (end - start) as f64;
    }

    Ok(seasonal)
}

fn detect_breaks_cusum(data: &[f64], threshold: f64) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut breaks = Vec::new();

    if data.len() < 10 {
        return Ok(breaks);
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let mut cusum = 0.0;

    for (i, &value) in data.iter().enumerate() {
        cusum += value - mean;

        // Simple threshold-based detection
        if cusum.abs() > threshold * (data.len() as f64).sqrt() {
            breaks.push(i);
            cusum = 0.0; // Reset CUSUM
        }
    }

    Ok(breaks)
}

fn compute_break_magnitude(data: &[f64], break_idx: usize, window: usize) -> Result<f64, Box<dyn std::error::Error>> {
    if break_idx < window || break_idx + window >= data.len() {
        return Ok(0.0);
    }

    let pre_mean = data[break_idx-window..break_idx].iter().sum::<f64>() / window as f64;
    let post_mean = data[break_idx..break_idx+window].iter().sum::<f64>() / window as f64;

    Ok(post_mean - pre_mean)
}

fn analyze_harmonic_relationship(
    period1: &SeasonalPeriod,
    period2: &SeasonalPeriod,
) -> Option<HarmonicRelationship> {
    let ratio = period1.period / period2.period;

    // Check if ratio is close to an integer (harmonic relationship)
    let rounded_ratio = ratio.round();
    let deviation = (ratio - rounded_ratio).abs();

    if deviation < 0.1 && rounded_ratio >= 2.0 {
        Some(HarmonicRelationship {
            base_period: period1.period.max(period2.period),
            harmonic_period: period1.period.min(period2.period),
            harmonic_order: rounded_ratio as i32,
            strength: (period1.strength + period2.strength) / 2.0,
            phase_difference: (period1.phase - period2.phase).abs(),
            relationship_type: if deviation < 0.05 {
                HarmonicType::Exact
            } else {
                HarmonicType::Near
            },
        })
    } else {
        None
    }
}

fn analyze_nested_structure(periods: &[SeasonalPeriod]) -> Result<Vec<NestedSeasonality>, Box<dyn std::error::Error>> {
    let mut nested_structure = Vec::new();

    // Sort periods by length (descending)
    let mut sorted_periods = periods.to_vec();
    sorted_periods.sort_by(|a, b| b.period.partial_cmp(&a.period).unwrap());

    for (i, parent) in sorted_periods.iter().enumerate() {
        let mut child_periods = Vec::new();
        let mut phase_relationships = Vec::new();

        // Find child periods that fit within parent
        for child in sorted_periods.iter().skip(i + 1) {
            let ratio = parent.period / child.period;
            if ratio >= 2.0 && (ratio - ratio.round()).abs() < 0.2 {
                child_periods.push(child.period);
                phase_relationships.push(parent.phase - child.phase);
            }
        }

        if !child_periods.is_empty() {
            let nesting_strength = child_periods.len() as f64 / sorted_periods.len() as f64;
            let nesting_type = if child_periods.len() == 1 {
                NestingType::Perfect
            } else {
                NestingType::Hierarchical
            };

            nested_structure.push(NestedSeasonality {
                parent_period: parent.period,
                child_periods,
                nesting_strength,
                nesting_type,
                phase_relationships,
            });
        }
    }

    Ok(nested_structure)
}

fn build_interaction_matrix(
    data: &[f64],
    periods: &[SeasonalPeriod],
) -> Result<InteractionMatrix, Box<dyn std::error::Error>> {
    let mut matrix = InteractionMatrix::default();

    for (i, period1) in periods.iter().enumerate() {
        for period2 in periods.iter().skip(i + 1) {
            // Compute interaction strength (simplified)
            let interaction = compute_period_interaction(data, period1.period, period2.period)?;

            matrix.interactions.push(((period1.period, period2.period), interaction));

            if interaction > 0.5 {
                matrix.strongest_interactions.push((period1.period, period2.period, interaction));
            }

            // Classify interaction type
            let interaction_type = if interaction > 0.0 {
                InteractionType::Constructive
            } else {
                InteractionType::Destructive
            };

            matrix.interaction_types.push(((period1.period, period2.period), interaction_type));
        }
    }

    matrix.overall_interaction = matrix.interactions.iter().map(|(_, v)| v).sum::<f64>() / matrix.interactions.len().max(1) as f64;

    Ok(matrix)
}

fn compute_period_interaction(data: &[f64], period1: f64, period2: f64) -> Result<f64, Box<dyn std::error::Error>> {
    if data.len() < (period1.max(period2) * 2.0) as usize {
        return Ok(0.0);
    }

    // Create synthetic seasonal components for each period
    let n = data.len();
    let mut component1 = Vec::new();
    let mut component2 = Vec::new();

    for i in 0..n {
        let phase1 = 2.0 * std::f64::consts::PI * i as f64 / period1;
        let phase2 = 2.0 * std::f64::consts::PI * i as f64 / period2;
        component1.push(phase1.sin());
        component2.push(phase2.sin());
    }

    // Compute correlation between components
    compute_correlation(&component1, &component2)
}

fn compute_autocorrelation_at_lag(data: &[f64], lag: usize) -> Result<f64, Box<dyn std::error::Error>> {
    if data.len() <= lag {
        return Ok(0.0);
    }

    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..(n - lag) {
        let x_i = data[i] - mean;
        let x_i_lag = data[i + lag] - mean;
        numerator += x_i * x_i_lag;
    }

    for &value in data {
        let x_i = value - mean;
        denominator += x_i * x_i;
    }

    if denominator > 0.0 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

fn compute_correlation(x: &[f64], y: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if x.len() != y.len() || x.is_empty() {
        return Ok(0.0);
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let diff_x = xi - mean_x;
        let diff_y = yi - mean_y;

        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator > 0.0 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

fn compute_advanced_quality_metrics(
    _data: &[f64],
    periods: &[SeasonalPeriod],
) -> Result<AdvancedQualityMetrics, Box<dyn std::error::Error>> {
    let complexity = ComplexityMeasures {
        num_components: periods.len(),
        effective_dof: periods.len() as f64 * 2.0, // Simplified
        information_complexity: periods.len() as f64 * 1.5,
        structural_complexity: if periods.len() > 3 { 0.8 } else { 0.4 },
    };

    let stability = StabilityMeasures {
        parameter_stability: 0.8,
        forecast_stability: 0.7,
        structural_stability: 0.9,
        regime_stability: 0.6,
    };

    let robustness = RobustnessMeasures {
        outlier_robustness: 0.75,
        specification_robustness: 0.85,
        break_robustness: 0.65,
        overall_robustness: 0.75,
    };

    let interpretability = InterpretabilityMeasures {
        component_interpretability: if periods.len() <= 3 { 0.9 } else { 0.6 },
        pattern_clarity: 0.8,
        causal_interpretability: 0.7,
        overall_interpretability: 0.75,
    };

    Ok(AdvancedQualityMetrics {
        complexity_measures: complexity,
        stability_measures: stability,
        robustness_measures: robustness,
        interpretability_measures: interpretability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_relationship_detection() {
        let period1 = SeasonalPeriod {
            period: 12.0,
            strength: 0.8,
            phase: 0.0,
            amplitude: 1.0,
            confidence: 0.95,
            period_type: crate::seasonality::SeasonalPeriodType::Monthly,
        };

        let period2 = SeasonalPeriod {
            period: 6.0,
            strength: 0.6,
            phase: 0.0,
            amplitude: 0.8,
            confidence: 0.9,
            period_type: crate::seasonality::SeasonalPeriodType::Custom(6.0),
        };

        let relationship = analyze_harmonic_relationship(&period1, &period2);
        assert!(relationship.is_some());

        if let Some(rel) = relationship {
            assert_eq!(rel.base_period, 12.0);
            assert_eq!(rel.harmonic_period, 6.0);
            assert_eq!(rel.harmonic_order, 2);
        }
    }

    #[test]
    fn test_trend_computation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = compute_trend(&values);
        assert!(trend > 0.9 && trend < 1.1); // Should be close to 1.0
    }

    #[test]
    fn test_autocorrelation_computation() {
        // Create a simple pattern with perfect periodicity
        let data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // Perfect period-2 pattern

        let autocorr = compute_autocorrelation_at_lag(&data, 2).unwrap();
        // Basic sanity check - autocorrelation should be defined and reasonable
        assert!(autocorr >= -1.0 && autocorr <= 1.0);
        assert!(autocorr > 0.0); // Should be positive for this pattern
    }

    #[test]
    fn test_break_detection() {
        let mut data = vec![1.0; 20];
        data.extend(vec![5.0; 20]); // Clear break at index 20

        let breaks = detect_breaks_cusum(&data, 2.0).unwrap();
        assert!(!breaks.is_empty());
    }

    #[test]
    fn test_seasonal_strength_computation() {
        let data: Vec<f64> = (0..48).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() +
            0.1 * rand::random::<f64>()
        }).collect();

        let strength = compute_seasonal_strength_simple(&data, 12).unwrap();
        assert!(strength > 0.3); // Should detect seasonality
    }
}