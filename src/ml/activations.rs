//! Activation functions for neural networks

use crate::ml::{MLError, MLResult};
use crate::ml::tensor::Tensor;
use crate::ml::types::ActivationType;

/// Activation function trait
pub trait ActivationFunction {
    /// Apply activation function
    fn forward(&self, input: &Tensor) -> MLResult<Tensor>;

    /// Compute derivative for backpropagation
    fn backward(&self, input: &Tensor, grad_output: &Tensor) -> MLResult<Tensor>;
}

/// Apply activation function based on type
pub fn apply_activation(input: &Tensor, activation: ActivationType) -> MLResult<Tensor> {
    match activation {
        ActivationType::ReLU => relu(input),
        ActivationType::Sigmoid => sigmoid(input),
        ActivationType::Tanh => tanh(input),
        ActivationType::Softmax => softmax(input),
        ActivationType::Swish => swish(input),
        ActivationType::GELU => gelu(input),
        ActivationType::LeakyReLU { alpha } => leaky_relu(input, alpha),
        ActivationType::ELU { alpha } => elu(input, alpha),
        ActivationType::Linear => Ok(input.clone()),
    }
}

/// ReLU activation: max(0, x)
pub fn relu(input: &Tensor) -> MLResult<Tensor> {
    let result = input.inner().relu()
        .map_err(|e| MLError::tensor_operation(format!("ReLU failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(input: &Tensor) -> MLResult<Tensor> {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let neg_input = input.inner().neg()
        .map_err(|e| MLError::tensor_operation(format!("Negation failed: {}", e)))?;

    let exp = neg_input.exp()
        .map_err(|e| MLError::tensor_operation(format!("Exp failed: {}", e)))?;

    let one_plus_exp = (exp + 1.0)
        .map_err(|e| MLError::tensor_operation(format!("Addition failed: {}", e)))?;

    let result = one_plus_exp.recip()
        .map_err(|e| MLError::tensor_operation(format!("Reciprocal failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

/// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
pub fn tanh(input: &Tensor) -> MLResult<Tensor> {
    let result = input.inner().tanh()
        .map_err(|e| MLError::tensor_operation(format!("Tanh failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

/// Softmax activation: exp(x_i) / sum(exp(x_j))
pub fn softmax(input: &Tensor) -> MLResult<Tensor> {
    // Apply softmax along the last dimension
    let dims = input.shape().dims.len();
    if dims == 0 {
        return Err(MLError::invalid_input("Cannot apply softmax to scalar"));
    }

    let last_dim = dims - 1;
    let result = candle_nn::ops::softmax(input.inner(), last_dim)
        .map_err(|e| MLError::tensor_operation(format!("Softmax failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

/// Swish activation: x * sigmoid(x)
pub fn swish(input: &Tensor) -> MLResult<Tensor> {
    let sigmoid_x = sigmoid(input)?;
    let result = (input.inner() * sigmoid_x.inner())
        .map_err(|e| MLError::tensor_operation(format!("Swish multiplication failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

/// GELU activation: x * Φ(x) where Φ is the CDF of standard normal distribution
pub fn gelu(input: &Tensor) -> MLResult<Tensor> {
    let result = input.inner().gelu()
        .map_err(|e| MLError::tensor_operation(format!("GELU failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

/// Leaky ReLU: max(alpha * x, x)
pub fn leaky_relu(input: &Tensor, alpha: f32) -> MLResult<Tensor> {
    // Compute alpha * x
    let alpha_x = (input.inner() * alpha as f64)
        .map_err(|e| MLError::tensor_operation(format!("Multiplication failed: {}", e)))?;

    // max(alpha * x, x)
    let result = input.inner().maximum(&alpha_x)
        .map_err(|e| MLError::tensor_operation(format!("Maximum failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

/// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise
pub fn elu(input: &Tensor, alpha: f32) -> MLResult<Tensor> {
    let result = input.inner().elu(alpha as f64)
        .map_err(|e| MLError::tensor_operation(format!("ELU failed: {}", e)))?;

    Ok(Tensor::from_candle(result, input.shape().clone(), input.dtype()))
}

// Helper methods for Tensor
impl Tensor {
    /// Create Tensor from Candle tensor (internal helper)
    pub(crate) fn from_candle(
        inner: candle_core::Tensor,
        shape: crate::ml::tensor::Shape,
        dtype: crate::ml::tensor::DataType,
    ) -> Self {
        Self { inner, shape, dtype }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::tensor::{Shape, DataType};
    use crate::ml::types::Device;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_relu() {
        let device = Device::CPU;
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = Tensor::new(data, Shape::new(vec![5]), &device).unwrap();

        let result = relu(&tensor).unwrap();
        let output = result.to_vec1().unwrap();

        assert!(approx_eq(output[0], 0.0, 1e-6)); // -2 -> 0
        assert!(approx_eq(output[1], 0.0, 1e-6)); // -1 -> 0
        assert!(approx_eq(output[2], 0.0, 1e-6)); // 0 -> 0
        assert!(approx_eq(output[3], 1.0, 1e-6)); // 1 -> 1
        assert!(approx_eq(output[4], 2.0, 1e-6)); // 2 -> 2
    }

    #[test]
    fn test_sigmoid() {
        let device = Device::CPU;
        let data = vec![0.0];
        let tensor = Tensor::new(data, Shape::new(vec![1]), &device).unwrap();

        let result = sigmoid(&tensor).unwrap();
        let output = result.to_vec1().unwrap();

        // sigmoid(0) = 0.5
        assert!(approx_eq(output[0], 0.5, 1e-6));
    }

    #[test]
    fn test_tanh() {
        let device = Device::CPU;
        let data = vec![0.0];
        let tensor = Tensor::new(data, Shape::new(vec![1]), &device).unwrap();

        let result = tanh(&tensor).unwrap();
        let output = result.to_vec1().unwrap();

        // tanh(0) = 0
        assert!(approx_eq(output[0], 0.0, 1e-6));
    }

    #[test]
    fn test_softmax() {
        let device = Device::CPU;
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, Shape::new(vec![3]), &device).unwrap();

        let result = softmax(&tensor).unwrap();
        let output = result.to_vec1().unwrap();

        // Sum of probabilities should be 1
        let sum: f32 = output.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));

        // Values should be in (0, 1)
        for val in output {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_leaky_relu() {
        let device = Device::CPU;
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = Tensor::new(data, Shape::new(vec![5]), &device).unwrap();

        let alpha = 0.01;
        let result = leaky_relu(&tensor, alpha).unwrap();
        let output = result.to_vec1().unwrap();

        assert!(approx_eq(output[0], -0.02, 1e-6)); // -2 * 0.01 = -0.02
        assert!(approx_eq(output[1], -0.01, 1e-6)); // -1 * 0.01 = -0.01
        assert!(approx_eq(output[2], 0.0, 1e-6));   // 0
        assert!(approx_eq(output[3], 1.0, 1e-6));   // 1
        assert!(approx_eq(output[4], 2.0, 1e-6));   // 2
    }

    #[test]
    fn test_gelu() {
        let device = Device::CPU;
        let data = vec![0.0, 1.0, -1.0];
        let tensor = Tensor::new(data, Shape::new(vec![3]), &device).unwrap();

        let result = gelu(&tensor).unwrap();
        let output = result.to_vec1().unwrap();

        // GELU(0) ≈ 0
        assert!(approx_eq(output[0], 0.0, 1e-5));

        // GELU(1) ≈ 0.8413
        assert!(approx_eq(output[1], 0.8413, 1e-2));
    }

    #[test]
    fn test_apply_activation() {
        let device = Device::CPU;
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, Shape::new(vec![3]), &device).unwrap();

        // Test different activation types
        let relu_result = apply_activation(&tensor, ActivationType::ReLU).unwrap();
        assert!(relu_result.to_vec1().is_ok());

        let sigmoid_result = apply_activation(&tensor, ActivationType::Sigmoid).unwrap();
        assert!(sigmoid_result.to_vec1().is_ok());

        let linear_result = apply_activation(&tensor, ActivationType::Linear).unwrap();
        let output = linear_result.to_vec1().unwrap();
        // Linear activation should return input unchanged
        assert!(approx_eq(output[0], 1.0, 1e-6));
        assert!(approx_eq(output[1], 2.0, 1e-6));
        assert!(approx_eq(output[2], 3.0, 1e-6));
    }
}
