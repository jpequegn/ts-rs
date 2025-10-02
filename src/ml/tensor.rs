//! Tensor operations and GPU acceleration support

use candle_core::{DType, Device as CandleDevice, Tensor as CandleTensor, Shape as CandleShape};
use serde::{Deserialize, Serialize};
use crate::ml::{MLError, MLResult};
use crate::ml::types::Device;

/// Data type for tensors
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U8,
}

impl DataType {
    /// Convert to Candle DType
    pub fn to_candle_dtype(&self) -> DType {
        match self {
            DataType::F32 => DType::F32,
            DataType::F64 => DType::F64,
            DataType::I32 => DType::U32, // Use U32 as I32 is not available in candle 0.9
            DataType::I64 => DType::I64,
            DataType::U8 => DType::U8,
        }
    }
}

/// Shape of a tensor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Convert to Candle shape
    pub fn to_candle_shape(&self) -> CandleShape {
        CandleShape::from(self.dims.as_slice())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

/// GPU backend types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GpuBackend {
    CUDA,
    Metal,
}

/// Tensor wrapper for Candle tensors
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) inner: CandleTensor,
    pub(crate) shape: Shape,
    pub(crate) dtype: DataType,
}

impl Tensor {
    /// Create a new tensor from raw data
    pub fn new(data: Vec<f32>, shape: Shape, device: &Device) -> MLResult<Self> {
        let candle_device = Self::device_to_candle(device)?;
        let candle_shape = shape.to_candle_shape();

        let inner = CandleTensor::from_vec(data, &candle_shape, &candle_device)
            .map_err(|e| MLError::tensor_operation(format!("Failed to create tensor: {}", e)))?;

        Ok(Self {
            inner,
            shape,
            dtype: DataType::F32,
        })
    }

    /// Create a tensor of zeros
    pub fn zeros(shape: Shape, dtype: DataType, device: &Device) -> MLResult<Self> {
        let candle_device = Self::device_to_candle(device)?;
        let candle_shape = shape.to_candle_shape();

        let inner = CandleTensor::zeros(&candle_shape, dtype.to_candle_dtype(), &candle_device)
            .map_err(|e| MLError::tensor_operation(format!("Failed to create zeros tensor: {}", e)))?;

        Ok(Self {
            inner,
            shape,
            dtype,
        })
    }

    /// Create a tensor of ones
    pub fn ones(shape: Shape, dtype: DataType, device: &Device) -> MLResult<Self> {
        let candle_device = Self::device_to_candle(device)?;
        let candle_shape = shape.to_candle_shape();

        let inner = CandleTensor::ones(&candle_shape, dtype.to_candle_dtype(), &candle_device)
            .map_err(|e| MLError::tensor_operation(format!("Failed to create ones tensor: {}", e)))?;

        Ok(Self {
            inner,
            shape,
            dtype,
        })
    }

    /// Create a tensor with random values
    pub fn randn(shape: Shape, mean: f64, std: f64, device: &Device) -> MLResult<Self> {
        let candle_device = Self::device_to_candle(device)?;
        let candle_shape = shape.to_candle_shape();

        let inner = CandleTensor::randn(mean, std, &candle_shape, &candle_device)
            .map_err(|e| MLError::tensor_operation(format!("Failed to create random tensor: {}", e)))?;

        Ok(Self {
            inner,
            shape,
            dtype: DataType::F32,
        })
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the data type
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the device
    pub fn device(&self) -> Device {
        Self::candle_to_device(self.inner.device())
    }

    /// Convert internal Device to Candle Device
    fn device_to_candle(device: &Device) -> MLResult<CandleDevice> {
        match device {
            Device::CPU => Ok(CandleDevice::Cpu),
            Device::CUDA(idx) => {
                CandleDevice::new_cuda(*idx)
                    .map_err(|e| MLError::device(format!("CUDA device not available: {}", e)))
            },
            Device::Metal => {
                CandleDevice::new_metal(0)
                    .map_err(|e| MLError::device(format!("Metal device not available: {}", e)))
            },
            Device::Auto => {
                // Try CUDA first, then Metal, fallback to CPU
                if let Ok(dev) = CandleDevice::new_cuda(0) {
                    Ok(dev)
                } else if let Ok(dev) = CandleDevice::new_metal(0) {
                    Ok(dev)
                } else {
                    Ok(CandleDevice::Cpu)
                }
            },
        }
    }

    /// Convert Candle Device to internal Device
    fn candle_to_device(device: &CandleDevice) -> Device {
        if device.is_cpu() {
            Device::CPU
        } else if device.is_cuda() {
            // Extract CUDA device index if possible
            Device::CUDA(0) // Simplified - actual implementation would extract index
        } else if device.is_metal() {
            Device::Metal
        } else {
            Device::CPU
        }
    }

    /// Get reference to inner Candle tensor
    pub fn inner(&self) -> &CandleTensor {
        &self.inner
    }

    /// Convert to CPU device
    pub fn to_cpu(&self) -> MLResult<Self> {
        let cpu_tensor = self.inner.to_device(&CandleDevice::Cpu)
            .map_err(|e| MLError::tensor_operation(format!("Failed to move to CPU: {}", e)))?;

        Ok(Self {
            inner: cpu_tensor,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    /// Get tensor data as Vec<f32>
    pub fn to_vec1(&self) -> MLResult<Vec<f32>> {
        self.inner.to_vec1()
            .map_err(|e| MLError::tensor_operation(format!("Failed to convert to vec: {}", e)))
    }

    /// Get tensor data as 2D Vec
    pub fn to_vec2(&self) -> MLResult<Vec<Vec<f32>>> {
        self.inner.to_vec2()
            .map_err(|e| MLError::tensor_operation(format!("Failed to convert to 2D vec: {}", e)))
    }
}

/// Tensor operations trait
pub trait TensorOps {
    /// Add two tensors
    fn add(&self, other: &Self) -> MLResult<Self> where Self: Sized;

    /// Subtract two tensors
    fn sub(&self, other: &Self) -> MLResult<Self> where Self: Sized;

    /// Multiply two tensors element-wise
    fn mul(&self, other: &Self) -> MLResult<Self> where Self: Sized;

    /// Divide two tensors element-wise
    fn div(&self, other: &Self) -> MLResult<Self> where Self: Sized;

    /// Matrix multiplication
    fn matmul(&self, other: &Self) -> MLResult<Self> where Self: Sized;

    /// Transpose tensor
    fn transpose(&self, dim0: usize, dim1: usize) -> MLResult<Self> where Self: Sized;

    /// Reshape tensor
    fn reshape(&self, shape: Shape) -> MLResult<Self> where Self: Sized;
}

impl TensorOps for Tensor {
    fn add(&self, other: &Self) -> MLResult<Self> {
        let result = (&self.inner + &other.inner)
            .map_err(|e| MLError::tensor_operation(format!("Addition failed: {}", e)))?;

        Ok(Self {
            inner: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    fn sub(&self, other: &Self) -> MLResult<Self> {
        let result = (&self.inner - &other.inner)
            .map_err(|e| MLError::tensor_operation(format!("Subtraction failed: {}", e)))?;

        Ok(Self {
            inner: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    fn mul(&self, other: &Self) -> MLResult<Self> {
        let result = (&self.inner * &other.inner)
            .map_err(|e| MLError::tensor_operation(format!("Multiplication failed: {}", e)))?;

        Ok(Self {
            inner: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    fn div(&self, other: &Self) -> MLResult<Self> {
        let result = (&self.inner / &other.inner)
            .map_err(|e| MLError::tensor_operation(format!("Division failed: {}", e)))?;

        Ok(Self {
            inner: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    fn matmul(&self, other: &Self) -> MLResult<Self> {
        let result = self.inner.matmul(&other.inner)
            .map_err(|e| MLError::tensor_operation(format!("Matrix multiplication failed: {}", e)))?;

        // Calculate output shape
        let result_shape = result.shape();
        let dims: Vec<usize> = result_shape.dims().to_vec();

        Ok(Self {
            inner: result,
            shape: Shape::new(dims),
            dtype: self.dtype,
        })
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> MLResult<Self> {
        let result = self.inner.transpose(dim0, dim1)
            .map_err(|e| MLError::tensor_operation(format!("Transpose failed: {}", e)))?;

        let result_shape = result.shape();
        let dims: Vec<usize> = result_shape.dims().to_vec();

        Ok(Self {
            inner: result,
            shape: Shape::new(dims),
            dtype: self.dtype,
        })
    }

    fn reshape(&self, shape: Shape) -> MLResult<Self> {
        let candle_shape = shape.to_candle_shape();
        let result = self.inner.reshape(&candle_shape)
            .map_err(|e| MLError::tensor_operation(format!("Reshape failed: {}", e)))?;

        Ok(Self {
            inner: result,
            shape,
            dtype: self.dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn test_tensor_zeros() {
        let shape = Shape::new(vec![2, 3]);
        let device = Device::CPU;
        let tensor = Tensor::zeros(shape, DataType::F32, &device).unwrap();

        assert_eq!(tensor.shape().dims, vec![2, 3]);
        assert_eq!(tensor.dtype(), DataType::F32);
    }

    #[test]
    fn test_tensor_ones() {
        let shape = Shape::new(vec![2, 3]);
        let device = Device::CPU;
        let tensor = Tensor::ones(shape, DataType::F32, &device).unwrap();

        assert_eq!(tensor.shape().dims, vec![2, 3]);
    }

    #[test]
    fn test_tensor_addition() {
        let shape = Shape::new(vec![2, 2]);
        let device = Device::CPU;

        let t1 = Tensor::ones(shape.clone(), DataType::F32, &device).unwrap();
        let t2 = Tensor::ones(shape, DataType::F32, &device).unwrap();

        let result = t1.add(&t2).unwrap();
        let data = result.to_vec2().unwrap();

        // All elements should be 2.0
        for row in data {
            for val in row {
                assert!((val - 2.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_tensor_matmul() {
        let device = Device::CPU;

        // Create 2x3 matrix
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = Tensor::new(data1, Shape::new(vec![2, 3]), &device).unwrap();

        // Create 3x2 matrix
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t2 = Tensor::new(data2, Shape::new(vec![3, 2]), &device).unwrap();

        // Result should be 2x2
        let result = t1.matmul(&t2).unwrap();
        assert_eq!(result.shape().dims, vec![2, 2]);
    }

    #[test]
    fn test_tensor_reshape() {
        let device = Device::CPU;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let tensor = Tensor::new(data, Shape::new(vec![2, 3]), &device).unwrap();
        let reshaped = tensor.reshape(Shape::new(vec![3, 2])).unwrap();

        assert_eq!(reshaped.shape().dims, vec![3, 2]);
    }
}
