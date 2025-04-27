# Gradient Descent Scratch Implementation

## Overview
This repository contains implementations of three main gradient descent optimization algorithms used in machine learning, specifically for linear regression. The implementations include batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

## What is Gradient Descent?
Gradient Descent is a fundamental optimization algorithm used in machine learning to minimize a cost function. It works by iteratively adjusting parameters in the direction of steepest descent of the cost function.

## Types of Gradient Descent

### 1. Batch Gradient Descent (BGD)
- **Description**: Updates model parameters using all training examples in each iteration
- **Advantages**:
  - Stable convergence
  - Guaranteed to reach global minimum for convex problems
  - Computationally efficient for small datasets
- **Disadvantages**:
  - Slow for large datasets
  - Requires entire dataset to fit in memory
- **When to Use**:
  - Small to medium-sized datasets
  - When computational resources are not a constraint
  - When you need stable and deterministic results

### 2. Stochastic Gradient Descent (SGD)
- **Description**: Updates parameters using one randomly selected training example at a time
- **Advantages**:
  - Fast iteration speed
  - Can handle large datasets
  - Can escape local minima due to noise
- **Disadvantages**:
  - High variance in parameter updates
  - May not converge to exact minimum
  - Requires careful learning rate scheduling
- **When to Use**:
  - Very large datasets
  - Online learning scenarios
  - When computational resources are limited

### 3. Mini-batch Gradient Descent (MBGD)
- **Description**: Updates parameters using a small random subset of training data
- **Advantages**:
  - Best of both BGD and SGD
  - More stable than SGD
  - More memory efficient than BGD
  - Allows for GPU acceleration
- **Disadvantages**:
  - Requires tuning of batch size
  - Still needs careful learning rate selection
- **When to Use**:
  - **Industry Standard Choice**: This is the most commonly used variant in modern machine learning
  - Deep learning applications
  - Large-scale machine learning problems

## Industry Best Practices

### Choosing the Right Variant
1. **Default Choice**: Start with Mini-batch Gradient Descent
   - Typical batch sizes: 32, 64, 128, 256
   - Provides good balance between speed and stability

2. **Use Batch Gradient Descent when**:
   - Dataset fits in memory
   - Need deterministic results
   - Working with simple models

3. **Use Stochastic Gradient Descent when**:
   - Dealing with extremely large datasets
   - Need real-time learning capability
   - Resources are very limited

### Optimization Tips
1. **Learning Rate Selection**:
   - Start with a small learning rate (e.g., 0.01, 0.001)
   - Use learning rate scheduling for better convergence
   - Consider adaptive learning rate methods (Adam, RMSprop)

2. **Batch Size Guidelines**:
   - Smaller batches: Better generalization, more noise
   - Larger batches: More stable, might overfit
   - Popular sizes: 32 (default), 64, 128

3. **Convergence Monitoring**:
   - Track loss/cost function
   - Monitor validation metrics
   - Use early stopping when needed

## Implementation Details
This repository includes:
- Type-hinted Python implementations
- Visualization tools for convergence
- Performance comparison utilities
- Example usage with real-world datasets

## Usage Example
```python
from gradient import GradientDescent

# Initialize the model
gd = GradientDescent()

# Train using mini-batch gradient descent (recommended)
weights, intercept, cost_history, _, _ = gd.mini_batch_gradient(
    X=X_train,
    Y=y_train,
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# Visualize convergence
gd.plot_convergence(cost_history)
```

## Requirements
- NumPy
- Matplotlib
- scikit-learn (for data preprocessing and evaluation)
