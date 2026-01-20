import numpy as np

def single_neuron(inputs, weights, bias):
    """
    A simple implementation of a single neuron forward pass.
    
    Args:
        inputs (numpy.ndarray): Input features, shape (n_features,).
        weights (numpy.ndarray): Weights for the neuron, shape (n_features,).
        bias (float): Bias term for the neuron.
        
    Returns: activation (float): The output of the neuron after applying the activation function.
    """
    
    # Calculate the weighted sum
    weighted_sum = np.dot(inputs, weights) + bias
    # Apply activation function (ReLU in this case)
    activation = max(0, weighted_sum)
    return activation

# Example usage
if __name__ == "__main__":
    # Define input features, weights, and bias
    inputs  = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.2, 0.8, -0.5])
    bias = 2.0
                
    # Compute the neuron's output
    output = single_neuron(inputs, weights, bias)
    print("Neuron output:", output)
    