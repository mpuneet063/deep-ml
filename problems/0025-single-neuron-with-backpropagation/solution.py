import numpy as np

def train_neuron(features, labels, initial_weights, initial_bias,
                 learning_rate, epochs):

    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []

    for _ in range(epochs):

        # Forward pass
        z = features @ weights + bias
        pred = 1 / (1 + np.exp(-z))

        # MSE BEFORE update
        error = pred - labels
        mse = np.mean(error ** 2)
        mse_values.append(mse)

        # Backpropagation
        grad_z = (2 / len(labels)) * error * pred * (1 - pred)
        grad_w = features.T @ grad_z
        grad_b = np.sum(grad_z)

        # Update
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias, mse_values