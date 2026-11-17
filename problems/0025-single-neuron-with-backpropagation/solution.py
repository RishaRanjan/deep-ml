import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    x = features
    y = labels
    w = initial_weights
    b = initial_bias

    mse_values = []
    n = len(x)

    for _ in range(epochs):
        # forward
        z = x @ w + b
        sigmoid = 1/(1 + np.exp(-z))
        # mse
        errors = sigmoid - y
        mse = 1/n * np.sum(errors**2)
        mse_values.append(mse)
        # backward
        sigmoid_der = sigmoid * (1- sigmoid)
        delta = 2/n * errors * sigmoid_der
        grad_w = x.T @ delta
        grad_b = np.sum(delta)

        #update
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

    return np.round(w,4), round(b, 4), mse_values