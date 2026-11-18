# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    def _loss(true, pred):
        return np.mean((pred - true)**2)
    epochs = 5000
    lr = 0.01
    # using momentum based gradient-descent
    v_w, v_b = 0.0, 0.0
    beta = 0.9    
    for epoch in range(epochs):
        train_loss = 0.0
        pred = X @ W + b
        error = pred - y
        train_loss = _loss(y, pred)
        lw = 2 * X.T @ error / len(y)
        lb = 2 * np.sum(error) / len(y)
        # update velocities
        v_w = beta * v_w + (1-beta) * lw
        v_b = beta * v_b + (1-beta) * lb
        W -= lr * v_w
        b -= lr * v_b
        # if epoch % 10 == 0:
            # print(f'Epoch {epoch + 1} | Loss = {train_loss}')
    return W, b