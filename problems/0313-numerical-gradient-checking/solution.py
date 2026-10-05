import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    """
    numerical_grad = np.zeros_like(x, dtype=np.float64)
    
    # Use np.nditer to handle scalar, 1D, or multi-dimensional arrays
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        # Evaluate f(x + epsilon)
        x[idx] = old_val + epsilon
        fx_plus = f(x)
        
        # Evaluate f(x - epsilon)
        x[idx] = old_val - epsilon
        fx_minus = f(x)
        
        # Reset original value
        x[idx] = old_val
        
        # Centered finite difference formula
        numerical_grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        it.iternext()
    
    # Compute relative error
    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = np.maximum(1e-8, np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad))
    relative_error = np.max(numerator / denominator)
    
    return numerical_grad, relative_error