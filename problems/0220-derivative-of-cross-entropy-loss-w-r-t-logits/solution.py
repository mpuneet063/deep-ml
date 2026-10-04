import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	y = [0]*len(logits)
	y[target] = 1
	def softmax(x1, x):
		exp_sum = sum([np.exp(a) for a in x])
		return np.exp(x1) / exp_sum

	q = [softmax(logits[i], logits) for i in range(len(logits))]

	return [(q[i] - y[i]) for i in range(len(y))]