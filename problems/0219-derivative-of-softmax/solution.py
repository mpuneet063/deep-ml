import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	def softmax(x1, x):
		exp_sum = sum([np.exp(a) for a in x])
		return np.exp(x1) / exp_sum

	derivative = [[0] * len(x) for _ in range(len(x))]
	
	for i in range(len(x)):
		for j in range(len(x)):
			if i == j:
				derivative[i][j] = softmax(x[i], x)*(1-softmax(x[j], x))
			else:
				derivative[i][j] = softmax(x[i], x)*(-softmax(x[j], x))

	return derivative