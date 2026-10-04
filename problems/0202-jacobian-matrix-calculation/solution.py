import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	
	jacob = [[0] * len(x) for _ in range(len(f(x)))]
	for i in range(len(f(x))):
		for j in range(len(x)):
			x_plus = x.copy()
			x_plus[j] += h
			jacob[i][j] = (f(x_plus)[i] - f(x)[i])/h 

	return jacob