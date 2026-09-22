def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	import numpy as np
	A = np.array(matrix)
	return np.dot(scalar, A)