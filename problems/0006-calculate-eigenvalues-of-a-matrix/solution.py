import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	eigenvalues = np.real(np.linalg.eigvals(matrix))
	return eigenvalues