import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	dot_product = np.dot(v1,v2)
	norm_A = np.linalg.norm(v1)
	norm_B = np.linalg.norm(v2)

	# Calculate cosine similarity
	cosine_sim = dot_product / (norm_A * norm_B)

	return  cosine_sim