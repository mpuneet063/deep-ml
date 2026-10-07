import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	gradient = np.array(gradient)
	magn = np.linalg.norm(gradient)
	if magn < 1e-10:
		direc = np.zeros_like(gradient)
		desc = np.zeros_like(gradient)

	else:
		direc = gradient / magn
		desc = -direc

	return {
		'magnitude':float(magn),
		'direction':direc.tolist(),
		'descent_direction':desc.tolist()
	}