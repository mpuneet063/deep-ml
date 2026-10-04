import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	func_map = {
		'square': (lambda x: x**2, lambda x: 2*x),
		'sin': (lambda x: np.sin(x), lambda x: np.cos(x)),
		'exp': (lambda x: np.exp(x), lambda x: np.exp(x)),
		'log': (lambda x: np.log(x), lambda x: 1/x)
	}
	value, grad = x, 1

	for func in reversed(functions):
		f, d = func_map[func]

		grad *= d(value)
		value = f(value)


	return grad
