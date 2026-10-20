import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	entropy = -1 * sum([p*np.log(p) for p in P]) if 0.0 not in P else 0.0
	cross_entropy = -1 * sum([p*np.log(q) for p, q in zip(P, Q)]) if 0.0 not in Q else 0.0

	return (entropy, cross_entropy)