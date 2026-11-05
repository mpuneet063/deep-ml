import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	
	if x0 is None:
		x = np.zeros_like(b, dtype = float)
	else:
		x = np.array(x0, dtype=float)
	r = b - A @ x
	p = r.copy()			# initial search direction
	for k in range(n):
		alpha = (r.T @ r) / (p.T@A@p)		# step size
		x = x + alpha * p 					# update solution
		r_k = r - alpha * (A @ p) 			# update residue
		if np.linalg.norm(r_k) < tol:
			break
		beta = (r_k.T @ r_k) / (r.T @ r)
		p = r_k + beta * p 					# update search direction 
		r = r_k

	return x