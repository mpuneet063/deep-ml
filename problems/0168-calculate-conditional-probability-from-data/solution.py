def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    xs = [(i,j) for (i,j) in data if i==x]
    ys = [(i,j) for (i,j) in xs if j==y]
    if len(xs) == 0:
      return 0
    return len(ys) / len(xs)
