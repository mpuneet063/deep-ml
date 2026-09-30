def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    from collections import Counter
    count = Counter(samples)

    res = []
    for c in count:
        res.append((c,count[c]/len(samples)))
    return res
