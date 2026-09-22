def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    
    import numpy as np
    a, b = np.array(a), np.array(b)
    if a.shape[1] != b.shape[0]:
        return -1
    
    c = np.matmul(a, b)
    return c
    