import numpy as np

def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    return (
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
        - 0.5
    )