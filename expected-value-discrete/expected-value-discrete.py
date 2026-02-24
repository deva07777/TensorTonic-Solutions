import numpy as np

def expected_value_discrete(x, p):
    """
    x : numpy array of values
    p : numpy array of probabilities
    Returns: float expected value
    """
    x=np.array(x)
    p=np.array(p)
    if not np.isclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1")

    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")

    return np.sum(x * p)