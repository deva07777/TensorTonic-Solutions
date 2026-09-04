import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a=np.array(a,dtype=float)
    b=np.array(b,dtype=float)
    dot =np.dot(a,b)
    a1=np.linalg.norm(a)
    b1=np.linalg.norm(b)
    if a1==0 or b1==0:
        return 0.0
    return float((dot)/(a1*b1))
    pass