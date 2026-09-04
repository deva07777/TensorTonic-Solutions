import math
import numpy as np

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    a = np.array(x1, dtype=float)
    b = np.array(x2, dtype=float)
    
    # Compute cosine similarity: (a . b) / (||a|| * ||b||)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    cos_sim = dot_product / (norm_a * norm_b)
    
    # Compute loss based on label
    if label == 1:
        loss = 1.0 - cos_sim
    else:
        loss = max(0.0, cos_sim - margin)
        
    return float(loss)