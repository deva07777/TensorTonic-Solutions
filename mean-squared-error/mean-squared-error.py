import numpy as np

def mean_squared_error(y_pred, y_true):
    predictions = np.asarray(y_pred, dtype=float)
    targets = np.asarray(y_true, dtype=float)
    return float(np.mean((predictions - targets) ** 2))
