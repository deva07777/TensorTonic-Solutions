import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    e=y_true-y_pred
    abs_err=np.abs(e)
    losses=np.where(abs_err <= delta,
                   0.5*e**2,
                   delta*(abs_err-0.5*delta),

                   )
    return float(np.mean(losses))
    