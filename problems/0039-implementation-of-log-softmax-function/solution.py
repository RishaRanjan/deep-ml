import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    max_val = np.max(scores)
    log_softmax = scores - max_val - np.log(np.sum(np.exp(scores - np.max(scores))))
    return log_softmax
