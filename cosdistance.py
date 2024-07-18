import numpy as np
def find_cos_dist(m: np.ndarray, n: np.ndarray):
    """
    Function to measure cosine distance between face descriptors. 
    It is useful to be able to take in a shape-(M, D) array of M descriptor 
    vectors and a shape-(N, D) array of N descriptor vectors, and compute a shape-(M, N) 
    array of cosine distances – this holds all MxN combinations of pairwise cosine distances.

    """
    m_norm = m / np.linalg.norm(m, axis=1, keepdims=True) # m / | m |
    n_norm = n / np.linalg.norm(n, axis =1, keepdims=True) # n / | n |

    dot_mn = np.dot(m_norm, n_norm.T) # gives shape (m, n)

    return 1 - dot_mn







    





