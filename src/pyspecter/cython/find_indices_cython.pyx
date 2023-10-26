# find_indices_cython.pyx
cimport numpy as np

def find_indices(np.ndarray[np.float32_t, ndim=1] X, np.ndarray[np.float32_t, ndim=1] Y):
    cdef int i, j
    cdef int X_length = X.shape[0]
    cdef int Y_length = Y.shape[0]
    cdef list result = []

    i = 1
    j = 1

    while i < X_length and j < Y_length:
        if X[i] > Y[j-1] and Y[j] > X[i-1]:
            result.append((i, j))
            i += 1
        elif X[i] <= Y[j-1]:
            i += 1
        else:
            j += 1

    return result
