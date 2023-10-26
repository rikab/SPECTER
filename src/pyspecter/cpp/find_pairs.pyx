# find_pairs.pyx
cimport cython
from libcpp.vector cimport vector



cdef extern from "find_pairs.h":
    cdef vector[tuple[int, int]] findPairs(vector[double], vector[double])

def find_pairs(list X, list Y):
    cdef vector[double] X_c = X
    cdef vector[double] Y_c = Y
    return findPairs(X_c, Y_c)
