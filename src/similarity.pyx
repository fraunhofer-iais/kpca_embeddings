"""
Similarity functions implemented in cython
"""
#import pyximport; pyximport.install()
import numpy as np
from libc.stdlib cimport malloc, free
import multiprocessing 
from scipy.spatial.distance import cosine

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

cdef chain(*iterables):
    """Make an iterator that returns elements from the first iterable
    until it is exhausted, then proceeds to the next iterable, until
    all of the iterables are exhausted. Used for treating consecutive
    sequences as a single sequence.
    """
    for it in iterables:
        for element in it:
            yield element

cpdef double bisim1(a, b, normalized=True) except? -1:
    """computes the binary version of bigram similarity.
    """
    cdef int i,j,la,lb,n,count
    
    pad_symbol = "-"

    n = 2    
    la = len(a) + 1
    lb = len(b) + 1
    s_a = chain((pad_symbol,) * (n - 1), a)
    s_a = chain(s_a, (pad_symbol,) * (n - 1))
    s_a = list(s_a)
    s_b = chain((pad_symbol,) * (n - 1), b)
    s_b = chain(s_b, (pad_symbol,) * (n - 1))
    s_b = list(s_b)
    count = max(0, len(s_a) - n + 1)
    s_a = [tuple(s_a[i:i + n]) for i in range(count)]
    count = max(0, len(s_b) - n + 1)
    s_b = [tuple(s_b[i:i + n]) for i in range(count)]

    m_np = np.zeros([la, lb], dtype=np.intc)
    cdef int [:,:] m = m_np

    for i in range(1, la):
        for j in range(1, lb):
            if (s_a[i - 1] == s_b[j - 1]):
                m[i][j] = m[i - 1][j - 1] + 1
            else:
                m[i][j] = max(m[i][j - 1], m[i - 1][j])
    la = la - 1
    lb = lb - 1
    if not normalized:
        return  float(m[la][lb]) - float(max(la, lb))
    return float(m[la][lb]) / float(max(la, lb))


cdef projectWord(word, kpcaModel, maxoids , degree=2):
    pair_sim = np.array([ bisim1(word,t) for t in maxoids])
    k = pair_sim**degree
    return k.dot(alphas / lambdas)

cpdef most_similar(w, kpcaModel, maxoids, topn=10):   
    v1 = projectWord(w)    
   
    D = np.array([(t, cosine(v1, kpcaModel[t])  ) for t in kpcaModel ])    

    D_sorted= sorted(D, key=lambda tup: float(tup[1]))
    return D_sorted[:topn]