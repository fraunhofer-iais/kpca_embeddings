#   Copyright 2018 Fraunhofer IAIS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np

cdef ngrams(s, int n):
    """List of ngrams of the string s after prepending and appending a filler (repetitions included). Order of ngrams is kept .
    """
    cdef int i
    s2 = " "+s+" "        
    return [s2[i:i+n] for i in range(len(s2)-n+1)]

cpdef double ngram_sim(x, y, int n= 2) except? -1:
    """Binary version of n-gram similarity
    """
    cdef int x_len,y_len,i,j

    ng_a = ngrams(x,n)
    ng_b = ngrams(y,n)
    x_len = len(ng_a)
    y_len = len(ng_b)

    np_mem = np.zeros([x_len + 1, y_len + 1], dtype=np.intc)
    cdef int [:,:] mem_table = np_mem

    for i in range(1, x_len + 1):
        for j in range(1, y_len + 1):
            mem_table[i][j] = max(mem_table[i][j - 1], mem_table[i - 1][j], mem_table[i - 1][j - 1] + (ng_a[i - 1] == ng_b[j - 1]) )
    
    return float(mem_table[x_len][y_len]) / float(max(x_len, y_len))
