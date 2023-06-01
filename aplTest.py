# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:21:56 2023

@author: benja
"""
import nets 
import numpy as np
import scipy.sparse as sparse

WI = sparse.csr_matrix(np.random.rand(10, 10) * .19)
print(nets.APL(WI))