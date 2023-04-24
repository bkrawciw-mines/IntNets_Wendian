#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nets
Benjamin Krawciw
9/29/22

The necessary network theory functions for performing the Interferometer Small-
World Tests
"""

#Library imports
import numpy as np
import numpy.ma as ma
import scipy.linalg as sp
import scipy.sparse as sparse
from time import time

#Create a random number generator
rng = np.random.default_rng(int(time()))

#Tolerance for numerical things
TOL = 1e-8
    
#Creates a ring network
def makeRing(N, khalf):
    #Ensure that khalf is a valid input
    if khalf <= (N // 2):
        khalf_checked = khalf // 1
    else:
        khalf_checked = N // 2
    
    #Function that determines if entry (i, j) is included in ring
    def ringLogic(i, j):
        dist = np.minimum((j - i) % N, (i - j) % N) 
        far_enough = np.greater(dist, 0)
        close_enough = np.less_equal(dist, khalf_checked)
        return(np.logical_and(far_enough, close_enough))
    
    #Unweighted adjacency matrix
    A = np.fromfunction(ringLogic, (N, N))
    
    return(A)

#Function that randomly rewires ring according to beta
def rewire(A, beta):
    N = (A.shape)[0]
    
    #Select edges to rewire
    B = (np.less_equal(rng.random((N, N)), beta)) * A
        
    #Do not rewire self edges
    B = (np.ones((N, N)) - np.eye(N)) * B
    
    #Remove self-edges from rewire pool
    candidates = ma.masked_array(B, mask = np.eye(N))
        
    #Shuffle candidates
    choices = rng.permuted(candidates[~candidates.mask], axis = 0)
    Shuff = np.zeros_like(B)
    Shuff[~candidates.mask] = choices
    
    #Generate the rewired matrix
    return(A.astype(int) - B.astype(int) + Shuff.astype(int))


#Returns the adjacency matrix of a small-world network with the given parameters
def ws(params):
    #Unpack parameters
    N, khalf, beta, phi, weighting = (params[0], params[1], params[2], 
                                      params[3], params[4])
    
    #Generate ring
    ring = makeRing(N, khalf)
    
    #Rewire ring
    rewired = rewire(ring, beta)
    
    #Add phase
    w = (weighting * (1.0 /  khalf)) * np.exp(1.0j * phi)
    W = rewired.astype(complex) * w 
    
    #Return adjacency matrix as a sparse matrix
    return(sparse.csr_matrix(W))

#General clutering formula
def __clustHelper(W):
    #These functions only work for dense matrices
    Wsparse = W
    if not(sparse.issparse(W)):
        Wsparse = sparse.csr_matrix(W)
    
    #Record the network size
    N = np.min(W.shape)
    
    #Compute the numerator
    num = np.zeros(N)
    with np.nditer(num, op_flags = ['writeonly'], flags = ['f_index']) as iterator:
        for entry in iterator:
            #Record index 
            i = iterator.index
            #Compare paths through node i to shortcuts
            throughPaths = Wsparse[:, i] * Wsparse[i, :]
            adjShortcuts = np.abs(throughPaths + Wsparse) - np.abs(throughPaths)
            mat = (abs(throughPaths)).multiply(adjShortcuts)
            #Sum all triads on node i
            norm = mat.sum()
            entry[...] = norm
    
    #Compute the denominator
    den = np.zeros(N)
    with np.nditer(den, op_flags = ['writeonly'], flags = ['f_index']) as iterator:
        for entry in iterator:
            #Record index 
            i = iterator.index
            #Compare paths through node i to path of 1
            throughPaths = Wsparse[:, i] * Wsparse[i, :]
            adjShortcuts = np.ones(N)
            adjShortcuts[i] = 0 #Do not anticipate self-loops
            mat = (abs(throughPaths)).multiply(adjShortcuts)
            #Sum and record
            norm = mat.sum()
            entry[...] = norm
    
    #Meshing vector, scaled and shifted to match unweighted clustering
    Mvec = np.divide(num, den + TOL, out=np.zeros_like(num), where = den != 0.0)
    avg = np.mean(Mvec)
    return avg
    
#Returns the traditional, real-valued clustering coefficient
def Creal(W):
    return __clustHelper(np.abs(W))

#Returns the magnitude of the interferometric meshing
def Mesh(W):
    return __clustHelper(W)

#Returns the real-valued strongest path strength
def SPL(W):
    #Convert multiplicative path lengths to additive ones by taking a logarithm
    Wlog = sparse.csr_matrix((-np.log(np.abs(W.data)), 
                              W.indices, 
                              W.indptr))
    
    #Ensure the matrix is square
    wlogShape = Wlog.get_shape()
    newSize = min(wlogShape[0], wlogShape[1])
    #print(Wlog.get_shape())
    Wlog.resize(newSize, newSize)
    
    #Use Dijkstra's algorithm to solve for shortest paths in log space
    logShorts = sparse.csgraph.shortest_path(csgraph = Wlog)
    #Return the average
    SPL = np.mean(logShorts, (0, 1), where = (np.abs(logShorts) != np.inf))
    return(SPL)

#Returns the magnitude of the apparent path strength
def APL(W):
    #This measure comes from treating the network as an interferometer
    #This involves solving for the node-signal-value vector E
    #A constant-source-to-node vector S is also supplied
    #E is the solution to WE + S = E
    #So, the apparent paths are the entries of the MP inverse of (I-W)
    N = (W.shape)[0]
    WI = sparse.eye(N, format = "csr") - W
    
    #Ensure the matrix is square
    wShape = WI.get_shape()
    newSize = min(wShape[0], wShape[1])
    WI.resize((newSize, newSize))
    
    #Compute matrix inverse
    P = sp.inv(np.nan_to_num(WI.todense()))
    #print(P @ WI)
    APL = np.mean(-np.log(np.abs(P) + TOL), (0, 1))
    return(APL)

'''
#Testing the functions
ring = makeRing(100, 18)
#print(ring)
rewired = rewire(ring, 0.1)
#print(rewired)
W = ws([100, 8, 1.0, np.pi / 4, 0.95])
#print(W.diagonal())
#print(W)
Cr = Creal(W)
print(Cr)
M = Mesh(W)
print(M)
ap = APL(W)
#print(ap)
sps = SPL(W)
#print(sps)
'''