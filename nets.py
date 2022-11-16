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

#Create a random number generator
rng = np.random.default_rng(0)

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
    N, khalf, beta, phi = params[0], params[1], params[2], params[3]
    
    #Generate ring
    ring = makeRing(N, khalf)
    
    #Rewire ring
    rewired = rewire(ring, beta)
    
    #Add phase
    w = ((0.25 /  khalf)) * np.exp(1.0j * phi)
    W = rewired.astype(complex) * w 
    
    #Return adjacency matrix as a sparse matrix
    return(sparse.csr_matrix(W))

#General clutering formula
def __clustHelper(W):
    num = W @ W.getH() @ W #Numerator
    O = sparse.csr_matrix(np.ones_like(W.todense())) #Ones matrix
    norm = (np.abs(W) @ O @ np.abs(W)).diagonal() #Normalization term
    norm += (norm < TOL) #Preventing division by zero
    output = num.diagonal() / norm
    avg = np.mean(abs(output))
    return avg
    
#Returns the traditional, real-valued clustering coefficient
def Creal(W):
    return __clustHelper(abs(W))

#Returns the magnitude of the interferometric meshing
def Mesh(W):
    return __clustHelper(W)

#Returns the real-valued strongest path strength
def SPS(W):
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
    strongs = np.exp(-logShorts)
    #Return the average
    SPS = np.mean(np.abs(strongs), (0, 1))
    return(SPS)

#Returns the magnitude of the apparent path strength
def APS(W):
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
    APS = np.mean(np.abs(P), (0, 1))
    return(APS)

'''
#Testing the functions
ring = makeRing(100, 18)
print(ring)
rewired = rewire(ring, 0.1)
print(rewired)
W = ws([1000, 4, 1.0, np.pi / 4])
print(W.diagonal())
print(W)
Cr = Creal(W)
print(Cr)
M = Mesh(W)
print(M)
ap = APL(W)
print(ap)
sps = SPS(W)
print(sps)
'''