# -*- coding: utf-8 -*-
"""
Phase Variability Tests
Benjamin Krawciw
9/13/2023

This program runs a set of tests to see if introducing some variability to the
edge-weight phases in a Watts-Strogatz interferometer affects how interference
changes the small-world effect. 
"""

#Library imports 
import numpy as np
import nets
import csv
import itertools
import concurrent.futures as fut
from time import time
import scipy.sparse as sparse

#File to write results to
outFileName = "phaseVar.csv"

#Function that creates a numpy array with randomized phase shifts
def randPhaseMat(dev, N):
    randAngles = nets.rng.normal(0.0, dev, (N, N))
    randPhase = np.exp(1.0j * randAngles)
    return(randPhase)

#Function for WS model with phase variability
def ws_dev(params):
    #Unpack parameters
    N, khalf, beta, phi, weighting, dev = (params[0], params[1], params[2], 
                                           params[3], params[4], params[5])
    
    #Create a normal ws network first
    params0 = (N, khalf, beta, phi, weighting)
    ws0 = np.array((nets.ws(params0)).todense())
    
    #Now, apply a random phase shift to it
    phase = randPhaseMat(dev, N)
    ws = sparse.csr_matrix(phase * ws0)
    return(ws)

#Create a parameter space for these tests
Nrange = [500]
#Redundancy for tighter statistics
redundancy = 100
Ns = []
#Small networks need more trials
for Nval in Nrange:
    reps = redundancy * ((Nrange[-1]) // (Nval))
    [Ns.append(Nval) for i in range(reps)]
Ns = np.array(Ns)
kRange = [6]
#Create a beta space
betaRange = np.logspace(-5.0, 0.0, num = 10, base = 10, endpoint = True)
#Create a phi space
phiRange = [0.0, np.pi]
weighting = [0.9]
devSpace = [np.pi / 5.0]
#Total parameter space for tests
pSpace = itertools.product(Ns, kRange, betaRange, phiRange, weighting, 
                           devSpace)

#Function for each independent test
def Stest(params):
    print('sTest', params)
    #Unpack parameters
    N, khalf, beta, phi, weighting, dev = (params[0], params[1], params[2], 
                                           params[3], params[4], params[5])
    
    #Create a normal ws network first
    params0 = (N, khalf, beta, phi, weighting)
    
    #Create a WS interferometer 
    W = ws_dev(params)
    
    #Calculate small-world coefficients
    C, M = nets.Creal(W), nets.Mesh(W)
    SPL, APL = nets.SPL(W), nets.APL(W)
    
    #print(psutil.virtual_memory().used)
    return(params0 + (C, M, SPL, APL))

#Run all tests
if __name__ == '__main__':
    #Start a timer
    tStart = time()
    
    results = map(Stest, pSpace)

    #Creating the file to log data
    with open(outFileName, 'w', newline= '' ) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['N', 'kHalf', 'beta', 'phi', 'weighting', 'C',
                         'M', 'SPL', 'APL'])
        writer.writerows(results)
  
    #Report the execution time
    print("Tests completed. Time: %f s" % (time() - tStart))