#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interferometer Small-World Tests
Benjamin Krawciw
9/30/2022

Creates a suite of watts-strogatz interferometers, computes their network 
measures, and saves them using an HPC cluster.
"""

#Library imports
import numpy as np
import nets
import csv
import itertools
from mpi4py.futures import MPIPoolExecutor
from time import time

#Name for the output file
outFileName = 'thick500.csv'

#Create the parameter space for testing
Nrange = [500]
#Redundancy for tighter statistics
redundancy = 1
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
phiRange = np.linspace(0.0, 2.0 * np.pi, num = 10, endpoint = False)
weighting = [0.95]
#Total parameter space for tests
pSpace = itertools.product(Ns, kRange, betaRange, phiRange, weighting)

#Function for each independent test
def Stest(params):
    print('sTest', params)
    #Create a WS interferometer 
    W = nets.ws(params)
    
    #Calculate small-world coefficients
    C, M = nets.Creal(W), nets.Mesh(W)
    SPL, APL = nets.SPL(W), nets.APL(W)
    
    #print(psutil.virtual_memory().used)
    return(params + (C, M, SPL, APL))

#Run all tests
if __name__ == '__main__':
    #Start a timer
    tStart = time()
    
    #Run an MPI executor on the available nodes
    with MPIPoolExecutor() as executor:
        #Use the executor to run the Stest process on parameters in pSpace
        results = executor.map(Stest, pSpace, 
                               #Limit executions to __ seconds
                               timeout = 10.0,
                               #Send tasks this many tasks at a time 
                               chunksize = 10,
                               #I don't care about the order tests run in
                               unordered = True)
    
        #Creating the file to log data
        with open(outFileName, 'w', newline= '' ) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['N', 'kHalf', 'beta', 'phi', 'weighting', 'C', 'M', 'SPL', 'APL'])
            writer.writerows(results)
    
    #Report the execution time
    print("Tests completed. Time: %f s" % (time() - tStart))
    
