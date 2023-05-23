# -*- coding: utf-8 -*-
"""
Beta Patch
Benjamin Krawciw
3/4/2023

This program creates a file with supplementary data near some critical points,
particularly beta between beta = e^-5 and beta = 1. I call
these files "data patches." 
"""

#Library imports
import numpy as np
import csv
from time import time
import itertools
from mpi4py.futures import MPIPoolExecutor

#Base output on parent program IntTests.py
import IntTests

#Naming output file
bpName = 'betaPatch_redundancy2.csv'

#Identifying existing gridpoints in critical areas
betaMinIndex = np.argmin(np.abs(IntTests.betaRange - np.exp(-5.0)))
betaMaxIndex = np.argmax(IntTests.betaRange)
betaRough = IntTests.betaRange[betaMinIndex:betaMaxIndex + 1]

#Filling in grid more densely
betaNum = (500 // len(betaRough)) * len(betaRough)
betaFine = np.geomspace(betaRough[0], betaRough[-1], endpoint=True, 
                        num = betaNum)

phiRoughLow = IntTests.phiRange[0:5]
phiRoughHigh = IntTests.phiRange[-5:]
phiRough = np.concatenate((phiRoughLow, phiRoughHigh))

#Increasing redundancy
#Redundancy for tighter statistics
redundancy = 4000
Ns = []
#Small networks need more trials
for Nval in IntTests.Nrange:
    reps = redundancy * ((IntTests.Nrange[-1]) // (Nval))
    [Ns.append(Nval) for i in range(reps)]
Ns = np.array(Ns)

#Creating full parameter space for patch testing
betaParams = itertools.product(
    Ns, 
    IntTests.kRange, 
    IntTests.betaRange, 
    phiRough, 
    IntTests.weighting)

#Run all tests
if __name__ == '__main__':
    #Start a timer
    tStart = time()
    
    #Run an MPI executor on the available nodes
    with MPIPoolExecutor() as executor:
        #Use the executor to run the Stest on parameters in betaParams
        betaResults = executor.map(IntTests.Stest, betaParams)
    
        #Creating the file to log beta patch data
        with open(bpName, 'w', newline= '' ) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['N', 'kHalf', 'beta', 'phi', 'weighting', 'C', 'M', 'SPL', 'APL'])
            writer.writerows(betaResults)
        
        executor.shutdown(wait = False)
  
    #Report the execution time
    print("Tests completed. Time: %f s" % (time() - tStart))
