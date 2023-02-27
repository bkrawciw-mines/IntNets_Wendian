# -*- coding: utf-8 -*-
"""
Patches
Benjamin Krawciw
2/27/2023

This program creates a file with supplementary data near some critical points,
particularly around phi = 0 and beta between beta = e^-5 and beta = 1. I call
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

#Naming output files
bpName = 'betaPatch.csv'
ppName = 'phiPatch.csv'

#Identifying existing gridpoints in critical areas
betaMinIndex = np.argmin(np.abs(IntTests.betaRange - np.exp(-5.0)))
betaMaxIndex = np.argmax(IntTests.betaRange)
betaRough = IntTests.betaRange[betaMinIndex:betaMaxIndex + 1]

phiRoughLow = IntTests.phiRange[0:5]
phiRoughHigh = IntTests.phiRange[-5:]
phiRough = np.concatenate((phiRoughLow, phiRoughHigh))

#Filling in grid more densely
betaNum = (500 // len(betaRough)) * len(betaRough)
betaFine = np.geomspace(betaRough[0], betaRough[-1], endpoint=True, 
                        num = betaNum)

phiFineLow = np.linspace(0.0, phiRoughLow[-1], endpoint=True,
                         num = 50)
phiFineHigh = np.linspace(phiRoughHigh[0], phiRoughHigh[-1], endpoint=True,
                          num = 50)
phiFine = np.concatenate((phiFineLow, phiFineHigh))

#Creating full parameter space for patch testing
betaParams = itertools.product(
    IntTests.Ns, 
    IntTests.kRange, 
    betaFine, 
    phiRough, 
    IntTests.weighting)

phiParams = itertools.product(
    IntTests.Ns, 
    IntTests.kRange, 
    betaRough, 
    phiFine, 
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
        
        '''
        #Running Stest on the parameters in phiParams
        phiResults = executor.map(IntTests.Stest, phiParams)
        
        #Creating the file to log phi patch data
        with open(bpName, 'w', newline= '' ) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['N', 'kHalf', 'beta', 'phi', 'weighting', 'C', 'M', 'SPL', 'APL'])
            writer.writerows(betaResults)
        '''
        
        executor.shutdown(wait = False)
  
    #Report the execution time
    print("Tests completed. Time: %f s" % (time() - tStart))
