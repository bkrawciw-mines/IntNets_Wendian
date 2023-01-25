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
#from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
#import os
#import psutil

#Name for the output file
outFileName = 'IntTestsData.csv'

#Create the parameter space for testing
Nrange = np.arange(start = 100, stop = 550, step = 50)
#Small networks need more trials
Ns = []
for Nval in Nrange:
    reps = 100 * (Nrange[-1]**2) // (Nval**2)
    [Ns.append(Nval) for i in range(reps)]
Ns = np.array(Ns)
kRange = np.arange(start = 1, stop = 22, step = 2)
betaRange = np.logspace(-5.0, 0.0, num = 10, base = 10, endpoint = True)
phiRange = np.linspace(0.0, 2.0 * np.pi, num = 10, endpoint = False)
pSpace = itertools.product(Ns, kRange, betaRange, phiRange)
#print(len([params for params in pSpace]))

#Function for each independent test
def Stest(params):
    print('sTest', params)
    #Create a WS interferometer 
    W = nets.ws(params)
    
    #Calculate small-world coefficients
    C, M = nets.Creal(W), nets.Mesh(W)
    SPS, APS = nets.SPS(W), nets.APS(W)
    
    #print(psutil.virtual_memory().used)
    return(params + (C, M, SPS, APS))

#Run all tests
if __name__ == '__main__':
    with MPIPoolExecutor() as executor:
        results = executor.map(Stest, pSpace)
    
        #Creating the file to log data
        with open(outFileName, 'w', newline= '' ) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['N', 'kHalf', 'beta', 'phi', 'C', 'M', 'SPS', 'APS'])
            writer.writerows(results)
        
        print("Tests completed.")
    
