#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interferometer Small-World Tests
Benjamin Krawciw
9/30/2022

Creates a suite of watts-strogatz interferometers, computes their small-world 
coefficients, and saves them. This version runs in serial on a laptop.
"""

#Library imports
import numpy as np
import nets
import csv
import itertools

#Name for the output file
outFileName = 'IntTestsData.csv'

#Create the parameter space for testing
Nrange = np.arange(start = 100, stop = 200, step = 100)
#Small networks need more trials
Ns = []
for Nval in Nrange:
    reps = Nrange[-1] // Nval
    [Ns.append(Nval) for i in range(reps)]
Ns = np.array(Ns)
kRange = np.arange(start = 2, stop = 20, step = 2)
betaRange = np.logspace(-5.0, 0.0, num = 20, base = 10, endpoint = False)
phiRange = np.linspace(0.0, 2.0 * np.pi, num = 20, endpoint = False)
pSpace = itertools.product(Ns, kRange, betaRange, phiRange)
#print([params for params in pSpace])
    

#Function for each independent test
def Stest(params):
    print('sTest', params)
    #Create a WS interferometer 
    W = nets.ws(params)
    
    #Calculate small-world coefficients
    C, M = nets.Creal(W), nets.Mesh(W)
    SPS, APS = nets.SPS(W), nets.APS(W)
    
    return(params + (C, M, SPS, APS))

#Run all tests
if __name__ == '__main__':
    results = map(Stest, pSpace)
    
    #Creating the file to log data
    with open(outFileName, 'w', newline= '' ) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['N', 'kHalf', 'beta', 'phi', 'C', 'M', 'SPS', 'APS'])
        writer.writerows(results)
        