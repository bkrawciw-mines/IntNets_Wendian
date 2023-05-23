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
outFileName = 'thick100.csv'

#Name for the output file
outFileName = 'thick100.csv'

#Create the parameter space for testing
Nrange = [100]
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
betaRange = np.logspace(-5.0, 0.0, num = 20, base = 10, endpoint = True)
#Create a phi space
phiRange = np.linspace(0.0, 2.0 * np.pi, num = 20, endpoint = False)
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
    results = map(Stest, pSpace)
    
    #Creating the file to log data
    with open(outFileName, 'w', newline= '' ) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['N', 'kHalf', 'beta', 'phi', 'weighting', 'C', 'M', 
                         'SPL', 'APL'])
        writer.writerows(results)
        