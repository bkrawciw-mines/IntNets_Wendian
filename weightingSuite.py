# -*- coding: utf-8 -*-
"""
Weighting Tests
Benjamin Krawciw
5/23/2023

This program runs a suite of tests over different values of the weighting 
parameter s to demonstrate the effect of the weighting parameter on the shape
of the curve for the interferometric small-world coefficient as it changes 
with respect to rewiring parameter beta." 
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
fName = 'weightingSuite.csv'

#Creating new parameter space for these tests
phiRange = [0.0] #Only curious about phi = 0.0, where effect is strangest
sRange = np.linspace(0.01, 1.0, num = 20, endpoint = False)

#Increasing redundancy
#Redundancy for tighter statistics
redundancy = 1000
Ns = []
#Small networks need more trials
for Nval in IntTests.Nrange:
    reps = redundancy * ((IntTests.Nrange[-1]) // (Nval))
    [Ns.append(Nval) for i in range(reps)]
Ns = np.array(Ns)

#Creating full parameter space for patch testing
params = itertools.product(
    Ns, 
    IntTests.kRange, 
    IntTests.betaRange, 
    phiRange, 
    sRange)

#Run all tests
if __name__ == '__main__':
    #Start a timer
    tStart = time()
    
    #Run an MPI executor on the available nodes
    with MPIPoolExecutor() as executor:
        #Use the executor to run the Stest on parameters in params
        sResults = executor.map(IntTests.Stest, params)
    
        #Creating the file to log beta patch data
        with open(fName, 'w', newline= '' ) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['N', 'kHalf', 'beta', 'phi', 'weighting', 'C', 'M', 'SPL', 'APL'])
            writer.writerows(sResults)
        
        executor.shutdown(wait = False)
  
    #Report the execution time
    print("Tests completed. Time: %f s" % (time() - tStart))
