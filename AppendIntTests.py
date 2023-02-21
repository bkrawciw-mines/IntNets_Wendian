#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interferometer Small-World Tests (Append Mode)
Benjamin Krawciw
9/30/2022

Creates a suite of watts-strogatz interferometers, computes their network 
measures, and saves them using an HPC cluster. Appends data to previous tests,
allowing accumulation of data over several runs.
"""

#Library imports
import csv
from mpi4py.futures import MPIPoolExecutor
from time import time

#Base output on parent program IntTests.py
import IntTests

#Run all tests
if __name__ == '__main__':
    #Start a timer
    tStart = time()
    #Run an MPI executor on the available nodes
    with MPIPoolExecutor() as executor:
        #Use the executor to run the Stest process on parameters in pSpace
        results = executor.map(IntTests.Stest, IntTests.pSpace, 
                               #Limit executions to __ seconds
                               timeout = 10.0,
                               #Send tasks this many tasks at a time 
                               chunksize = 10,
                               #I don't care about the order tests run in
                               unordered = True)
    
        #Open the file to log data
        with open(IntTests.outFileName, 'a', newline= '' ) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(results)
    
    #Report program execution time
    print("Tests completed. Time: %f s" % (time() - tStart))
    
