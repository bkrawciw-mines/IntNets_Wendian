#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Visualization
Benjamin Krawciw
10/3/2022

This program plots the results for the small-world interferometer tests over
several values of the weighting parameter s
"""

#Library imports
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt

#Numerical tolerance
TOL = 1e-8

'''
CHECKING FOR PRE-COMPLETED COMPUTATIONS
Computing the small-world coefficients for all trials can take a few minutes.
For running this code over and over again (for instance, in debugging), it
makes sense to save the computation's outputs to a file and load that file for
later runs instead of wasting the time to compute everything again. Here, we
are going to check to see if we can find the computations and load them. 
Otherwise, we will proceed with computing the small-world coefficients. 
'''
#File name for computations
compFile = "weightingSuite_compfile.npy"
#Check if output file exists
outFileExists = os.path.isfile(compFile)
if outFileExists:
    loadResults = np.load(compFile)
    (Ns, ks, betas, phis, weightings, Cs, Ms, SPLs, APLs, 
     baseCs, baseMs, baseSPLs, baseAPLs, 
     gammaReal, gammaInt, lambdaReal, lambdaInt,
     Sreal, Sint) = loadResults
    needToCompute = False
    print('Computations already exist; loading file.')
else:
    needToCompute = True
    print('No computation file found. Computing small-world coefficients.')
    
#If necessary, compute measures
if needToCompute:   
    
    #Reading in the CSV data
    inFileName = 'weightingSuite.csv'
    dataFrame = pd.read_csv(inFileName, delimiter = ',')
    #Treat infinite values as invalid
    dataFrame.replace([np.inf, -np.inf], np.nan, inplace = True)
    
    '''
    RANDOM BASELINES FOR GAMMA AND LAMBDA
    To compute the small-world coefficient, a random baseline clustering and path 
    length is required. To find these baselines, we must average clustering and 
    path length at beta=1 for each set of parameters.
    '''
    #Restrict to random case
    randNets = dataFrame.loc[dataFrame['beta'] == 1.0]
    #Sort by parameters
    sortRand = randNets.groupby(['N', 'kHalf', 'beta', 'phi', 'weighting'], 
                                 as_index = True)
    #Compute averages
    baselines = sortRand.mean()
    '''
    SMALL-WORLD COEFFICIENTS FOR EACH NETWORK TRIAL
    Now that we have the random baselines, we can compute the interferometric
    small-world coefficients for each tested network.
    '''
    
    #Turn data into numpy array for faster computation
    ndData = dataFrame.to_numpy()
    #Break out the individual measures and parameters
    params = ndData[:, 0:5]
    Ns = params[:, 0]
    ks = 2.0 * params[:, 1]
    betas = params[:, 2]
    phis = params[:, 3]
    weightings = params[:, 4]
    Cs = ndData[:, 5]
    Ms = ndData[:, 6]
    SPLs = ndData[:, 7]
    APLs = ndData[:, 8]
    
    #Look up reference values for each test
    def findBaseline(i):
        #Break out the parameters
        N, khalf, beta, phi, weighting = params[i]
        #Find the corresponding results
        outBaseline = baselines.loc[int(N), int(khalf), 1.0, phi, weighting].to_numpy()
        #Report the progress on finding all baselines
        print(("%0.2f %% of baselines found" %(100.0 * i / len(params))))
        return(outBaseline)
    
    #Execute the search for all baselines
    baseData = np.array([findBaseline(i) for i in range(len(params))])
    #Break out the baseline data
    baseCs = baseData[:, 0]
    baseMs = baseData[:, 1]
    baseSPLs = baseData[:, 2]
    baseAPLs = baseData[:, 3]
    
    #Implements version of gamma (or lambda) that can handle negative numbers
    def GammaLambda(val, ref):
        out = (val + np.abs(ref) - ref) / (np.abs(ref) + TOL)
        return(out)
    
    #Compute gammas and lambdas
    gammaReal = GammaLambda(Cs, baseCs)
    gammaInt = GammaLambda(Ms, baseMs)
    lambdaReal = GammaLambda(SPLs, baseSPLs)
    lambdaInt = GammaLambda(APLs, baseAPLs)
    
    #Compute small-world coefficients
    Sreal = gammaReal / (lambdaReal + TOL)
    Sint = gammaInt / (lambdaInt + TOL)
    
    #Save results to a file
    fileResults = np.array([Ns, ks, betas, phis, weightings, Cs, Ms, SPLs, APLs, 
                            baseCs, baseMs, baseSPLs, baseAPLs, 
                            gammaReal, gammaInt, lambdaReal, lambdaInt,
                            Sreal, Sint])
    np.save(compFile, fileResults)

'''
STATISTICS
Now, we need to group the trials together by parameter, then we need to compute
an average for each parameter. We also compute the standard deviation of the 
mean to quantify the precision of that average.
'''

#Create pandas dataframe to sort trials
resultArray = np.array([Ns, ks, betas, phis, weightings, Cs, Ms, SPLs, APLs, 
                        baseCs, baseMs, baseSPLs, baseAPLs, 
                        gammaReal, gammaInt, lambdaReal, lambdaInt,
                        Sreal, Sint]).T
resultFrame = pd.DataFrame(resultArray, columns = ['N', 'k', 'beta', 'phi',
                                                   'weighting', 'C', 'M', 'SPL',
                                                   'APL', 'baseC', 'baseM',
                                                   'baseSPL', 'baseAPL', 
                                                   'gammaReal', 'gammaInt',
                                                   'lambdaReal', 'lambdaInt',
                                                   'Sreal', 'Sint'])
#There are a few apparent outliers. Let's remove them
#TODO: Justify these bounds
resultFrame = resultFrame.clip(-50, 200)

sortedResults = resultFrame.groupby(['N', 'k', 'beta', 'phi', 'weighting'], 
                                    as_index = False)
#Compute averages and standard deviation of the mean (SDOM)
paramMeans = sortedResults.mean()
paramMeansArray = paramMeans.to_numpy().T
paramStd = sortedResults.std()
#To go from standard deviation to standard deviation of the mean,
#we also need the number of trials in each grouping
paramCounts = sortedResults.count()
paramSDOM = paramStd / np.sqrt(paramCounts)
paramSDOMarray = paramSDOM.to_numpy().T
print(paramMeans)
print(len(paramMeans['beta'].to_numpy()))

'''
#Create histograms of measures
resultFrame['baseAPL'].loc[resultFrame['phi'] == 0.0].hist(bins=20)
plt.figure()
resultFrame['baseM'].loc[resultFrame['phi'] == 0.0].hist(bins=20)
plt.figure()
resultFrame['M'].loc[resultFrame['phi'] == 0.0].hist(bins=20)
plt.figure()
resultFrame['APL'].loc[resultFrame['phi'] == 0.0].hist(bins=20)
plt.figure()
resultFrame['Sint'].loc[resultFrame['phi'] == 0.0].hist(bins=20)
print(len(paramMeans['APL'].loc[paramMeans['phi'] == 0.0]))
print(paramCounts.loc[paramCounts['phi'] == 0.0])
'''

#Plot M and APL over beta for phi = 0
plt.figure()
pDat = paramMeans.loc[paramMeans['phi'] == paramMeans['phi'].unique()[0]]
pDatErr = paramSDOM.loc[paramMeans['phi'] == paramMeans['phi'].unique()[0]]
betax = np.log(pDat['beta'].to_numpy())
plt.errorbar(betax, 
             pDat['M'], 
             pDatErr['M'], 
             label = (r"$C_{int}$"),
             elinewidth=1,
             capsize=5,
             marker='.', 
             lw = 0)
plt.errorbar(betax, 
             pDat['APL'], 
             pDatErr['APL'], 
             label = (r"APL"),
             elinewidth=1,
             capsize=5,
             marker='.', 
             lw = 0)


'''
S OVER BETA FOR A FEW S VALUES
Here, we plot the interferometric small-world coefficient over beta for a few 
values of s.
'''

#Plot APL over beta for a few interesting places at phi = 0
plt.figure()
#Identify all unique phi values
weightings = paramMeans['weighting'].unique()
numWeightings = 10
for index in range(numWeightings):
    weightDex = index * (len(weightings) // numWeightings)
    s = weightings[weightDex]
    pDat = paramMeans.loc[(paramMeans['phi'] == 0.0) & 
                          (paramMeans['weighting'] == s)]
    print(pDat)
    pDatErr = paramSDOM.loc[(paramMeans['phi'] == 0.0) & 
                            (paramMeans['weighting'] == s)]
    betax = np.log(pDat['beta'].to_numpy())
    plt.errorbar(betax, 
                 pDat['APL'], 
                 pDatErr['APL'], 
                 label = (r"s = %0.2f" %s),
                 elinewidth=1,
                 capsize=5,
                 marker='.', 
                 lw = 0)
plt.legend(loc = 'lower left')
plt.xlabel(r"log $\beta$")
plt.ylabel(r'APL')
plt.savefig("weightings_APL_overbeta.pdf")