#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Visualization
Benjamin Krawciw
10/3/2022

This program plots the results for the small-world interferometer tests
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
compFile = "compfile.npy"
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
    inFileName = 'full500.csv'
    dataFrame = pd.read_csv(inFileName, delimiter = ',')
    #Including data patches (supplemental data in sensitive areas)
    betaPatch = 'betaPatch_redundancy.csv'
    betaPatchFrame = pd.read_csv(betaPatch, delimiter = ',')
    dataFrame = pd.concat((dataFrame, betaPatchFrame))
    phiPatch = 'phiPatch.csv'
    phiPatchFrame = pd.read_csv(phiPatch, delimiter = ',')
    dataFrame = pd.concat((dataFrame, phiPatchFrame))
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
#print(paramMeans)
#print(len(paramMeans['beta'].to_numpy()))

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
#print(len(paramMeans['APL'].loc[paramMeans['phi'] == 0.0]))
#print(paramCounts.loc[paramCounts['phi'] == 0.0])


plt.figure()
phiDex = 0
#print(paramMeans['phi'].unique()[phiDex])
pDat = paramMeans.loc[paramMeans['phi'] == paramMeans['phi'].unique()[phiDex]]
pDatErr = paramSDOM.loc[paramMeans['phi'] == paramMeans['phi'].unique()[phiDex]]
betax = np.log(pDat['beta'].to_numpy())
plt.errorbar(betax, 
             pDat['M'] / np.max(pDat['M']), 
             pDatErr['M'] / np.max(pDat['M']) , 
             label = (r"$C_{int}$"),
             elinewidth=1,
             capsize=5,
             marker='.', 
             lw = 0)
plt.errorbar(betax, 
             pDat['APL'] / np.max(pDat['APL']) , 
             pDatErr['APL'] / np.max(pDat['APL']) , 
             label = (r"APL"),
             elinewidth=1,
             capsize=5,
             marker='.', 
             lw = 0)


'''
S OVER BETA FOR A FEW PHI VALUES
Here, we plot the interferometric small-world coefficient over beta for a few 
values of phi close to zero.
'''

#Plot S over beta for a few interesting places
plt.figure()
#Identify all unique phi values
phis = paramMeans['phi'].unique()
print(phis)
for index in np.linspace(0, (len(phis)-1) // 4, num=5, dtype=int):
    phi = phis[index]
    pDat = paramMeans.loc[paramMeans['phi'] == phi]
    pDatErr = paramSDOM.loc[paramMeans['phi'] == phi]
    betax = np.log(pDat['beta'].to_numpy())
    plt.errorbar(betax, 
                 pDat['Sint'], 
                 pDatErr['Sint'], 
                 label = (r"$\phi$ = %0.2f" %phi),
                 elinewidth=1,
                 capsize=5,
                 marker='.', 
                 ecolor = 'k',
                 lw = 0)
plt.legend(loc = 'lower left')
plt.xlabel(r"log $\beta$")
plt.ylabel(r'$S_{{int}}$')
plt.savefig("sw_overbeta.pdf")

'''
MAX S OVER PHI
First, for a given phi value, we look at the average small-world coefficient
over beta, and we take the maximum. Then, we plot the maximum small-world
coefficient over phi.
'''
#Group by phi values (not beta)
phiGroups = paramMeans.groupby(['N', 'k', 'phi', 'weighting'],
                               as_index = False)
#Find maximum measures over beta
betaMaxes = phiGroups.max()

#Locate those maxima in paramMeans
def locateMaxes(series, array):
    locations = []
    for item in series:
        location = int(np.argwhere(array == item)[0])
        locations.append(location)
    return np.array(locations)
maxSintLoc = locateMaxes(betaMaxes['Sint'], paramMeansArray[18])
maxSrealLoc = locateMaxes(betaMaxes['Sreal'], paramMeansArray[17])

#Record the corresponding SDOM
maxSintSDOM = np.array([paramSDOMarray[18, index] for index in maxSintLoc])
maxSrealSDOM = np.array([paramSDOMarray[17, index] for index in maxSrealLoc])

#Since this plot involves maxima on a grid, additional error in case true max
#is between grid points. 
#Estimate this error with grid spacing and derivative of S
def gridErrs(maxes, locs, vals):
    #Create output vector for errors
    out = np.zeros(len(maxes))
    
    #Identify the beta scale
    betas = paramMeans['beta'].unique()
    
    #For each maxVal,
    for index, loc in enumerate(locs):
        
        #Find the associated beta value
        beta = paramMeansArray[2, index]
        #print(beta)
        
        #Place in val for max
        valLoc = np.argmin(np.abs(vals[index] - maxes[index]))
        #print(valLoc)
        
        #Estimate abs of derivative around max and grid spacing
        #If max is at the left end, only use forward points
        if beta == np.min(betas):
            spacing = np.abs(betas[1] - betas[0])
            Splus = vals[index][valLoc + 1]
            Sderiv = (Splus - maxes[index]) / spacing
            err = Sderiv * spacing
        #If max is at the right end, only use the backward points
        elif beta == np.max(betas):
            spacing = np.abs(betas[-1] - betas[-2])
            Sminus = vals[index][valLoc - 1]
            Sderiv = np.abs(maxes[index] - Sminus) / spacing
            err = Sderiv * spacing
        #Otherise, average forward and backward points
        else:
            betaLoc = np.argmin(np.abs(betas - beta))
            spacePlus = np.abs(betas[betaLoc + 1] - beta)
            Splus = vals[index][valLoc + 1]
            Sminus = vals[index][valLoc - 1]
            spaceMinus = np.abs(beta - betas[betaLoc - 1])
            spacing = 0.5 * (spacePlus + spaceMinus)
            derivPlus = np.abs(Splus - maxes[index]) / spacePlus
            derivMinus = np.abs(maxes[index] - Sminus) / spaceMinus
            err = (derivPlus + derivMinus) * spacing / 2.0
        #Record error
        out[index] = err
    return(out)

#Create arrays of the S values for each trial
phiGroupSreals = phiGroups['Sreal'].agg(lambda x: list(x))['Sreal'].to_numpy()
phiGroupSints = phiGroups['Sint'].agg(lambda x: list(x))['Sint'].to_numpy()
#Compute the grid errors
realGridErr = gridErrs(betaMaxes['Sreal'], maxSrealLoc, phiGroupSreals)
intGridErr = gridErrs(betaMaxes['Sint'], maxSintLoc, phiGroupSints)
#print(intGridErr)

#Combine all sources of error
maxSintSDOM = np.sqrt(maxSintSDOM**2 + intGridErr**2)
maxSrealSDOM = np.sqrt(maxSrealSDOM**2 + realGridErr**2)
 

#Plot Smax over phi
plt.figure()
plt.errorbar(betaMaxes['phi'], 
             betaMaxes['Sint'], 
             maxSintSDOM, 
             label = r'$S_{int}$',
             elinewidth=1,
             capsize=5,
             marker='.', 
             ecolor = 'k',
             lw = 0)
plt.errorbar(betaMaxes['phi'], 
             betaMaxes['Sreal'], 
             maxSrealSDOM, 
             label = r'$S_{real}$',
             elinewidth=1,
             capsize=5,
             marker='.', 
             ecolor = 'k',
             lw = 0)
plt.xlabel(r'$\phi$')
plt.ylabel('S')
plt.legend()
plt.savefig("sw_overphi.pdf")
