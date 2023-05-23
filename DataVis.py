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
import matplotlib.pyplot as plt
from scipy import stats #For filtering outliers

#Numerical tolerance
TOL = 1e-8

#Reading in the CSV data
inFileName = 'full500.csv'
dataFrame = pd.read_csv(inFileName, delimiter = ',')
#Including data patches (supplemental data in sensitive areas)
betaPatch = 'betaPatch_redundancy.csv'
betaPatchFrame = pd.read_csv(betaPatch, delimiter = ',')
#dataFrame = pd.concat((dataFrame, betaPatchFrame))
phiPatch = 'phiPatch.csv'
phiPatchFrame = pd.read_csv(phiPatch, delimiter = ',')
#dataFrame = pd.concat((dataFrame, phiPatchFrame))
#Treat infinite values as invalid
dataFrame.replace([np.inf, -np.inf], np.nan, inplace = True)
#Find outliers in APL and M
MzScore = np.abs(stats.zscore(dataFrame['M']))
APLzScore = np.abs(stats.zscore(dataFrame['APL'], nan_policy='omit'))
dataFrame = dataFrame.loc[(APLzScore < 3) & (MzScore < 3)]

#Group trials by input parameters, then compute means
sortNets = dataFrame.groupby(['N', 'kHalf', 'beta', 'phi', 'weighting'], 
                             as_index = False)
#Using a more stable version of the average to handle volatility in APL
measures = sortNets.mean()
measuresErrs = sortNets.std()
numTests = sortNets.count()
print(measures)

#Case study: How do N and kHalf change things?
#Select a particular beta and phi value
NkData = measures.loc[(measures['beta'] == max(measures['beta']))
                       & (measures['phi'] == 0)]

NkN, Nkk = NkData['N'].to_numpy(), NkData['kHalf'].to_numpy()
NkM, NkSPL = NkData['M'].to_numpy(), NkData['SPL'].to_numpy()
NkAPL = NkData['APL'].to_numpy()

fig, ax = plt.subplots()
scat = ax.scatter(NkN, Nkk, c = NkM, s = 100)
ax.set(
        title = "Clustering over kHalf and N",
        xlabel = 'N',
        ylabel = 'kHalf'
        )
fig.colorbar(scat, label = 'Clustering')
fig.show()

fig, ax = plt.subplots()
scat = ax.scatter(NkN, Nkk, c = NkSPL, s = 100)
ax.set(
        title = "SPL over kHalf and N",
        xlabel = 'N',
        ylabel = 'kHalf'
        )
fig.colorbar(scat, label = 'SPL')
fig.show()

fig, ax = plt.subplots()
scat = ax.scatter(NkN, Nkk, c = NkAPL, s = 100)
ax.set(
        title = "APL over kHalf and N",
        xlabel = 'N',
        ylabel = 'kHalf'
        )
fig.colorbar(scat, label = 'SPL')
fig.show()


#Case study: How do beta and phi change things?
bpData = measures.loc[(measures['N'] == max(measures['N'])) 
                      & (measures['kHalf'] == 6)]
bpb, bpp = bpData['beta'].to_numpy(), bpData['phi'].to_numpy()
bpM, bpSPL = bpData['M'].to_numpy(), bpData['SPL'].to_numpy()
bpAPL = bpData['APL'].to_numpy()
fig, ax = plt.subplots()
scat = ax.scatter(np.log(bpb), bpp, c = bpM, s = 100)
ax.set(
        title = "Clustering over beta and phi",
        xlabel = 'log(beta)',
        ylabel = 'phi'
        )
fig.colorbar(scat, label = 'Clustering')
fig.show()

fig, ax = plt.subplots()
scat = ax.scatter(np.log(bpb), bpp, c = bpSPL, s = 100)
ax.set(
        title = "SPL over beta and phi",
        xlabel = 'log(beta)',
        ylabel = 'phi'
        )
fig.colorbar(scat, label = 'SPL')
fig.show()

fig, ax = plt.subplots()
scat = ax.scatter(np.log(bpb), bpp, c = bpAPL, s = 100)
ax.set(
        title = "APL over beta and phi",
        xlabel = 'log(beta)',
        ylabel = 'phi'
        )
fig.colorbar(scat, label = 'APL')

#Recreating previous results
SmaxDat = measures.loc[(measures['N'] == max(measures['N'])) 
                      & (measures['kHalf'] == 6)]
SmaxErrs = measuresErrs.loc[(measuresErrs['N'] == max(measures['N'])) 
                      & (measuresErrs['kHalf'] == 6)]
SmaxCounts = numTests.loc[(measuresErrs['N'] == max(measures['N'])) 
                      & (measuresErrs['kHalf'] == 6)]
M = SmaxDat['M']
SPL = SmaxDat['SPL']
APL = SmaxDat['APL']

#Create set of random baselines
randDat = SmaxDat.loc[SmaxDat['beta'] == 1.0]
randM = M.loc[(SmaxDat['beta'] == 1.0)]
randSPL = SPL.loc[(SmaxDat['beta'] == 1.0)]
randAPL = APL.loc[(SmaxDat['beta'] == 1.0)]

#Function to compute gamma and lambda for a single config (real and complex)
def GammaLambda(row):

    #Unpack row
    N, kHalf, beta, phi, weighting, C, M, SPL, APL = row

    
    #Reference row
    ref = measures.loc[
        (measures['N'] == N) &
        (measures['kHalf'] == kHalf) &
        (measures['beta'] == 1.0) & 
        (measures['phi'] == phi)
        ].to_numpy()[0]
    refN, refK, refB, refPhi, refWeight, refC, refM, refSPL, refAPL = ref
    
    #Create output row
    out = np.array([
        N, 
        kHalf, 
        beta, 
        phi,
        weighting, 
        C / refC,
        (M + np.abs(refM) - refM) / (np.abs(refM) + TOL),
        SPL / refSPL,
        (APL + np.abs(refAPL) - refAPL) / (np.abs(refAPL) + TOL)
        ])
    
    return(out)

#Compute gammas and lambdas
GamLam = SmaxDat.apply(GammaLambda, axis = 'columns', raw = True, 
                       result_type = 'expand')

#Compute small-world coefficients for a single config
def Scomp(gammaLamRow):
    #Unpack row
    N, kHalf, beta, phi, weighting, GamReal, GamComp, LambReal, LambComp = gammaLamRow
    #Create output row
    out = pd.Series([N, kHalf, beta, phi, weighting, 
                    GamReal / (LambReal + TOL),
                    GamComp / (LambComp + TOL)])
    
    return(out)

#Computing small-world coefficients
Svals = GamLam.agg(Scomp, axis = 'columns')

#Relabeling svals
Svals = pd.DataFrame(Svals.values, 
                     columns = ['N', 'kHalf', 'beta', 'phi', 'weighting', 
                                'Sreal', 'Scomp'])

#Find maximum over beta for each phi
Smax = Svals.groupby('phi', as_index = False).max()

#Characterize S over beta
SmaxBeta = Svals.groupby(['N', 'kHalf', 'phi'], as_index = False).max()
SmaxBetaErr = Svals.groupby(['N', 'kHalf', 'phi'], as_index = False).std()

#Plot S over beta for a few interesting places
plt.figure()
#Identify all unique phi values
phis = Svals['phi'].unique()
for i in [-2, -1, 0, 1, 2]:
    index = i
    phi = phis[index]
    pDat = Svals.loc[Svals['phi'] == phi]
    #Estimate swcoeff error as the product of relative C and P errors times S.
    Mdat = M.loc[Svals['phi'] == phi]
    Pdat = APL.loc[Svals['phi'] == phi]
    Cerr = (SmaxErrs.loc[SmaxErrs['phi'] == phi])['M'] / Mdat
    Perr = (SmaxErrs.loc[SmaxErrs['phi'] == phi])['APL'] / Pdat
    pDatErr = pDat['Scomp'] * np.sqrt(Cerr**2 + Perr**2)
    #Divide by sqrt(numTests) to get standard dev of the mean
    nTests = min(SmaxCounts.loc[SmaxCounts['phi']==phi]['M'])
    pDatErr = pDatErr / np.sqrt(nTests)
    betax = np.log(pDat['beta'].to_numpy())
    plt.errorbar(betax, pDat['Scomp'], pDatErr, label = (r"$\phi$ = %0.2f" %phi),
                 elinewidth=1,
                 capsize=5,
                 marker='.', 
                 lw = 0)
plt.legend(loc = 'lower left')
plt.xlabel(r"log $\beta$")
plt.ylabel(r'$S_{{int}}$')
#plt.ylim(0, 300)
plt.savefig("sw_overbeta100.pdf")

#Plot Smax over phi
plt.figure()
plt.errorbar(SmaxBeta['phi'], SmaxBeta['Scomp'], SmaxBetaErr['Scomp'], label = r'$S_{int}$',
             elinewidth=1,
             capsize=5,
             marker='.', 
             lw = 0)
plt.errorbar(SmaxBeta['phi'], SmaxBeta['Sreal'], 0, label = r'$S_{real}$',
             elinewidth=1,
             capsize=5,
             marker='.', 
             lw = 0)
plt.xlabel(r'$\phi$')
plt.ylabel('S')
plt.legend()
plt.savefig("sw_overphi100.pdf")
