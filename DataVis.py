#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 07:25:11 2022

@author: kachow
"""

#Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading in the CSV data
inFileName = "IntTestsData.csv"
dataFrame = pd.read_csv(inFileName, delimiter = ',')

#Group trials by input parameters, then compute means
sortNets = dataFrame.groupby(['N', 'kHalf', 'beta', 'phi'], as_index = False)
measures = sortNets.mean()

#Case study: How do N and kHalf change things?
#Select a particular beta and phi value
NkData = measures.loc[(measures['beta'] == measures['beta'][1]) 
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
bpData = measures.loc[(measures['N'] == 300) & (measures['kHalf'] == 12)]
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
SmaxDat = measures.loc[(measures['N'] == 300) 
                      & (measures['kHalf'] == 12)]
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
    N, kHalf, beta, phi, C, M, SPL, APL = row

    
    #Reference row
    ref = measures.loc[
        (measures['N'] == N) &
        (measures['kHalf'] == kHalf) &
        (measures['beta'] == 1.0) & 
        (measures['phi'] == phi)
        ].to_numpy()[0]
    refN, refK, refB, refPhi, refC, refM, refSPL, refAPL = ref
    
    #Create output row
    out = np.array([
        N, 
        kHalf, 
        beta, 
        phi,
        C / refC,
        M / refM,
        (np.abs(SPL)**np.sign(SPL)) / (np.abs(refSPL)**np.sign(SPL)),
        (np.abs(APL)**np.sign(APL)) / (np.abs(refAPL)**np.sign(APL))
        ])
    
    return(out)

#Compute gammas and lambdas
GamLam = SmaxDat.apply(GammaLambda, axis = 'columns', raw = True, 
                       result_type = 'expand')

#Compute small-world coefficients for a single config
def Scomp(gammaLamRow):
    #Unpack row
    N, kHalf, beta, phi, GamReal, GamComp, LambReal, LambComp = gammaLamRow
    #Create output row
    out = np.array([N, kHalf, beta, phi, 
                    GamReal * LambReal,
                    GamComp * LambComp])
    
    return(out)

#Computing small-world coefficients
Svals = GamLam.agg(Scomp, axis = 'columns')

#Relabeling svals
Svals = pd.DataFrame(Svals.to_list(), 
                     columns = ['N', 'kHalf', 'beta', 'phi', 'Sreal', 'Scomp'])

#Find maximum over beta for each phi
Smax = Svals.groupby('phi').max()
print(Smax)

#Plot S over beta for a few interesting places

#Plot Smax over phi