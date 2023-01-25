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
inFileName = "IntTestsData_pc.csv"
dataFrame = pd.read_csv(inFileName, delimiter = ',')

#Group trials by input parameters, then compute means
sortNets = dataFrame.groupby(['N', 'kHalf', 'beta', 'phi'], as_index = False)
measures = sortNets.mean()

#Case study: How do N and kHalf change things?
#Select a particular beta and phi value
NkData = measures.loc[(measures['beta'] == measures['beta'][1]) 
                       & (measures['phi'] == 0)]

NkN, Nkk = NkData['N'].to_numpy(), NkData['kHalf'].to_numpy()
NkM, NkSPS = NkData['M'].to_numpy(), NkData['SPS'].to_numpy()
NkAPS = NkData['APS'].to_numpy()

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
scat = ax.scatter(NkN, Nkk, c = NkSPS, s = 100)
ax.set(
        title = "SPS over kHalf and N",
        xlabel = 'N',
        ylabel = 'kHalf'
        )
fig.colorbar(scat, label = 'SPS')
fig.show()

fig, ax = plt.subplots()
scat = ax.scatter(NkN, Nkk, c = NkAPS, s = 100)
ax.set(
        title = "APS over kHalf and N",
        xlabel = 'N',
        ylabel = 'kHalf'
        )
fig.colorbar(scat, label = 'SPS')
fig.show()


#Case study: How do beta and phi change things?
bpData = measures.loc[(measures['N'] == 300) & (measures['kHalf'] == 6)]
bpb, bpp = bpData['beta'].to_numpy(), bpData['phi'].to_numpy()
bpM, bpSPS = bpData['M'].to_numpy(), bpData['SPS'].to_numpy()
bpAPS = bpData['APS'].to_numpy()
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
scat = ax.scatter(np.log(bpb), bpp, c = bpSPS, s = 100)
ax.set(
        title = "SPS over beta and phi",
        xlabel = 'log(beta)',
        ylabel = 'phi'
        )
fig.colorbar(scat, label = 'SPL')
fig.show()

fig, ax = plt.subplots()
scat = ax.scatter(np.log(bpb), bpp, c = bpAPS, s = 100)
ax.set(
        title = "APS over beta and phi",
        xlabel = 'log(beta)',
        ylabel = 'phi'
        )
fig.colorbar(scat, label = 'APS')

#Recreating previous results
SmaxDat = measures.loc[(measures['N'] == 300) 
                      & (measures['kHalf'] == 12)]
M = SmaxDat['M']
SPS = SmaxDat['SPS']
APS = SmaxDat['APS']

#Create set of random baselines
randDat = SmaxDat.loc[SmaxDat['beta'] == 1.0]
randM = M.loc[(SmaxDat['beta'] == 1.0)]
randSPS = SPS.loc[(SmaxDat['beta'] == 1.0)]
randAPS = APS.loc[(SmaxDat['beta'] == 1.0)]

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
    refN, refK, refB, refPhi, refC, refM, refSPS, refAPS = ref
    
    #Create output row
    out = np.array([
        N, 
        kHalf, 
        beta, 
        phi,
        C / refC,
        M / refM,
        (np.abs(SPS)**np.sign(SPS)) / (np.abs(refSPS)**np.sign(SPS)),
        (np.abs(APS)**np.sign(APS)) / (np.abs(refAPS)**np.sign(APS))
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

print(Svals)

#Relabeling svals
Svals = pd.DataFrame(Svals.values, 
                     columns = ['N', 'kHalf', 'beta', 'phi', 'Sreal', 'Scomp'])

#Find maximum over beta for each phi
Smax = Svals.groupby('phi').max()
print(Smax)

#Plot S over beta for a few interesting places

#Plot Smax over phi