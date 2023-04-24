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

#Numerical tolerance to avoid dividing by zero
TOL = 10e-8

#Reading in the CSV data
inFileName = 'randParams.csv'
dataFrame = pd.read_csv(inFileName, delimiter = ',')
#Treat infinite values as invalid
dataFrame.replace([np.inf, -np.inf], np.nan, inplace = True)

#Group trials by input parameters, then compute means
sortNets = dataFrame.groupby(['N', 'kHalf', 'beta', 'phi', 'weighting'], 
                             as_index = False)
measures = sortNets.max()

#Recreating previous results
SmaxDat = measures
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

#Plot S over beta for a couple interesting places
plt.figure()
#Identify all unique phi values
phis = Svals['phi'].unique()
for i in range(2):
    index = i * (len(phis)) // 2
    phi = phis[index]
    pDat = Svals.loc[Svals['phi'] == phi]
    betax = np.log(pDat['beta'].to_numpy())
    plt.scatter(betax, pDat['Scomp'], label = (r"$\phi$ = %0.2f" %phi))
plt.legend()
plt.xlabel(r"log $\beta$")
plt.ylabel(r'$S_{{complex}}$')
plt.title(r'''Small-World Effect over $\beta$,
N = 500, k = 12''')
#plt.savefig("sw_overbeta.pdf")

#Plot Smax over phi
plt.figure()
plt.scatter(SmaxBeta['phi'], SmaxBeta['Scomp'], label = r'$S_{complex}$')
plt.scatter(SmaxBeta['phi'], SmaxBeta['Sreal'], label = r'$S_{real}$')
plt.xlabel(r'$\phi$')
plt.ylabel('S')
plt.title(r'''The Change in Peak S over $\phi$,
N = 500, k=12''')
plt.legend()
#plt.savefig("sw_overphi.pdf")

#Compute S ratios
def Sratios(Smax):
    #Find the phi = 0 and phi = pi entries
    phi_0 = Smax.loc[(Smax['phi'] == 0.0)]
    phi_pi = Smax.loc[(Smax['phi'] == np.pi)]
    
    #Compute the ratios
    Smaxes = (phi_0['Scomp']).to_numpy()
    Smins = (phi_pi['Scomp']).to_numpy()
    logratios = (np.log(Smaxes / Smins) / np.log(10.0))
    
    #Collect other important pieces of data
    Ns = phi_0['N'].to_numpy()
    ks = 2 * phi_0['kHalf'].to_numpy()
    data = np.array([Ns, ks, logratios])
    
    #Create and return the final dataframe
    out = pd.DataFrame(data.T, columns = ['N', 'k', 'Sratio'])
    return(out)

Srats = Sratios(SmaxBeta)

#Create histogram of S ratios
plt.figure()
Srats['Sratio'].plot.hist()
plt.xlabel(r"$\log_{10} \left( \frac{S(\phi = 0)}{S(\phi = \pi)} \right)$",
           fontdict = {'fontsize': 16})
plt.xlim(0.0, 3.0)
plt.savefig('RandomizedHistogram.pdf', bbox_inches = 'tight')

#Plot the N, K pairs tested
fig, ax = plt.subplots()
scat = ax.scatter((Srats['N'].to_numpy()),
                  (Srats['k'].to_numpy()),
                  c = (Srats['Sratio']).to_numpy())
fig.colorbar(scat, label = 'log of S ratio')
ax.set(xlabel = 'N',
       ylabel = 'k'
       #title = 'Networks tested for Small-World Dissipation')
       )
fig.savefig('RandParams.pdf', bbox_inches = 'tight')