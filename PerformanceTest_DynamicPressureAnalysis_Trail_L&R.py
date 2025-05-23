# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Kate.Harrison
Last Updated: Eric Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox
import scipy.signal as sig
import scipy.interpolate
import scipy
from XSENSORFunctions import readXSENSORFile, delimitTrial, createTSmat, zeroInsoleForce, findGaitEvents


save_on = 0
data_check = 0

# Read in files
# only read .csv files for this work
fPath = 'Z:\\Testing Segments\\EndurancePerformance\\2024\\EH_Trail_AltraMidsole_Perf_Mar24\\Xsensor\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) ]

GPStiming = pd.read_csv('Z:\\Testing Segments\\EndurancePerformance\\2024\\EH_Trail_AltraMidsole_Perf_Mar24\\GPS\\CombinedGPS.csv')
SeshConf = GPStiming[['Config', 'Sesh']]

### set plot font size ###
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




###############################################################################
# Function list specific for this code
def intp_steps(var,landings,takeoffs,GS):
    """
    Function to interpolate the variable of interest across a step
    (from foot contact to toe off) in order to plot the 
    variable of interest over top each other

    Parameters
    ----------
    var : list or numpy array
        Variable of interest. Can be taken from a dataframe or from a numpy array
    landings : list
        Foot contact indicies
    takeoffs : list
        Toe-off indicies

    Returns
    -------
    intp_var : numpy array
        Interpolated variable to 101 points with the number of columns dictated
        by the number of strides.

    """
    # Preallocate
    intp_var = np.zeros((101,len(GS)))
    # Index through the strides
    for count,ii in enumerate(GS):
        dum = var[landings[ii]:takeoffs[ii]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,count] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var

###############################################################################



badFileList = []
config = []
subject = []
ct = []
movement = []
side = []
sesh = []
oLabel = np.array([])

toePmidstance = []
toeAreamidstance = []
ffAreaLate = []
ffPLate = []
ffPMaxLate = []
heelAreaLate = []
heelPLate = []
heelPmax = []
maxmaxToes = []

ffAreaMid = []
ffPMid = []

mfAreaLate = []
mfPLate = []
mfAreaMid = []
mfPMid = []
mfmax = []

latPmidstance = []
latAreamidstance = []
latPLate = []
latAreaLate = []
medPmidstance = []
medAreamidstance = []
medPLate = []
medAreaLate = []

latPropMid = []
medPropMid = []



for fName in entries[0:2]:
    print(fName)
    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
    
    # Find the correct GPS trial
    #GPStrial = np.array(GPStiming.Subject == subName) * np.array(GPStiming.Config == ConfigTmp) * np.array(GPStiming.Sesh == Sesh)
    GPStrial = np.array(GPStiming.Subject == subName) * np.array(GPStiming.Config == ConfigTmp)
    
    tmpDat = readXSENSORFile(fName,fPath)
    # tmpDat = delimitTrial(tmpDat,fName,fPath) make one specfic for trail running
    tmpDat = createTSmat(fName, fPath, tmpDat)
    
    # Estimate the frequency from the time stamps
    freq = 1/np.mean(np.diff(tmpDat.time))
    if freq < 50:
        print('Time off for: ' + fName)
        print('Previous freq est:' + str(freq))
        print('Defaulting to 50 Hz for contact event estimations')
        freq = 50
    
    # Compute foot contact events
    tmpDat.LForce = zeroInsoleForce(tmpDat.LForce,freq)
    [LHS,LTO] = findGaitEvents(tmpDat.LForce,freq)
    
    tmpDat.RForce = zeroInsoleForce(tmpDat.RForce,freq)
    [RHS,RTO] = findGaitEvents(tmpDat.RForce,freq)

    
    start_LHS = []; start_RHS = []
    if subName == 'ChadPrichard': # change 25 to more or less depending on time standing around before start of run
        checkwnd = 25
        checkless = 20
    else:  
        checkwnd = 15 
        checkless = 15

        
        
    for jj in range(checkwnd): 
        jump_check = np.where(np.abs(LHS[jj] - np.array(RHS[0:checkwnd])) < checkless) 
        if jump_check[0].size > 0:
            print('Jump Found')
            start_LHS = jj+1
            start_RHS = np.argmin(np.abs(LHS[jj] - np.array(RHS[0:checkwnd])))+1
        
    
    LHS = LHS[start_LHS:]
    LTO = LTO[start_LHS:]
    RHS = RHS[start_RHS:]
    RTO = RTO[start_RHS:]
       
    # Remove strides that have a peak GRF below 1000 N or over 1900
    # Remove strides that are below 0.5 and above 1.25 seconds    
    pk = 1000
    upper = 1900
    timemin = 0.5
    timemax = 1.25

    LGS = []    
    # May need to exclude poor toe-off dectections here as well
    for jj in range(len(LHS)-1):
        if np.max(tmpDat.LForce[LHS[jj]:LTO[jj]]) > pk and np.max(tmpDat.LForce[LHS[jj]:LTO[jj]]) < upper:
            if (LHS[jj+1] - LHS[jj]) > timemin*freq and LHS[jj+1] - LHS[jj] < timemax*freq:
                LGS.append(jj)
    
    RGS = []
    for jj in range(len(RHS)-1):
        if np.max(tmpDat.RForce[RHS[jj]:RTO[jj]]) > pk and np.max(tmpDat.RForce[RHS[jj]:RTO[jj]]) < upper :
            if (RHS[jj+1] - RHS[jj]) > timemin*freq and RHS[jj+1] - RHS[jj] < timemax*freq:
                RGS.append(jj)
    
    answer = True # if data check is off. 
    if data_check == 1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(intp_steps(tmpDat.LForce,LHS,LTO,LGS))
        plt.ylabel('Total Left Insole Force (N)')
        
        plt.subplot(1,2,2)
        plt.plot(intp_steps(tmpDat.RForce,RHS,RTO,RGS))
        plt.ylabel('Total Right Insole Force (N)')

        answer = messagebox.askyesno("Question","Is data clean?")
    
    
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(fName)

    if answer == True:
        plt.close('all')
        print('Estimating point estimates')
        
        # Create Labels
        Rlabel = np.zeros([len(RGS),1])
        Llabel = np.zeros([len(LGS),1])
        

        for kk in RGS:
            config.append(tmpDat.config)
            subject.append(tmpDat.subject)
            sesh.append(moveTmp)
            #movement.append(moveTmp)
            movement.append('run')
            side.append('R')
            frames = RTO[kk] - RHS[kk]
            ct.append(tmpDat.time[RTO[kk]]-tmpDat.time[RHS[kk]])
            pct40 = RHS[kk] + round(frames*.4)
            pct50 = RHS[kk] + round(frames*.5)
            pct60 = RHS[kk] + round(frames*.6)
            pct90 = RHS[kk] + round(frames*.9)
            
            maxmaxToes.append(np.max(tmpDat.RplantarToe[RHS[kk]:RTO[kk]])*6.895)
            toePmidstance.append(np.mean(tmpDat.RplantarToe[pct40:pct60,:,:])*6.895)
            
            heelAreaLate.append(np.count_nonzero(tmpDat.RplantarHeel[pct50:RTO[kk], :, :])/(RTO[kk] - pct50)/tmpDat.RplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
            heelPLate.append(np.mean(tmpDat.RplantarHeel[pct90:RTO[kk], :, :])*6.895)

            latPmidstance.append(np.mean(tmpDat.RplantarLateral[pct40:pct60, :, :])*6.895)
            latAreamidstance.append(np.count_nonzero(tmpDat.RplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarLateralSensNo*100)
            medPmidstance.append(np.mean(tmpDat.RplantarMedial[pct40:pct60, :, :])*6.895)
            medAreamidstance.append(np.count_nonzero(tmpDat.RplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarMedialSensNo*100)
            
            latPropMid.append(np.sum(tmpDat.RplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
            medPropMid.append(np.sum(tmpDat.RplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
            
        for kk in LGS:
            config.append(tmpDat.config)
            subject.append(tmpDat.subject)
            sesh.append(moveTmp)
            #movement.append(moveTmp)
            movement.append('run')
            side.append('L')
            frames = LTO[kk] - LHS[kk]
            ct.append(tmpDat.time[LTO[kk]]-tmpDat.time[LHS[kk]])
            pct40 = LHS[kk] + round(frames*.4)
            pct50 = LHS[kk] + round(frames*.5)
            pct60 = LHS[kk] + round(frames*.6)
            pct90 = LHS[kk] + round(frames*.9)
            
            maxmaxToes.append(np.max(tmpDat.LplantarToe[LHS[kk]:LTO[kk]])*6.895)
            toePmidstance.append(np.mean(tmpDat.LplantarToe[pct40:pct60,:,:])*6.895)
            
            heelAreaLate.append(np.count_nonzero(tmpDat.LplantarHeel[pct50:LTO[kk], :, :])/(LTO[kk] - pct50)/tmpDat.LplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
            heelPLate.append(np.mean(tmpDat.LplantarHeel[pct90:LTO[kk], :, :])*6.895)
        
            latPmidstance.append(np.mean(tmpDat.LplantarLateral[pct40:pct60, :, :])*6.895)
            latAreamidstance.append(np.count_nonzero(tmpDat.LplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarLateralSensNo*100)
            medPmidstance.append(np.mean(tmpDat.LplantarMedial[pct40:pct60, :, :])*6.895)
            medAreamidstance.append(np.count_nonzero(tmpDat.LplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarMedialSensNo*100)
            latPropMid.append(np.sum(tmpDat.LplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
            medPropMid.append(np.sum(tmpDat.LplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
                    
        # Create Labels
        Llabel = np.zeros([len(LGS),1])
        LHS = np.array(LHS)
        LGS = np.array(LGS)
        # Uphill label
        idx = LHS[LGS]/freq < float(GPStiming.EndS1[GPStrial])
        Llabel[idx] = 1
        # Top label
        idx = (LHS[LGS]/freq > float(GPStiming.StartS2[GPStrial]))*(LHS[LGS]/freq < float(GPStiming.EndS2[GPStrial]))
        Llabel[idx] = 2
        # Bottom label
        idx = LHS[LGS]/freq > float(GPStiming.StartS3[GPStrial])
        Llabel[idx] = 3
        
        Rlabel = np.zeros([len(RGS),1])
        RHS = np.array(RHS)
        RGS = np.array(RGS)
        # Uphill label
        idx = RHS[RGS]/freq < float(GPStiming.EndS1[GPStrial])
        Rlabel[idx] = 1
        
        # Top label
        idx = (RHS[RGS]/freq > float(GPStiming.StartS2[GPStrial]))*(RHS[RGS]/freq < float(GPStiming.EndS2[GPStrial]))
        Rlabel[idx] = 2
        
        # Bottom label
        idx = RHS[RGS]/freq > float(GPStiming.StartS3[GPStrial])
        Rlabel[idx] = 3
        oLabel = np.concatenate((oLabel,Llabel,Rlabel),axis = None)
    

outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order' : list(sesh), 'Side':list(side), 'Label':list(oLabel), 'ContactTime':list(ct),
                         'toeP_mid':list(toePmidstance), 'maxmaxToes':list(maxmaxToes),
                         'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate),  
                         'latP_mid':list(latPmidstance), 'latArea_mid':list(latAreamidstance), 'latPropMid':list(latPropMid),
                         'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medPropMid':list(medPropMid)
                         })

outfileName = fPath + '0_CompiledResults.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
    
        outcomes.to_csv(outfileName, header=True, index = False)

    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            

