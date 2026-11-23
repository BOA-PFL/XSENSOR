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
import tkinter as tk
from tkinter import messagebox
import scipy.signal as sig
import scipy.interpolate
import scipy
from XSENSORFunctions import (readXSENSORFile, createTSmat, zeroInsoleForce, findGaitEvents,
                              loadProcessingDecisions, saveProcessingDecision)

save_on = 0
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about trials with no recorded decision (the normal setting)
# 2 = re-review EVERY trial, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied.
data_check = 1

# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'DynamicPressure_Trail_LR'

# Read in files
# only read .csv files for this work
fPath = 'Z:\\Testing Segments\\Outdoor\\TrailRunning\\2024\\EH_Trail_Kailas_Perf_Jun24\\XSENSOR\\RemoteFiles\\'
fileExt = r".csv"
# Data-quality decisions recorded by previous runs of this script. Loading
# these is what stops a trial being judged twice. Rejected files are left
# where they are so they can still be opened and looked at - it is this
# record, not the folder, that decides what gets processed.
decisions = loadProcessingDecisions(fPath, scriptName)

# Results from an out-of-date copy of this script will not match anyone
# else's. Asked only when nothing has been decided here yet: a decision file
# means the question was answered on an earlier run of this dataset.
if len(decisions) == 0:
    # Give this dialog its own window and put it in front. It is the first
    # thing the script does, so unlike the data check further down there is no
    # figure on screen to pull focus, and a borrowed default root can open the
    # box behind Spyder - which just looks like the script has hung.
    dialogRoot = tk.Tk()
    dialogRoot.withdraw()
    dialogRoot.attributes('-topmost', True)
    upToDate = messagebox.askyesno("Question", "Is this script up to date with main?",
                                   parent = dialogRoot)
    dialogRoot.destroy()
    # A plain error rather than SystemExit: these scripts get run a cell at a
    # time, and SystemExit can take the console with it. This stops just as
    # firmly, prints the reason, and leaves the console alone.
    if upToDate == False:
        raise RuntimeError('Pull this script from main before processing')

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and not fName.startswith('0_')]

GPStiming = pd.read_csv('Z:\\Testing Segments\\Outdoor\\TrailRunning\\2024\\EH_Trail_Kailas_Perf_Jun24\\CombinedGPS.csv')
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
def delimitTrialTrailRun(inputDF,FName,FilePath):
    """
    This function uses ginput to delimit the start and end of a trial
    You will need to indicate on the plot when to start/end the trial. 
    You must use tkinter or plotting outside the console to use this function
    Parameters
    ----------
    inputDF : Pandas DataFrame
        DF containing all desired output variables.
    fName : str 
        File name
    FilePath : str
        Path of the desired file

    Returns
    -------
    outputDat: dataframe subset to the beginning and end of jumps.

    """

    # generic function to plot and start/end trial #
    if os.path.exists(FilePath+FName+'TrialSeg.npy'):
        trial_segment_old = np.load(FilePath+FName+'TrialSeg.npy', allow_pickle =True)
        trialStart = trial_segment_old[1][0,0]
        trialEnd = trial_segment_old[1][1,0]
        inputDF = inputDF.iloc[int(np.floor(trialStart)) : int(np.floor(trialEnd)),:]
        outputDat = inputDF.reset_index(drop = True)
        
    else: 
        fig, ax = plt.subplots()
        leftForce = inputDF['Est. Load (lbf)']*4.44822
        rightForce = inputDF['Est. Load (lbf).1']*4.44822
                   
        print('Select a point on the plot to represent the beginning & end of trial')

        ax.plot(leftForce, label = 'Left Force')
        ax.plot(rightForce, label = 'Right Force')
        fig.legend()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        outputDat = outputDat.reset_index(drop = True)
        trial_segment = np.array([FName, pts], dtype = object)
        np.save(FilePath+FName+'TrialSeg.npy',trial_segment)

    return(outputDat)

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

for fName in entries:
    print(fName)
    # A trial rejected in a previous run is skipped. The file is still there
    # to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(fName, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + fName)
        continue
    
    # Clear the previous trial's cyclic detections. They are assigned
    # inside conditionals, so a trial that skips a detection would keep
    # the last trial's events and be processed against them silently.
    LHS = []
    LTO = []
    LGS = []
    RHS = []
    RTO = []
    RGS = []
    start_LHS = 0
    start_RHS = 0

    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    trialTmp = int(fName.split(sep = "_")[2].split(sep = '.')[0].lower())
    
    # Find the correct GPS trial. The three comparisons multiply out to a
    # True/False mask over the GPS rows, so indexing a column with it hands
    # back a one-element Series rather than a number. float() used to take
    # one of those and no longer does, so the matching row is pulled out
    # here once and its times are read by name below.
    GPSmask = np.array(GPStiming.Subject == subName) * np.array(GPStiming.Config == ConfigTmp) * np.array(GPStiming.Sesh == trialTmp) # This will be the go-to in the future
    GPSmatches = np.count_nonzero(GPSmask)
    if GPSmatches != 1:
        raise ValueError('Expected one GPS timing row for ' + fName
                         + ', found ' + str(GPSmatches))
    GPStrial = GPStiming[GPSmask].iloc[0]
    
    tmpDat = readXSENSORFile(fName,fPath)
    tmpDat = delimitTrialTrailRun(tmpDat,fName,fPath)
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
    
    checkwnd = 15 # Number of foot contacts to check
    checkless = 15 # Number of frames between foot contacts to check between
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

    LGS = [] # GS indicates good step
    for jj in range(len(LHS)-1):
        if np.max(tmpDat.LForce[LHS[jj]:LTO[jj]]) > pk and np.max(tmpDat.LForce[LHS[jj]:LTO[jj]]) < upper:
            if (LHS[jj+1] - LHS[jj]) > timemin*freq and LHS[jj+1] - LHS[jj] < timemax*freq:
                LGS.append(jj)
    
    RGS = [] # GS indicates good step
    for jj in range(len(RHS)-1):
        if np.max(tmpDat.RForce[RHS[jj]:RTO[jj]]) > pk and np.max(tmpDat.RForce[RHS[jj]:RTO[jj]]) < upper :
            if (RHS[jj+1] - RHS[jj]) > timemin*freq and RHS[jj+1] - RHS[jj] < timemax*freq:
                RGS.append(jj)
    
    answer = True # if data check is off. 
    if priorDecision == 'good' and data_check == 1:
        print('Previously accepted, not asking again: ' + fName)
    elif data_check >= 1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(intp_steps(tmpDat.LForce,LHS,LTO,LGS))
        plt.ylabel('Total Left Insole Force (N)')
        
        plt.subplot(1,2,2)
        plt.plot(intp_steps(tmpDat.RForce,RHS,RTO,RGS))
        plt.ylabel('Total Right Insole Force (N)')
        plt.tight_layout()
        plt.margins(0.1)
        
        saveFolder= fPath + 'SubStepForce'
        
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)
            
        plt.savefig(saveFolder + '/' + subName +' '+ ConfigTmp +' '+ str(trialTmp) + '.png')

        answer = messagebox.askyesno("Question","Is data clean?")
    
    # Record the decision straight away, inside the loop, so a run that
    # fails later keeps every decision made so far
    if priorDecision == '' or data_check == 2:
        if answer == False:
            saveProcessingDecision(fPath, scriptName, fName, 'bad',
                                   Subject=subName, Config=ConfigTmp,
                                   reason='rejected at the data check')
        else:
            saveProcessingDecision(fPath, scriptName, fName, 'good',
                                   Subject=subName, Config=ConfigTmp)
    
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
            sesh.append(trialTmp)
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
            sesh.append(trialTmp)
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
        idx = LHS[LGS]/freq < float(GPStrial.EndS1)
        Llabel[idx] = 1
        # Top label
        idx = (LHS[LGS]/freq > float(GPStrial.StartS2))*(LHS[LGS]/freq < float(GPStrial.EndS2))
        Llabel[idx] = 2
        # Bottom label
        idx = LHS[LGS]/freq > float(GPStrial.StartS3)
        Llabel[idx] = 3
        
        Rlabel = np.zeros([len(RGS),1])
        RHS = np.array(RHS)
        RGS = np.array(RGS)
        # Uphill label
        idx = RHS[RGS]/freq < float(GPStrial.EndS1)
        Rlabel[idx] = 1
        
        # Top label
        idx = (RHS[RGS]/freq > float(GPStrial.StartS2))*(RHS[RGS]/freq < float(GPStrial.EndS2))
        Rlabel[idx] = 2
        
        # Bottom label
        idx = RHS[RGS]/freq > float(GPStrial.StartS3)
        Rlabel[idx] = 3
        oLabel = np.concatenate((oLabel,Llabel,Rlabel),axis = None)
    

outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order' : list(sesh), 'Side':list(side), 'Label':list(oLabel), 'ContactTime':list(ct),
                         'toeP_mid':list(toePmidstance), 'maxmaxToes':list(maxmaxToes),
                         'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate),  
                         'latP_mid':list(latPmidstance), 'latArea_mid':list(latAreamidstance), 'latPropMid':list(latPropMid),
                         'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medPropMid':list(medPropMid)
                         })

outfileName = fPath + '0_CompiledResults.csv'

if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    outcomes.to_csv(outfileName, header=True, index = False)
            

