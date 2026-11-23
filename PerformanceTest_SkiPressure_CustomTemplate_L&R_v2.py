# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import scipy.signal as sig
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import trapezoid as trapz
from XSENSORFunctions import (readXSENSORFile, createTSmat, delimitTrial,
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
scriptName = 'SkiPressure_LR'

turn_det_nob = 0.5 # filter frequency for turn detection: may need to be different for different subjects

# Read in files
# only read .csv files for this work
fPath = 'Z:/Testing Segments/SnowSports/Ski/2025/2025_Mech_AlpineRace/SuperG/XSENSOR/'
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

# list of functions 
 
def findSkiTurns(dat, cut_hz):
    """
    Determine ski turn detection based on the gyroscope onboard the XSENSOR SPK

    Parameters
    ----------
    dat : dataclass
        Created from the function "createTSmat
    cut_hz : int
        Cut-off frequency for gyroscope to find turning indicies

    Returns
    -------
    Rpeaks : array
        Peaks detected from the right gyroscope
    Lpeaks : array
        Peaks detected from the left gyroscope
    Rgyro_det : array
        Detection signal from the right gyroscope
    Lgyro_det : array
        Detection signal from the left gyroscope

    """
    # Set Up the filter
    fs = 1.0 / np.mean(np.diff(dat.time))
    w = cut_hz / (fs / 2.0)  # normalized cutoff
    b, a = sig.butter(2, w, 'low') # 2nd order filter
    
    # Filter @ Cut-off frequency: Recommended at 0.5 Hz
    Lgyro_det = sig.filtfilt(b, a, dat.Lgyr[:,1])
    Rgyro_det = sig.filtfilt(b, a, dat.Rgyr[:,1])

    # Peaks: height=15, distance=200
    Lpeaks, _ = sig.find_peaks(Lgyro_det, height=15, distance=200)
    Rpeaks, _ = sig.find_peaks(Rgyro_det, height=15, distance=200)

    return Rpeaks, Lpeaks, Rgyro_det, Lgyro_det

# Keeping this function just in case

def crop_turns_fft(var,turns,freq):
    """
    Function to crop the intended variable into turns, pad the turns with 
    zeros and perform the FFT on the variable of interest

    Parameters
    ----------
    var : list or numpy array
        Variable of interest
    turns : list
        foot-contact or landing indices

    Returns
    -------
    fft_out : numpy array
        FFT of the variable of interest during the stride
    xf : numpy array
        Frequency vector from the FFT in [Hz]

    """
    nn = 2000 # number for padding out the array with 0's
    
    # Preallocate
    intp_var = np.zeros((nn,len(turns)-1))
    fft_out = np.zeros((int(nn/2),len(turns)-1))
    f90 = []
    # Index through the strides
    for ii in range(len(turns)-1):
        intp_var[0:turns[ii+1]-turns[ii],ii] = var[turns[ii]:turns[ii+1]]
        fft_out[:,ii] = abs(fft(intp_var[:,ii])[0:int(nn/2)])
        fft_out[:,ii] = fft_out[:,ii]/np.max(fft_out[:,ii])
        xf = fftfreq(nn,1/freq)[0:int(nn/2)]
        
        # Find the cut-off for 90% of the power
        cPower = cumtrapz(fft_out[:,ii])/trapz(fft_out[:,ii])
        idx = np.where(cPower > 0.9)[0][0]
        f90.append(xf[idx])
    return [fft_out,xf,f90]

#############################################################################################################################################################

badFileList = []
config = []
subject = []
trialNo = []
TurnDir = []

# initialize outcome variables
OutTotMaxForce = []
InsTotMaxForce = []
maxmaxToes = []
medPropMid = []

RFD = []
RFDtime = []
heelAreaP = []
TurnTime = []

TurnDirection = []

freq90 = []


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
    LTurnIdx = []
    RTurnIdx = []
    Lturn_sig = []
    Rturn_sig = []

    ####

    subName = fName.split(sep = "_")[0]
    configTmp = fName.split(sep="_")[1]
    trialTmp = fName.split(sep="_")[2][0]
    
    # Obtained from the master function list
    tmpDat = readXSENSORFile(fName,fPath)
    tmpDat = delimitTrial(tmpDat,fName,fPath)
    tmpDat = createTSmat(fName, fPath, tmpDat)
    
    # Estimate the frequency from the time stamps
    freq = 1/np.mean(np.diff(tmpDat.time))

    # Turn detection: Turn indicies are     
    [RTurnIdx, LTurnIdx, Rturn_sig, Lturn_sig] = findSkiTurns(tmpDat, turn_det_nob)
        
    answer = True # if data check is off. 
    if priorDecision == 'good' and data_check == 1:
        print('Previously accepted, not asking again: ' + fName)
    elif data_check >= 1:
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(Rturn_sig)
        plt.plot(RTurnIdx,Rturn_sig[RTurnIdx],'rs')
        plt.ylabel('Right')
        plt.title('Turn Signal')
        
        plt.subplot(2,2,2)
        plt.plot(tmpDat.RForce)
        plt.plot(RTurnIdx,tmpDat.RForce[RTurnIdx],'rs')
        plt.title('Force Signal')
        
        plt.subplot(2,2,3)
        plt.plot(Lturn_sig)
        plt.plot(LTurnIdx,Lturn_sig[LTurnIdx],'rs')
        plt.ylabel('Left')
        
        plt.subplot(2,2,4)
        plt.plot(tmpDat.LForce)
        plt.plot(LTurnIdx,tmpDat.LForce[LTurnIdx],'rs')
        
        # Save the debugging figure
        saveFolder= fPath + 'ForcePlots'
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)    
        plt.savefig(saveFolder + '/' + subName + '_' + configTmp + '_' + trialTmp +'.png')
            
        answer = messagebox.askyesno("Question","Is data clean?")
    
    # Record the decision straight away, inside the loop, so a run that
    # fails later keeps every decision made so far
    if priorDecision == '' or data_check == 2:
        if answer == False:
            saveProcessingDecision(fPath, scriptName, fName, 'bad',
                                   Subject=subName, Config=configTmp,
                                   reason='rejected at the data check')
        else:
            saveProcessingDecision(fPath, scriptName, fName, 'good',
                                   Subject=subName, Config=configTmp)
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(fName)

    if answer == True:
        plt.close('all')
        print('Estimating point estimates')
        
        [Lfft_out,Lxf,Lfreq90] = crop_turns_fft(tmpDat.LForce,LTurnIdx,freq)
        [Rfft_out,Rxf,Rfreq90] = crop_turns_fft(tmpDat.RForce,RTurnIdx,freq)
        
        freq90.extend(Lfreq90)
        freq90.extend(Rfreq90)
        
        # Filter the forces
        # Set Up the filter
        w = 20 / (freq / 2.0)  # normalized cutoff
        b, a = sig.butter(3, w, 'low') # 3rd order filter
        
        LForce = sig.filtfilt(b, a, tmpDat.LForce)
        RForce = sig.filtfilt(b, a, tmpDat.RForce)
        
        
        for jj, value in enumerate(LTurnIdx[:-1]):
            # Loop through all cleaned turns to calculate discrete outcome measures.
            # using right side only (left turns). No longer assuming left to right
            # transitions are constant and only using data from one side. 
            # Extend this to Left Turns as well
            # variables of interest: outside force (downhill foot peak force) higher is better
            # Outside medial peak force (higher is better)
            # Outside heel force lower is better with an average higher force on forefoot Tricky**
            
            # Evaluate the downhill ski force
            OutTotMaxForce.append(np.nanmax(LForce[value:LTurnIdx[jj+1]]))
            # Evaluate the peak uphill ski force: between the current turn and
            # when the athlete transitions onto the opposite edge
            nextRTurn = np.where(RTurnIdx > value)[0][0]
            InsTotMaxForce.append(np.nanmax(RForce[value:RTurnIdx[nextRTurn]]))
            
            # Peak Toe Pressure
            maxmaxToes.append(np.nanmax(tmpDat.LplantarToe[value:LTurnIdx[jj+1],:,:])*6.895)
            
            # Heel Hold
            heelArea = np.count_nonzero(tmpDat.LplantarHeel[value:LTurnIdx[jj+1], :, :], axis = (1,2))
            heelAreaP.append(np.mean(heelArea)/tmpDat.LplantarHeelSensNo*100)
            
            # Find the peak force in the turn
            idx_max = np.argmax(LForce[value:LTurnIdx[jj+1]])+value
            
            # Examine a 0.25 second window around the peak of the turn
            # for fraction of force applied medial/lateral
            medPropMid.append(np.sum(tmpDat.LplantarMedial[idx_max-25:25+idx_max, :, :])/np.sum(tmpDat.LplantarMat[idx_max-25:25+idx_max, :, :])*100)
                    
            # Rate of force development
            RFD.append(np.mean(np.gradient(LForce[value:idx_max],tmpDat.time[value:idx_max])))
            RFDtime.append(tmpDat.time[idx_max]-tmpDat.time[value])
            
            # turntime
            TurnTime.append(tmpDat.time[LTurnIdx[jj+1]]-tmpDat.time[value])
                        
            TurnDir.append('R')
            subject.append(subName)
            config.append(configTmp)
            trialNo.append(trialTmp)
        
        for jj, value in enumerate(RTurnIdx[:-1]):
            # Loop through all cleaned turns to calculate discrete outcome measures.
            # using right side only (left turns). No longer assuming left to right
            # transitions are constant and only using data from one side. 
            # Extend this to Left Turns as well
            # variables of interest: outside force (downhill foot peak force) higher is better
            # Outside medial peak force (higher is better)
            # Outside heel force lower is better with an average higher force on forefoot Tricky**
            
            # Evaluate the downhill ski force
            OutTotMaxForce.append(np.nanmax(RForce[value:RTurnIdx[jj+1]]))
            # Evaluate the peak uphill ski force: between the current turn and
            # when the athlete transitions onto the opposite edge
            nextLTurn = np.where(LTurnIdx > value)[0][0]
            InsTotMaxForce.append(np.nanmax(LForce[value:LTurnIdx[nextLTurn]]))
            
            # Peak Toe Pressure
            maxmaxToes.append(np.nanmax(tmpDat.RplantarToe[value:RTurnIdx[jj+1],:,:])*6.895)
            
            # Heel Hold
            heelArea = np.count_nonzero(tmpDat.RplantarHeel[value:RTurnIdx[jj+1], :, :], axis = (1,2))
            heelAreaP.append(np.mean(heelArea)/tmpDat.RplantarHeelSensNo*100)
            
            # Find the peak force in the turn
            idx_max = np.argmax(RForce[value:RTurnIdx[jj+1]])+value
            
            # Examine a 0.25 second window around the peak of the turn
            # for fraction of force applied medial/lateral
            medPropMid.append(np.sum(tmpDat.RplantarMedial[idx_max-25:25+idx_max, :, :])/np.sum(tmpDat.RplantarMat[idx_max-25:25+idx_max, :, :])*100)
                    
            # Rate of force development
            RFD.append(np.mean(np.gradient(RForce[value:idx_max],tmpDat.time[value:idx_max])))
            RFDtime.append(tmpDat.time[idx_max]-tmpDat.time[value])
            
            # turntime
            TurnTime.append(tmpDat.time[RTurnIdx[jj+1]]-tmpDat.time[value])
                        
            TurnDir.append('L')
            subject.append(subName)
            config.append(configTmp)
            trialNo.append(trialTmp)
        

outcomes = pd.DataFrame({'Subject': list(subject), 'Config':list(config),'trialNo':list(trialNo) , 'TurnDirection':list(TurnDir), 'TurnTime':list(TurnTime),
                          'OutTotMaxForce':list(OutTotMaxForce), 'InsTotMaxForce':list(InsTotMaxForce),
                          'maxmaxToes':list(maxmaxToes), 'heelAreaP':list(heelAreaP), 'medPropMid':list(medPropMid),
                          'RFD':list(RFD), 'RFDtime':list(RFDtime)
                          })

outfileName = fPath + '0_CompiledResults_7.csv'
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    outcomes.to_csv(outfileName, header=True, index = False)
            
            
            
            
            
        


