# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tkinter import messagebox
import scipy.signal as sig
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import trapz
from XSENSORFunctions import readXSENSORFile, createTSmat, delimitTrial 


save_on = 0
data_check = 0

turn_det_nob = 0.5 # filter frequency for turn detection: may need to be different for different subjects

# Read in files
# only read .csv files for this work
fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/Snow Performance/2025_Mech_AlpineRace/SuperG/XSENSOR/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]

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
def EnsureTurnsAlternate(turndet1,turndet2,peaks1,peaks2):
    """
    This function takes 2 signals that alternate (such as left and right) with
    events that have been detected within those signals. It makes sure that the
    detected events (or peaks) oscillates between the two signals. If there are
    multiple events detected between events from the other signal, the event
    with the highest peak will be kept and the others eliminated.

    Parameters
    ----------
    turndet1 : numpy array
        Signal associated with the detected peaks from peaks1
    turndet2 : numpy array
        Signal associated with the detected peaks from peaks2
    peaks1 : numpy array
        Peaks detected from the function "find_peaks" (scipy library) from turndet1
    peaks2 : numpy array
        Peaks detected from the function "find_peaks" (scipy library) from turndet2

    Returns
    -------
    peaks1 : numpy array
        Cleaned peaks1 that oscillates with peaks2
    peaks2 : numpy array
        Cleaned peaks1 that oscillates with peaks1

    """
    # If the are multiple detected peaks after the last turn detection
    if peaks1[-2] > peaks2[-1]:
        idx_multiple = np.where(peaks1>peaks2[-1])[0]
        idx = np.argmax(turndet1[peaks1[idx_multiple]])
        # Remove the other detected peaks
        idx_remove = np.delete(idx_multiple,idx)
        peaks1 = np.delete(peaks1,idx_remove)
    
    if peaks2[-2] > peaks1[-1]:
        idx_multiple = np.where(peaks2>peaks1[-1])[0]
        idx = np.argmax(turndet2[peaks2[idx_multiple]])
        # Remove the other detected peaks
        idx_remove = np.delete(idx_multiple,idx)
        peaks2 = np.delete(peaks2,idx_remove)
    
    # Ensure that the entire signal oscillates between peaks1 and peaks2
    jj = 0
    it = 0
    while jj <= len(peaks2)-2 and it < 1000:
        it = it + 1
        if peaks1[jj] < peaks2[jj] and peaks1[jj+1] > peaks2 [jj] and peaks1[jj+1] < peaks2 [jj+1]:
            # A normal turn
            jj = jj+1
        else:
            if peaks2[jj+1] < peaks1[jj+1]:
                # Multiple detected peaks from the following signal before a peak from the leading signal
                idx_multiple = np.where((peaks2>peaks1[jj])*(peaks2<peaks1[jj+1])==True)[0]
                # Figure out which of multiple peaks is higher
                idx = np.argmax(turndet2[peaks2[idx_multiple]])
                # Remove the other detected peaks
                idx_remove = np.delete(idx_multiple,idx)
                peaks2 = np.delete(peaks2,idx_remove)
            else:
                # Multiple detected peaks from the leading signal before a peak from the following signal
                if jj == 0:
                    idx_multiple = np.where(peaks1<peaks2[jj])[0]  
                else:
                    idx_multiple = np.where((peaks1>peaks2[jj-1])*(peaks1<peaks2[jj])==True)[0]    
                idx = np.argmax(turndet1[peaks1[idx_multiple]])
                # Remove the other detected peaks
                idx_remove = np.delete(idx_multiple,idx)
                peaks1 = np.delete(peaks1,idx_remove)
        
    return(peaks1,peaks2)

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
    ####

    subName = fName.split(sep = "_")[0]
    configTmp = fName.split(sep="_")[1]
    trialTmp = fName.split(sep="_")[2][0]
    
    # Obtained from the master function list
    tmpDat = readXSENSORFile(fName,fPath)
    tmpDat = delimitTrial(tmpDat,fName,fPath)
    tmpDat = createTSmat(fName, fPath, tmpDat)

    
    # Turn detection: Turn indicies are     
    [RTurnIdx, LTurnIdx, Rturn_sig, Lturn_sig] = findSkiTurns(tmpDat, turn_det_nob)
        
    answer = True # if data check is off. 
    if data_check == 1:
        
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
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(fName)

    if answer == True:
        plt.close('all')
        print('Estimating point estimates')
        
        [Lfft_out,Lxf,Lfreq90] = crop_turns_fft(tmpDat.LForce,LTurnIdx,1/np.mean(np.diff(tmpDat.time)))
        [Rfft_out,Rxf,Rfreq90] = crop_turns_fft(tmpDat.RForce,RTurnIdx,1/np.mean(np.diff(tmpDat.time)))
        
        freq90.extend(Lfreq90)
        freq90.extend(Rfreq90)
        
        # Filter the forces
        # Set Up the filter
        fs = 1.0 / np.mean(np.diff(tmpDat.time))
        w = 20 / (fs / 2.0)  # normalized cutoff
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
if save_on == 1:
    if os.path.exists(outfileName) == False:
    
        outcomes.to_csv(outfileName, header=True, index = False)

    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
            
            
            
            
        


