# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox
import scipy.signal as sig


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


# Read in files
# only read .asc files for this work
fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/Snow Performance/EH_Alpine_FullBootvsShell_Mech_Jan2024/XSENSOR/Cropped/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) ]

# turn_det_nob = 0.75 # filter frequency for turn detection: may need to be different for different subjects

save_on = 1
data_check = 0


freq = 100 # sampling frequency
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold

def delimitTrial(inputDF,FName):
    """
     This function uses ginput to delimit the start and end of a trial
    You will need to indicate on the plot when to start/end the trial. 
    You must use tkinter or plotting outside the console to use this function
    Parameters
    ----------
    inputDF : Pandas DataFrame
        DF containing all desired output variables.
    zForce : numpy array 
        of force data from which we will subset the dataframe

    Returns
    -------
    outputDat: dataframe subset to the beginning and end of jumps.

    """

    # generic function to plot and start/end trial #
    if os.path.exists(fPath+FName+'TrialSeg.npy'):
        trial_segment_old = np.load(fPath+FName+'TrialSeg.npy', allow_pickle =True)
        trialStart = trial_segment_old[1][0,0]
        trialEnd = trial_segment_old[1][1,0]
        if data_check == 1:
            plt.figure()
            plt.plot(np.mean(inputDF.iloc[:,18:238], axis = 1)*6895*0.014699)
            plt.plot(np.mean(inputDF.iloc[:,250:470], axis = 1)*6895*0.014699)
            plt.axvspan(int(np.floor(trialStart)), int(np.floor(trialEnd)), color = 'lightgray', alpha = 0.5)
        
        
        inputDF = inputDF.iloc[int(np.floor(trialStart)) : int(np.floor(trialEnd)),:]
        outputDat = inputDF.reset_index(drop = True)
                
    else: 
        
        #inputDF = dat
        fig, ax = plt.subplots()
        
        LForce = np.mean(inputDF.iloc[:,18:238], axis = 1)*6895*0.014699
        RForce = np.mean(inputDF.iloc[:,250:470], axis = 1)*6895*0.014699
       
        print('Select a point on the plot to represent the beginning & end of trial')


        ax.plot(LForce, label = 'Left Force')
        ax.plot(RForce, label = 'Rigth Force')
        fig.legend()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        outputDat = outputDat.reset_index(drop = True)
        trial_segment = np.array([FName, pts], dtype = object)
        np.save(fPath+FName+'TrialSeg.npy',trial_segment)

    return(outputDat)




    
def findTurns(RForce, LForce, freq, cut_off_turn):
    """
    Function to determine foot contacts (here HS or heel strikes) and toe-off (TO)
    events using a gradient decent formula

    Parameters
    ----------
    vForce : list
        Vertical force (or approximate vertical force)
    freq : int
        Data collection frequency

    Returns
    -------
    Rpeaks:

    """

    
    fs = freq
    fc = cut_off_turn
    w = fc / (fs / 2)
    b, a = sig.butter(2, w, 'low')
    
    Lturn_detect = sig.filtfilt(b, a, LForce)
    Rturn_detect = sig.filtfilt(b, a, RForce)
    
    Lpeaks,_ = sig.find_peaks(Lturn_detect, prominence=25)
    Rpeaks,_ = sig.find_peaks(Rturn_detect, prominence=25)
    
    # Remove right detections if the peak is lower than the left signal at that
    # time, and vice versa
    to_delete = []
    for ii, peak in enumerate(Lpeaks):
        if Lturn_detect[peak] < Rturn_detect[peak]:
            to_delete.append(ii)
    Lpeaks = np.delete(Lpeaks,to_delete)
    
    to_delete = []
    for ii, peak in enumerate(Rpeaks):
        if Rturn_detect[peak] < Lturn_detect[peak]:
            to_delete.append(ii)
    
    Rpeaks = np.delete(Rpeaks,to_delete)
    
            
    # Clean up the turn detection: ensure they oscillate
    if Lpeaks[0] < Rpeaks[0]:
        Lpeaks, Rpeaks = EnsureTurnsAlternate(Lturn_detect,Rturn_detect,Lpeaks,Rpeaks)
    
    elif Lpeaks[0] > Rpeaks[0]:
        Rpeaks, Lpeaks = EnsureTurnsAlternate(Rturn_detect,Lturn_detect,Rpeaks,Lpeaks)
            
    return(Rpeaks,Lpeaks)

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




   
@dataclass    
class tsData:

    
    LMat: np.array
    LToe: np.array 
    LToeLat: np.array 
    LToeMed: np.array 
    LForefoot: np.array 
    LForefootLat : np.array 
    LForefootMed: np.array 
    LffToe: np.array
    LffToeLat: np.array
    LffToeMed:np.array
    LMidfoot: np.array 
    LMidfootLat: np.array 
    LMidfootMed: np.array
    LHeel: np.array 
    LHeelLat: np.array 
    LHeelMed: np.array 
    LLateral: np.array
    LMedial: np.array
    
    RMat: np.array
    RToe: np.array 
    RToeLat: np.array 
    RToeMed: np.array 
    RForefoot: np.array 
    RForefootLat : np.array 
    RForefootMed: np.array 
    RffToe: np.array
    RffToeLat: np.array
    RffToeMed: np.array
    RMidfoot: np.array 
    RMidfootLat: np.array 
    RMidfootMed: np.array
    RHeel: np.array 
    RHeelLat: np.array 
    RHeelMed: np.array 
    
    RLateral: np.array
    RMedial: np.array
    
    LForce: np.array
    RForce: np.array
    
    # RTurns: np.array 
    # LTurns: np.array
    
    config: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    # def plotAvgPressure(self):
        
    #     earlyTurnLeft = np.zeros([len(self.RTurns), 31, 9])
    #     midTurnLeft = np.zeros([len(self.RTurns), 31, 9])
    #     lateTurnLeft = np.zeros([len(self.RTurns), 31, 9])
        
    #     earlyTurnRight = np.zeros([len(self.RTurns), 31, 9])
    #     midTurnRight = np.zeros([len(self.RTurns), 31, 9])
    #     lateTurnRight = np.zeros([len(self.RTurns), 31, 9])
        
       
        
    #     for i in range(len(self.RTurns)):
            
    #         earlyTurnLeft[i,:,:] = self.LMat[self.RTurns[i],:,:]
    #         midTurnLeft[i,:,:] = self.LMat[self.RTurns[i] + round((self.LTurns[i]-self.RTurns[i])/2),:,:]
    #         lateTurnLeft[i,:,:] = self.LMat[self.LTurns[i],:,:]
            
    #         earlyTurnRight[i,:,:] = self.RMat[self.RTurns[i],:,:]
    #         midTurnRight[i,:,:] = self.RMat[self.RTurns[i] + round((self.LTurns[i]-self.RTurns[i])/2),:,:]
    #         lateTurnRight[i,:,:] = self.RMat[self.LTurns[i],:,:]
           
            
    #     earlyTurnLeftAvg = np.mean(earlyTurnLeft, axis = 0)
    #     midTurnLeftAvg = np.mean(midTurnLeft, axis = 0)
    #     lateTurnLeftAvg = np.mean(lateTurnLeft, axis = 0)
        
    #     earlyTurnRightAvg = np.mean(earlyTurnRight, axis = 0)
    #     midTurnRightAvg = np.mean(midTurnRight, axis = 0)
    #     lateTurnRightAvg = np.mean(lateTurnRight, axis = 0)
       
        
    #     fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    #     ax1 = sns.heatmap(earlyTurnLeftAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax1.set_title('Downhill Ski Pressure') 
    #     ax1.set_ylabel('Early Turn')
        
    #     ax2 = sns.heatmap(earlyTurnRightAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax2.set_title('Uphill Ski Pressure') 
        
    #     ax3 = sns.heatmap(midTurnLeftAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax3.set_ylabel('Mid Turn')
        
    #     ax4 = sns.heatmap(midTurnRightAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        
    #     ax5 = sns.heatmap(lateTurnLeftAvg, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax5.set_ylabel('Late Turn')
    
    #     ax6 = sns.heatmap(lateTurnRightAvg, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        
    #     fig.set_size_inches(5, 10)
        
    #     plt.suptitle(self.subject +' '+ self.config)
    #     plt.tight_layout()
    #     plt.margins(0.1)
        
    #     saveFolder= fPath + '2DPlots'
        
    #     if os.path.exists(saveFolder) == False:
    #         os.mkdir(saveFolder)
            
    #     plt.savefig(saveFolder + '/' + self.subject +' '+ self.config + '.png')
    #     return fig  



def createTSmat(inputName):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """

    #inputName = entries[0]

    # dat = pd.read_csv(fPath+inputName, sep=',', usecols=(columns) )  
    dat = pd.read_csv(fPath+inputName, sep=',', header = 0 , low_memory=False)
    
    if dat.shape[1] <= 2:    
        dat = pd.read_csv(fPath+inputName, sep=',', header = 1 , low_memory=False)
   
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]

    Lsensel = dat.iloc[:,18:238]
    Rsensel = dat.iloc[:,250:470]
    
    # Set up filters: Note - 6 Hz is the cut-off
    w1 = 6 / (100 / 2)
    b1, a1 = sig.butter(2, w1, 'low')
    
    headers = Lsensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    LMat_raw = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)):
        LMat_raw[:, store_r[ii],store_c[ii]] = Lsensel.iloc[:,ii]
    
    LMat_raw[LMat_raw < 1] = 0
    ################### Need to filter the signals here.
    LMat = sig.filtfilt(b1, a1, LMat_raw,axis=0)
    
    
    headers = Rsensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    RMat_raw = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)):
        RMat_raw[:, store_r[ii],store_c[ii]] = Rsensel.iloc[:,ii]
    
    RMat_raw[RMat_raw <1] = 0
    ################### Need to filter the signals here.  
    RMat = sig.filtfilt(b1, a1, RMat_raw,axis=0)
    
    LToe = LMat[:,:7,:]  
    LToeLat = LMat[:,:7, :5]
    LToeMed = LMat[:,:7,5:] 
    LForefoot = LMat[:,7:15, :] 
    LForefootLat = LMat[:,7:15,:5] #Opposite ":," sequence from R side
    LForefootMed = LMat[:,7:15,5:] 
    LffToe = LMat[:,:15,:]
    LffToeLat = LMat[:, :15, :5]
    LffToeMed = LMat[:, :15, 5:]
    LMidfoot = LMat[:,15:25,:] 
    LMidfootLat = LMat[:,15:25,:5] #Opposite ":," sequence from R side
    LMidfootMed = LMat[:,15:25,5:] 
    LHeel = LMat[:,25:, :] 
    LHeelLat = LMat[:,25:,:5] #Opposite ":," sequence from R side
    LHeelMed = LMat[:,25:, 5:]
    LLateral = LMat[:,:5]
    LMedial = LMat[:,:,5:]
           
    RToe = RMat[:,:7,:]
    RToeLat = RMat[:,:7,4:]
    RToeMed = RMat[:,:7,:4]
    RForefoot = RMat[:,7:15, :]
    RForefootLat = RMat[:,7:15,4:]
    RffToe = RMat[:,:15, :]
    RffToeLat = RMat[:, :15, 4:]
    RffToeMed = RMat[:, :15, :4]
    RForefootMed = RMat[:,7:15,:4]
    RMidfoot = RMat[:,15:25,:]
    RMidfootLat = RMat[:,15:25,4:]
    RMidfootMed = RMat[:,15:25,:4]
    RHeel = RMat[:,25:, :]
    RHeelLat = RMat[:,25:,4:]
    RHeelMed = RMat[:,25:, :4]
    RLateral = RMat[:,:,4:]
    RMedial = RMat[:,:,:4]
    
    RForce = np.mean(RMat, axis = (1,2))*6895*0.014699
    LForce = np.mean(LMat, axis = (1,2))*6895*0.014699
    
    # The indices were garbled on the output. Disabling feature.
    # [RT,LT] = findTurns(RForce, LForce, freq)
    
    result = tsData(LMat, LToe, LToeLat, LToeMed, 
                     LForefoot, LForefootLat, LForefootMed, LffToe, LffToeLat, LffToeMed, 
                     LMidfoot, LMidfootLat, LMidfootMed, 
                     LHeel, LHeelLat, LHeelMed, LLateral, LMedial,
                     RMat, RToe, RToeLat, RToeMed,
                     RForefoot, RForefootLat, RForefootMed, RffToe, RffToeLat, RffToeMed,
                     RMidfoot, RMidfootLat, RMidfootMed,
                     RHeel, RHeelLat, RHeelMed, RLateral, RMedial, RForce, LForce,
                     config, subj, dat
                     
                    )
    
    return(result)



#############################################################################################################################################################



badFileList = []

for fName in entries:
    print(fName)
    
    config = []
    subject = []
    trialNo = []
    TurnDir = []

    # initialize outcome variables
    #### New List
    OutTotMaxForce = []
    InsTotMaxForce = []
    maxmaxToes = []
    medPropMid = []
    
    RFD = []
    RFDtime = []
    heelAreaP = []
    TurnTime = []

    TurnDirection = []
    ####

    subName = fName.split(sep = "_")[0]
    configTmp = fName.split(sep="_")[1]
    trialTmp = fName.split(sep="_")[2][0]
    
    # Create turn detection exceptions:
    if subName == 'TrapperSteinle':
        turn_det_nob = 1
    elif subName == 'ZackStetson':
        turn_det_nob = 1.5
    else:
        turn_det_nob = 0.75
    
    
    # Make sure the files are named FirstLast_Config_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
    
    #dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')

    tmpDat = createTSmat(fName)
    # tmpDat.plotAvgPressure()
    
    # Turn detection
    w = turn_det_nob / (freq / 2)
    b, a = sig.butter(2, w, 'low')
    
    Rturn_sig = sig.filtfilt(b, a, tmpDat.RForce)
    Lturn_sig = sig.filtfilt(b, a, tmpDat.LForce)
    
    [Rmax,Lmax] = findTurns(tmpDat.RForce, tmpDat.LForce, freq, turn_det_nob)
    
    
    
    answer = True # if data check is off. 
    if data_check == 1:
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(Rturn_sig)
        plt.plot(Rmax,Rturn_sig[Rmax],'rs')
        plt.ylabel('Right')
        plt.title('Turn Signal')
        
        plt.subplot(2,2,2)
        plt.plot(tmpDat.RForce)
        plt.plot(Rmax,tmpDat.RForce[Rmax],'rs')
        plt.title('Raw Signal')
        
        plt.subplot(2,2,3)
        plt.plot(Lturn_sig)
        plt.plot(Lmax,Lturn_sig[Lmax],'rs')
        plt.ylabel('Left')
        
        plt.subplot(2,2,4)
        plt.plot(tmpDat.LForce)
        plt.plot(Lmax,tmpDat.LForce[Lmax],'rs')
            
        answer = messagebox.askyesno("Question","Is data clean?")
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(fName)

    if answer == True:
        plt.close('all')
        print('Estimating point estimates')
        
        for jj, value in enumerate(Lmax):
            # Loop through all cleaned turns to calculate discrete outcome measures.
            # using right side only (left turns). No longer assuming left to right
            # transitions are constant and only using data from one side. 
            # Extend this to Left Turns as well
            # variables of interest: outside force (downhill foot peak force) higher is better
            # Outside medial peak force (higher is better)
            # Outside heel force lower is better with an average higher force on forefoot Tricky**

            try:
                if jj>0:
                    # Look at a 100 frame window for the true max force
                    # Look at the Outside (or downhill) ski
                    if value < 100:
                        window = value
                    else:
                        window = 50
                    
                    # Outside (downhill) ski force
                    OutTotMaxForce.append(np.nanmax(tmpDat.LForce[value-window:value+window]))
                    
                    # Look at the Inside (or uphill) ski
                    InsTotMaxForce.append(np.nanmax(tmpDat.RForce[value-window:value+window]))
                    
                    # Peak toe pressure
                    maxmaxToes.append(np.nanmax(tmpDat.LToe[value-window:value+window])*6.895)
                    
                    # Examine a 0.25 second window around the peak of the turn
                    # for fraction of force applied medial/lateral
                    # Medial Fraction
                    medPropMid.append(np.sum(tmpDat.LMedial[value-25:25+value, :, :])/np.sum(tmpDat.LMat[value-25:25+value, :, :])*100)
                    
                    # Rate of force development
                    # Find the index of the true local maxima
                    idx_max = np.argmax(tmpDat.LForce[value-window:value+window])+value-window
                    idx_min = np.argmin(tmpDat.LForce[Lmax[jj-1]:idx_max-1])+Lmax[jj-1]
                    RFD.append(np.mean(np.gradient(tmpDat.LForce[idx_min:idx_max],1/freq)))
                    RFDtime.append((idx_max-idx_min)/freq)
                    
                    # Heel Hold
                    heelAreaP.append(np.count_nonzero(tmpDat.LHeel[Lmax[jj-1]:value, :, :])/(value-Lmax[jj-1])/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                    
                    # turntime
                    TurnTime.append((value-Lmax[jj-1])/freq)
                    
                    
                    TurnDir.append('R')
                    subject.append(subName)
                    config.append(configTmp)
                    trialNo.append(trialTmp)
                
                
            except Exception as e: print(e)
        
        
        for jj, value in enumerate(Rmax):
            # Loop through all cleaned turns to calculate discrete outcome measures.
            # using right side only (left turns). No longer assuming left to right
            # transitions are constant and only using data from one side. 
            # Extend this to Left Turns as well
            # variables of interest: outside force (downhill foot peak force) higher is better
            # Outside medial peak force (higher is better)
            # Outside heel force lower is better with an average higher force on forefoot Tricky**

            try:
                if jj>0:
                    # Look at a 100 frame window for the true max force
                    # Look at the Outside (or downhill) ski
                    if value < 100:
                        window = value
                    else:
                        window = 50
                    
                    # Outside (downhill) ski force
                    OutTotMaxForce.append(np.nanmax(tmpDat.RForce[value-window:value+window]))
                    
                    # Look at the Inside (or uphill) ski
                    InsTotMaxForce.append(np.nanmax(tmpDat.LForce[value-window:value+window]))
                    
                    # Peak toe pressure
                    maxmaxToes.append(np.nanmax(tmpDat.RToe[value-window:value+window])*6.895)
                    
                    # Examine a 0.25 second window around the peak of the turn
                    # for fraction of force applied medial/lateral
                    # Medial Fraction
                    medPropMid.append(np.sum(tmpDat.RMedial[value-25:25+value, :, :])/np.sum(tmpDat.RMat[value-25:25+value, :, :])*100)
                    
                    # Rate of force development
                    # Find the index of the true local maxima
                    idx_max = np.argmax(tmpDat.RForce[value-window:value+window])+value-window
                    idx_min = np.argmin(tmpDat.RForce[Lmax[jj-1]:idx_max])+Lmax[jj-1]
                    RFD.append(np.mean(np.gradient(tmpDat.RForce[idx_min:idx_max],1/freq)))
                    RFDtime.append((idx_max-idx_min)/freq)
                    
                    # Heel Hold
                    heelAreaP.append(np.count_nonzero(tmpDat.RHeel[Lmax[jj-1]:value, :, :])/(value-Rmax[jj-1])/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                    
                    # turntime
                    TurnTime.append((value-Rmax[jj-1])/freq)
                    
                    
                    TurnDir.append('L')
                    subject.append(subName)
                    config.append(configTmp)
                    trialNo.append(trialTmp)
                
                
            except Exception as e: print(e)

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
        
        
        
    # except:
    #         print('Not usable data')
    #         badFileList.append(fName)             
            
            
            
            
            
        


