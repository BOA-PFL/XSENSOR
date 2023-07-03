# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Kate.Harrison
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

save_on = 1

data_check = 0

fwdLook = 30
fThresh = 50
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
        inputDF = inputDF.iloc[int(np.floor(trialStart)) : int(np.floor(trialEnd)),:]
        outputDat = inputDF.reset_index(drop = True)
        
    else: 
        fig, ax = plt.subplots()

        totForce = np.mean(inputDF.iloc[:,210:430], axis = 1)*6895*0.014699
        print('Select a point on the plot to represent the beginning & end of trial')


        ax.plot(totForce, label = 'Total Force')
        fig.legend()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        outputDat = outputDat.reset_index(drop = True)
        trial_segment = np.array([FName, pts], dtype = object)
        np.save(fPath+FName+'TrialSeg.npy',trial_segment)

    return(outputDat)



def zeroInsoleForce(vForce,freq):
    """
    Function to detect "foot-off" pressure to zero the insole force during swing
    This is expecially handy when residual pressue on the foot "drops out" 
    during swing

    Parameters
    ----------
    vForce : list
        Vertical force (or approximate vertical force)
    freq : int
        Data collection frequency

    Returns
    -------
    newForce : list
        Updated vertical force
        

    """
    newForce = vForce
    # Quasi-constants
    zcoeff = 0.7
    
    windowSize = round(0.8*freq)
    
    zeroThresh = 50
    
    # Ensure that the minimum vertical force is zero
    vForce = vForce - np.min(vForce)
    
    # Filter the forces
    # Set up a 2nd order 10 Hz low pass buttworth filter
    cut = 10
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    # Filter the force signal
    vForce = sig.filtfilt(b, a, vForce)
        
    # Set the threshold to start to find possible gait events
    thresh = zcoeff*np.mean(vForce)
    
    # Allocate gait variables
    nearHS = []
    # Index through the force    
    for ii in range(len(vForce)-1):
        if vForce[ii] <= thresh and vForce[ii+1] > thresh: 
            nearHS.append(ii+1)
    
    nearHS = np.array(nearHS)
    nearHS = nearHS[(nearHS > windowSize)*(nearHS < len(vForce)-windowSize)]
    
    HS = []
    for idx in nearHS:
        HS_unity_slope = (vForce[idx-windowSize] - vForce[idx])/windowSize*np.array(range(windowSize))
        HS_sig = HS_unity_slope - vForce[idx-windowSize:idx]
        HS.append(idx-windowSize+np.argmax(HS_sig))
    
    newForce = newForce - np.median(newForce[HS])
    newForce[newForce < zeroThresh] = 0
    
    return(newForce)
    
def findGaitEvents(vForce,freq):
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
    HS : numpy array
        Heel strike (or foot contact) indicies
    TO : numpy array
        Toe-off indicies

    """
    # Quasi-constants
    zcoeff = 0.9
    
    
        
    # Set the threshold to start to find possible gait events
    thresh = zcoeff*np.mean(vForce)
    
    # Allocate gait variables
    nearHS = []
    nearTO = []
    # Index through the force 
    for ii in range(len(vForce)-1):
        if vForce[ii] <= thresh and vForce[ii+1] > thresh:
            nearHS.append(ii+1)
        if vForce[ii] > thresh and vForce[ii+1] <= thresh:
            nearTO.append(ii)
    
    nearHS = np.array(nearHS); nearTO = np.array(nearTO)
    # # Remove events that are too close to the start/end of the trial
    # nearHS = nearHS[(nearHS > windowSize)*(nearHS < len(vForce)-windowSize)]
    # nearTO = nearTO[(nearTO > windowSize)*(nearTO < len(vForce)-windowSize)]
    
    HS = []
    for idx in nearHS:
        #if idx < len(vForce)-windowSize:
            for ii in range(idx,0,-1):
                if vForce[ii] == 0 :
                    HS.append(ii)
                    break
    
    # Only utilize unique event detections
    HS = np.unique(HS)
    HS = np.array(HS)
    # Used to eliminate dections that could be due to the local minima in the 
    # middle of a walking step 
    HS = HS[vForce[HS]<100]
    
    # Only search for toe-offs that correspond to heel-strikes
    TO = []
    newHS = []
    for ii in range(len(HS)):
        tmp = np.where(nearTO > HS[ii])[0]
        if len(tmp) > 0:
            idx = nearTO[tmp[0]]
            #if idx < len(vForce)-windowSize:
            for jj in range(idx,len(vForce)):
                if vForce[jj] == 0 :
                    newHS.append(HS[ii])
                    TO.append(jj)
                    break
                # if (len(TO)-1) < ii:
                #     np.delete(HS,ii)
        else:
            np.delete(HS,ii)
    
    if newHS[-1] > TO[-1]:
        newHS = newHS[:-1]
    
    
            
    return(newHS,TO)





   
@dataclass    
class tsData:
    dorsalMat: np.array
    dorsalForefoot: np.array
    dorsalForefootLat: np.array 
    dorsalForefootMed: np.array 
    dorsalMidfoot: np.array
    dorsalMidfootLat: np.array 
    dorsalMidfootMed: np.array 
    dorsalInstep: np.array 
    dorsalInstepLat: np.array 
    dorsalInstepMed: np.array 
    
    plantarMat: np.array
    plantarToe: np.array 
    plantarToeLat: np.array 
    plantarToeMed: np.array 
    plantarForefoot: np.array 
    plantarForefootLat : np.array 
    plantarForefootMed: np.array 
    plantarMidfoot: np.array 
    plantarMidfootLat: np.array 
    plantarMidfootMed: np.array
    plantarHeel: np.array 
    plantarHeelLat: np.array 
    plantarHeelMed: np.array 
    
    plantarLateral: np.array
    plantarMedial: np.array
    
    RForce: np.array 
    
    RHS: np.array 
    RTO: np.array
    
    config: str
    movement: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        
        earlyPlantar = np.zeros([len(self.RHS), 31, 9])
        midPlantar = np.zeros([len(self.RHS), 31, 9])
        latePlantar = np.zeros([len(self.RHS), 31, 9])
        
        earlyDorsal = np.zeros([len(self.RHS), 18, 10])
        midDorsal = np.zeros([len(self.RHS), 18, 10])
        lateDorsal = np.zeros([len(self.RHS), 18, 10])
        
        for i in range(len(self.RHS)):
            
            earlyPlantar[i,:,:] = self.plantarMat[self.RHS[i],:,:]
            midPlantar[i,:,:] = self.plantarMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]
            latePlantar[i,:,:] = self.plantarMat[self.RTO[i],:,:]
            earlyDorsal[i,:,:] = self.dorsalMat[self.RHS[i],:,:]
            midDorsal[i,:,:] = self.dorsalMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]
            lateDorsal[i,:,:] = self.dorsalMat[self.RTO[i],:,:]
            
        earlyPlantarAvg = np.mean(earlyPlantar, axis = 0)
        midPlantarAvg = np.mean(midPlantar, axis = 0)
        latePlantarAvg = np.mean(latePlantar, axis = 0)
        earlyDorsalAvg = np.mean(earlyDorsal, axis = 0)
        midDorsalAvg = np.mean(midDorsal, axis = 0)
        lateDorsalAvg = np.mean(lateDorsal, axis = 0)
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
        ax1 = sns.heatmap(earlyDorsalAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(earlyDorsalAvg))
        ax1.set_title('Dorsal Pressure') 
        ax1.set_ylabel('Initial Contact')
        
        
        ax2 = sns.heatmap(earlyPlantarAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(earlyPlantarAvg))
        ax2.set_title('Plantar Pressure') 
        
        ax3 = sns.heatmap(midDorsalAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(midDorsalAvg))
        ax3.set_ylabel('Midstance')
        
        ax4 = sns.heatmap(midPlantarAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(midPlantarAvg))
        
        ax5 = sns.heatmap(lateDorsalAvg, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(latePlantarAvg))
        ax5.set_ylabel('Toe off')
        
        
        ax6 = sns.heatmap(latePlantarAvg, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(lateDorsalAvg))
        
        fig.set_size_inches(5, 10)
        
        plt.suptitle(self.subject +' '+ self. movement +' '+ self.config)
        plt.tight_layout()
        plt.margins(0.1)
        
        saveFolder= fPath + '2DPlots'
        
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)
            
        plt.savefig(saveFolder + '/' + self.subject +' '+ self. movement +' '+ self.config + '.png')
        return fig  



def createTSmat(inputName):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """
    
    
    # inputName = entries[2]

  
    # dat = pd.read_csv(fPath+inputName, sep=',', usecols=(columns) )  
    dat = pd.read_csv(fPath+inputName, sep=',',  header = 'infer')
   
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2] 
    
    
    
    if ('Insole Side' == 'Right'): 
        dorsalSensel = dat.iloc[:,18:198]
        plantarSensel = dat.iloc[:,210:430] 
    else:  
        # Left side
        plantarSensel = dat.iloc[:,18:238]
        dorsalSensel = dat.iloc[:,250:430]
    

        
   
    
    headers = plantarSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    plantarMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)):
        plantarMat[:, store_r[ii],store_c[ii]] = plantarSensel.iloc[:,ii]
    
    plantarMat[plantarMat < 1] = 0
    
    
    headers = dorsalSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    dorsalMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)):
        dorsalMat[:, store_r[ii],store_c[ii]] = dorsalSensel.iloc[:,ii]
    
    
    dorsalMat = np.flip(dorsalMat, axis = 0) 
    dorsalMat[dorsalMat <1] = 0  
    
    
    if ('Insole Side' == 'Right'): 
        
         
        plantarToe = plantarMat[:,:7,:]
        plantarToeLat = plantarMat[:,:7,4:]
        plantarToeMed = plantarMat[:,:7,:4]
        plantarForefoot = plantarMat[:,7:15, :]
        plantarForefootLat = plantarMat[:,7:15,4:]
        plantarForefootMed = plantarMat[:,7:15,:4]
        plantarMidfoot = plantarMat[:,15:25,:]
        plantarMidfootLat = plantarMat[:,15:25,4:]
        plantarMidfootMed = plantarMat[:,15:25,:4]
        plantarHeel = plantarMat[:,25:, :]
        plantarHeelLat = plantarMat[:,25:,4:]
        plantarHeelMed = plantarMat[:,25:, :4]
        
        dorsalForefoot = dorsalMat[:,:6,:]
        dorsalForefootLat = dorsalMat[:,:6,5:]
        dorsalForefootMed = dorsalMat[:,:6,:5]
        dorsalMidfoot = dorsalMat[:,6:12, :]
        dorsalMidfootLat = dorsalMat[:,6:12,:5]
        dorsalMidfootMed = dorsalMat[:, 6:12,5:]
        dorsalInstep = dorsalMat[:,12:, :]
        dorsalInstepLat = dorsalMat[:,12:,5:]
        dorsalInstepMed = dorsalMat[:,12:,:5]
        
        plantarLateral = plantarMat[:,:,4:]
        plantarMedial = plantarMat[:,:,:4]
    
    else:  
               
        plantarToe = plantarMat[:,:7,:] 
        
        plantarToeLat = plantarMat[:,:7, :4]
        plantarToeMed = plantarMat[:,:7,4:] 
        
        plantarForefoot = plantarMat[:,7:15, :] 
        
        plantarForefootLat = plantarMat[:,7:15,:4] #Opposite ":," sequence from R side
        plantarForefootMed = plantarMat[:,7:15,4:] 
        
        plantarMidfoot = plantarMat[:,15:25,:] 
        
        plantarMidfootLat = plantarMat[:,15:25,:4] #Opposite ":," sequence from R side
        plantarMidfootMed = plantarMat[:,15:25,4:] 
        
        
        plantarHeel = plantarMat[:,25:, :] 
        
        plantarHeelLat = plantarMat[:,25:,:4] #Opposite ":," sequence from R side
        plantarHeelMed = plantarMat[:,25:, 4:]
        
        
        dorsalForefoot = dorsalMat[:,:6,:] 
        
        dorsalForefootLat = dorsalMat[:,:6,:5]
        dorsalForefootMed = dorsalMat[:,:6,5:]
        dorsalMidfoot = dorsalMat[:,6:12, :]  
        
        dorsalMidfootLat = dorsalMat[:,6:12,:5]
        dorsalMidfootMed = dorsalMat[:, 6:12,5:] 
        
        dorsalInstep = dorsalMat[:,12:, :] 
        
        dorsalInstepLat = dorsalMat[:,12:,:5]
        dorsalInstepMed = dorsalMat[:,12:,5:]
        
        plantarLateral = plantarMat[:,:4]
        plantarMedial = plantarMat[:,:,4:]
    
    
    
    
    RForce = np.mean(plantarMat, axis = (1,2))*6895*0.014699
    RForce = zeroInsoleForce(RForce,freq)
    [RHS,RTO] = findGaitEvents(RForce,freq)
    
    
    
    
    
    result = tsData(dorsalMat, dorsalForefoot, dorsalForefootLat, dorsalForefootMed, 
                     dorsalMidfoot, dorsalMidfootLat, dorsalMidfootMed, 
                     dorsalInstep, dorsalInstepLat, dorsalInstepMed, 
                     plantarMat, plantarToe, plantarToeLat, plantarToeMed,
                     plantarForefoot, plantarForefootLat, plantarForefootMed,
                     plantarMidfoot, plantarMidfootLat, plantarMidfootMed,
                     plantarHeel, plantarHeelLat, plantarHeelMed, plantarLateral, plantarMedial, RForce, RHS, RTO,
                     config, movement, subj, dat)
    
    return(result)



#############################################################################################################################################################

# Read in files
# only read .asc files for this work
fPath = 'C:/Users/bethany.kilpatrick/Boa Technology Inc/PFL - General/Testing Segments/AgilityPerformanceData/AS_Trail_DorsalPressureVariationII_PFLMech_June2023/Xsensor/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]




badFileList = []

for fName in entries:
    
    
    config = []
    subject = []
    ct = []
    movement = []

    toePmidstance = []
    toeAreamidstance = []
    ffAreaLate = []
    ffPLate = []
    ffPMaxLate = []
    heelAreaLate = []
    heelPLate = []
    maxmaxToes = []
    
    ffAreaMid = []
    ffPMid = []
    
    mfAreaLate = []
    mfPLate = []
    mfAreaMid = []
    mfPMid = []

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
    dorsalVar = []
    maxDorsal = []
    
    ffDorsalEarlyP = []
    ffDorsalMidP = []
    ffDorsalLateP = []
    mfDorsalEarlyP = []
    mfDorsalMidP = []
    mfDorsalLateP = []
    instepEarlyP = []
    instepMidP = []
    instepLateP = []
    
    ffDorsalMax=[]
    mfDorsalMax=[]
    instepMax=[]
    

    try: 
        #fName = entries[205]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
        
        # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        if ('skater' in moveTmp) or ('cmj' in moveTmp) or ('run' in moveTmp) or ('walk' in moveTmp):
            #dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
        
            tmpDat = createTSmat(fName)
            tmpDat.plotAvgPressure()
            
            answer = True # if data check is off. 
            if data_check == 1:
                plt.figure()
                plt.plot(tmpDat.RForce, label = 'Right Foot Total Force')
                for i in range(len(tmpDat.RHS)):
    
                    plt.axvspan(tmpDat.RHS[i], tmpDat.RTO[i], color = 'lightgray', alpha = 0.5)
                answer = messagebox.askyesno("Question","Is data clean?")
            
            
            
            if answer == False:
                plt.close('all')
                print('Adding file to bad file list')
                badFileList.append(fName)
        
            if answer == True:
                plt.close('all')
                print('Estimating point estimates')
                tmpDat.plotAvgPressure()
    
                for i in range(len(tmpDat.RHS)):
                    
                    #i = 5
                    config.append(tmpDat.config)
                    subject.append(tmpDat.subject)
                    movement.append(moveTmp)
                    frames = tmpDat.RTO[i] - tmpDat.RHS[i]
                    ct.append(frames/200)
                    pct10 = tmpDat.RHS[i] + round(frames*.1)
                    pct40 = tmpDat.RHS[i] + round(frames*.4)
                    pct50 = tmpDat.RHS[i] + round(frames*.5)
                    pct60 = tmpDat.RHS[i] + round(frames*.6)
                    pct90 = tmpDat.RHS[i] + round(frames*.9)
                    

                    
                    maxmaxToes.append(np.max(tmpDat.plantarToe[tmpDat.RHS[i]:tmpDat.RTO[i]])*6.895)
                    toePmidstance.append(np.mean(tmpDat.plantarToe[pct40:pct60,:,:])*6.895)
                    toeAreamidstance.append(np.count_nonzero(tmpDat.plantarToe[pct40:pct60,:,:])/(pct60 - pct40)/39*100)
                    ffAreaLate.append(np.count_nonzero(tmpDat.plantarForefoot[pct90:tmpDat.RTO[i], :,:])/(tmpDat.RTO[i] - pct90)/68*100)
                    ffPLate.append(np.mean(tmpDat.plantarForefoot[pct90:tmpDat.RTO[i], :, :])*6.895)
                    ffPMaxLate.append(np.max(tmpDat.plantarForefoot[pct90:tmpDat.RTO[i], :, :]))
                    ffAreaMid.append(np.count_nonzero(tmpDat.plantarForefoot[pct40:pct60, :,:])/(pct60 - pct40)/68*100)
                    ffPMid.append((np.mean(tmpDat.plantarForefoot[pct40:pct60, :, :]))*6.895)
                    
                    mfAreaLate.append(np.count_nonzero(tmpDat.plantarMidfoot[pct90:tmpDat.RTO[i], :,:])/(tmpDat.RTO[i] - pct90)/70*100)
                    mfPLate.append(np.mean(tmpDat.plantarMidfoot[pct90:tmpDat.RTO[i], :, :])*6.895)
                    mfAreaMid.append(np.count_nonzero(tmpDat.plantarMidfoot[pct40:pct60, :,:])/(pct60 - pct40)/70*100)
                    mfPMid.append((np.mean(tmpDat.plantarMidfoot[pct40:pct60, :, :]))*6.895)
                    
                    heelAreaLate.append(np.count_nonzero(tmpDat.plantarHeel[pct50:tmpDat.RTO[i], :, :])/(tmpDat.RTO[i] - pct50)/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                    heelPLate.append(np.mean(tmpDat.plantarHeel[pct90:tmpDat.RTO[i], :, :])*6.895)
    
                    latPmidstance.append(np.mean(tmpDat.plantarLateral[pct40:pct60, :, :])*6.895)
                    latAreamidstance.append(np.count_nonzero(tmpDat.plantarLateral[pct40:pct60, :, :])/(pct60-pct40)/138*100)
                    latPLate.append(np.mean(tmpDat.plantarLateral[pct90:tmpDat.RTO[i], :, :])*6.895)
                    latAreaLate.append(np.count_nonzero(tmpDat.plantarLateral[pct90:tmpDat.RTO[i], :, :])/(tmpDat.RTO[i] - pct90)/138*100)
                    medPmidstance.append(np.mean(tmpDat.plantarMedial[pct40:pct60, :, :])*6.895)
                    medAreamidstance.append(np.count_nonzero(tmpDat.plantarMedial[pct40:pct60, :, :])/(pct60-pct40)/82*100)
                    medPLate.append(np.mean(tmpDat.plantarMedial[pct90:tmpDat.RTO[i], :, :])*6.895)
                    medAreaLate.append(np.count_nonzero(tmpDat.plantarMedial[pct90:tmpDat.RTO[i], :, :])/(tmpDat.RTO[i]-pct90)/82*100)
                    
                    latPropMid.append(np.sum(tmpDat.plantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.plantarMat[pct40:pct60, :, :]))
                    medPropMid.append(np.sum(tmpDat.plantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.plantarMat[pct40:pct60, :, :]))
                    
                    dorsalVar.append(np.std(tmpDat.dorsalMat[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])/np.mean(tmpDat.dorsalMat[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    maxDorsal.append(np.max(tmpDat.dorsalMat[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    
                    ffDorsalEarlyP.append(np.mean(tmpDat.dorsalForefoot[tmpDat.RHS[i]:pct10, :, :])*6.895)
                    ffDorsalMidP.append(np.mean(tmpDat.dorsalForefoot[pct40:pct60, :, :])*6.895)
                    ffDorsalLateP.append(np.mean(tmpDat.dorsalForefoot[pct90:tmpDat.RTO[i], :, :])*6.895)
                    mfDorsalEarlyP.append(np.mean(tmpDat.dorsalMidfoot[tmpDat.RHS[i]:pct10, :, :])*6.895)
                    mfDorsalMidP.append(np.mean(tmpDat.dorsalMidfoot[pct40:pct60, :, :])*6.895)
                    mfDorsalLateP.append(np.mean(tmpDat.dorsalMidfoot[pct90:tmpDat.RTO[i], :, :])*6.895)
                    instepEarlyP.append(np.mean(tmpDat.dorsalInstep[tmpDat.RHS[i]:pct10, :, :])*6.895)
                    instepMidP.append(np.mean(tmpDat.dorsalInstep[pct40:pct60, :, :])*6.895)
                    instepLateP.append(np.mean(tmpDat.dorsalInstep[pct90:tmpDat.RTO[i], :, :])*6.895)
                    
                    ffDorsalMax.append(np.max(tmpDat.dorsalForefoot[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    mfDorsalMax.append(np.max(tmpDat.dorsalMidfoot[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    instepMax.append(np.max(tmpDat.dorsalInstep[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    
    
                    
    
            

                outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'ContactTime':list(ct),
                                         'toeP_mid':list(toePmidstance),'toeArea_mid':list(toeAreamidstance), 'maxmaxToes':list(maxmaxToes),
                                         'ffP_late':list(ffPLate), 'ffArea_late':list(ffAreaLate), 'ffP_Mid':list(ffPMid), 'ffArea_Mid':list(ffAreaMid), 'ffPMax_late':list(ffPMaxLate),
                                         'mfP_late':list(mfPLate), 'mfArea_late':list(mfAreaLate), 'mfP_Mid':list(mfPMid), 'mfArea_Mid':list(mfAreaMid),
                                         'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate),  
                                         'latP_mid':list(latPmidstance), 'latArea_mid':list(latAreamidstance), 'latP_late':list(latPLate), 'latArea_late':list(latAreaLate), 'latPropMid':list(latPropMid),
                                         'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medP_late':list(medPLate), 'medArea_late':list(medAreaLate), 'medPropMid':list(medPropMid),
                                         'dorsalVar':list(dorsalVar), 'maxDorsalP':list(maxDorsal),
                                         'ffDorsalEarlyP':list(ffDorsalEarlyP), 'ffDorsalMidP':list(ffDorsalMidP), 'ffDorsalLateP':list(ffDorsalLateP),
                                         'mfDorsalEarlyP':list(mfDorsalEarlyP), 'mfDorsalMidP':list(mfDorsalMidP), 'mfDorsalLateP':list(mfDorsalLateP),

                                         'instepEarlyP':list(instepEarlyP), 'instepMidP':list(instepMidP), 'instepLateP':list(instepLateP),
                                         'ffDorsalMax':list(ffDorsalMax), 'mfDorsalMax':list(mfDorsalMax), 'instepMax':list(instepMax)
                                         
                                         })

                outfileName = fPath + '0_CompiledResults1.csv'
                if save_on == 1:
                    if os.path.exists(outfileName) == False:
                    
                        outcomes.to_csv(outfileName, header=True, index = False)
                
                    else:
                        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
                
        
        
    except:
            print('Not usable data')             
            
            
            
            
            
        


