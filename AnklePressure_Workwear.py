# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:01 2023

@author: Eric.Honert

The purpose of this code is to evaluate the pressure at the ankles using two
dorsal pressure pads
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
        
        #inputDF = dat
        fig, ax = plt.subplots()
        
        insoleSide = inputDF['Insole Side'][0]
                   
        
        if (insoleSide == 'Left'): 
            
            # Left side
            totForce = np.mean(inputDF.iloc[:,18:238], axis = 1)*6895*0.014699
        else:  
            
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
    
    
    #inputName = entries[2]

  
    # dat = pd.read_csv(fPath+inputName, sep=',', usecols=(columns) )  
    dat = pd.read_csv(fPath+inputName, sep=',', header = 'infer', low_memory=False)
   
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2] 
    
    
    
    insoleSide = dat['Insole Side'][0]
    
        
    
    if (insoleSide == 'Left'): 
        
        # Left side
        plantarSensel = dat.iloc[:,18:238]
        dorsalSensel = dat.iloc[:,250:430]
    else:  
        dorsalSensel = dat.iloc[:,18:198]
        plantarSensel = dat.iloc[:,210:430] 

        
   
    
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
    
    
    if ('Insole Side' == 'Left'): 
        plantarToe = plantarMat[:,:7,:] 
        
        plantarToeLat = plantarMat[:,:7, :5]
        plantarToeMed = plantarMat[:,:7,5:] 
        
        plantarForefoot = plantarMat[:,7:15, :] 
        
        plantarForefootLat = plantarMat[:,7:15,:5] #Opposite ":," sequence from R side
        plantarForefootMed = plantarMat[:,7:15,5:] 
        
        plantarMidfoot = plantarMat[:,15:25,:] 
        
        plantarMidfootLat = plantarMat[:,15:25,:5] #Opposite ":," sequence from R side
        plantarMidfootMed = plantarMat[:,15:25,5:] 
        
        
        plantarHeel = plantarMat[:,25:, :] 
        
        plantarHeelLat = plantarMat[:,25:,:5] #Opposite ":," sequence from R side
        plantarHeelMed = plantarMat[:,25:, 5:]
        
        
        dorsalForefoot = dorsalMat[:,:6,:] 
        
        dorsalForefootLat = dorsalMat[:,:6,:5]
        dorsalForefootMed = dorsalMat[:,:6,5:]
        dorsalMidfoot = dorsalMat[:,6:12, :]  
        
        dorsalMidfootLat = dorsalMat[:,6:12,:5]
        dorsalMidfootMed = dorsalMat[:, 6:12,5:] 
        
        dorsalInstep = dorsalMat[:,12:, :] 
        
        dorsalInstepLat = dorsalMat[:,12:,:5]
        dorsalInstepMed = dorsalMat[:,12:,5:]
        
        plantarLateral = plantarMat[:,:5]
        plantarMedial = plantarMat[:,:,5:]
    
         
       
    
    else:  
               
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
fPath = 'Z:/Testing Segments/WorkWear_Performance/EH_Workwear_MidCutStabilityII_CPDMech_Sept23_AnkPress/XSENSOR/'
fileExt = r".csv"
Lat_entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and fName.count('Lateral') and fName.count('Walk')]
Med_entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and fName.count('Medial') and fName.count('Walk')]

# Preallocate variables
badFileList = []

Lat_PeakPressure = []
Lat_AvgPressure = []
Lat_PeakTotForce = []
Lat_ConArea = []
Lat_Var = []

Med_PeakPressure = []
Med_AvgPressure = []
Med_PeakTotForce = []
Med_ConArea = []
Med_Var = []

Subject = []
Config = []


for ii in range(len(Lat_entries)):
    print(Lat_entries[ii])
    print(Med_entries[ii])
    
    TMPsubj = Lat_entries[ii].split(sep="_")[0]
    TMPconfig = Lat_entries[ii].split(sep="_")[2]
    
    # Note: the lateral side has both the lateral pressure and plantar
    Latdat = pd.read_csv(fPath+Lat_entries[ii], sep=',',skiprows = 1, header = 'infer')
    Meddat = pd.read_csv(fPath+Med_entries[ii], sep=',',skiprows = 1, header = 'infer')
    
    # Pull the plantar pressure from the Latdat
    PP = np.array(np.sum(Latdat.iloc[:,210:430],axis = 1))*4.44822
    LatSum = np.array(np.sum(Latdat.iloc[:,18:198],axis = 1))
    MedSum = np.array(np.sum(Meddat.iloc[:,18:198],axis = 1))
    
    
    
    # Try cross-correlation between the summed medial pressure and plantar pressure
    corr_arr = sig.correlate(PP,MedSum,mode='full')
    lags = sig.correlation_lags(len(PP),len(MedSum),mode='full')
    lag = lags[np.argmax(corr_arr)]
    
    if lag > 0:
        PP = PP[lag:]
        MedSum = MedSum[:-lag]        
    elif lag < 0:
        PP = PP[:lag]
        MedSum = MedSum[-lag:]
    
    # Already checked the cross correlations. They worked well
    # plt.figure(1)
    # plt.plot(PP)
    # plt.plot(MedSum)
    # plt.legend(['PP','MedSum'])
    # answer = messagebox.askyesno("Question","Is data clean?")
    # plt.close('all')
    
    # Shorten the data frames to be the same length
    if len(PP) > len(MedSum):
        PP = PP[0:len(MedSum)-1]
        Latdat = Latdat.iloc[0:len(MedSum)-1]
    elif len(MedSum) > len(PP):
        Meddat = Meddat.iloc[0:len(PP)-1,:]
    
    # Compute gait events from the summed plantar pressure
    PP = zeroInsoleForce(PP,freq)
    [RHS,RTO] = findGaitEvents(PP,freq)
    RHS = np.array(RHS); RTO = np.array(RTO)
    
    # Only evaluate strides within a certain length
    RGS = []
    for jj in range(len(RHS)-1):
        if np.max(PP[RHS[jj]:RTO[jj]]) > 1000:
            if (RHS[jj+1] - RHS[jj]) > 0.5*freq and RHS[jj+1] - RHS[jj] < 2*freq:
                RGS.append(jj)
    
    RHS = RHS[np.array(RGS)]
    RTO = RTO[np.array(RGS)]
    
    # Evaluate Metrics from the Medial & Lateral Pressure

    # tmpDat = createTSmat(fName)
    # tmpDat.plotAvgPressure()
    
    answer = True # if data check is off. 
    if data_check == 1:
        plt.figure()
        plt.plot(PP, label = 'Right Foot Total Force')
        for jj in range(len(RHS)):
            plt.axvspan(RHS[jj], RTO[jj], color = 'lightgray', alpha = 0.5)
        
        answer = messagebox.askyesno("Question","Is data clean?")
    
    plt.close('all')
    if answer == False:
        print('Adding file to bad file list')
        badFileList.append(Lat_entries[ii])

    if answer == True:
        print('Estimating point estimates')
        
        for jj in range(len(RHS)):
            LatPress = np.array(Latdat.iloc[RHS[jj]:RTO[jj],18:198])*6.89476
            MedPress = np.array(Meddat.iloc[RHS[jj]:RTO[jj],18:198])*6.89476
            
            Lat_PeakPressure.append(np.max(LatPress))
            Lat_AvgPressure.append(np.mean(LatPress))
            Lat_PeakTotForce.append(np.max(np.sum(LatPress,axis = 1)))
            Lat_ConArea.append(np.mean(np.count_nonzero(LatPress,axis = 1)/180)*100)
            Lat_Var.append(np.std(np.sum(LatPress,axis = 1),axis=0)/np.mean(np.sum(LatPress,axis = 1)))

            Med_PeakPressure.append(np.max(MedPress))
            Med_AvgPressure.append(np.mean(MedPress))
            Med_PeakTotForce.append(np.max(np.sum(MedPress,axis = 1)))
            Med_ConArea.append(np.mean(np.count_nonzero(MedPress,axis = 1)/180)*100)
            Med_Var.append(np.std(np.sum(MedPress,axis = 1),axis=0)/np.mean(np.sum(MedPress,axis = 1)))
            
            Subject.append(TMPsubj)
            Config.append(TMPconfig)
            
            
# Combine outcomes
outcomes = pd.DataFrame({'Subject':list(Subject), 'Config': list(Config), 
                         'Lat_PeakPressure': list(Lat_PeakPressure), 'Lat_AvgPressure': list(Lat_AvgPressure), 'Lat_PeakTotForce': list(Lat_PeakTotForce), 'Lat_ConArea': list(Lat_ConArea), 'Lat_Var': list(Lat_Var),
                         'Med_PeakPressure': list(Med_PeakPressure), 'Med_AvgPressure': list(Med_AvgPressure), 'Med_PeakTotForce': list(Med_PeakTotForce), 'Med_ConArea': list(Med_ConArea), 'Med_Var': list(Med_Var)})
            
if save_on == 1:
    outcomes.to_csv(fPath + 'DynamicPressureOutcomes.csv',header=True)
elif save_on == 2: 
    outcomes.to_csv(fPath + 'DynamicPressureOutcomes.csv',mode = 'a', header=False)    
    
    
    
    
    
    
    
    
    
    
    
    

