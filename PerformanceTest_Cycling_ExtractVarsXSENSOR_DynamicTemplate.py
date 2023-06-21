

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

        totForce = np.mean(inputDF.iloc[:,214:425], axis = 1)*6895*0.014699
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
    
    
    #inputName = entries[205]
    dat = pd.read_csv(fPath+inputName, sep=',', skiprows = 1, header = 'infer')
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2]
    dorsalSensel = dat.iloc[:,17:197]
    plantarSensel = dat.iloc[:,214:429]
    
    
    headers = plantarSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    plantarMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        plantarMat[:, store_r[ii],store_c[ii]] = plantarSensel.iloc[:,ii]
    
    plantarMat[plantarMat < 1] = 0
    
    
    headers = dorsalSensel.columns
    store_r = []
    store_c = []

    for name in headers[1:]:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    dorsalMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        dorsalMat[:, store_r[ii],store_c[ii]] = dorsalSensel.iloc[:,ii]
    
    
    dorsalMat = np.flip(dorsalMat, axis = 0)
    
    dorsalMat[dorsalMat <1] = 0    
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

def findLandings(force, fThreshold):
    ric = []
    for step in range(len(force)-1):
        if force[step] < fThreshold and force[step + 1] >= fThreshold:
            ric.append(step)
    return ric

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force, fThreshold):
    rto = []
    for step in range(len(force)-1):
        if force[step] >= fThreshold and force[step + 1] < fThreshold:
            rto.append(step + 1)
    return rto

def trimTakeoffs(landings, takeoffs):
    if takeoffs[0] < landings[0]:
        del(takeoffs[0])
    return(takeoffs)

def trimLandings(landings, trimmedTakeoffs):
    if landings[len(landings)-1] > trimmedTakeoffs[len(trimmedTakeoffs)-1]:
        del(landings[-1])
    return(landings)

def trimForce(inputDFCol, threshForce):
    forceTot = inputDFCol
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)


#############################################################################################################################################################

# Read in files
# only read .asc files for this work
fPath = 'C:/Users/bethany.kilpatrick/Boa Technology Inc/PFL - General/Testing Segments/Cycling Performance Tests/PP_Cycling_PFSBoundary_Mech_May23/Xsensor/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]




badFileList = []

HeelConArea_Sprint = []
HeelConArea_Steady = []
HeelTotConArea = []
Upstroke =[] 
heelconArea = []


steadySub = []
steadyConfig = []
steadyTrial = []
steadyInitialSTDV = []
steadyInitialPkP = []
steadyPeakSTDV = []
steadyPeakPkP = []
steadyEndSTDV = []
steadyEndPkP = []
steadyOverallHeelSTDV = []
steadyOverallPeak = []

sprintSub = []
sprintConfig = []
sprintTrial = []
sprintInitialSTDV = []
sprintInitialPkP = []
sprintPeakSTDV = []
sprintPeakPkP = []
sprintEndSTDV = []
sprintEndPkP = []
sprintOverallHeelSTDV = []
sprintOverallPeak = []  

pkToeP_Steady = []
ffDorsalMax = []
mfDorsalMax = []
instepMax = []


instepEarlyP = []

for fName in entries:
        try:
            fName = entries[2] #Load one file at a time
            dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            subName = fName.split(sep = "_")[0]
            ConfigTmp = fName.split(sep="_")[1]
            moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower() 
            
            tmpDat = createTSmat(fName)
            tmpDat.plotAvgPressure()
            
            plt.figure()
            plt.plot(tmpDat.RForce, label = 'Right Foot Total Force') 
    
            print('click the start of as many steady state periods are recorded in the file. Press enter when done')
            steadyStart = plt.ginput(-1)
            steadyStart = steadyStart[0]
            steadyStart= round(steadyStart[0])
            
            print('click the start of as many sprints are recorded in the file. Press enter when done')
            sprintStart = plt.ginput(-1) 
            sprintStart = sprintStart[0]
            sprintStart = round(sprintStart[0])
            plt.close()
            
            
             
            
          
            
            steadyLandings = findLandings(tmpDat.RForce[steadyStart:steadyStart+freq*30], fThresh) 
            
            
            steadyTakeoffs = findTakeoffs(tmpDat.RForce[steadyStart:steadyStart+freq*30], fThresh)
            sprintLandings = findLandings(tmpDat.RForce[sprintStart:sprintStart+freq*10], fThresh)
            sprintTakeoffs = findTakeoffs(tmpDat.RForce[sprintStart:sprintStart+freq*10], fThresh)
            
            steadyTakeoffs = trimTakeoffs(steadyLandings, steadyTakeoffs)
            steadyLandings = trimLandings(steadyLandings, steadyTakeoffs)
            
            sprintTakeoffs = trimTakeoffs(sprintLandings, sprintTakeoffs)
            sprintLandings = trimLandings(sprintLandings, sprintTakeoffs) 
            
            
            
            
            #for i in range(len(steadyLandings)-1):  
                # for i, steadyLand in enumerate(steadyLandings):
            for i, steadyLand in enumerate(steadyLandings): 
               
                # i = 0
                # tmpForce = (tmpDat.RForce[steadyStart+steadyLand : steadyStart+steadyTakeoffs[i]])
                # tmpPk = max(tmpForce)
                # timePk = list(tmpForce).index(tmpPk) #indx of max force applied during that pedal stroke
                try: 
                    
                    frames = tmpDat.RTO[i] - tmpDat.RHS[i]
            
                   
                     
                # steadyOverallHeelSTDV.append(np.std(tmpDat.plantarHeel[steadyStart+steadyLand:steadyStart+steadyLandings[i+1]])) 
                    
                    
                    HeelConArea_Steady.append(np.count_nonzero(tmpDat.plantarHeel[tmpDat.RTO[i],:, : tmpDat.RHS[i]+1])/36*100) # Heel con. area from agility temp.
                    pkToeP_Steady.append(np.max(tmpDat.plantarToe[tmpDat.RHS[i]:tmpDat.RTO[i]])*6.895)    
                    
                # steadyOverallPeak.append(np.nanmax(dat.PeakP_RF[steadyStart+steadyLand:steadyStart+steadyLandings[i+1]])) 
                
                
                    # ffDorsalP.append(np.mean(tmpDat.dorsalForefoot[tmpDat.RHS[i]:, :, :])*6.895)
                 
                 
                    # mfDorsalPEarly.append(np.mean(tmpDat.dorsalMidfoot[tmpDat.RHS[i]:, :, :])*6.895)
                    # mfDorsalMidP.append(np.mean(tmpDat.dorsalMidfoot[ :, :])*6.895)
                    # mfDorsalLateP.append(np.mean(tmpDat.dorsalMidfoot[tmpDat.RTO[i], :, :])*6.895)
                    # instepEarlyP.append(np.mean(tmpDat.dorsalInstep[tmpDat.RHS[i]:, :, :])*6.895)
                    # instepMidP.append(np.mean(tmpDat.dorsalInstep[ :, :])*6.895)
                    # instepLateP.append(np.mean(tmpDat.dorsalInstep[tmpDat.RTO[i], :, :])*6.895)
                    
                    ffDorsalMax.append(np.max(tmpDat.dorsalForefoot[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    mfDorsalMax.append(np.max(tmpDat.dorsalMidfoot[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    instepMax.append(np.max(tmpDat.dorsalInstep[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                    
                # HeelConArea_Steady.append(np.mean(tmpDat.plantarHeel[steadyTakeoffs[i] : steadyLandings[i]+1])) #Old heel contact area 
                    
                    
                    # steadyInitialSTDV.append( dat.StdDevRF[steadyStart+steadyLand+1])# / RForce[steadyStart+steadyLandings[i]+1] )
                    # steadyInitialPkP.append( dat.PeakP_RF[steadyStart+steadyLand + 1])
                    # steadyPeakSTDV.append( dat.StdDevRF[steadyStart+steadyLand + timePk])# / RForce[steadyStart+steadyLandings[i]+timePk] )
                    # steadyPeakPkP.append( dat.PeakP_RF[steadyStart+steadyLand + timePk])
                    # steadyEndSTDV.append( dat.StdDevRF[steadyStart+steadyLand-1])# / RForce[steadyTakeoffs[i]-1]  )
                    # steadyEndPkP.append( dat.PeakP_RF[steadyStart+steadyLand -1])
                                
                    steadySub.append( fName.split('_')[0] )
                    steadyConfig.append( fName.split('_')[1])
                    steadyTrial.append( fName.split('_')[2][0])
                except:
                    print("reached end of landings")
                
            for i, sprintLand in enumerate(sprintLandings):
                
                # i = 0
                # tmpForce = tmpDat.RForce[sprintStart+sprintLand : sprintStart+sprintTakeoffs[i]]
                # tmpPk = max(tmpForce)
                # timePk = list(tmpForce).index(tmpPk) #indx of max force applied during that pedal stroke
                try:
                    sprintOverallHeelSTDV.append(np.std(dat.R_Heel_EstLoad[sprintStart+sprintLand:sprintStart+sprintLandings[i+1]]))
                    sprintOverallPeak.append(np.nanmax(dat.PeakP_RF[sprintStart+sprintLand:sprintStart+sprintLandings[i+1]]))
                                    
               #Heel Contact 
               # 'R_Heel_TotalArea', 'R_Heel_Contact'
                HeelConArea_Sprint.append(np.mean(([tmpDat.RTO[i],:, : tmpDat.RHS[i]+1])/36*100) )
                instepEarlyP.append(np.mean(tmpDat.dorsalInstep[tmpDat.RHS[i]:, :, :])*6.895) 
                
                
                
                    # sprintSub.append( fName.split('_')[0] )
                    # sprintConfig.append( fName.split('_')[1])
                    # sprintTrial.append( fName.split('_')[2][0]) 
                    
                    
                    sprintInitialSTDV.append( dat.StdDevRF[sprintStart+sprintLand+1])# / RForce[steadyStart+steadyLandings[i]+1] )
               sprintInitialPkP.append( np.max(tmpDat.plantarMat[sprintStart+sprintLand + 1]))
                    sprintPeakSTDV.append( dat.StdDevRF[sprintStart+sprintLand + timePk])# / RForce[steadyStart+steadyLandings[i]+timePk] )
                    sprintPeakPkP.append( dat.PeakP_RF[sprintStart+sprintLand + timePk])
                    sprintEndSTDV.append( dat.StdDevRF[sprintStart+sprintLand-1])# / RForce[steadyTakeoffs[i]-1]  )
                    sprintEndPkP.append( dat.PeakP_RF[sprintStart+sprintLand -1])
                except:
                    print("reached end of sprint landings")
                
            
                
        except:
            print(fName) 