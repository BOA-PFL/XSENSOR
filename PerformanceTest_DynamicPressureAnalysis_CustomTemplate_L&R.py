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
from datetime import datetime


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

save_on = 0
data_check = 1

fwdLook = 30
fThresh = 50
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold

# Read in files
# only read .asc files for this work
fPath = 'Z:\\Testing Segments\\WorkWear_Performance\\2025_Performance_HighCutPFSWorkwearI_TimberlandPro\\Xsensor\\cropped\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]

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
        
        insoleSide = inputDF['Insole'][0]
                   
        
        if (insoleSide == 'Left'): 
            
            # Left side
            totForce = inputDF['Est. Load (lbf)']*4.44822
        else:  
            
            totForce = inputDF['Est. Load (lbf).1']*4.44822
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

## setting up data classes for 6 possible combos: DorsalRightCOP, DorsalLeftCOP, RightLeftCOP, DorsalRightnoCOP, DorsalLeftnoCOP, RightLeftnoCOP
   
@dataclass    
class tsData:
    time: np.array
    
    dorsalMat: np.array
    dorsalForefoot: np.array
    dorsalMidfoot: np.array
    dorsalInstep: np.array 
    dorsalSensNo: int
    dorsalForefootSensNo: int
    dorsalMidfootSensNo: int
    dorsalInstepSensNo: int
    
    LplantarMat: np.array
    LplantarToe: np.array 
    LplantarForefoot: np.array 
    LplantarMidfoot: np.array 
    LplantarHeel: np.array 
    LplantarLateral: np.array
    LplantarMedial: np.array
    LplantarSensNo: int
    LplantarToeSensNo: int
    LplantarForefootSensNo: int
    LplantarMidfootSensNo: int
    LplantarHeelSensNo: int
    LplantarLateralSensNo: int
    LplantarMedialSensNo: int
    
    RplantarMat: np.array
    RplantarToe: np.array 
    RplantarForefoot: np.array 
    RplantarMidfoot: np.array 
    RplantarHeel: np.array 
    RplantarLateral: np.array
    RplantarMedial: np.array
    RplantarSensNo: int
    RplantarToeSensNo: int
    RplantarForefootSensNo: int
    RplantarMidfootSensNo: int
    RplantarHeelSensNo: int
    RplantarLateralSensNo: int
    RplantarMedialSensNo: int
 
    LForce: np.array 
    LHS: np.array 
    LTO: np.array
    
    RForce: np.array 
    RHS: np.array 
    RTO: np.array
    
    LCOP_X: np.array 
    LCOP_Y: np.array 

    RCOP_X: np.array 
    RCOP_Y: np.array 

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
            
            earlyPlantar[i,:,:] = self.RplantarMat[self.RHS[i],:,:]
            midPlantar[i,:,:] = self.RplantarMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]
            latePlantar[i,:,:] = self.RplantarMat[self.RTO[i],:,:]
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
    
    dat = pd.read_csv(fPath+inputName, sep=',', header = 0, low_memory=False)
    if dat.shape[1] == 2:
        dat = pd.read_csv(fPath+inputName, sep=',', header = 1, low_memory=False)
   
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2] 
    
    time = np.array([((datetime.strptime(timestr,' %H:%M:%S.%f')-datetime(1900,1,1)).total_seconds()) for timestr in dat.Time])
    time = time - time[0]
    freq = 1/np.mean(np.diff(time))
    if freq < 50:
        print('Time off for: ' + inputName)
        print('Previous freq est:' + str(freq))
        print('Defaulting to 50 Hz for contact event estimations')
        freq = 50
    
    RplantarMat = []
    RplantarToe = []
    RplantarForefoot = []
    RplantarMidfoot = []
    RplantarHeel = []
    RplantarLateral = []
    RplantarMedial = []
    RplantarSensNo = []
    RplantarToeSensNo = []
    RplantarForefootSensNo = []
    RplantarMidfootSensNo = []
    RplantarHeelSensNo = []
    RplantarLateralSensNo = []
    RplantarMedialSensNo = []
    RForce = []
    RForce = []
    RHS = []
    RTO = []
    RCOP_Y = []
    RCOP_X = []

    LplantarMat = []
    LplantarToe = []
    LplantarForefoot = []
    LplantarMidfoot = []
    LplantarHeel = []
    LplantarLateral = []
    LplantarMedial = []
    LplantarSensNo = []
    LplantarToeSensNo = []
    LplantarForefootSensNo = []
    LplantarMidfootSensNo = []
    LplantarHeelSensNo = []
    LplantarLateralSensNo = []
    LplantarMedialSensNo = []
    LForce = []
    LHS = []
    LTO = []
    LCOP_Y = []
    LCOP_X = []
    
    dorsalMat = []
    dorsalForefoot = []
    dorsalMidfoot = []
    dorsalInstep = []

    if 'Insole' in dat.columns:
        if  dat['Insole'][0] == 'Left':      # check to see if right insole used
           
            LplantarSensel = dat.loc[:,'S_1_5':'S_31_5']
            
            headers = LplantarSensel.columns
            store_r = []
            store_c = []
           
            for name in headers:
                store_r.append(int(name.split(sep = "_")[1])-1)
                store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
            
            LplantarMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
            
            for ii in range(len(headers)):
                LplantarMat[:, store_r[ii],store_c[ii]] = LplantarSensel.iloc[:,ii]
            
            LplantarMat[LplantarMat < 1] = 0
            LplantarToe = LplantarMat[:,:7,:]
            LplantarForefoot = LplantarMat[:,7:15, :]
            LplantarMidfoot = LplantarMat[:,15:25,:]
            LplantarHeel = LplantarMat[:,25:, :]
            LplantarLateral = LplantarMat[:,:,:4:]
            LplantarMedial =LplantarMat[:,:,4]
            # Sensel Number Computation: note this needs to match the column/row
            # callouts from the lines above
            store_r = np.array(store_r); store_c = np.array(store_c)
            LplantarSensNo = len(store_r)
            LplantarToeSensNo = len(np.where(store_r < 7)[0])
            LplantarForefootSensNo = len(np.where((store_r >= 7)*(store_r < 15))[0])
            LplantarMidfootSensNo = len(np.where((store_r >= 15)*(store_r < 25))[0])
            LplantarHeelSensNo = len(np.where(store_r >= 25)[0])
            LplantarLateralSensNo = len(np.where(store_c >= 4)[0])
            LplantarMedialSensNo = len(np.where(store_c < 4)[0])
            
            LForce = dat['Est. Load (lbf)']*4.44822
            LForce = zeroInsoleForce(LForce,freq)
            [LHS,LTO] = findGaitEvents(LForce,freq)
        
        if dat['Insole'][0] != 'Right' and dat['Insole'][0] != 'Left' :       # check to see if dorsal pad was used
            
            dorsalSensel = dat.loc[:,'S_1_1':'S_18_10']
            
        elif 'Insole.1' in dat.columns:
            if dat['Insole.1'][0] != 'Right' and dat['Insole.1'][0] != 'Left' :
    
                dorsalSensel = dat.loc[:,'S_1_1':'S_18_10']
                
        if 'dorsalSensel' in locals():        
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
            
            dorsalForefoot = dorsalMat[:,:6,:]
            dorsalMidfoot = dorsalMat[:,6:12, :]
            dorsalInstep = dorsalMat[:,12:, :]
            # Sensel Number Computation: note this needs to match the column/row
            # callouts from the lines above
            store_r = np.array(store_r)
            dorsalSensNo = len(store_r)
            dorsalForefootSensNo = len(np.where(store_r < 6)[0])
            dorsalMidfootSensNo = len(np.where((store_r >= 6)*(store_r < 12))[0])
            dorsalInstepSensNo = len(np.where(store_r >= 12)[0])
            
        
        if  dat['Insole'][0] == 'Right':  # check to see if left insole used
            
            RplantarSensel = dat.loc[:, 'S_1_2':'S_31_7'] 
        
        elif  'Insole.1' in dat.columns:
            if dat['Insole.1'][0] == 'Right':  
                
                RplantarSensel = dat.loc[:, 'S_1_2.1':'S_31_7']
            
        if 'RplantarSensel' in locals():  
            headers = RplantarSensel.columns
            store_r = []
            store_c = []
              
            for name in headers:
               store_r.append(int(name.split(sep = "_")[1])-1)
               store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
           
            RplantarMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
           
            for ii in range(len(headers)):
               RplantarMat[:, store_r[ii],store_c[ii]] = RplantarSensel.iloc[:,ii]
            
            RplantarMat[RplantarMat < 1] = 0
            RplantarToe = RplantarMat[:,:7,:]
            RplantarForefoot = RplantarMat[:,7:15, :]
            RplantarMidfoot = RplantarMat[:,15:25,:]
            RplantarHeel = RplantarMat[:,25:, :]
            RplantarLateral = RplantarMat[:,:,4:]
            RplantarMedial = RplantarMat[:,:,:4]
            # Sensel Number Computation: note this needs to match the column/row
            # callouts from the lines above
            store_r = np.array(store_r); store_c = np.array(store_c)
            RplantarSensNo = len(store_r)
            RplantarToeSensNo = len(np.where(store_r < 7)[0])
            RplantarForefootSensNo = len(np.where((store_r >= 7)*(store_r < 15))[0])
            RplantarMidfootSensNo = len(np.where((store_r >= 15)*(store_r < 25))[0])
            RplantarHeelSensNo = len(np.where(store_r >= 25)[0])
            RplantarLateralSensNo = len(np.where(store_c >= 4)[0])
            RplantarMedialSensNo = len(np.where(store_c < 4)[0])
          
            RForce = dat['Est. Load (lbf).1']*4.44822
            RForce = zeroInsoleForce(RForce,freq)
            [RHS,RTO] = findGaitEvents(RForce,freq)


        if 'COP Row' in dat.columns:  
            
            if dat['Insole'][0] == 'Left':
                
                LCOP_Y = dat['COP Column']
                LCOP_X = dat['COP Row']
                
            if dat['Insole'][0] == 'Right':
                
                RCOP_Y = dat['COP Column']
                RCOP_X = dat['COP Row']
               
            if 'Insole.1' in dat.columns:
                if dat['Insole.1'][0] == 'Right':
                
                    RCOP_Y = dat['COP Column.1']
                    RCOP_X = dat['COP Row.1']
                
    result = tsData(time,dorsalMat, dorsalForefoot, dorsalMidfoot, dorsalInstep, 
                     dorsalSensNo, dorsalForefootSensNo, dorsalMidfootSensNo, dorsalInstepSensNo,
                     LplantarMat, LplantarToe, LplantarForefoot, LplantarMidfoot, LplantarHeel, LplantarLateral, LplantarMedial,
                     LplantarSensNo, LplantarToeSensNo, LplantarForefootSensNo, LplantarMidfootSensNo, LplantarHeelSensNo, LplantarLateralSensNo, LplantarMedialSensNo,
                     RplantarMat, RplantarToe, RplantarForefoot, RplantarMidfoot, RplantarHeel,  RplantarLateral, RplantarMedial,
                     RplantarSensNo, RplantarToeSensNo, RplantarForefootSensNo, RplantarMidfootSensNo, RplantarHeelSensNo, RplantarLateralSensNo, RplantarMedialSensNo,
                     LForce, LHS, LTO, RForce, RHS, RTO,
                     LCOP_X, LCOP_Y, RCOP_X, RCOP_Y,
                     config, movement, subj, dat)
    
    return(result)



#############################################################################################################################################################


badFileList = []

for fName in entries:
    
    config = []
    subject = []
    ct = []
    movement = []
    oOrder = []
    side = []

    toePmidstance = []
    heelAreaLate = []
    heelPLate = []
    maxmaxToes = []
    
    latPmidstance = []
    latAreamidstance = []
    medPmidstance = []
    medAreamidstance = []
    
    latPropMid = []
    medPropMid = []
    dorsalVar = []
    maxDorsal = []
    
    try: 
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
        orderTmp = fName.split(sep = "_")[3]
        
        # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        # if ('skater' in moveTmp) or ('cmj' in moveTmp) or ('run' in moveTmp) or ('walk' in moveTmp):
        if ('dh' in moveTmp) or ('uh' in moveTmp) or ('walk' in moveTmp):
            #dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
        
            tmpDat = createTSmat(fName)
            answer = True # if data check is off. 
            if data_check == 1:
                tmpDat.plotAvgPressure()
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
                
    
                if len(tmpDat.RplantarMat) != 0:
                    
                    for i in range(len(tmpDat.RHS)):
                        side.append('Right')
                        config.append(tmpDat.config)
                        subject.append(tmpDat.subject)
                        movement.append(moveTmp)
                        frames = tmpDat.RTO[i] - tmpDat.RHS[i]
                        ct.append(tmpDat.time[tmpDat.RTO[i]]-tmpDat.time[tmpDat.RHS[i]])
                        pct10 = tmpDat.RHS[i] + round(frames*.1)
                        pct40 = tmpDat.RHS[i] + round(frames*.4)
                        pct50 = tmpDat.RHS[i] + round(frames*.5)
                        pct60 = tmpDat.RHS[i] + round(frames*.6)
                        pct90 = tmpDat.RHS[i] + round(frames*.9)
                        
                        maxmaxToes.append(np.max(tmpDat.RplantarToe[tmpDat.RHS[i]:tmpDat.RTO[i]])*6.895)
                        toePmidstance.append(np.mean(tmpDat.RplantarToe[pct40:pct60,:,:])*6.895)
                                           
                        heelAreaLate.append(np.count_nonzero(tmpDat.RplantarHeel[pct50:tmpDat.RTO[i], :, :])/(tmpDat.RTO[i] - pct50)/tmpDat.RplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                        heelPLate.append(np.mean(tmpDat.RplantarHeel[pct90:tmpDat.RTO[i], :, :])*6.895)
        
                        latPmidstance.append(np.mean(tmpDat.RplantarLateral[pct40:pct60, :, :])*6.895)
                        latAreamidstance.append(np.count_nonzero(tmpDat.RplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarLateralSensNo*100)
                        medPmidstance.append(np.mean(tmpDat.RplantarMedial[pct40:pct60, :, :])*6.895)
                        medAreamidstance.append(np.count_nonzero(tmpDat.RplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarMedialSensNo*100)
                        latPropMid.append(np.sum(tmpDat.RplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
                        medPropMid.append(np.sum(tmpDat.RplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
                        
                        if len(tmpDat.dorsalMat) != 0: 
                            
                            dorsalVar.append(np.std(tmpDat.dorsalMat[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])/np.mean(tmpDat.dorsalMat[tmpDat.RHS[i]:tmpDat.RTO[i], :, :]))
                            maxDorsal.append(np.max(tmpDat.dorsalMat[tmpDat.RHS[i]:tmpDat.RTO[i], :, :])*6.895)
                            
                        else:
                            dorsalVar.append('nan')
                            maxDorsal.append('nan')
                            
                if len(tmpDat.LplantarMat) != 0:
                    
                    for i in range(len(tmpDat.LHS)):
                        side.append('Left')
                        config.append(tmpDat.config)
                        subject.append(tmpDat.subject)
                        movement.append(moveTmp)
                        frames = tmpDat.LTO[i] - tmpDat.LHS[i]
                        ct.append(tmpDat.time(tmpDat.LTO[i])-tmpDat.time(tmpDat.LHS[i]))
                        pct10 = tmpDat.LHS[i] + round(frames*.1)
                        pct40 = tmpDat.LHS[i] + round(frames*.4)
                        pct50 = tmpDat.LHS[i] + round(frames*.5)
                        pct60 = tmpDat.LHS[i] + round(frames*.6)
                        pct90 = tmpDat.LHS[i] + round(frames*.9)
                        
                        maxmaxToes.append(np.max(tmpDat.LplantarToe[tmpDat.LHS[i]:tmpDat.LTO[i]])*6.895)
                        toePmidstance.append(np.mean(tmpDat.LplantarToe[pct40:pct60,:,:])*6.895)
                                           
                        heelAreaLate.append(np.count_nonzero(tmpDat.LplantarHeel[pct50:tmpDat.LTO[i], :, :])/(tmpDat.LTO[i] - pct50)/tmpDat.LplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                        heelPLate.append(np.mean(tmpDat.LplantarHeel[pct90:tmpDat.LTO[i], :, :])*6.895)
        
                        latPmidstance.append(np.mean(tmpDat.LplantarLateral[pct40:pct60, :, :])*6.895)
                        latAreamidstance.append(np.count_nonzero(tmpDat.LplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarLateralSensNo*100)
                        medPmidstance.append(np.mean(tmpDat.LplantarMedial[pct40:pct60, :, :])*6.895)
                        medAreamidstance.append(np.count_nonzero(tmpDat.LplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarMedialSensNo*100)
                        latPropMid.append(np.sum(tmpDat.LplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
                        medPropMid.append(np.sum(tmpDat.LplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
                        
                        if len(tmpDat.dorsalMat) != 0: 
                            
                            dorsalVar.append(np.std(tmpDat.dorsalMat[tmpDat.LHS[i]:tmpDat.LTO[i], :, :])/np.mean(tmpDat.dorsalMat[tmpDat.LHS[i]:tmpDat.LTO[i], :, :]))
                            maxDorsal.append(np.max(tmpDat.dorsalMat[tmpDat.LHS[i]:tmpDat.LTO[i], :, :])*6.895)
                            
                        else:
                            dorsalVar.append('nan')
                            maxDorsal.append('nan')
    
        
            outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(oOrder), 'ContactTime':list(ct),
                                     'toeP_mid':list(toePmidstance),'toeArea_mid':list(toeAreamidstance), 'maxmaxToes':list(maxmaxToes),
                                     'ffP_late':list(ffPLate), 'ffArea_late':list(ffAreaLate), 'ffP_Mid':list(ffPMid), 'ffArea_Mid':list(ffAreaMid), 'ffPMax_late':list(ffPMaxLate),
                                     'mfP_late':list(mfPLate), 'mfArea_late':list(mfAreaLate), 'mfP_Mid':list(mfPMid), 'mfArea_Mid':list(mfAreaMid),
                                     'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate),  
                                     'latP_mid':list(latPmidstance), 'latArea_mid':list(latAreamidstance), 'latPropMid':list(latPropMid),
                                     'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medPropMid':list(medPropMid),
                                     'dorsalVar':list(dorsalVar), 'maxDorsalP':list(maxDorsal)
                                     
                                     })
    
            outfileName = fPath + '0_CompiledResults.csv'
            if save_on == 1:
                if os.path.exists(outfileName) == False:
                
                    outcomes.to_csv(outfileName, header=True, index = False)
            
                else:
                    outcomes.to_csv(outfileName, mode='a', header=False, index = False) 

    except:
            print('Not usable data')
            badFileList.append(fName)             
            
            
            
            
            
        


