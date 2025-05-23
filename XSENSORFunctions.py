# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:31:04 2025

@author: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
import scipy.signal as sig
from datetime import datetime

# list of functions 
def delimitTrial(inputDF,FName,FilePath):
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
        np.save(FilePath+FName+'TrialSeg.npy',trial_segment)

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

def readXSENSORFile(inputName,FilePath):
    """
    Open and provide a dataframe for an XSENSOR file

    Parameters
    ----------
    inputName : str
        File name (.csv)
    FilePath : str
        Destination where the file is stored

    Returns
    -------
    dat : pandas dataframe
        Dataframe of the XSENSOR data

    """
    dat = pd.read_csv(FilePath+inputName, sep=',', header = 0, low_memory=False)
    if dat.shape[1] == 2:
        dat = pd.read_csv(FilePath+inputName, sep=',', header = 1, low_memory=False)
    return(dat)

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
    RForce: np.array
    
    LCOP_X: np.array 
    LCOP_Y: np.array 

    RCOP_X: np.array 
    RCOP_Y: np.array 
    
    Lacc: np.array
    Lgyr: np.array
    
    Racc: np.array
    Rgyr: np.array

    config: str
    movement: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 

def createTSmat(inputName,FilePath,dat):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """       
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2] 
    
    time = np.array([((datetime.strptime(timestr,' %H:%M:%S.%f')-datetime(1900,1,1)).total_seconds()) for timestr in dat.Time])
    time = time - time[0]
    
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
    RCOP_Y = []
    RCOP_X = []
    Racc = []
    Rgyr = []

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
    LCOP_Y = []
    LCOP_X = []
    Lacc = []
    Lgyr = []
    
    dorsalMat = []
    dorsalForefoot = []
    dorsalMidfoot = []
    dorsalInstep = []
    dorsalSensNo =[]
    dorsalForefootSensNo = []
    dorsalMidfootSensNo = []
    dorsalInstepSensNo = []

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
            LplantarMedial =LplantarMat[:,:,:4]
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
            
            if 'IMU AX' in dat.columns:
                Lacc = np.array([dat['IMU AX'], dat['IMU AY'], dat['IMU AZ']]).T
                Lgyr = np.array([dat['IMU GX'], dat['IMU GY'], dat['IMU GZ']]).T
        
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
            dorsalMat[dorsalMat < 0.1] = 0  # 0.1 used here due to the lower calibration range of the dorsal sensor pad
            
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
                if 'dorsalSensel' in locals():
                    RplantarSensel = dat.loc[:, 'S_1_2.1':'S_31_7']
                else:
                    RplantarSensel = dat.loc[:, 'S_1_2':'S_31_7']
            
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
            
            if 'IMU AX.1' in dat.columns:
                Racc = np.array([dat['IMU AX'], dat['IMU AY'], dat['IMU AZ']]).T
                Rgyr = np.array([dat['IMU GX'], dat['IMU GY'], dat['IMU GZ']]).T
                

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
                     LForce, RForce,
                     LCOP_X, LCOP_Y, RCOP_X, RCOP_Y,
                     Lacc,Lgyr,Racc,Rgyr,
                     config, movement, subj, dat)
    
    return(result)
