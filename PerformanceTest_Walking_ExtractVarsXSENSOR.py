# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:28:44 2021
Script to process XSENSOR pressures during gait.
Updates to accomidate gradient decent event detection along with dialog boxes
for verifying data quality.

@author: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
from scipy import interpolate
import os
from tkinter import messagebox

fwdLook = 30
fThresh = 50
freq = 100 # sampling frequency
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force, fThreshold):
    """
    The purpose of this function is to determine the landings (foot contacts)
    events when the vertical force exceeds the force threshold
    
    Parameters
    ----------
    force : list
        vertical ground reaction force. 
    
    fThreshold : float
        threshold to detect landings
    
    Returns
    -------
    ric : list
        indices of the landings (foot contacts)

    """
    ric = []
    for step in range(len(force)-1):
        if force[step] < fThreshold and force[step + 1] >= fThreshold:
            ric.append(step)
    return ric

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force, fThreshold):
    """
    The purpose of this function is to determine the take-off
    events when the vertical force exceeds the force threshold

    Parameters
    ----------
    force : list
        vertical ground reaction force. 
    
    fThreshold : float
        threshold to detect landings
    
    Returns
    -------
    ric : list
        indices of the landings (foot contacts)

    """
    rto = []
    for step in range(len(force)-1):
        if force[step] >= fThreshold and force[step + 1] < fThreshold:
            rto.append(step + 1)
    return rto

def trimTakeoffs(landings, takeoffs):
    """
    Function to ensure that the first take-off index is greater than the first 
    landing index

    Parameters
    ----------
    landings : list
        indices of the landings
    takeoffs : list
        indices of the take-offs

    Returns
    -------
    takeoff

    """
    if takeoffs[0] < landings[0]:
        del(takeoffs[0])
    return(takeoffs)

def trimLandings(landings, trimmedTakeoffs):
    """
    Function to ensure that the first landing index is greater than the first 
    take-off index

    Parameters
    ----------
    landings : list
        indices of the landings
    takeoffs : list
        indices of the take-offs

    Returns
    -------
    landings: list
        updated indices of the landings

    """
    if landings[len(landings)-1] > trimmedTakeoffs[len(trimmedTakeoffs)-1]:
        del(landings[-1])
    return(landings)

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
    
    windowSize = round(0.8*freq)
    
    n = 5
    
    # Filter the forces
    # Set up a 2nd order 10 Hz low pass buttworth filter
    # cut = 10
    # w = cut / (freq / 2) # Normalize the frequency
    # b, a = sig.butter(2, w, 'low')
    # # Filter the force signal
    # vForce = sig.filtfilt(b, a, vForce)
        
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
    # Remove events that are too close to the start/end of the trial
    nearHS = nearHS[(nearHS > windowSize)*(nearHS < len(vForce)-windowSize)]
    nearTO = nearTO[(nearTO > windowSize)*(nearTO < len(vForce)-windowSize)]
    
    HS = []
    for idx in nearHS:
        for ii in range(idx,idx-windowSize,-1):
            if vForce[ii] == 0 or vForce[ii] < vForce[ii-n] or ii == idx-windowSize+1:
                HS.append(ii)
                break
    
    # Only utilize unique event detections
    HS = np.unique(HS)
    HS = np.array(HS)
    # Used to eliminate dections that could be due to the local minima in the 
    # middle of a walking step 
    HS = HS[vForce[HS]<200]
    
    # Only search for toe-offs that correspond to heel-strikes
    TO = []
    for ii in range(len(HS)):
        tmp = np.where(nearTO > HS[ii])[0]
        if len(tmp) > 0:
            idx = nearTO[tmp[0]]
            for jj in range(idx,idx+windowSize):
                if vForce[jj] == 0 or vForce[jj] < vForce[jj+n] or jj == idx+windowSize-1:
                    TO.append(jj)
                    break
        else:
            np.delete(HS,ii)
            
            
    return(HS,TO)

def intp_steps(var,landings,takeoffs,GS):
    """
    Function to interpolate the variable of interest across a stride
    (from foot contact to subsiquent foot contact) in order to plot the 
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

# Read in files
# only read .asc files for this work
fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/Hike/FocusAnkleDualDial_Midcut_Sept2022/XSENSOR/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

debug = 1
save_on = 0

# Initialize Variables
sdHeel = []
meanToes = []
meanFf = []
maxmeanHeel = []
maxmaxHeel = []
maxmeanToes = []
maxmaxMet = []
maxmaxMid = []
maxmaxToes = []
cvHeel = []
heelArea = []
heelAreaP = []
ffAreaEarly = []
ffAreaIC = []
meanTotalP = []
Subject = []
Config = []
Side = []
Movement = []
badFileList = []

for fName in entries:
        try:
            subName = fName.split(sep = "_")[0]
            ConfigTmp = fName.split(sep="_")[1]
            moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
            
            # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
            if (moveTmp == 'run') or (moveTmp == 'Run') or (moveTmp == 'running') or (moveTmp == 'Running') or (moveTmp == 'walk') or (moveTmp == 'Walk') or (moveTmp == 'walking') or (moveTmp == 'Walking') or (moveTmp == 'Trail') or (moveTmp == 'UH') or (moveTmp == 'DH'):
                dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
                print('Processing:' + fName)
                dat.columns = ['Frame', 'Date',	'Time',	'Units', 'Threshold', 
                           'SensorLF', 'SideLF', 'RowsLF',	'ColumnsLF', 'AverageP_LF',	'MinP_LF',	'PeakP_LF', 'ContactArea_LF', 'TotalArea_LF', 'ContactPct_LF', 'EstLoadLF',	'StdDevLF',	
                           'SensorRF', 'SideRF', 'RowsRF', 'ColumnsRF', 'AverageP_RF', 'MinP_RF',	'PeakP_RF', 'ContactArea_RF', 'TotalArea_RF',	'ContactPct_RF', 'EstLoadRF', 'StdDevRF',	 
                           
                            'L_Heel', 'SideLHeel', 'RowsLHeel', 'ColumnsLHeel', 'L_Heel_Average',	'L_Heel_MIN','L_Heel_MAX', 'L_Heel_ContactArea',
                            'L_Heel_TotalArea', 'L_Heel_Contact',	'L_Heel_EstLoad',	'L_Heel_StdDev',	
                           
                            'R_Heel', 'SideRHeel', 'RowsRHeel', 'ColumnsRHeel','R_Heel_Average',	'R_Heel_MIN',	'R_Heel_MAX',	'R_Heel_ContactArea',
                            'R_Heel_TotalArea', 'R_Heel_Contact',	'R_Heel_EstLoad', 'R_Heel_StdDev',	 
                           
                            'L_Midfoot', 'SideLMidfoot', 'RowsLMidfoot', 'ColumnsLMidfoot',	'L_Midfoot_Average', 'L_Midfoot_MIN', 'L_Midfoot_MAX',	'L_Midfoot_ContactArea',	
                            'L_Midfoot_TotalArea', 'L_Midfoot_Contact',	'L_Midfoot_EstLoad', 'L_Midfoot_StdDev', 
                                                
                            'R_Midfoot', 'SideRMidfoot', 'RowsRMidfoot', 'ColumnsRMidfoot',	'R_Midfoot_Average',	'R_Midfoot_MIN',	'R_Midfoot_MAX', 'R_Midfoot_ContactArea', 	
                            'R_Midfoot_TotalArea',	'R_Midfoot_Contact',	'R_Midfoot_EstLoad', 'R_Midfoot_StdDev',	
                           
                            'L_Metatarsal', 'SideLMets', 'RowsLMets', 'ColumnsLMets','L_Metatarsal_Average',	'L_Metatarsal_MIN',	'L_Metatarsal_MAX', 	
                            'L_Metatarsal_ContactArea',	'L_Metatarsal_TotalArea', 'L_Metatarsal_Contact',	
                            'L_Metatarsal_EstLoad',	'L_Metatarsal_StdDev',	
                           
                            'R_Metatarsal', 'SideRMets', 'RowsRMets', 'ColumnsRMets','R_Metatarsal_Average',	'R_Metatarsal_MIN',	'R_Metatarsal_MAX','R_Metatarsal_ContactArea',	
                            'R_Metatarsal_TotalArea',	'R_Metatarsal_Contact',	'R_Metatarsal_EstLoad',	'R_Metatarsal_StdDev',	
                           
                            'L_Toe', 'SideLToes', 'RowsLToes', 'ColumnsLToes',	'L_Toe_Average', 'L_Toe_MIN', 'L_Toe_MAX', 'L_Toe_ContactArea', 'L_Toe_TotalArea',	
                            'L_Toe_L_Toe_Contact',	'L_Toe_EstLoad', 'L_Toe_StdDev',

                            'R_Toe', 'SideRToes', 'RowsRToes', 'ColumnsRToes','R_Toe_Average', 'R_Toe_MIN',	'R_Toe_MAX',	'R_Toe_Contact Area','R_Toe_TotalArea',	
                            'R_Toe_Contact',	'R_Toe_EstLoad', 'R_Toe_StdDev'
                           
                             ]
                #__________________________________________________________________
                # Right foot:
                # Est. Load (lbf) conversion to N = 1lbf * 4.44822N 
                # Convert the output load to Newtons from lbf
                RForce = np.array(dat.EstLoadRF)*4.44822
                RForce = zeroInsoleForce(RForce,freq)
                [RHS,RTO] = findGaitEvents(RForce,freq)
                                
                # Remove strides that have a peak GRF below 1000 N
                # Remove strides that are below 0.5 and above 1.5 seconds
                RGS = []    
                # May need to exclude poor toe-off dectections here as well
                for jj in range(len(RHS)-1):
                    if np.max(RForce[RHS[jj]:RTO[jj]]) > 500 and RHS[jj] > 2000:
                        if (RHS[jj+1] - RHS[jj]) > 0.5*freq and RHS[jj+1] - RHS[jj] < 1.5*freq:
                            RGS.append(jj)
                
                #__________________________________________________________________
                # Left foot:
                # Est. Load (lbf) conversion to N = 1lbf * 4.44822N 
                # Convert the output load to Newtons from lbf
                LForce = np.array(dat.EstLoadLF)*4.44822
                LForce = zeroInsoleForce(LForce,100)
                [LHS,LTO] = findGaitEvents(LForce,100)
                
                # Remove strides that have a peak GRF below 1000 N
                # Remove strides that are below 0.5 and above 1.5 seconds
                LGS = []    
                # May need to exclude poor toe-off dectections here as well
                for jj in range(len(LHS)-1):
                    if np.max(LForce[LHS[jj]:LTO[jj]]) > 500 and LHS[jj] > 2000:
                        if (LHS[jj+1] - LHS[jj]) > 0.5*freq and LHS[jj+1] - LHS[jj] < 1.5*freq:
                            LGS.append(jj)
                
                #______________________________________________________________
                # Debugging: Creation of dialog box for looking where foot contact are accurate
                answer = True # Defaulting to true: In case "debug" is not used
                if debug == 1:
                
                    plt.figure
                    plt.subplot(1,2,1)
                    plt.plot(intp_steps(LForce,LHS,LTO,LGS))
                    plt.ylabel('Insole Force [N]')
                    plt.xlabel('% Step')
                    plt.title('Left Interpolated Steps')
                    
                    plt.subplot(1,2,2)
                    plt.plot(intp_steps(RForce,RHS,RTO,RGS))
                    plt.xlabel('% Step')
                    plt.title('Right Interpolated Steps')
                    answer = messagebox.askyesno("Question","Is data clean?")
                    plt.close()                
                    
                    if answer == False:
                        plt.close('all')
                        print('Adding file to bad file list')
                        badFileList.append(fName)
                
                if answer == True:
                    print('Estimating point estimates')
                    #__________________________________________________________
                    # Right Side Metrics
                    for i in RGS:
                        try:
                            # Note: converting psi to kpa with 1psi=6.89476kpa                            
                            peakFidx = round((RTO[i] - RHS[i])/2)
                            
                            heelAreaLate = np.mean(dat.R_Heel_ContactArea[RHS[i]+peakFidx:RTO[i]])
                            heelPAreaLate = np.mean(dat.R_Heel_Contact[RHS[i]+peakFidx:RTO[i]])
                            ffAreaEarly.append(np.mean(dat.R_Metatarsal_ContactArea[RHS[i]:RHS[i]+peakFidx]))
                            ffAreaIC.append(dat.R_Metatarsal_ContactArea[RHS[i]])
                            meanHeel = np.mean(dat.R_Heel_Average[RHS[i]:RTO[i]])*6.89476
                            meanMidfoot = np.mean(dat.R_Midfoot_Average[RHS[i]:RTO[i]])*6.89476    
                            meanForefoot = np.mean(dat.R_Metatarsal_Average[RHS[i]:RTO[i]])*6.89476 
                            meanToe = np.mean(dat.R_Toe_Average[RHS[i]:RTO[i]])*6.89476   
                            
                            stdevHeel = np.std(dat.R_Heel_Average[RHS[i]:RTO[i]])*6.89476
                            maximummeanHeel = np.max(dat.R_Heel_Average[RHS[i]:RTO[i]])*6.89476
                            maximummaxHeel = np.max(dat.R_Heel_MAX[RHS[i]:RTO[i]])*6.89476
                            maximummaxMet = np.max(dat.R_Metatarsal_MAX[RHS[i]:RTO[i]])*6.89476
                            maximummaxMid = np.max(dat.R_Midfoot_MAX[RHS[i]:RTO[i]])*6.89476
                            maximummeanToe = np.max(dat.R_Toe_Average[RHS[i]:RTO[i]])*6.89476
                            maximummaxToe = np.max(dat.R_Toe_MAX[RHS[i]:RTO[i]])*6.89476
                            meanFoot = (meanHeel + meanMidfoot + meanForefoot + meanToe)/4
                            
                            meanTotalP.append(meanFoot)
                            sdHeel.append(stdevHeel)
                            meanToes.append(meanToe/meanFoot)
                            meanFf.append(meanForefoot)
                            maxmeanHeel.append(maximummeanHeel)
                            maxmaxHeel.append(maximummaxHeel)
                            cvHeel.append(stdevHeel/meanFoot)
                            heelArea.append(heelAreaLate)
                            heelAreaP.append(heelPAreaLate)
                        
                            maxmaxMid.append(maximummaxMid)
                            maxmaxMet.append(maximummaxMet)
                            
                            maxmeanToes.append(maximummeanToe)
                            maxmaxToes.append(maximummaxToe)
                        
                            Subject.append(subName)
                            Config.append(ConfigTmp)
                            Side.append('Right')
                            Movement.append(moveTmp)
                        except:
                            print(fName, RHS[i])
                    #__________________________________________________________
                    # Left Side Metrics
                    for i in LGS:
                        try:
                            # Note: converting psi to kpa with 1psi=6.89476kpa
                            # Updating to 50% of stance
                            peakFidx = round((LTO[i] - LHS[i])/2)
                            heelAreaLate = np.mean(dat.L_Heel_ContactArea[LHS[i]+peakFidx:LTO[i]])
                            heelPAreaLate = np.mean(dat.L_Heel_Contact[LHS[i]+peakFidx:LTO[i]])
                            ffAreaEarly.append(np.mean(dat.L_Metatarsal_ContactArea[LHS[i]:LHS[i]+peakFidx]))
                            ffAreaIC.append(dat.L_Metatarsal_ContactArea[LHS[i]])
                            meanHeel = np.mean(dat.L_Heel_Average[LHS[i]:LTO[i]])*6.89476
                            meanMidfoot = np.mean(dat.L_Midfoot_Average[LHS[i]:LTO[i]])*6.89476    
                            meanForefoot = np.mean(dat.L_Metatarsal_Average[LHS[i]:LTO[i]])*6.89476 
                            meanToe = np.mean(dat.L_Toe_Average[LHS[i]:LTO[i]])*6.89476   
                            
                            stdevHeel = np.std(dat.L_Heel_Average[LHS[i]:LTO[i]])*6.89476
                            maximummeanHeel = np.max(dat.L_Heel_Average[LHS[i]:LTO[i]])*6.89476
                            maximummaxHeel = np.max(dat.L_Heel_MAX[LHS[i]:LTO[i]])*6.89476
                            maximummaxMet = np.max(dat.L_Metatarsal_MAX[LHS[i]:LTO[i]])*6.89476
                            maximummaxMid = np.max(dat.L_Midfoot_MAX[LHS[i]:LTO[i]])*6.89476
                            maximummeanToe = np.max(dat.L_Toe_Average[LHS[i]:LTO[i]])*6.89476
                            maximummaxToe = np.max(dat.L_Toe_MAX[LHS[i]:LTO[i]])*6.89476
                            meanFoot = (meanHeel + meanMidfoot + meanForefoot + meanToe)/4
                            
                            meanTotalP.append(meanFoot)
                            sdHeel.append(stdevHeel)
                            meanToes.append(meanToe/meanFoot)
                            meanFf.append(meanForefoot)
                        
                            maxmaxMid.append(maximummaxMid)
                            maxmaxMet.append(maximummaxMet)
                            
                            maxmeanHeel.append(maximummeanHeel)
                            maxmaxHeel.append(maximummaxHeel)
                            cvHeel.append(stdevHeel/meanFoot)
                            heelArea.append(heelAreaLate)
                            heelAreaP.append(heelPAreaLate)
                            
                            maxmeanToes.append(maximummeanToe)
                            maxmaxToes.append(maximummaxToe)
                        
                            Subject.append(subName)
                            Config.append(ConfigTmp)
                            Side.append('Left')
                            Movement.append(moveTmp)
                        except:
                            print(fName, LHS[i])
            
            else:
                    print('Only anlayzing running')
        except:
            print(fName) 
            
             
outcomes = pd.DataFrame({'Subject':list(Subject),'Config':list(Config),'Side':list(Side), 'Movement':list(Movement), 'meanTotalP':list(meanTotalP),
                         'sdHeel': list(sdHeel),'cvHeel':list(cvHeel), 'heelArea':list(heelArea), 'heelAreaP':list(heelAreaP),'ffAreaEarly':list(ffAreaEarly), 'ffAreaIC':list(ffAreaIC),
                         'meanToes':list(meanToes), 'meanFf':list(meanFf),
                         'maxmeanHeel':list(maxmeanHeel), 'maxmeanToes':list(maxmeanToes),
                         'maxmaxHeel':list(maxmaxHeel), 'maxmaxToes':list(maxmaxToes),
                         'maxmaxMid':list(maxmaxMid), 'maxmaxMet':list(maxmaxMet)
                         })  

if save_on == 1:
    outFileName = fPath + 'CompiledPressureData_allwalking.csv'
    outcomes.to_csv(outFileName, index = False)          
            
        
