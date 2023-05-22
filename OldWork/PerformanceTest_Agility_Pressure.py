# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 07:02:29 2022

This code compiles data from XSENSOR insoles during CMJ and skater jumps. Point estimates are extracted and saved as 0_CompiledPressureData.csv.
Times series data are exported for each metric as 1_ ..... .csv. These can be plotted using the code PerformanceTest_Agility_PressurePlotting for visual examination.

@author: Kate.Harrison


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
import os
from tkinter import messagebox

fwdLook = 30
fThresh = 50
freq = 100 # sampling frequency
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold


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
fPath = 'C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/AgilityPerformanceData/2022_Tests/CPDMech_ForefootMechII_Nov2022/XSENSOR/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

debug = 1
save_on = 1

# Initialize Variables

badFileList = []


for fName in entries:
        try:
            #fName = entries[1]
            subName = fName.split(sep = "_")[0]
            ConfigTmp = fName.split(sep="_")[1]
            moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
            
            # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
            if (moveTmp == 'Skater') or (moveTmp == 'cmj') or (moveTmp == 'CMJ'):
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

                            'R_Toe', 'SideRToes', 'RowsRToes', 'ColumnsRToes','R_Toe_Average', 'R_Toe_MIN',	'R_Toe_MAX',	'R_Toe_ContactArea','R_Toe_TotalArea',	
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
                    ct = RTO[jj] - RHS[jj]
                    peak = 1.5*np.mean(RForce)
                    if np.max(RForce[RHS[jj]:RHS[jj]+round(0.25*ct)]) > peak  and np.max(RForce[RHS[jj] + round(0.75*ct):RTO[jj]]) > peak:
                        if (RHS[jj+1] - RHS[jj]) > 0.5*freq and RHS[jj+1] - RHS[jj] < 1.5*freq:
                            RGS.append(jj)
                
               

                
                #______________________________________________________________
                # Debugging: Creation of dialog box for looking where foot contact are accurate
                answer = True # Defaulting to true: In case "debug" is not used
                if debug == 1:
                
                    plt.figure()
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
                    
                    ct = []
                    toePmidstance = []
                    toeAreamidstance = []
                    ffAreaLate = []
                    ffPLate = []
                    heelAreaLate = []
                    heelPLate = []
                    movement = []
                
                    
                    for i in RGS:
                        
                        frames = RTO[i] - RHS[i]
                        ct.append(frames/200)
                        toePmidstance.append(np.mean( dat.R_Toe_Average[RHS[i] + round(0.4*frames):RHS[i] + round(0.8*frames)]))
                        toeAreamidstance.append(np.mean( dat.R_Toe_ContactArea[RHS[i] + round(0.4*frames):RHS[i] + round(0.8*frames)]))
                        movement.append(moveTmp)
                        ffAreaLate.append(np.mean(dat.R_Metatarsal_ContactArea[RHS[i] + round(0.9*frames):RTO[i]]))
                        ffPLate.append(np.mean(dat.R_Metatarsal_Average[RHS[i] + round(0.9*frames):RTO[i]]))
                        heelAreaLate.append(np.mean(dat.R_Heel_ContactArea[RHS[i] + round(0.9*frames):RTO[i]]))
                        heelPLate.append(np.mean(dat.R_Heel_Average[RHS[i] + round(0.9*frames):RTO[i]]))
                    #__________________________________________________________
                    # Right Side Metrics
                    
                    toeP_ts = (intp_steps(dat.R_Toe_Average, RHS, RTO, RGS ).transpose())
                    ffP_ts = (intp_steps(dat.R_Metatarsal_Average, RHS, RTO, RGS ).transpose())
                    mfP_ts = ( intp_steps(dat.R_Midfoot_Average, RHS, RTO, RGS).transpose())
                    heelP_ts = (intp_steps(dat.R_Heel_Average, RHS, RTO, RGS).transpose())
                    
                    Subject = [subName]*toeP_ts.shape[0]
                    Config = [ConfigTmp]*toeP_ts.shape[0]
                    
                    toeP_ts = pd.DataFrame(np.stack(toeP_ts))
                    toeP_ts['Subject'] = Subject
                    toeP_ts['Movement'] = movement
                    toeP_ts['Config'] = Config
                    toeP_ts['ContactTime'] = ct
                    
                    ffP_ts = pd.DataFrame(np.stack(ffP_ts))
                    ffP_ts['Subject'] = Subject
                    ffP_ts['Movement'] = movement
                    ffP_ts['Config'] = Config
                    ffP_ts['ContactTime'] = ct
                    
                    mfP_ts = pd.DataFrame(np.stack(mfP_ts))
                    mfP_ts['Subject'] = Subject
                    mfP_ts['Movement'] = movement
                    mfP_ts['Config'] = Config
                    mfP_ts['ContactTime'] = ct
                    
                    heelP_ts = pd.DataFrame(np.stack(heelP_ts))
                    heelP_ts['Subject'] = Subject
                    heelP_ts['Movement'] = movement
                    heelP_ts['Config'] = Config
                    heelP_ts['ContactTime'] = ct
                    
                    toeArea_ts = (intp_steps(dat.R_Toe_Contact, RHS, RTO, RGS ).transpose())
                    ffArea_ts = (intp_steps(dat.R_Metatarsal_Contact, RHS, RTO, RGS ).transpose())
                    mfArea_ts = ( intp_steps(dat.R_Midfoot_Contact, RHS, RTO, RGS).transpose())
                    heelArea_ts = (intp_steps(dat.R_Heel_Contact, RHS, RTO, RGS).transpose())
                    
                    toeArea_ts = pd.DataFrame(np.stack(toeArea_ts))
                    toeArea_ts['Subject'] = Subject
                    toeArea_ts['Movement'] = movement
                    toeArea_ts['Config'] = Config
                    toeArea_ts['ContactTime'] = ct
                    
                    ffArea_ts = pd.DataFrame(np.stack(ffArea_ts))
                    ffArea_ts['Subject'] = Subject
                    ffArea_ts['Movement'] = movement
                    ffArea_ts['Config'] = Config
                    ffArea_ts['ContactTime'] = ct
                   
                    
                    mfArea_ts = pd.DataFrame(np.stack(mfArea_ts))
                    mfArea_ts['Subject'] = Subject
                    mfArea_ts['Movement'] = movement
                    mfArea_ts['Config'] = Config
                    mfArea_ts['ContactTime'] = ct
                    
                    heelArea_ts = pd.DataFrame(np.stack(heelArea_ts))
                    heelArea_ts['Subject'] = Subject
                    heelArea_ts['Movement'] = movement
                    heelArea_ts['Config'] = Config
                    heelArea_ts['ContactTime'] = ct
                    
                    pointEst = pd.DataFrame({'Subject':list(Subject), 'Config':list(Config), 'Movement':list(movement), 'ContactTime':list(ct),
                                             
                                             'toePmidstance':list(toePmidstance), 'toeAreamidstance':list(toeAreamidstance),
                                             'ffPLate':list(ffPLate), 'ffAreaLate':list(ffAreaLate),
                                             'heelPLate':list(heelPLate), 'heelAreaLate':list(heelAreaLate)})
                    
                    if save_on == 1:
                        
                        if os.path.exists(fPath + '1_ToePressureTimeSeries.csv') == False:
                            toeP_ts.to_csv(fPath + '1_ToePressureTimeSeries.csv', mode='a', header=True, index = False)
                            ffP_ts.to_csv(fPath + '1_ForefootPressureTimeSeries.csv', mode='a', header=True, index = False)
                            mfP_ts.to_csv(fPath + '1_MidfootPressureTimeSeries.csv', mode='a', header=True, index = False)
                            heelP_ts.to_csv(fPath + '1_HeelPressureTimeSeries.csv', mode='a', header=True, index = False)
                            toeArea_ts.to_csv(fPath + '1_ToeContactTimeSeries.csv', mode='a', header=True, index = False)
                            ffArea_ts.to_csv(fPath + '1_ForefootContactTimeSeries.csv', mode='a', header=True, index = False)
                            mfArea_ts.to_csv(fPath + '1_MidfootContactTimeSeries.csv', mode='a', header=True, index = False)
                            heelArea_ts.to_csv(fPath + '1_HeelContactTimeSeries.csv', mode='a', header=True, index = False)
                            pointEst.to_csv(fPath + '0_CompiledPressureData.csv', mode = 'a', header=True, index = False)

                        else:
                            toeP_ts.to_csv(fPath + '1_ToePressureTimeSeries.csv', mode='a', header=False, index = False)
                            ffP_ts.to_csv(fPath + '1_ForefootPressureTimeSeries.csv', mode='a', header=False, index = False)
                            mfP_ts.to_csv(fPath + '1_MidfootPressureTimeSeries.csv', mode='a', header=False, index = False)
                            heelP_ts.to_csv(fPath + '1_HeelPressureTimeSeries.csv', mode='a', header=False, index = False)
                            toeArea_ts.to_csv(fPath + '1_ToeContactTimeSeries.csv', mode='a', header=False, index = False)
                            ffArea_ts.to_csv(fPath + '1_ForefootContactTimeSeries.csv', mode='a', header=False, index = False)
                            mfArea_ts.to_csv(fPath + '1_MidfootContactTimeSeries.csv', mode='a', header=False, index = False)
                            heelArea_ts.to_csv(fPath + '1_HeelContactTimeSeries.csv', mode='a', header=False, index = False)
                            pointEst.to_csv(fPath + '0_CompiledPressureData.csv', mode = 'a', header=False, index = False)
                  
                    
            else:
                    print('Not Analyzing this movement')
        except:
            print(fName) 
            
             
  

         
            
        
