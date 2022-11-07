# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:28:44 2021
Script to process MVA files from cycling pilot test

@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
from scipy import interpolate
import os

fwdLook = 30
fThresh = 50
freq = 100 # sampling frequency (intending to change to 150 after this study)
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
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

def zeroInsoleForce(vForce,freq):
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
    # Eliminate detections above 100 N
    HS = np.unique(HS)
    HS = np.array(HS)
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
            
            
            
            
    #     1
    
    
    # for idx in nearTO:       
    #     for ii in range(idx,idx+windowSize):
    #         if vForce[ii] == 0 or vForce[ii] < vForce[ii+n] or ii == idx+windowSize-1:
    #             TO.append(ii)
    #             break
    # # Eliminate detections above 100 N
    # TO = np.array(TO)
    # TO = TO[vForce[TO]<200]
    
    # # Only take unique events
    
    # TO = np.unique(TO)
      
    # # Remove extra HS/TO
    # if TO[0] < HS[0]:
    #     TO = TO[1:]
    # if HS[-1] > TO[-1]:
    #     HS = HS[0:-1]
    return(HS,TO)

def intp_strides(var,landings,takeoffs,GS):
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

save_on = 1

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

for fName in entries:
        try:
            #fName = entries[3] #Load one file at a time
            dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            subName = fName.split(sep = "_")[0]
            ConfigTmp = fName.split(sep="_")[1]
            moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
            
            # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
            if (moveTmp == 'run') or (moveTmp == 'Run') or (moveTmp == 'running') or (moveTmp == 'Running') or (moveTmp == 'walk') or (moveTmp == 'Walk') or (moveTmp == 'walking') or (moveTmp == 'Walking') or (moveTmp == 'Trail') or (moveTmp == 'UH') or (moveTmp == 'DH'):
                print(fName)
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
                # #__________________________________________________________________
                # # Right foot:
                # # Est. Load (lbf) conversion to N = 1lbf * 4.44822N 
                # # Convert the output load to Newtons from lbf
                # RForce = np.array(dat.EstLoadRF)*4.44822
                # RForce = RForce-np.min(RForce)
            
                # # Compute landings and takeoffs
                # landings = findLandings(RForce, fThresh)
                # takeoffs = findTakeoffs(RForce, fThresh)

                # landings[:] = [x for x in landings if x < takeoffs[-1]]
                # takeoffs[:] = [x for x in takeoffs if x > landings[0]]
                
                # landings = np.array(landings)
                # takeoffs = np.array(takeoffs)
                
                # plt.plot(intp_strides(RForce,landings,takeoffs,np.array(range(len(landings)))))
                # plt.close()
            
                # for i in range(len(landings)):
                #     try:
                #         # Note: converting psi to kpa with 1psi=6.89476kpa
                #         #i = 0
                        
                #         peakFidx = np.argmax(dat.EstLoadRF[landings[i]:takeoffs[i]])
                #         heelAreaLate = np.mean(dat.R_Heel_ContactArea[landings[i]+peakFidx:takeoffs[i]])
                #         heelPAreaLate = np.mean(dat.R_Heel_Contact[landings[i]+peakFidx:takeoffs[i]]) ######
                #         heelPAreaLate = np.mean(dat.R_Heel_Contact[landings[i]+peakFidx:takeoffs[i]]) ######
                #         ffAreaEarly.append(np.mean(dat.R_Metatarsal_ContactArea[landings[i]:landings[i]+peakFidx]))
                #         ffAreaIC.append(dat.R_Metatarsal_ContactArea[landings[i]])
                #         meanHeel = np.mean(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                #         meanMidfoot = np.mean(dat.R_Midfoot_Average[landings[i]:takeoffs[i]])*6.89476    
                #         meanForefoot = np.mean(dat.R_Metatarsal_Average[landings[i]:takeoffs[i]])*6.89476 
                #         meanToe = np.mean(dat.R_Toe_Average[landings[i]:takeoffs[i]])*6.89476   
                        
                #         stdevHeel = np.std(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                #         maximummeanHeel = np.max(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                #         maximummaxHeel = np.max(dat.R_Heel_MAX[landings[i]:takeoffs[i]])*6.89476
                #         maximummaxMet = np.max(dat.R_Metatarsal_MAX[landings[i]:takeoffs[i]])*6.89476
                #         maximummaxMid = np.max(dat.R_Midfoot_MAX[landings[i]:takeoffs[i]])*6.89476
                #         maximummeanToe = np.max(dat.R_Toe_Average[landings[i]:takeoffs[i]])*6.89476
                #         maximummaxToe = np.max(dat.R_Toe_MAX[landings[i]:takeoffs[i]])*6.89476
                #         meanFoot = (meanHeel + meanMidfoot + meanForefoot + meanToe)/4
                        
                #         meanTotalP.append(meanFoot)
                #         sdHeel.append(stdevHeel)
                #         meanToes.append(meanToe/meanFoot)
                #         meanFf.append(meanForefoot)
                #         maxmeanHeel.append(maximummeanHeel)
                #         maxmaxHeel.append(maximummaxHeel)
                #         cvHeel.append(stdevHeel/meanFoot)
                #         heelArea.append(heelAreaLate)
                #         heelAreaP.append(heelPAreaLate)
                    
                #         maxmaxMid.append(maximummaxMid)
                #         maxmaxMet.append(maximummaxMet)
                        
                #         maxmeanToes.append(maximummeanToe)
                #         maxmaxToes.append(maximummaxToe)
                    
                #         Subject.append(subName)
                #         Config.append(ConfigTmp)
                #         Side.append('Right')
                #         Movement.append(moveTmp)
                #     except:
                #         print(fName, landings[i])
            
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
                
                plt.figure
                plt.plot(intp_strides(LForce,LHS,LTO,LGS))
                plt.ylabel('Insole Force [N]')
                plt.close()
                        
            
                for i in LGS:
                    # try:
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
                    # except:
                    #     print(fName, landings[i])
            
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
            
        
