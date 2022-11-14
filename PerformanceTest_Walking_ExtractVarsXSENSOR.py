# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:28:44 2021
Script to process MVA files from cycling pilot test

@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import messagebox

fwdLook = 30
fThresh = 50
<<<<<<< Updated upstream
freq = 75 # sampling frequency (intending to change to 150 after this study)
=======
freq = 100 # sampling frequency

>>>>>>> Stashed changes
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

def makeVizPlot(inputForce, inputLandings, inputTakeoffs):
    plt.plot(inputForce)
    plt.vlines(x = inputLandings, ymin = 0, ymax = np.max(RForce),
           color = 'red', label = 'Landings')
    plt.vlines(x = inputTakeoffs, ymin = 0, ymax = np.max(RForce),
           color = 'green', label = 'Takeoffs')

# Read in files
# only read .asc files for this work
fPath = 'C:/Users/daniel.feeney/Boa Technology Inc/PFL Team - General/Testing Segments/Hike/ZonalFit_Midcut_Aug2022/Xsensor/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

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
ffAreaEarly = []
ffAreaIC = []
meanTotalP = []
Subject = []
Config = []
Side = []

badFileList = []

for fName in entries:
        try:
            # fName = entries[3] #Load one file at a time
            dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            subName = fName.split(sep = "_")[0]
            ConfigTmp = fName.split(sep="_")[2]
            moveTmp = fName.split(sep = "_")[1]
            
            # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
            if (moveTmp == 'run') or (moveTmp == 'Run') or (moveTmp == 'running') or (moveTmp == 'Running') or (moveTmp == 'walk') or (moveTmp == 'Walk') or (moveTmp == 'walking') or (moveTmp == 'Walking') or (moveTmp == 'Trail') or (moveTmp == 'Uphill'):
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
                RForce = RForce-np.min(RForce)
            
                # Compute landings and takeoffs
                landings = findLandings(RForce, fThresh)
                takeoffs = findTakeoffs(RForce, fThresh)

                landings[:] = [x for x in landings if x < takeoffs[-1]]
                takeoffs[:] = [x for x in takeoffs if x > landings[0]]
                
                makeVizPlot(RForce, landings, takeoffs)
                answer = messagebox.askyesno("Question","Is data clean?")
                
                if answer == False:
                    plt.close('all')
                    print('Adding file to bad file list')
                    badFileList.append(fName)
                
                if answer == True:
                    plt.close('all')
                    print('Estimating point estimates')
            
                    for i in range(len(landings)):
                        try:
                            # Note: converting psi to kpa with 1psi=6.89476kpa
                            #i = 0
                            
                            peakFidx = np.argmax(dat.EstLoadRF[landings[i]:takeoffs[i]])
                            heelAreaLate = np.mean(dat.R_Heel_ContactArea[landings[i]+peakFidx:takeoffs[i]])
                            ffAreaEarly.append(np.mean(dat.R_Metatarsal_ContactArea[landings[i]:landings[i]+peakFidx]))
                            ffAreaIC.append(dat.R_Metatarsal_ContactArea[landings[i]])
                            meanHeel = np.mean(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                            meanMidfoot = np.mean(dat.R_Midfoot_Average[landings[i]:takeoffs[i]])*6.89476    
                            meanForefoot = np.mean(dat.R_Metatarsal_Average[landings[i]:takeoffs[i]])*6.89476 
                            meanToe = np.mean(dat.R_Toe_Average[landings[i]:takeoffs[i]])*6.89476   
                            
                            stdevHeel = np.std(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                            maximummeanHeel = np.max(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                            maximummaxHeel = np.max(dat.R_Heel_MAX[landings[i]:takeoffs[i]])*6.89476
                            maximummaxMet = np.max(dat.R_Metatarsal_MAX[landings[i]:takeoffs[i]])*6.89476
                            maximummaxMid = np.max(dat.R_Midfoot_MAX[landings[i]:takeoffs[i]])*6.89476
                            maximummeanToe = np.max(dat.R_Toe_Average[landings[i]:takeoffs[i]])*6.89476
                            maximummaxToe = np.max(dat.R_Toe_MAX[landings[i]:takeoffs[i]])*6.89476
                            meanFoot = (meanHeel + meanMidfoot + meanForefoot + meanToe)/4
                            
                            meanTotalP.append(meanFoot)
                            sdHeel.append(stdevHeel)
                            meanToes.append(meanToe/meanFoot)
                            meanFf.append(meanForefoot)
                            maxmeanHeel.append(maximummeanHeel)
                            maxmaxHeel.append(maximummaxHeel)
                            cvHeel.append(stdevHeel/meanFoot)
                            heelArea.append(heelAreaLate)
                        
                            maxmaxMid.append(maximummaxMid)
                            maxmaxMet.append(maximummaxMet)
                            
                            maxmeanToes.append(maximummeanToe)
                            maxmaxToes.append(maximummaxToe)
                        
                            Subject.append(subName)
                            Config.append(ConfigTmp)
                            Side.append('Right')
                        except:
                            print(fName, landings[i])
                
                    #__________________________________________________________________
                    # Left foot:
                    # Est. Load (lbf) conversion to N = 1lbf * 4.44822N 
                    # Convert the output load to Newtons from lbf
                    LForce = np.array(dat.EstLoadLF)*4.44822
                    LForce = LForce-np.min(LForce)
                
                    # Compute landings and takeoffs
                    landings = findLandings(LForce, fThresh)
                    takeoffs = findTakeoffs(LForce, fThresh)
    
                    landings[:] = [x for x in landings if x < takeoffs[-1]]
                    takeoffs[:] = [x for x in takeoffs if x > landings[0]]
                
                    for i in range(len(landings)):
                        try:
                            # Note: converting psi to kpa with 1psi=6.89476kpa
                            peakFidx = np.argmax(dat.EstLoadLF[landings[i]:takeoffs[i]])
                            heelAreaLate = np.mean(dat.L_Heel_ContactArea[landings[i]+peakFidx:takeoffs[i]])
                            ffAreaEarly.append(np.mean(dat.L_Metatarsal_ContactArea[landings[i]:landings[i]+[peakFidx]]))
                            ffAreaIC.append(dat.L_Metatarsal_ContactArea[landings[i]])
                            meanHeel = np.mean(dat.L_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                            meanMidfoot = np.mean(dat.L_Midfoot_Average[landings[i]:takeoffs[i]])*6.89476    
                            meanForefoot = np.mean(dat.L_Metatarsal_Average[landings[i]:takeoffs[i]])*6.89476 
                            meanToe = np.mean(dat.L_Toe_Average[landings[i]:takeoffs[i]])*6.89476   
                            
                            stdevHeel = np.std(dat.L_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                            maximummeanHeel = np.max(dat.L_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                            maximummaxHeel = np.max(dat.L_Heel_MAX[landings[i]:takeoffs[i]])*6.89476
                            maximummaxMet = np.max(dat.L_Metatarsal_MAX[landings[i]:takeoffs[i]])*6.89476
                            maximummaxMid = np.max(dat.L_Midfoot_MAX[landings[i]:takeoffs[i]])*6.89476
                            maximummeanToe = np.max(dat.L_Toe_Average[landings[i]:takeoffs[i]])*6.89476
                            maximummaxToe = np.max(dat.L_Toe_MAX[landings[i]:takeoffs[i]])*6.89476
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
                            
                            maxmeanToes.append(maximummeanToe)
                            maxmaxToes.append(maximummaxToe)
                        
                            Subject.append(subName)
                            Config.append(ConfigTmp)
                            Side.append('Left')
                        except:
                            print(fName, landings[i])
                
                else:
                        print('Only anlayzing running')
        except:
            print(fName) 
            
             
outcomes = pd.DataFrame({'Subject':list(Subject),'Config':list(Config),'Side':list(Side), 'meanTotalP':list(meanTotalP),
                         'sdHeel': list(sdHeel),'cvHeel':list(cvHeel), 'heelArea':list(heelArea), 'ffAreaEarly':list(ffAreaEarly), 'ffAreaIC':list(ffAreaIC),
                         'meanToes':list(meanToes), 'meanFf':list(meanFf),
                         'maxmeanHeel':list(maxmeanHeel), 'maxmeanToes':list(maxmeanToes),
                         'maxmaxHeel':list(maxmaxHeel), 'maxmaxToes':list(maxmaxToes),
                         'maxmaxMid':list(maxmaxMid), 'maxmaxMet':list(maxmaxMet)
                         })  

outFileName = fPath + 'CompiledPressureData.csv'
outcomes.to_csv(outFileName, index = False)          
            
        
