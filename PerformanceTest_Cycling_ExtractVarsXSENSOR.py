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
import pandas as pd
import scipy.signal as sig
import seaborn as sns 



fwdLook = 30
fThresh = 40
freq = 75 # sampling frequency (intending to change to 150 after this study)
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

# Read in files
# only read .asc files for this work

fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Cycling Performance Tests\\CyclingHL_May2022\Xsensor\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

HeelConArea_Sprint = []
HeelConArea_Steady = []
HeelTotConArea = []
Upstroke =[]

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



for fName in entries:
        try:
           # fName = entries[] #Load one file at a time
            dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            
            
            dat.columns = ['Frame', 'Date',	'Time',	'Units', 'Threshold', 'SensorLF','Insole Side',
                           	'RowsLF',	'ColumnsLF', 'AverageP_LF',	'MinP_LF',	'PeakP_LF', 'ContactArea_LF', 'TotalArea_LF', 'ContactPct_LF', 'EstLoadLF',	'StdDevLF',	
                           'SensorRF','Incole SIde', 'RowsRF', 'ColumnsRF', 'AverageP_RF', 'MinP_RF',	'PeakP_RF', 'ContactArea_RF', 'TotalArea_RF',	'ContactPct_RF', 'EstLoadRF', 'StdDevRF',	 
                           
                           'L_Heel','Insole Side','Rows','Columns', 'L_Heel_Average',	'L_Heel_MIN','L_Heel_MAX', 'L_Heel_ContactArea',
                           'L_Heel_TotalArea', 'L_Heel_Contact',	'L_Heel_EstLoad',	'L_Heel_StdDev',	
                           
                           'R_Heel',	'Insole Side','Rows','Columns','R_Heel_Average',	'R_Heel_MIN',	'R_Heel_MAX',	'R_Heel_ContactArea',
                           'R_Heel_TotalArea', 'R_Heel_Contact',	'R_Heel_EstLoad', 'R_Heel_StdDev',	 
                           
                           'L_MidFoot','Insole Side','Rows','Columns',	'L_MidFoot_Average', 'L_MidFoot_MIN', 'L_MidFoot_MAX',	'L_MidFoot_ContactArea',	
                           'L_MidFoot_TotalArea', 'L_MidFoot_Contact',	'L_MidFoot_EstLoad', 'L_MidFoot_StdDev', 
                                                
                           'R_Midfoot','Insole Side','Rows','Columns',	'R_Midfoot_Average',	'R_Midfoot_MIN',	'R_Midfoot_MAX', 'R_Midfoot_ContactArea', 	
                           'R_Midfoot_TotalArea',	'R_Midfoot_Contact',	'R_Midfoot_EstLoad', 'R_Midfoot_StdDev',	
                           
                           'L_Metatarsal','Insole Side','Rows','Columns',	'L_Metatarsal_Average',	'L_Metatarsal_MIN',	'L_Metatarsal_MAX', 	
                           'L_Metatarsal_ContactArea',	'L_Metatarsal_TotalArea', 'L_Metatarsal_Contact',	
                           'L_Metatarsal_EstLoad',	'L_Metatarsal_StdDev',	
                           
                           'R_Metatarsal','Insole Side','Rows','Columns',	'R_Metatarsal_Average',	'R_Metatarsal_MIN',	'R_Metatarsal_MAX','R_Metatarsal_ContactArea',	
                           'R_Metatarsal_TotalArea',	'R_Metatarsal_Contact',	'R_Metatarsal_EstLoad',	'R_Metatarsal_StdDev',	
                           
                           'L_Toe',	'Insole Side','Rows','Columns','L_Toe_Average', 'L_Toe_MIN', 'L_Toe_MAX', 'L_Toe_ContactArea', 'L_Toe_TotalArea',	
                           'L_Toe_L_Toe_Contact',	'L_Toe_EstLoad', 'L_Toe_StdDev',

                           'R_Toe','Insole Side','Rows','Columns', 'R_Toe_Average', 'R_Toe_MIN',	'R_Toe_MAX',	'R_Toe_Contact Area','R_Toe_TotalArea',	
                           'R_Toe_Contact',	'R_Toe_EstLoad', 'R_Toe_StdDev',
                           
                            ]
            # Est. Load (lbf) conversion to N = 1lbf * 4.44822N 
             
            RForce = np.array(dat.EstLoadRF)
            
            plt.figure()
            plt.plot(RForce) 
    
            print('click the start of as many steady state periods are recorded in the file. Press enter when done')
            steadyStart = plt.ginput(-1)
            steadyStart = steadyStart[0]
            steadyStart= round(steadyStart[0])
            
            print('click the start of as many sprints are recorded in the file. Press enter when done')
            sprintStart = plt.ginput(-1) 
            sprintStart = sprintStart[0]
            sprintStart = round(sprintStart[0])
            plt.close()
            
            
         
            
            steadyLandings = findLandings(RForce[steadyStart:steadyStart+freq*30], fThresh)
            steadyTakeoffs = findTakeoffs(RForce[steadyStart:steadyStart+freq*30], fThresh)
            sprintLandings = findLandings(RForce[sprintStart:sprintStart+freq*10], fThresh)
            sprintTakeoffs = findTakeoffs(RForce[sprintStart:sprintStart+freq*10], fThresh)
            
            steadyTakeoffs = trimTakeoffs(steadyLandings, steadyTakeoffs)
            steadyLandings = trimLandings(steadyLandings, steadyTakeoffs)
            
            sprintTakeoffs = trimTakeoffs(sprintLandings, sprintTakeoffs)
            sprintLandings = trimLandings(sprintLandings, sprintTakeoffs)
            
            #for i in range(len(steadyLandings)-1):
            for i, steadyLand in enumerate(steadyLandings):    
                #i = 0
                tmpForce = RForce[steadyStart+steadyLand : steadyStart+steadyTakeoffs[i]]
                tmpPk = max(tmpForce)
                timePk = list(tmpForce).index(tmpPk) #indx of max force applied during that pedal stroke
                try:
                    steadyOverallHeelSTDV.append(np.std(dat.R_Heel_EstLoad[steadyStart+steadyLand:steadyStart+steadyLandings[i+1]]))
                    steadyOverallPeak.append(np.nanmax(dat.PeakP_RF[steadyStart+steadyLand:steadyStart+steadyLandings[i+1]]))
                    
                    HeelConArea_Steady.append(np.mean(dat.R_Heel_ContactArea[steadyTakeoffs[i] : steadyLandings[i+1]]))
                    steadyInitialSTDV.append( dat.StdDevRF[steadyStart+steadyLand+1])# / RForce[steadyStart+steadyLandings[i]+1] )
                    steadyInitialPkP.append( dat.PeakP_RF[steadyStart+steadyLand + 1])
                    steadyPeakSTDV.append( dat.StdDevRF[steadyStart+steadyLand + timePk])# / RForce[steadyStart+steadyLandings[i]+timePk] )
                    steadyPeakPkP.append( dat.PeakP_RF[steadyStart+steadyLand + timePk])
                    steadyEndSTDV.append( dat.StdDevRF[steadyStart+steadyLand-1])# / RForce[steadyTakeoffs[i]-1]  )
                    steadyEndPkP.append( dat.PeakP_RF[steadyStart+steadyLand -1])
                                
                    steadySub.append( fName.split('_')[0] )
                    steadyConfig.append( fName.split('_')[1])
                    steadyTrial.append( fName.split('_')[2])
                except:
                    print("reached end of landings")
                
            for i, sprintLand in enumerate(sprintLandings):
                
                #i = 0
                tmpForce = RForce[sprintStart+sprintLand : sprintStart+sprintTakeoffs[i]]
                tmpPk = max(tmpForce)
                timePk = list(tmpForce).index(tmpPk) #indx of max force applied during that pedal stroke
                try:
                    sprintOverallHeelSTDV.append(np.std(dat.R_Heel_EstLoad[sprintStart+sprintLand:sprintStart+sprintLandings[i+1]]))
                    sprintOverallPeak.append(np.nanmax(dat.PeakP_RF[sprintStart+sprintLand:sprintStart+sprintLandings[i+1]]))
                                    
               #Heel Contact 
               # 'R_Heel_TotalArea', 'R_Heel_Contact'
                    HeelConArea_Sprint.append(np.mean(dat.R_Heel_ContactArea[sprintTakeoffs[i] : sprintLandings[i+1]]))  
                    sprintSub.append( fName.split('_')[0] )
                    sprintConfig.append( fName.split('_')[1])
                    sprintTrial.append( fName.split('_')[2])
                    sprintInitialSTDV.append( dat.StdDevRF[sprintStart+sprintLand+1])# / RForce[steadyStart+steadyLandings[i]+1] )
                    sprintInitialPkP.append( dat.PeakP_RF[sprintStart+sprintLand + 1])
                    sprintPeakSTDV.append( dat.StdDevRF[sprintStart+sprintLand + timePk])# / RForce[steadyStart+steadyLandings[i]+timePk] )
                    sprintPeakPkP.append( dat.PeakP_RF[sprintStart+sprintLand + timePk])
                    sprintEndSTDV.append( dat.StdDevRF[sprintStart+sprintLand-1])# / RForce[steadyTakeoffs[i]-1]  )
                    sprintEndPkP.append( dat.PeakP_RF[sprintStart+sprintLand -1])
                except:
                    print("reached end of sprint landings")
                
            
                
        except:
            print(fName) 
            
             
            
            
        
steadyOutcomes = pd.DataFrame({ 'Subject':list(steadySub),'Config':list(steadyConfig), 'Trial': list(steadyTrial),
                   'InitialSTDV': list(steadyInitialSTDV), 'InitialPeak':list(steadyInitialPkP), 'peakSTDV': list(steadyPeakSTDV),'peakPk':list(steadyPeakPkP), 'endSTDV': list(steadyEndSTDV), 'endPk': list(steadyEndPkP),
                   'overallHeelVar':list(steadyOverallHeelSTDV), 'overallPeakP':list(steadyOverallPeak), 'Steady_HeelContactArea' : list(HeelConArea_Steady)})  

steadyFileName = fPath + 'SteadyPressureData_PFHL_2.csv'
steadyOutcomes.to_csv(steadyFileName, header = True)

sprintOutcomes = pd.DataFrame({ 'Subject':list(sprintSub),'Config':list(sprintConfig), 'Trial': list(sprintTrial),
                   'InitialSTDV': list(sprintInitialSTDV), 'InitialPeak':list(sprintInitialPkP), 'peakSTDV': list(sprintPeakSTDV),'peakPk':list(sprintPeakPkP), 'endSTDV': list(sprintEndSTDV), 'endPk': list(sprintEndPkP),
                   'overallHeelVar':list(sprintOverallHeelSTDV), 'overallPeakP':list(sprintOverallPeak), 'Sprint_HeelContactArea' : list(HeelConArea_Sprint),})  
 
sprintFileName = fPath + 'SprintPressureData_PFHL_1.csv'
sprintOutcomes.to_csv(sprintFileName, header = True)
