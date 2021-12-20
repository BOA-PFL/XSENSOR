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

fwdLook = 30
fThresh = 10

# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force, fThreshold):
    ric = []
    for step in range(len(force)-1):
        if force[step] == 0 and force[step + 1] >= fThreshold:
            ric.append(step)
    return ric

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force, fThreshold):
    rto = []
    for step in range(len(force)-1):
        if force[step] >= fThreshold and force[step + 1] == 0:
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
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\Cycling2021\\DH_PressureTest_Sept2021\\Novel\\'
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\Cycling2021\\EH_CyclingPilot_2021\\Pressures\\'
fileExt = r".mva"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]
sub = []
config = []
condition = []
trial = []
initialPct = []
peakPct = []
endPct = []

for file in entries:
        try:
            fName = file #Load one file at a time
            dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            
            
            dat.columns = ['Frame',	'Time',	'Units', 'Threshold', 'Sensor',	'Rows',	'Columns',	
                           'Average Pressure',	'Minimum Pressure',	'Peak Pressure', 'Contact Area (cm²)',	
                           'Total Area (cm²)',	'Contact %', 'Est. Load (lbf)',	'Std Dev.',	'Sensor', '	Rows',	
                           'Columns', 'Average Pressure', 'Minimum Pressure',	'Peak Pressure', 'Contact Area (cm²)',	
                           'Total Area (cm²)',	'Contact %',	'Est. Load (lbf)',	'Std Dev.',	 
                           
                           'L_Heel', 'L_Heel_Average',	'L_Heel_MIN','L_Heel_MAX', 'L_Heel_ContactArea',
                           'L_Heel_TotalArea', 'L_Heel_Contact',	'L_Heel_EstLoad',	'L_Heel_StdDev',	
                           
                           'R_Heel',	'R_Heel_Average',	'R_Heel_MIN',	'R_Heel_MAX',	'R_Heel_ContactArea',
                           'R_Heel_TotalArea', 'R_Heel_Contact',	'R_Heel_EstLoad', 'R_Heel_StdDev',	 
                           
                           'L_MidFoot',	'L_MidFoot_Average', 'L_MidFoot_MIN', 'L_MidFoot_MAX',	'L_MidFoot_ContactArea',	
                           'L_MidFoot_TotalArea', 'L_MidFoot_Contact',	'L_MidFoot_EstLoad', 'L_MidFoot_StdDev', 
                                                
                           'R_Midfoot',	'R_Midfoot_Average',	'R_Midfoot_MIN',	'R_Midfoot_MAX', 'R_Midfoot_ContactArea', 	
                           'R_Midfoot_TotalArea',	'R_Midfoot_Contact',	'R_Midfoot_EstLoad', 'R_Midfoot_StdDev',	
                           
                           'L_Metatarsal',	'L_Metatarsal_Average',	'L_Metatarsal_MIN',	'L_Metatarsal_MAX', 	
                           'L_Metatarsal_ContactArea',	'L_Metatarsal_TotalArea', 'L_Metatarsal_Contact',	
                           'L_Metatarsal_EstLoad',	'L_Metatarsal_StdDev',	
                           
                           'R_Metatarsal',	'R_Metatarsal_Average',	'R_Metatarsal_MIN',	'R_Metatarsal_MAX',	'R_Metatarsal_ContactArea',	
                           'R_Metatarsal_TotalArea',	'R_Metatarsal_Contact',	'R_Metatarsal_EstLoad',	'R_Metatarsal_StdDev',	
                           
                           'L_Toe',	'L_Toe_Average', 'L_Toe_MIN', 'L_Toe_MAX', 'L_Toe_ContactArea',	'L_Toe_TotalArea',	
                           'L_Toe_L_Toe_Contact',	'L_Toe_EstLoad', 'L_Toe_StdDev',

                           'R_Toe', 'R_Toe_Average', 'R_Toe_MIN',	'R_Toe_MAX',	'R_Toe_Contact Area',	'R_Toe_TotalArea',	
                           'R_Toe_Contact',	'R_Toe_EstLoad', 'R_Toe_StdDev',
                           
                            ]
            
            forceCol = dat.RForce
            newForce = trimForce(forceCol, fThresh)
            
            landings = findLandings(newForce, fThresh)
            takeoffs = findTakeoffs(newForce, fThresh)
            
            trimmedTakeoffs = trimTakeoffs(landings, takeoffs)
            trimmedLandings = trimLandings(landings, trimmedTakeoffs)
            
            for countVar, landing in enumerate(trimmedLandings):
                
                tmpForce = dat.RForce[landing : landing + fwdLook]
                tmpPk = max(tmpForce)
                timePk = list(tmpForce).index(tmpPk) #indx of max force applied during that pedal stroke
                
                initialPct.append( dat.RPctMean[landing+1] / dat.RForce[landing+1] )
                peakPct.append( dat.RPctMean[landing + timePk] / dat.RForce[landing+timePk] )
                try:
                    endPct.append( dat.RPctMean[trimmedTakeoffs[countVar]-1] /dat.RForce[trimmedTakeoffs[countVar]-1]  )
                except:
                    endPct.append(0)
            
                sub.append( fName.split('_')[0] )
                config.append( fName.split('_')[1].split('.')[0] )
                condition.append( fName.split('_')[2] )
                trial.append( fName.split('_')[3].split('.')[0] )
                
        except:
            print(file)
        
        
outcomes = pd.DataFrame({ 'Subject':list(sub),'config':list(config), 'condition': list(condition), 'trial': list(trial),
                   'initialPct': list(initialPct), 'peakPct': list(peakPct),'endPct': list(endPct) })

#outcomes = pd.DataFrame({ 'Subject':list(sub),'config':list(config),
#                'initialPct': list(initialPct), 'peakPct': list(peakPct),'endPct': list(endPct) })
        
        
outcomes.to_csv('C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\Cycling2021\\EH_CyclingPilot_2021\\mvaResults.csv')
