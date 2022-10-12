# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:45:48 2022

@author: Bethany.Kilpatrick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

fwdLook = 30
fThresh = 50
freq = 75 # sampling frequency (intending to change to 150 after this study) 

hitThresh = 800


# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold

def trimForce(inputDFCol, threshForce):
    forceTot = inputDFCol
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

 
 
    
def findHit(force, hitThresh):
    """
    This function finds the landings from force plate data
    it uses a heuristic to determine landings from when the smoothed force is
    0 and then breaches a threshold
    
    Parameters
    ----------
    force : Pandas Series
        Vertical force from force plate.
    fThresh: integer
        Value force has to be greater than to count as a takeoff/landing

    Returns
    -------
    lic : list
        Indices of landings.

    """
    lic = [] 
    
    for swing in range(len(force)-1):
        if len(lic) == 0: 
            
            if force[swing] >1000 and force[swing + 1] >= hitThresh and force [swing] > 200:
                lic.append(swing)
    
        else:
        
            if force[swing] >1000 and force[swing + 1] >= hitThresh and swing > lic[-1] + 200 and force [swing] > 200:
                lic.append(swing)
    return lic
 

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\bethany.kilpatrick\\Boa Technology Inc\\PFL - General\\Testing Segments\\Baseball\\Xsensor\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

# Initialize Variables
sdHeel = []
meanToes = []
maxmeanHeel = []
maxmaxHeel = []
maxmeanToes = []
maxmaxMet = []
maxmaxMid = []
maxmaxToes = []
cvHeel = []
heelArea = []
meanTotalP = []
Subject = []
Config = [] 
movements = []
Side = [] 

hitMax =[]

for fName in entries:
        try:
            fName = entries[0] #Load one file at a time
            dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            subName = fName.split(sep = "_")[0]
            ConfigTmp = fName.split(sep="_")[3]
            moveTmp = fName.split(sep = "_")[2]
#            dat = dat.drop(columns = ['Insole Side', 'Insole Side.1', 'Insole Side.2' , 'Rows.2', 'Columns.2' , 'Insole Side.3' , 'Rows.3', 'Columns.3', 'Insole Side.4' , 'Rows.4', 'Columns.4', 'Insole Side.5' , 'Rows.5', 'Columns.5', 'Insole Side.6' , 'Rows.6', 'Columns.6', 'Insole Side.7' , 'Rows.7', 'Columns.7', 'Insole Side.8' , 'Rows.8', 'Columns.8', 'Insole Side.9' , 'Rows.9', 'Columns.9'])
            # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
            if (moveTmp == 'Bat') or (moveTmp == 'bat') or (moveTmp == 'batting'):
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
                
                hit = findHit(RForce, hitThresh) 
                
                

                            

                for i in range(len(hit)):
                    try:
                        # Note: converting psi to kpa with 1psi=6.89476kpa
                        #i = 0
                        # peakFidx = np.argmax(dat.EstLoadRF)
                        # heelAreaLate = np.mean(dat.R_Heel_ContactArea[hit [i]:hit[i]])
                        meanHeel = np.mean(dat.R_Heel_Average[hit:hit +100])*6.89476 
                        # meanMidfoot = np.mean(dat.R_Midfoot_Average[landings[i]:takeoffs[i]])*6.89476    
                        # meanForefoot = np.mean(dat.R_Metatarsal_Average[landings[i]:takeoffs[i]])*6.89476 
                        # meanToe = np.mean(dat.R_Toe_Average[landings[i]:takeoffs[i]])*6.89476   
                        
                        # stdevHeel = np.std(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                        # maximummeanHeel = np.max(dat.R_Heel_Average[landings[i]:takeoffs[i]])*6.89476
                        # maximummaxHeel = np.max(dat.R_Heel_MAX[landings[i]:takeoffs[i]])*6.89476
                        # maximummaxMet = np.max(dat.R_Metatarsal_MAX[landings[i]:takeoffs[i]])*6.89476
                        # maximummaxMid = np.max(dat.R_Midfoot_MAX[landings[i]:takeoffs[i]])*6.89476
                        # maximummeanToe = np.max(dat.R_Toe_Average[landings[i]:takeoffs[i]])*6.89476
                        # maximummaxToe = np.max(dat.R_Toe_MAX[landings[i]:takeoffs[i]])*6.89476
                        # meanFoot = (meanHeel + meanMidfoot + meanForefoot + meanToe)/4
                        
                        # meanTotalP.append(meanFoot)
                        # sdHeel.append(stdevHeel)
                        # meanToes.append(meanToe/meanFoot)
                        
                        # maxmeanHeel.append(maximummeanHeel)
                        # maxmaxHeel.append(maximummaxHeel)
                        # cvHeel.append(stdevHeel/meanFoot)
                        # heelArea.append(heelAreaLate)
                    
                        # maxmaxMid.append(maximummaxMid)
                        # maxmaxMet.append(maximummaxMet)
                        
                        # maxmeanToes.append(maximummeanToe)
                        # maxmaxToes.append(maximummaxToe)
                    
                        Subject.append(subName)
                        Config.append(ConfigTmp)
                        Side.append('Right')
                    except:
                        print(fName)
          
            else:
                    print('Only anlayzing running')
        except:
            print(fName) 
            
             
outcomes = pd.DataFrame({'Subject':list(Subject),'Config':list(Config),'Movement':list( movements),'Side':list(Side), 'meanTotalP':list(meanTotalP),
                         'sdHeel': list(sdHeel),'cvHeel':list(cvHeel), 'heelArea':list(heelArea),
                         'meanToes':list(meanToes), 
                         'maxmeanHeel':list(maxmeanHeel), 'maxmeanToes':list(maxmeanToes),
                         'maxmaxHeel':list(maxmaxHeel), 'maxmaxToes':list(maxmaxToes),
                         'maxmaxMid':list(maxmaxMid), 'maxmaxMet':list(maxmaxMet)
                         })  

outFileName = fPath + 'CompiledPressureDataHeelArea.csv'
outcomes.to_csv(outFileName, index = False)          
            
