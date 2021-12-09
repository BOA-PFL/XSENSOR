# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:46:45 2021

@author: Bethany.Kilpatrick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


fPath = 'C:/Users/bethany.kilpatrick/Boa Technology Inc/PFL - General/AgilityPerformanceData/XENSOR Test Trials/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]


Subject = []
Config = []

targetPeakFz = [] 

for fName in entries:

    try: 
        fName = entries[1]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]

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
        
                # 2. Find midpoint of COPx (anterior-posterioe) by weighting resultant forces.  
                # Est. Load (lbf) conversion to N = 1lbf * 4.44822N

        RearHeel_Force = ((dat.L_Heel_EstLoad + dat.R_Heel_EstLoad)*4.44822) 
        RearMidfoot_Force = ((dat.L_MidFoot_EstLoad + dat.R_Midfoot_EstLoad)*4.44822) 
        
        FrontMidfoot_Force = ((dat.L_Metatarsal_EstLoad + dat.R_Metatarsal_EstLoad)*4.44822) 
        FrontToe_Force = ((dat.L_Toe_EstLoad + dat.R_Toe_EstLoad)*4.44822)
        
        
        GRF_Rear = (RearHeel_Force + RearMidfoot_Force)
        GRF_Target = (FrontMidfoot_Force + FrontToe_Force)
        GRF_Tot = GRF_Rear + GRF_Target
        bc = np.argmax(GRF_Tot)

        start = bc - 5

        finish = bc
        # Find GRF peak and 2ms before and after to ID period of interest

        

        #5. Calculate variables
        Subject.append(subName)
        Config.append(ConfigTmp)
        targetPeakFz.append(max(GRF_Target[start:finish+1]))
        
    

    except:
        print(fName)  
        
        
        
outcomes = pd.DataFrame({'Subject':list(Subject),'Config':list(Config), 'targetPeakFz':list(targetPeakFz), 
                                         })

outFileName = fPath + 'CompiledGolfForceData.csv' 

outcomes.to_csv(outFileName, index = False)
