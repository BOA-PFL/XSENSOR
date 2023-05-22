# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 00:29:16 2023

This code analyzes data from static trials of the right plantar pressure insole. Will be updated in the future to include top of foot pressures. 

@author: Kate.Harrison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
import os
from tkinter import messagebox


fPath = 'C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/AgilityPerformanceData/2022_Tests/CPDMech_ForefootMechII_Nov2022/XSENSOR/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

debug = 1
save_on = 1

# Initialize Variables

badFileList = []

toeP = []
toeCA = []
metP = []
metCA = []
mfP = []
mfCA = []
heelP = []
heelCA = []
stdP = []

Subject = []
Config = []

for fName in entries:
    
    try:
        #fName = entries[15]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
        
    except:
        print('Not Analyzing: ', fName)
        moveTmp = 'NA'
    
    # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
    if (moveTmp == 'Static') or (moveTmp == 'static'):
        dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
        
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
        
        answer = True # Defaulting to true: In case "debug" is not used
        if debug == 1:
        
            
            
            
            
            fig, (ax, ax1) = plt.subplots(1, 2)
            ax.set_title('Pressures')
            ax.plot(dat.R_Toe_Average, color = 'b', label = 'Toes')
            ax.plot(dat.R_Metatarsal_Average, color = 'r', label = 'Metatarsals')
            ax.plot(dat.R_Midfoot_Average, color = 'g', label = 'Midfoot')
            ax.plot(dat.R_Heel_Average, color = 'k', label = 'Heel')
            ax.legend()
            ax.set_ylabel('lbf')
            ax.set_xlabel('frames')
                
            ax1.set_title('Contact Area')
            ax1.plot(dat.R_Toe_ContactArea, color = 'b')
            ax1.plot(dat.R_Metatarsal_ContactArea, color = 'r')
            ax1.plot(dat.R_Midfoot_ContactArea, color = 'g')
            ax1.plot(dat.R_Heel_ContactArea, color = 'k')
            ax1.set_ylabel('%')
            ax1.set_xlabel('frames')
                
            plt.tight_layout()
                
                
            answer = messagebox.askyesno("Question","Is data clean?")
            plt.close()                
                
        if answer == False:
            plt.close('all')
            print('Adding file to bad file list')
            badFileList.append(fName)
            
        if answer == True:
            print('Estimating point estimates')
                
            frames = dat.shape[0]
            
            for i in range(5):
                #i = 0
                t0 = round(frames - frames/5*(i+1))
                t1 = round(frames - frames/5*i)
                toeP.append(np.mean(dat.R_Toe_Average[t0:t1])*6.89476) # convert psi to kpa
                toeCA.append(np.mean(dat.R_Toe_ContactArea[t0:t1]))
                metP.append(np.mean(dat.R_Metatarsal_Average[t0:t1])*6.89476)
                metCA.append(np.mean(dat.R_Metatarsal_ContactArea[t0:t1]))
                mfP.append(np.mean(dat.R_Midfoot_Average[t0:t1])*6.89476)
                mfCA.append(np.mean(dat.R_Midfoot_ContactArea[t0:t1]))
                heelP.append(np.mean(dat.R_Heel_Average[t0:t1])*6.89476)
                heelCA.append(np.mean(dat.R_Heel_ContactArea[t0:t1]))
                stdP.append(np.mean(dat.StdDevRF[t0:t1])*6.89476)
            
                Subject.append(subName)
                Config.append(ConfigTmp)
       


outcomes = pd.DataFrame({'Subject':list(Subject),'Config':list(Config), 'ToePressure':list(toeP), 'ToeContactArea':list(toeCA),
                         'MetatarsalPressure':list(metP), 'MetatarsalContactArea':list(metCA),
                         'MidfootPressure':list(mfP), 'MidfootContactArea':list(mfCA),
                         'HeelPressure':list(heelP), 'HeelContactArea':list(heelCA),
                         'StdDevP':list(stdP)
                         })
         
  
outfileName = fPath + 'CompiledStaticData.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
        
        outcomes.to_csv(outfileName, mode='a', header=True, index = False)
    
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
 
