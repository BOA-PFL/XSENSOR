# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:55:43 2025

@author: Eric.Honert

The purpose of this code is to deliniate trials and name them based on the 
TrialNotes tab in the Qual excel document
"""


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\2025_AlpineRace_Mech\\XSENSOR\\Raw\\'
fSave = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\2025_AlpineRace_Mech\\XSENSOR\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]
cropped_entries = [fName for fName in os.listdir(fSave) if fName.endswith(fileExt)]

# Define Functions
def delimitTrial(inputDF,forceL,forceR,xaxislim):
    """
    Function to crop the data

    Parameters
    ----------
    inputDF : dataframe
        Original dataframe
    force : dataframe
        The force of the dataframe

    Returns
    -------
    outputDat: dataframe
        Dataframe that has been cropped based on selection

    """
    print('Select 2 points: the start and end of the trial')
    fig, ax = plt.subplots()
    ax.plot(forceL, label = 'Left Force')
    ax.plot(forceR, label = 'Right Force')
    plt.xlim(xaxislim)
    fig.legend()
    pts = np.asarray(plt.ginput(2, timeout=100))
    plt.close()
    outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
    outputDat = outputDat.reset_index(drop = True)
    return(outputDat)



for entry in entries:
    subject = entry.split('_')[0]
    config = entry.split('_')[1]
    noTrials = int(entry.split('_')[2][1])-int(entry.split('_')[2][0])+1
    firstTrial = int(entry.split('_')[2][0])
    
    # Don't redo cropping
    cropped_count = 0
    for sub_entry in cropped_entries:
        if subject+'_'+config in sub_entry:
            cropped_count = cropped_count + 1

    if cropped_count == 0:
        print(entry)
        dat = pd.read_csv(fPath+entry, sep=',', skiprows = 1, header = 'infer')
        
        for jj in range(noTrials):
            # Segment the data
            if len(dat.iloc[1,:]) > 210:
                datcrop = delimitTrial(dat,dat.iloc[:,25],dat.iloc[:,226],[0,len(dat.iloc[:,15])])
            else:
                datcrop = delimitTrial(dat,dat.iloc[:,25],np.zeros(len(dat.iloc[:,15])),[0,len(dat.iloc[:,15])])
            
            datcrop.to_csv(fSave+subject+'_'+config+'_'+str(firstTrial+jj)+'.csv', index = False)
            
            
            
            
         
            
