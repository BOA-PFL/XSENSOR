# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:42:21 2022

Script to crop and save the trail running pressure data from one whole trial
into several different trials

@author: Eric.Honert
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path
fPath = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\EH_Trail_HeelLockTrail_Perf_May23\\Xsensor\\Raw\\'
fSave = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\EH_Trail_HeelLockTrail_Perf_May23\\Xsensor\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]
cropped_entries = [fName for fName in os.listdir(fSave) if fName.endswith(fileExt)]

# Pull in the Config Names
configs_df = pd.read_excel('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\EH_Trail_HeelLockTrail_Perf_May23\\Qual_EH_Trail_HeelLockTrail_Perf_May23.xlsx')

config_no = len(np.unique(configs_df.Config))

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
    pts = np.asarray(plt.ginput(2, timeout=40))
    plt.close()
    outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
    outputDat = outputDat.reset_index()
    return(outputDat)



for entry in entries:
    subject = entry.split('_')[0]
    # Don't redo cropping
    cropped_count = 0
    for sub_entry in cropped_entries:
        if subject in sub_entry:
            cropped_count = cropped_count + 1

    if cropped_count == 0:
        print(entry)
        dat = pd.read_csv(fPath+entry, sep=',', skiprows = 1, header = 'infer')
        subconfig = pd.DataFrame.reset_index(configs_df[configs_df.Subject == subject])
        if 'Note' in dat:
            dat = dat.drop(['Note'],axis=1)

        # Look at the number of configurations to decide how to crop the data
        if config_no == 3:
            # Segment the data:
            dat1 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[0,round(0.4*len(dat.iloc[:,15]))])
            dat2 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.25*len(dat.iloc[:,15])),round(0.75*len(dat.iloc[:,15]))])
            dat3 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.55*len(dat.iloc[:,15])),len(dat.iloc[:,15])])
            # Save the data:
            dat1.to_csv(fSave+subject+'_'+subconfig.Config[0]+'_run_1.csv', index = False)
            dat2.to_csv(fSave+subject+'_'+subconfig.Config[1]+'_run_2.csv', index = False)
            dat3.to_csv(fSave+subject+'_'+subconfig.Config[2]+'_run_3.csv', index = False)
        elif config_no == 2:
            dat1 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[0,round(0.3*len(dat.iloc[:,15]))])
            dat2 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.25*len(dat.iloc[:,15])),round(0.6*len(dat.iloc[:,15]))])
            dat3 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.4*len(dat.iloc[:,15])),round(0.8*len(dat.iloc[:,15]))])
            dat4 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.75*len(dat.iloc[:,15])),len(dat.iloc[:,15])])
            # Save the data:
            dat1.to_csv(fSave+subject+'_'+subconfig.Config[0]+'_run_1.csv', index = False)
            dat2.to_csv(fSave+subject+'_'+subconfig.Config[1]+'_run_2.csv', index = False)
            dat3.to_csv(fSave+subject+'_'+subconfig.Config[1]+'_run_3.csv', index = False)
            dat4.to_csv(fSave+subject+'_'+subconfig.Config[0]+'_run_4.csv', index = False)
            
