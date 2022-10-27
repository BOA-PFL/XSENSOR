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


# Define the path
fPath = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\PressureData\\Raw\\'
fSave = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\PressureData\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

sub = 'S05'
cond = ['lace','pfs']


dat = pd.read_csv(fPath+entries[3], sep=',', skiprows = 1, header = 'infer')

# dat1 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[0,round(0.25*len(dat.iloc[:,15]))])
# dat2 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.25*len(dat.iloc[:,15])),round(0.5*len(dat.iloc[:,15]))])
# dat3 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.5*len(dat.iloc[:,15])),round(0.75*len(dat.iloc[:,15]))])
# dat4 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.75*len(dat.iloc[:,15])),len(dat.iloc[:,15])])

dat1 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[0,round(0.3*len(dat.iloc[:,15]))])
dat2 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.25*len(dat.iloc[:,15])),round(0.6*len(dat.iloc[:,15]))])
dat3 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.4*len(dat.iloc[:,15])),round(0.8*len(dat.iloc[:,15]))])
dat4 = delimitTrial(dat,dat.iloc[:,15],dat.iloc[:,27],[round(0.75*len(dat.iloc[:,15])),len(dat.iloc[:,15])])


dat1.to_csv(fSave+sub+'_'+cond[0]+'_1.csv', index = False)
dat2.to_csv(fSave+sub+'_'+cond[1]+'_1.csv', index = False)
dat3.to_csv(fSave+sub+'_'+cond[1]+'_2.csv', index = False)
dat4.to_csv(fSave+sub+'_'+cond[0]+'_2.csv', index = False)            
