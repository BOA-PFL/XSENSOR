# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:35:44 2022

@author: Dan.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox

save_on= 0

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\SkiValidation_Dec2022\\PrelimData\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

@dataclass
class avgData:
    avgBySensel: np.array
    config: str
    subject: str
    # below is a method of the dataclass
    def plotAvg(self):
        fig, ax1 = plt.subplots(1,1)
        ax1 = sns.heatmap(self.avgBySensel, cmap="mako", vmin = 0, vmax = np.max(self.avgBySensel) * 2)
        ax1.set(xticklabels=[])
        ax1.set_title(self.config) 
        return fig    

def createAvgMat(inputName):
    """ 
    Reads in file, creates average matrix data to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    """
    dat = pd.read_csv(fPath+inputName, sep=',', skiprows = 1, header = 'infer')
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1].split(sep=".")[0]
    sensel = dat.iloc[:,17:]
   
    avgMat = np.array(np.mean(sensel, axis = 0)).reshape((18,10))
    result = avgData(avgMat, config, subj)
    
    return(result)


meanPressure = []
maxPressure = [] 
sdPressure = []
totalPressure = []
config = []
subject = []

for entry in entries:
    
    tmpAvgMat = createAvgMat(entry)
    config.append(tmpAvgMat.config)
    subject.append(tmpAvgMat.subject)
    meanPressure.append(np.mean(tmpAvgMat.avgBySensel))
    maxPressure.append(np.max(tmpAvgMat.avgBySensel))
    sdPressure.append(np.std(tmpAvgMat.avgBySensel))
    totalPressure.append(np.sum(tmpAvgMat.avgBySensel))
    

outcomes = pd.DataFrame({'Subject':list(subject),'Config':list(config), 'MeanPressure':list(meanPressure), 
                         'MaxPressure':list(maxPressure),'SDPressure':list(sdPressure), 'TotalPressure':list(totalPressure)})
         
  
outfileName = fPath + 'CompiledResults.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
        
        outcomes.to_csv(outfileName, mode='a', header=True, index = False)
    
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
    
    
    
               