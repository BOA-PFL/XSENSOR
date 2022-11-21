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
import seaborn as sns
from dataclasses import dataclass


evalPlotting = 0 #if set to 1, will load test dataset to ensure reshape in correct order
save_on= 0

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\CuffPressure_Dec2022\\Prelim\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

@dataclass
class avgData:
    avgBySensel: np.array
    config: str

def createAvgMat(inputName):
    """ 
    Reads in file, creates average matrix data to be plotted 
    """
    dat = pd.read_csv(fPath+inputName, sep=',', skiprows = 1, header = 'infer')
    config = inputName.split(sep="_")[1].split(sep=".")[0]
    sensel = dat.iloc[:,17:]
   
    avgMat = np.array(np.mean(sensel, axis = 0)).reshape((18,10))
    result = avgData(avgMat, config)
    
    return(result)


meanPressure = []
maxPressure = [] 
sdPressure = []
totalPressure = []
config = []
for entry in entries:
    
    tmpAvgMat = createAvgMat(entry)
    config.append(tmpAvgMat.config)
    meanPressure.append(np.mean(tmpAvgMat.avgBySensel))
    maxPressure.append(np.max(tmpAvgMat.avgBySensel))
    sdPressure.append(np.std(tmpAvgMat.avgBySensel))
    totalPressure.append(np.sum(tmpAvgMat.avgBySensel))
    

outcomes = pd.DataFrame({'Config':list(config), 'MeanPressure':list(meanPressure), 'MaxPressure':list(maxPressure),
                         'SDPressure':list(sdPressure), 'TotalPressure':list(totalPressure)})
         
  
outfileName = fPath + 'CompiledResults.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
        
        outcomes.to_csv(outfileName, mode='a', header=True, index = False)
    
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
    
##############################################################################
## Testing section to ensure data is in correct order when using reshape ##    
if evalPlotting == 1:
    fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\CuffPressure_Dec2022\\Prelim\\'
    fileExt = r".csv"
    entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

    loc1 = createAvgMat(entries[0])
    loc2 = createAvgMat(entries[1])
    loc3 = createAvgMat(entries[2])
    loc4 = createAvgMat(entries[3])
    loc5 = createAvgMat(entries[4])
    loc6 = createAvgMat(entries[5])
    
    
    # Plots
    fig, ( (ax1, ax2, ax3), (ax4, ax5, ax6) ) = plt.subplots(2,3)
    ax1 = sns.heatmap(loc1.avgBySensel, cmap="jet", ax = ax1, vmin = 0, vmax = np.max(loc1.avgBySensel) * 2)
    ax1.set(xticklabels=[])
    ax1.set_title(loc1.config) 
    ax2 = sns.heatmap(loc2.avgBySensel, cmap="jet", ax = ax2, vmin = 0, vmax = np.max(loc1.avgBySensel) * 2)
    ax2.set(xticklabels=[])
    ax2.set_title(loc2.config) 
    ax3 = sns.heatmap(loc3.avgBySensel, cmap="jet", ax = ax3, vmin = 0, vmax = np.max(loc1.avgBySensel) * 2)
    ax3.set(xticklabels=[])
    ax3.set_title(loc3.config) 
    ax4 = sns.heatmap(loc4.avgBySensel, cmap="jet", ax = ax4, vmin = 0, vmax = np.max(loc1.avgBySensel) * 2)
    ax4.set(xticklabels=[])
    ax4.set_title(loc4.config)  
    ax5 = sns.heatmap(loc5.avgBySensel, cmap="jet", ax = ax5, vmin = 0, vmax = np.max(loc1.avgBySensel) * 2)
    ax5.set(xticklabels=[])
    ax5.set_title(loc5.config) 
    ax6 = sns.heatmap(loc6.avgBySensel, cmap="jet", ax = ax6, vmin = 0, vmax = np.max(loc1.avgBySensel) * 2)
    ax6.set(xticklabels=[])
    ax6.set_title(loc6.config) 
    plt.tight_layout()   
           
            
