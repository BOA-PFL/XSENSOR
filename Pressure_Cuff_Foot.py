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
    avgDorsal: np.array
    avgPlantar: np.array
    config: str
    subject: str
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1 = sns.heatmap(self.avgDorsal, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(self.avgDorsal) * 2)
        ax1.set(xticklabels=[])
        ax1.set_title('Dorsal Pressure') 
        ax2 = sns.heatmap(self.avgPlantar, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(self.avgPlantar) * 2)
        ax2.set(xticklabels=[])
        ax2.set_title('Plantar Pressure') 
        return fig  


def createAvgMat(inputName):
    """ 
    Reads in file, creates average matrix data to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    """
    dat = pd.read_csv(fPath+inputName, sep=',', skiprows = 1, header = 'infer')
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1].split(sep=".")[0]
    sensel = dat.iloc[:,17:197]
    plantarSensel = dat.iloc[:,214:425]
    
    headers = plantarSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    con_press = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        con_press[store_r[ii],store_c[ii]] = np.mean(plantarSensel.iloc[:,ii])
        
   
    avgDorsalMat = np.array(np.mean(sensel, axis = 0)).reshape((18,10))
    avgPlantarMat = np.array(np.flip(con_press))
    
    result = avgData(avgDorsalMat, avgPlantarMat, config, subj)
    
    return(result)


meanDorsalPressure = []
maxDorsalPressure = [] 
sdDorsalPressure = []
totalDorsalPressure = []
config = []
subject = []

for entry in entries:
    
    tmpAvgMat = createAvgMat(entry)
    tmpAvgMat.plotAvgPressure()
    
    config.append(tmpAvgMat.config)
    subject.append(tmpAvgMat.subject)
    meanDorsalPressure.append(np.mean(tmpAvgMat.avgDorsal))
    maxDorsalPressure.append(np.max(tmpAvgMat.avgDorsal))
    sdDorsalPressure.append(np.std(tmpAvgMat.avgDorsal))
    totalDorsalPressure.append(np.sum(tmpAvgMat.avgDorsal))
    

outcomes = pd.DataFrame({'Subject':list(subject),'Config':list(config), 'MeanPressure':list(meanDorsalPressure), 
                         'MaxPressure':list(maxDorsalPressure),'SDPressure':list(sdDorsalPressure), 'TotalPressure':list(totalDorsalPressure)})
         
  
outfileName = fPath + 'CompiledResults.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
        
        outcomes.to_csv(outfileName, mode='a', header=True, index = False)
    
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
    
    
    
               