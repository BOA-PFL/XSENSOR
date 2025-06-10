# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:29:20 2023

Visualization and comparison figure for dorsal and plantar pressures 

@author: Dan.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\SkiValidation_Dec2022\\InLabPressure\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

fileOne = entries[9]
fileTwo = entries[11]


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


@dataclass
class avgData:
    
    """ 
    Reads in file, creates average matrix data to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    
    inputName: string
        filename of the data you are analyzing.
        
    result: avgData (see dataclass above)
    """
    
    avgDorsal: np.array
    avgPlantar: np.array
    avgDorsal2: np.array
    avgPlantar2: np.array
    config: str
    config2: str
    subject: str
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        ax1 = sns.heatmap(self.avgDorsal, ax = ax1, cmap="inferno", vmin = 0, vmax = np.max(self.avgDorsal) * 1.5)
        ax1.set(xticklabels=[])
        ax1.set_title('Dorsal Pressure') 
        ax2 = sns.heatmap(self.avgPlantar, ax = ax2, cmap="inferno", vmin = 0, vmax = np.max(self.avgPlantar) * 1.5)
        ax2.set(xticklabels=[])
        ax2.set_title('Plantar Pressure') 
        
        ax3 = sns.heatmap(self.avgDorsal2, ax = ax3, cmap="inferno", vmin = 0, vmax = np.max(self.avgDorsal) * 1.5)
        ax3.set(xticklabels=[])
        ax3.set_title('Dorsal Pressure') 
        ax4 = sns.heatmap(self.avgPlantar2, ax = ax4, cmap="inferno", vmin = 0, vmax = np.max(self.avgPlantar) * 1.5)
        ax4.set(xticklabels=[])
        ax4.set_title('Plantar Pressure') 
        
        return fig  
    
    def sortDF(self, colName):
        """ 
        Grabs each individual grouping by location of foot from regions 
        specified in XSENSOR output
        """
        subsetDat = self.fullDat.iloc[:,self.fullDat.columns.get_loc(colName):self.fullDat.columns.get_loc(colName)+12]
        return(subsetDat)

# dd = test.sortDF('Group')
# filter_col = [col for col in dat if col.startswith('Group')]

@dataclass
class footLocData:
    """
    This class sorts pressure data by region of the foot.
    """
    RLHeel: pd.DataFrame
    RMHeel: pd.DataFrame
    RMMidfoot: pd.DataFrame
    RLMidfoot: pd.DataFrame
    RMMets: pd.DataFrame
    RLMets: pd.DataFrame
    RMToes: pd.DataFrame
    RLToes: pd.DataFrame
    

def createAvgMat(inputName, inputName2):
    """ 
    Reads in file, creates average matrix data to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    """
    dat = pd.read_csv(fPath+inputName, sep=',', skiprows = 1, header = 'infer')
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1].split(sep=".")[0]
    config2 = inputName2.split(sep="_")[1].split(sep=".")[0] 
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
    avgDorsalMat = np.fliplr(avgDorsalMat)
    avgPlantarMat = np.array(np.flip(con_press))
    avgPlantarMat = np.flip(avgPlantarMat)
    avgPlantarMat = np.fliplr(avgPlantarMat)
    
    # 2nd Filee #    
    dat2 = pd.read_csv(fPath+inputName2, sep=',', skiprows = 1, header = 'infer')
    sensel2 = dat2.iloc[:,17:197]
    plantarSensel2 = dat2.iloc[:,214:425]
    
    headers2 = plantarSensel2.columns
    store_r2 = []
    store_c2 = []

    for name in headers2:
        store_r2.append(int(name.split(sep = "_")[1])-1)
        store_c2.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    con_press2 = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers2)-1):
        con_press2[store_r2[ii],store_c2[ii]] = np.mean(plantarSensel2.iloc[:,ii])
   
    avgDorsalMat2 = np.array(np.mean(sensel2, axis = 0)).reshape((18,10))
    avgDorsalMat2 = np.fliplr(avgDorsalMat2)
    avgPlantarMat2 = np.array(np.flip(con_press2))
    avgPlantarMat2 = np.flip(avgPlantarMat2)
    avgPlantarMat2 = np.fliplr(avgPlantarMat2)
    
    result = avgData(avgDorsalMat, avgPlantarMat, avgDorsalMat2, avgPlantarMat2, config, config2 ,subj, dat)
    
    return(result)

# Create plot #
tmpAvgMat = createAvgMat(fileOne, fileTwo)
tmpAvgMat.plotAvgPressure() # plot heat maps of pressure data
