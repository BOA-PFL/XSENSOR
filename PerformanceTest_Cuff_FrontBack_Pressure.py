# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:35:44 2022

@author: Dan.Feeney

This code analyzes data collected from the front and back of the cuff of a boot - the shin and the calf. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox

save_on = 0

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\2022\\AlpinePressureMapping_Dec2022\\Pressure\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]


### set plot font size ###
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
    avgDorsal: np.array
    avgPlantar: np.array
    config: str
    subject: str
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis. Data is formatted in the shape of the pressure sensors for plotting and analysis by region. 
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1 = sns.heatmap(self.avgDorsal, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(self.avgDorsal) * 2)
        ax1.set(xticklabels=[])
        ax1.set_title('Shin Pressure') 
        ax2 = sns.heatmap(self.avgPlantar, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(self.avgPlantar) * 2)
        ax2.set(xticklabels=[])
        ax2.set_title('Calf Pressure') 
        plt.suptitle(self.config)
        plt.tight_layout()
        return fig  
    
    def sortDF(self, colName):
        """ 
        Grabs each individual grouping by location of foot from regions 
        specified in XSENSOR output
        """
        subsetDat = self.fullDat.iloc[:,self.fullDat.columns.get_loc(colName):self.fullDat.columns.get_loc(colName)+12]
        return(subsetDat)

    

def createAvgMat(inputName):
    """ 
    Reads in file, creates average matrix data to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    
    inputName: string
        filename of the data you are analyzing.
        
    result: avgData (see dataclass above)
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
    
    result = avgData(avgDorsalMat, avgPlantarMat, config, subj, dat)
    
    return(result)


#initialize outcome variable lists
meanShinPressure = []
maxShinPressure = [] 
sdShinPressure = []
totalShinPressure = []
config = []
subject = []

calfContact = []
calfPeakPressure = []
calfAvgPressure = []
calfSDPressure = []
calfTotalPressure = []

for entry in entries:
    
    tmpAvgMat = createAvgMat(entry)
    tmpAvgMat.plotAvgPressure()
    answer = messagebox.askyesno("Question","Is data clean?")
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        #badFileList.append(fName)
    
    if answer == True:
        plt.close('all')
        print('Estimating point estimates')

        config = (tmpAvgMat.config)
        subject = (tmpAvgMat.subject)
        meanShinPressure = float(np.mean(tmpAvgMat.avgDorsal))
        maxShinPressure = float(np.max(tmpAvgMat.avgDorsal))
        sdShinPressure = float(np.std(tmpAvgMat.avgDorsal))
        totalShinPressure = float(np.sum(tmpAvgMat.avgDorsal))
        
        calfContact = float(np.count_nonzero(tmpAvgMat.avgPlantar))
        calfPeakPressure = float(np.max(tmpAvgMat.avgPlantar))
        calfAvgPressure = float(np.mean(tmpAvgMat.avgPlantar))
        calfSDPressure = float(np.std(tmpAvgMat.avgPlantar))
        calfTotalPressure = float(np.sum(tmpAvgMat.avgPlantar))


        
        outcomes = pd.DataFrame([[subject,config,meanShinPressure,maxShinPressure,sdShinPressure,totalShinPressure,
                                  calfContact, calfPeakPressure, calfAvgPressure, calfSDPressure, calfTotalPressure]],
                                columns=['Subject','Config','meanShinPressure','maxShinPressure','sdShinPressure','totalShinPressure',
                                         'calfContact', 'calfPeakPressure', 'calfAvgPressure', 'calfSDPressure', 'calfTotalPressure'])
          
        outfileName = fPath + 'CompiledResults.csv'
        if save_on == 1:
            if os.path.exists(outfileName) == False:
                
                outcomes.to_csv(outfileName, header=True, index = False)
            
            else:
                outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
    
    
               