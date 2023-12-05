# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:44:43 2023

@author: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox
import addcopyfighandler

save_on = 1

# Read in files
# only read .asc files for this work

fPath = 'Z:/Testing Segments/WorkWear_Performance/EH_Workwear_MidCutStabilityII_CPDMech_Sept23_AnkPress/XSENSOR/'

fileExt = r".csv"
Lat_entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and fName.count('Lateral') and fName.count('Static')]
Med_entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and fName.count('Medial') and fName.count('Static')]


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

def plotAvgPressure(Med,Lat,configuration):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1 = sns.heatmap(Med, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(Med) * 2)
    ax1.set(xticklabels=[])
    ax1.set_title('Medial Pressure') 
    ax2 = sns.heatmap(Lat, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(Lat) * 2)
    ax2.set(xticklabels=[])
    ax2.set_title('Lateral Pressure') 
    plt.suptitle(configuration)
    plt.tight_layout()
    return fig  
    
    
    
    

@dataclass
class avgData:
    avgDorsal: np.array
    avgPlantar: np.array
    config: str
    subject: str
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis. 
    
    # below is a method of the dataclass
    
    
    def sortDF(self, colName):
        """ 
        Grabs each individual grouping by location of foot from regions 
        specified in XSENSOR output
        """
        subsetDat = self.fullDat.iloc[:,self.fullDat.columns.get_loc(colName):self.fullDat.columns.get_loc(colName)+12]
        return(subsetDat)

    

def createAvgMat(inputName):
    """ 
    Reads in file, creates average matrix data, in the shape of the pressure sensor(s), to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    
    inputName: string
        filename of data static trial you are processing. 
    """
   
        
    #inputName = entries[14]
    dat = pd.read_csv(fPath+inputName, sep=',', header = 'infer')
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[2]
    
    insoleSide = dat['Insole Side'][0]
    
    if (insoleSide == 'Left'): 
        
        # Left side
        # plantarSensel = dat.iloc[:,18:238]
        dorsalSensel = dat.iloc[:,250:430]
    elif (len(dat.iloc[0,:]) > 200): 
        dorsalSensel = dat.iloc[:,18:198]
        # plantarSensel = dat.iloc[:,210:430] 
    else:
        dorsalSensel = dat.iloc[:,18:198]
        # plantarSensel = np.zeros([len(dat.iloc[:,0]),220])
   
    avgDorsalMat = np.array(np.mean(dorsalSensel, axis = 0)).reshape((18,10))
   
    avgDorsalMat = np.flip(avgDorsalMat, axis = 0)
    
    result = avgData(avgDorsalMat, config, subj, dat)
    
    return(result)


# Preallocate variables
badFileList = []

Lat_PeakPressure = []
Lat_AvgPressure = []
Lat_TotForce = []
Lat_ConArea = []
Lat_Var = []

Med_PeakPressure = []
Med_AvgPressure = []
Med_TotForce = []
Med_ConArea = []
Med_Var = []

Subject = []
Config = []
Movement = []



for ii in range(len(Lat_entries)):
    print(Lat_entries[ii])
    #entry = entries[7]
    if 'tand' in Lat_entries[ii]:
        tmpMovement ='Standing'
    elif 'itting' in Lat_entries[ii]: 
        tmpMovement ='Sitting'
    
    tmpSubject = Lat_entries[ii].split(sep="_")[0]
    tmpConfig = Lat_entries[ii].split(sep="_")[2]

    Latdat = pd.read_csv(fPath+Lat_entries[ii], sep=',',skiprows = 1, header = 'infer')
    Meddat = pd.read_csv(fPath+Med_entries[ii], sep=',',skiprows = 1, header = 'infer')
    
    # Obtain and arrange average medial and lateral pressure
    LatAvg = np.flip(np.array(np.mean(Latdat.iloc[:,18:198],axis = 0)).reshape((18,10)),axis = 0)*6.895
    MedAvg = np.flip(np.array(np.mean(Meddat.iloc[:,18:198],axis = 0)).reshape((18,10)),axis = 0)*6.895
    
    answer = True
    plotAvgPressure(MedAvg,LatAvg,tmpConfig)
    answer = messagebox.askyesno("Question","Is data clean?") # If entire rows of sensels are blank, its not clean!
    plt.close('all')
    
    if answer == False:
        print('Adding file to bad file list')
        #badFileList.append(fName)
    
    if answer == True:
        print('Estimating point estimates')
        
        Subject.append(tmpSubject)
        Config.append(tmpConfig)
        Movement.append(tmpMovement)
        
        Lat_PeakPressure.append(np.max(LatAvg))
        Lat_AvgPressure.append(np.mean(LatAvg))
        Lat_TotForce.append(np.sum(LatAvg))
        Lat_ConArea.append(np.count_nonzero(LatAvg)/180*100)
        Lat_Var.append(np.std(LatAvg)/np.mean(LatAvg))
        
        Med_PeakPressure.append(np.max(MedAvg))
        Med_AvgPressure.append(np.mean(MedAvg))
        Med_TotForce.append(np.sum(MedAvg))
        Med_ConArea.append(np.count_nonzero(MedAvg)/180*100)
        Med_Var.append(np.std(MedAvg)/np.mean(MedAvg))
               

# Combine outcomes
outcomes = pd.DataFrame({'Subject':list(Subject), 'Config': list(Config), 'Movement': list(Movement),
                         'Lat_PeakPressure': list(Lat_PeakPressure), 'Lat_AvgPressure': list(Lat_AvgPressure), 'Lat_TotForce': list(Lat_TotForce), 'Lat_ConArea': list(Lat_ConArea), 'Lat_Var': list(Lat_Var),
                         'Med_PeakPressure': list(Med_PeakPressure), 'Med_AvgPressure': list(Med_AvgPressure), 'Med_TotForce': list(Med_TotForce), 'Med_ConArea': list(Med_ConArea), 'Med_Var': list(Med_Var)})

  
if save_on == 1:
    outcomes.to_csv(fPath + 'StaticPressureOutcomes.csv',header=True)
elif save_on == 2: 
    outcomes.to_csv(fPath + 'StaticPressureOutcomes.csv',mode = 'a', header=False)  
    
    
    
               
