# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Kate.Harrison
Last updated: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tkinter import messagebox
from XSENSORFunctions import readXSENSORFile, createTSmat

save_on = 0
data_check = 1

# Read in files
# only read .csv files for this work
fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/Cycling Performance Tests/2025_Performance_CyclingLacevBOA_Specialized/Xsensor/Static/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]


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

###############################################################################
# Function List   
def plotAvgStaticPressure(plantarMat, inputDC, FilePath, StaticPostion):
    """
    Plot the average static plantar and dorsal pressure
    Function dependencies: need to use createTSmat for the appropriate array
    shape for the plantar and dorsal pressure

    Parameters
    ----------
    plantarMat : numpy array
        DESCRIPTION.
    inputDC : dataclass
        Created from the function "createTSmat"
    FilePath : str
        file path string

    Returns
    -------
    fig : matplotlib figure
        Figure showing average dorsal and plantar pressure during the static trial

    """
    
    avgPlantar = np.mean(plantarMat, axis = 0)
    avgDorsal = np.mean(inputDC.dorsalMat, axis = 0)
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1 = sns.heatmap(avgDorsal, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(avgDorsal) * 2)
    ax1.set(xticklabels=[])
    ax1.set_title('Dorsal Pressure') 
    ax2 = sns.heatmap(avgPlantar, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(avgPlantar) * 2)
    ax2.set(xticklabels=[])
    ax2.set_title('Plantar Pressure') 
    plt.suptitle(inputDC.config)
    plt.tight_layout() 
    
    saveFolder= FilePath + '2DPlots'
    
    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)
        
    plt.savefig(saveFolder + '/' + inputDC.subject + inputDC.config + StaticPostion + '.png')
    return fig
###############################################################################
    

meanDorsalPressure = []
maxDorsalPressure = [] 
sdDorsalPressure = []
covDorsalPressure = []
totalDorsalPressure = []
config = []
subject = []
Movement = []

plantarContact = []
plantarPeakPressure = []
plantarAvgPressure = []
plantarSDPressure = []
plantarTotalPressure = []
plantarCOVPressure = []

heelArea = [] 

dorsalContact = []


ffDorsalContact = []
ffDorsalPressure = []
ffDorsalMaxPressure = []
mfDorsalContact = []
mfDorsalPressure = []
mfDorsalMaxPressure = []
instepDorsalContact = []
instepDorsalPressure = []
instepDorsalMaxPressure = []

toeContact = []
toePressure = []
ffContact = []
ffPressure = []
mfContact = []
mfPressure = []
heelContact = []
heelPressure = []

for entry in entries:
    print(entry)
    # Deliniate the static type
    if 'tanding' in entry:
        tmpMove = 'Standing'
    elif 'tand' in entry:
        tmpMove = 'Standing'
    elif 'itting' in entry: 
        tmpMove = 'Sitting'
    elif 'it' in entry: 
        tmpMove = 'Sitting'
    else:
        tmpMove = 'Sitting'
        
    tmpDat = readXSENSORFile(entry,fPath)
    tmpDat = createTSmat(entry, fPath, tmpDat)
    
    answer = True
    if data_check == 1:
        if len(tmpDat.LplantarMat) != 0:
            plotAvgStaticPressure(tmpDat.LplantarMat,tmpDat, fPath, tmpMove)
        if len(tmpDat.RplantarMat) != 0:
            plotAvgStaticPressure(tmpDat.RplantarMat,tmpDat, fPath, tmpMove)

        answer = messagebox.askyesno("Question","Is data clean?") # If entire rows of sensels are blank, its not clean!
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        #badFileList.append(fName)
    
    if answer == True:
        plt.close('all')
        print('Estimating point estimates')
        
        Movement.append(tmpMove)
        config.append(tmpDat.config)
        subject.append(tmpDat.subject)
        
        # Create averages:
        avgDorsal = np.mean(tmpDat.dorsalMat,axis = 0)
        avgDorsalff = np.mean(tmpDat.dorsalForefoot,axis = 0)
        avgDorsalmf = np.mean(tmpDat.dorsalMidfoot,axis = 0)
        avgDorsalin = np.mean(tmpDat.dorsalInstep,axis = 0)
        
        meanDorsalPressure.append(np.mean(avgDorsal)*6.895)
        maxDorsalPressure.append(np.max(avgDorsal)*6.895)
        sdDorsalPressure.append(np.std(avgDorsal)*6.895)
        covDorsalPressure.append(np.std(avgDorsal)/np.mean(avgDorsal))
        totalDorsalPressure.append(np.sum(avgDorsal)*6.895)
        dorsalContact.append(np.count_nonzero(avgDorsal)/tmpDat.dorsalSensNo*100)
        
        ffDorsalContact.append(np.count_nonzero(avgDorsalff)/tmpDat.dorsalForefootSensNo*100)
        ffDorsalPressure.append(np.mean(avgDorsalff)*6.895)
        ffDorsalMaxPressure.append(np.max(avgDorsalff)*6.895)
        mfDorsalContact.append(np.count_nonzero(avgDorsalmf)/tmpDat.dorsalMidfootSensNo*100)
        mfDorsalPressure.append(np.mean(avgDorsalmf)*6.895)
        mfDorsalMaxPressure.append(np.max(avgDorsalmf)*6.895)
        instepDorsalContact.append(np.count_nonzero(avgDorsalin)/tmpDat.dorsalInstepSensNo*100)
        instepDorsalPressure.append(np.mean(avgDorsalin)*6.895)
        instepDorsalMaxPressure.append(np.mean(avgDorsalin)*6.895)
        
        # Default to using the right side
        if len(tmpDat.RplantarMat) != 0:
            avgPlantar = np.mean(tmpDat.RplantarMat,axis = 0)
            avgPlantartoe = np.mean(tmpDat.RplantarToe,axis = 0)
            avgPlantarff = np.mean(tmpDat.RplantarForefoot,axis = 0)
            avgPlantarmf = np.mean(tmpDat.RplantarMidfoot,axis = 0)
            avgPlantarheel = np.mean(tmpDat.RplantarHeel,axis = 0)
            plantarSensNo = tmpDat.RplantarSensNo
            plantarToeSensNo = tmpDat.RplantarToeSensNo
            plantarForefootSensNo = tmpDat.RplantarForefootSensNo
            plantarMidfootSensNo = tmpDat.RplantarMidfootSensNo
            plantarHeelSensNo = tmpDat.RplantarHeelSensNo
        else:
            avgPlantar = np.mean(tmpDat.LplantarMat,axis = 0)
            avgPlantartoe = np.mean(tmpDat.LplantarToe,axis = 0)
            avgPlantarff = np.mean(tmpDat.LplantarForefoot,axis = 0)
            avgPlantarmf = np.mean(tmpDat.LplantarMidfoot,axis = 0)
            avgPlantarheel = np.mean(tmpDat.LplantarHeel,axis = 0)
            plantarSensNo = tmpDat.LplantarSensNo
            plantarToeSensNo = tmpDat.LplantarToeSensNo
            plantarForefootSensNo = tmpDat.LplantarForefootSensNo
            plantarMidfootSensNo = tmpDat.LplantarMidfootSensNo
            plantarHeelSensNo = tmpDat.LplantarHeelSensNo
        
        plantarContact.append(np.count_nonzero(avgPlantar)/plantarSensNo*100)
        plantarPeakPressure.append(np.max(avgPlantar)*6.895)
        plantarAvgPressure.append(np.mean(avgPlantar)*6.895)
        plantarSDPressure.append(np.std(avgPlantar)*6.895)
        plantarCOVPressure.append(np.std(avgPlantar)/np.mean(avgPlantar))
        plantarTotalPressure.append(np.sum(avgPlantar)*6.895)
        
        toeContact.append(np.count_nonzero(avgPlantartoe)/plantarToeSensNo*100)
        toePressure.append(np.mean(avgPlantartoe)*6.895)
        ffContact.append(np.count_nonzero(avgPlantarff)/plantarForefootSensNo*100)
        ffPressure.append(np.mean(avgPlantarff)*6.895)
        mfContact.append(np.count_nonzero(avgPlantarmf)/plantarMidfootSensNo*100)
        mfPressure.append(np.mean(avgPlantarmf)*6.895)
        heelContact.append(np.count_nonzero(avgPlantarheel)/plantarHeelSensNo*100)
        heelPressure.append(np.mean(avgPlantarheel)*6.895)




outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(Movement), 'Config':list(config),
                          'dorsalContact': list(dorsalContact), 'meanDorsalPressure':list(meanDorsalPressure), 'maxDorsalPressure':list(maxDorsalPressure),
                          'sdDorsalPressure':list(sdDorsalPressure),'covDorsalPressure':list(covDorsalPressure), 'totalDorsalPressure':list(totalDorsalPressure),
                          'ffDorsalContact':list(ffDorsalContact), 'ffDorsalPressure':list(ffDorsalPressure), 'ffDorsalMaxPressure':list(ffDorsalMaxPressure), 
                          'mfDorsalContact':list(mfDorsalContact), 'mfDorsalPressure':list(mfDorsalPressure), 'mfDorsalMaxPressure':list(mfDorsalMaxPressure),
                          'instepDorsalContact':list(instepDorsalContact), 'instepDorsalPressure':list(instepDorsalPressure),'instepDorsalMaxPressure':list(instepDorsalMaxPressure),
                          'plantarContact':list(plantarContact), 'meanPlantarPressure':list(plantarAvgPressure), 'maxPlantarPressure':list(plantarPeakPressure),
                          'sdPlantarPressure':list(plantarSDPressure), 'covPlantarPressure':list(plantarCOVPressure) , 'totalPlantarPressure':list(plantarTotalPressure),
                          'toeContact':list(toeContact), 'toePressure':list(toePressure), 'ffContact':list(ffContact), 'ffPressure':list(ffPressure),
                          'mfContact':list(mfContact), 'mfPressure':list(mfPressure), 'heelContact':list(heelContact), 'heelPressure':list(heelPressure)
                          })
                         
      
outfileName = fPath + '0_CompiledResults_Static.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
        outcomes.to_csv(outfileName, header=True, index = False)
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
    
    
               
