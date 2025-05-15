# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Kate.Harrison
last updated: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tkinter import messagebox
from XSENSORFunctions import readXSENSORFile, delimitTrial, createTSmat, zeroInsoleForce, findGaitEvents

save_on = 0
data_check = 0

# Read in files
# only read .csv files for this work
fPath = 'Z:\\Testing Segments\\WorkWear_Performance\\2025_Performance_HighCutPFSWorkwearI_TimberlandPro\\Xsensor\\cropped\\'
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
# Function list specific for this code
def plotAvgGaitPressure(plantarMat, inputDC, FilePath, HS, TO):
    """
    Plot the average pressure across all detections at early, mid and late stance 
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
    HS : numpy array
        foot contact detections
    TO : numpy array
        foot off detections

    Returns
    -------
    fig : matplotlib figure
        Figure showing average dorsal and plantar pressure throughout stance

    """
    
    # Make able to use both left and right
    earlyPlantar = np.zeros([len(HS), 31, 9])
    midPlantar = np.zeros([len(HS), 31, 9])
    latePlantar = np.zeros([len(HS), 31, 9])
    
    earlyDorsal = np.zeros([len(HS), 18, 10])
    midDorsal = np.zeros([len(HS), 18, 10])
    lateDorsal = np.zeros([len(HS), 18, 10])
    
    for ii in range(len(HS)):
        earlyPlantar[ii,:,:] = plantarMat[HS[ii],:,:]
        midPlantar[ii,:,:] = plantarMat[HS[ii] + round((TO[ii]-HS[ii])/2),:,:]
        latePlantar[ii,:,:] = plantarMat[TO[ii],:,:]
        earlyDorsal[ii,:,:] = inputDC.dorsalMat[HS[ii],:,:]
        midDorsal[ii,:,:] = inputDC.dorsalMat[HS[ii] + round((TO[ii]-HS[ii])/2),:,:]
        lateDorsal[ii,:,:] = inputDC.dorsalMat[TO[ii],:,:]
        
    earlyPlantarAvg = np.mean(earlyPlantar, axis = 0)
    midPlantarAvg = np.mean(midPlantar, axis = 0)
    latePlantarAvg = np.mean(latePlantar, axis = 0)
    earlyDorsalAvg = np.mean(earlyDorsal, axis = 0)
    midDorsalAvg = np.mean(midDorsal, axis = 0)
    lateDorsalAvg = np.mean(lateDorsal, axis = 0)
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    ax1 = sns.heatmap(earlyDorsalAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(earlyDorsalAvg))
    ax1.set_title('Dorsal Pressure') 
    ax1.set_ylabel('Initial Contact')
        
    ax2 = sns.heatmap(earlyPlantarAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(earlyPlantarAvg))
    ax2.set_title('Plantar Pressure') 
    
    ax3 = sns.heatmap(midDorsalAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(midDorsalAvg))
    ax3.set_ylabel('Midstance')
    
    ax4 = sns.heatmap(midPlantarAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(midPlantarAvg))
    
    ax5 = sns.heatmap(lateDorsalAvg, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(latePlantarAvg))
    ax5.set_ylabel('Toe off')
    
    ax6 = sns.heatmap(latePlantarAvg, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(lateDorsalAvg))
    
    fig.set_size_inches(5, 10)
    
    plt.suptitle(inputDC.subject +' '+ inputDC. movement +' '+ inputDC.config)
    plt.tight_layout()
    plt.margins(0.1)
    
    saveFolder= FilePath + '2DPlots'
    
    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)
        
    plt.savefig(saveFolder + '/' + inputDC.subject +' '+ inputDC. movement +' '+ inputDC.config + '.png')
    return fig
###############################################################################

badFileList = []
config = []
subject = []
ct = []
movement = []
oOrder = []
side = []

toePmidstance = []
heelAreaLate = []
heelPLate = []
maxmaxToes = []

latPmidstance = []
latAreamidstance = []
medPmidstance = []
medAreamidstance = []

latPropMid = []
medPropMid = []
dorsalVar = []
maxDorsal = []

for fName in entries:   
    # try:
    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
    orderTmp = fName.split(sep = "_")[3][0]
    
    # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
    # if ('skater' in moveTmp) or ('cmj' in moveTmp) or ('run' in moveTmp) or ('walk' in moveTmp):
    if ('dh' in moveTmp) or ('uh' in moveTmp) or ('walk' in moveTmp):    
        tmpDat = readXSENSORFile(fName,fPath)
        tmpDat = delimitTrial(tmpDat,fName,fPath)
        tmpDat = createTSmat(fName, fPath, tmpDat)
        
        # Estimate the frequency from the time stamps
        freq = 1/np.mean(np.diff(tmpDat.time))
        if freq < 50:
            print('Time off for: ' + fName)
            print('Previous freq est:' + str(freq))
            print('Defaulting to 50 Hz for contact event estimations')
            freq = 50
        
        # Compute cyclic events
        if len(tmpDat.LplantarMat) != 0:
            tmpDat.LForce = zeroInsoleForce(tmpDat.LForce,freq)
            [LHS,LTO] = findGaitEvents(tmpDat.LForce,freq)
        if len(tmpDat.RplantarMat) != 0:
            tmpDat.RForce = zeroInsoleForce(tmpDat.RForce,freq)
            [RHS,RTO] = findGaitEvents(tmpDat.RForce,freq)
        
        
        answer = True # if data check is off. 
        if data_check == 1:
            if len(tmpDat.RplantarMat) != 0:
                plotAvgGaitPressure(tmpDat.RplantarMat,tmpDat,fPath,RHS,RTO)
                plt.figure()
                plt.plot(tmpDat.RForce, label = 'Right Foot Total Force')
                for ii in range(len(RHS)):
                    plt.axvspan(RHS[ii], RTO[ii], color = 'lightgray', alpha = 0.5)
            elif len(tmpDat.LplantarMat) != 0:
                plotAvgGaitPressure(tmpDat.LplantarMat,tmpDat,fPath,LHS,LTO)
                plt.figure()
                plt.plot(tmpDat.LForce, label = 'Left Foot Total Force')
                for ii in range(len(LHS)):
                    plt.axvspan(LHS[ii], LTO[ii], color = 'lightgray', alpha = 0.5)
            answer = messagebox.askyesno("Question","Is data clean?")

        plt.close('all')
        if answer == False:
            print('Adding file to bad file list')
            badFileList.append(fName)
    
        if answer == True:
            print('Estimating point estimates')
            if len(tmpDat.RplantarMat) != 0: 
                for ii in range(len(RHS)):
                    side.append('Right')
                    config.append(tmpDat.config)
                    subject.append(tmpDat.subject)
                    movement.append(moveTmp)
                    oOrder.append(orderTmp)
                    frames = RTO[ii] - RHS[ii]
                    ct.append(tmpDat.time[RTO[ii]]-tmpDat.time[RHS[ii]])
                    pct40 = RHS[ii] + round(frames*.4)
                    pct50 = RHS[ii] + round(frames*.5)
                    pct60 = RHS[ii] + round(frames*.6)
                    pct90 = RHS[ii] + round(frames*.9)
                    
                    maxmaxToes.append(np.max(tmpDat.RplantarToe[RHS[ii]:RTO[ii]])*6.895)
                    toePmidstance.append(np.mean(tmpDat.RplantarToe[pct40:pct60,:,:])*6.895)
                                       
                    heelAreaLate.append(np.count_nonzero(tmpDat.RplantarHeel[pct50:RTO[ii], :, :])/(RTO[ii] - pct50)/tmpDat.RplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                    heelPLate.append(np.mean(tmpDat.RplantarHeel[pct90:RTO[ii], :, :])*6.895)
    
                    latPmidstance.append(np.mean(tmpDat.RplantarLateral[pct40:pct60, :, :])*6.895)
                    latAreamidstance.append(np.count_nonzero(tmpDat.RplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarLateralSensNo*100)
                    medPmidstance.append(np.mean(tmpDat.RplantarMedial[pct40:pct60, :, :])*6.895)
                    medAreamidstance.append(np.count_nonzero(tmpDat.RplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarMedialSensNo*100)
                    latPropMid.append(np.sum(tmpDat.RplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
                    medPropMid.append(np.sum(tmpDat.RplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
                    
                    if len(tmpDat.dorsalMat) != 0: 
                        dorsalVar.append(np.std(tmpDat.dorsalMat[RHS[ii]:RTO[ii], :, :])/np.mean(tmpDat.dorsalMat[RHS[ii]:RTO[ii], :, :]))
                        maxDorsal.append(np.max(tmpDat.dorsalMat[RHS[ii]:RTO[ii], :, :])*6.895)
                        
                    else:
                        dorsalVar.append('nan')
                        maxDorsal.append('nan')
                        
            if len(tmpDat.LplantarMat) != 0:
                for ii in range(len(LHS)):
                    side.append('Left')
                    config.append(tmpDat.config)
                    subject.append(tmpDat.subject)
                    movement.append(moveTmp)
                    oOrder.append(orderTmp)
                    frames = LTO[ii] - LHS[ii]
                    ct.append(tmpDat.time(LTO[ii])-tmpDat.time(LHS[ii]))
                    pct40 = LHS[ii] + round(frames*.4)
                    pct50 = LHS[ii] + round(frames*.5)
                    pct60 = LHS[ii] + round(frames*.6)
                    pct90 = LHS[ii] + round(frames*.9)
                    
                    maxmaxToes.append(np.max(tmpDat.LplantarToe[LHS[ii]:LTO[ii]])*6.895)
                    toePmidstance.append(np.mean(tmpDat.LplantarToe[pct40:pct60,:,:])*6.895)
                                       
                    heelAreaLate.append(np.count_nonzero(tmpDat.LplantarHeel[pct50:LTO[ii], :, :])/(LTO[ii] - pct50)/tmpDat.LplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                    heelPLate.append(np.mean(tmpDat.LplantarHeel[pct90:LTO[ii], :, :])*6.895)
    
                    latPmidstance.append(np.mean(tmpDat.LplantarLateral[pct40:pct60, :, :])*6.895)
                    latAreamidstance.append(np.count_nonzero(tmpDat.LplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarLateralSensNo*100)
                    medPmidstance.append(np.mean(tmpDat.LplantarMedial[pct40:pct60, :, :])*6.895)
                    medAreamidstance.append(np.count_nonzero(tmpDat.LplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarMedialSensNo*100)
                    latPropMid.append(np.sum(tmpDat.LplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
                    medPropMid.append(np.sum(tmpDat.LplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
                    
                    if len(tmpDat.dorsalMat) != 0: 
                        dorsalVar.append(np.std(tmpDat.dorsalMat[LHS[ii]:LTO[ii], :, :])/np.mean(tmpDat.dorsalMat[LHS[ii]:tmpDat.LTO[ii], :, :]))
                        maxDorsal.append(np.max(tmpDat.dorsalMat[LHS[ii]:LTO[ii], :, :])*6.895)
                        
                    else:
                        dorsalVar.append('nan')
                        maxDorsal.append('nan')

    # except:
    #         print('Not usable data')
    #         badFileList.append(fName)             
            
            
outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(oOrder), 'ContactTime':list(ct),
                         'toeP_mid':list(toePmidstance), 'maxmaxToes':list(maxmaxToes),
                         'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate),  
                         'latP_mid':list(latPmidstance), 'latArea_mid':list(latAreamidstance), 'latPropMid':list(latPropMid),
                         'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medPropMid':list(medPropMid),
                         'dorsalVar':list(dorsalVar), 'maxDorsalP':list(maxDorsal)
                         
                         })

outfileName = fPath + '0_CompiledResults.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
        outcomes.to_csv(outfileName, header=True, index = False)
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
            
        


