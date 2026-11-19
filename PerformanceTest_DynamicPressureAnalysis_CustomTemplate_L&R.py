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
import tkinter as tk
from tkinter import messagebox
from XSENSORFunctions import (readXSENSORFile, delimitTrial, createTSmat, zeroInsoleForce,
                              findGaitEvents, loadProcessingDecisions, saveProcessingDecision)

save_on = 1
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about trials with no recorded decision (the normal setting)
# 2 = re-review EVERY trial, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied. Nothing is saved in
#     this mode.
data_check = 1

# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'DynamicPressure_LR'

# Read in files
# only read .csv files for this work
fPath = 'Z:\\Testing Segments\\WorkWear\\2025\\2025_Performance_HighCutPFSWorkwearI_TimberlandPro\\Xsensor\\cropped\\'
fileExt = r".csv"
# Data-quality decisions recorded by previous runs of this script. Loading
# these is what stops a trial being judged twice. Rejected files are left
# where they are so they can still be opened and looked at - it is this
# record, not the folder, that decides what gets processed.
decisions = loadProcessingDecisions(fPath, scriptName)

# Results from an out-of-date copy of this script will not match anyone
# else's. Asked only when nothing has been decided here yet: a decision file
# means the question was answered on an earlier run of this dataset.
if len(decisions) == 0:
    # Give this dialog its own window and put it in front. It is the first
    # thing the script does, so unlike the data check further down there is no
    # figure on screen to pull focus, and a borrowed default root can open the
    # box behind Spyder - which just looks like the script has hung.
    dialogRoot = tk.Tk()
    dialogRoot.withdraw()
    dialogRoot.attributes('-topmost', True)
    upToDate = messagebox.askyesno("Question", "Is this script up to date with main?",
                                   parent = dialogRoot)
    dialogRoot.destroy()
    # A plain error rather than SystemExit: these scripts get run a cell at a
    # time, and SystemExit can take the console with it. This stops just as
    # firmly, prints the reason, and leaves the console alone.
    if upToDate == False:
        raise RuntimeError('Pull this script from main before processing')

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
def plotAvgGaitPressure(inputDC, FilePath, LHS, LTO, RHS, RTO):
    """
    Plot the average pressure across all detections at early, mid and late stance

    Every foot that carries data is drawn: dorsal pressure in the first column,
    then one column per plantar insole. A trial recorded with both insoles gets
    both side by side, because judging a two-insole trial on one foot alone
    hides a failure in the other - which is the thing the data check exists to
    catch.

    Function dependencies: need to use createTSmat for the appropriate array
    shape for the plantar and dorsal pressure

    Parameters
    ----------
    inputDC : dataclass
        Created from the function "createTSmat"
    FilePath : str
        file path string
    LHS : numpy array or list
        left foot contact detections. Empty when there is no left insole.
    LTO : numpy array or list
        left foot off detections
    RHS : numpy array or list
        right foot contact detections. Empty when there is no right insole.
    RTO : numpy array or list
        right foot off detections

    Returns
    -------
    fig : matplotlib figure
        Figure showing average dorsal and plantar pressure throughout stance.
        None when the trial has no plantar pressure at all.

    """
    # Work out which feet were actually recorded. Both insoles, one, or neither
    # are all possible, and the figure has to cope with each
    feetNames = []
    feetMats = []
    feetHS = []
    feetTO = []
    if len(inputDC.LplantarMat) != 0 and len(LHS) != 0:
        feetNames.append('Left Plantar')
        feetMats.append(inputDC.LplantarMat)
        feetHS.append(LHS)
        feetTO.append(LTO)
    if len(inputDC.RplantarMat) != 0 and len(RHS) != 0:
        feetNames.append('Right Plantar')
        feetMats.append(inputDC.RplantarMat)
        feetHS.append(RHS)
        feetTO.append(RTO)

    # Say so rather than leaving a blank screen in front of the data check
    # question - a silent empty plot reads as "nothing wrong"
    if len(feetNames) == 0:
        print('No plantar pressure detected for this trial, nothing plotted')
        return None

    # Average each foot at initial contact, midstance and toe off
    earlyPlantarAvg = []
    midPlantarAvg = []
    latePlantarAvg = []
    for ff in range(len(feetNames)):
        thisMat = feetMats[ff]
        thisHS = feetHS[ff]
        thisTO = feetTO[ff]
        # Preallocate
        earlyPlantar = np.zeros([len(thisHS), 31, 9])
        midPlantar = np.zeros([len(thisHS), 31, 9])
        latePlantar = np.zeros([len(thisHS), 31, 9])
        for ii in range(len(thisHS)):
            midIdx = thisHS[ii] + round((thisTO[ii]-thisHS[ii])/2)
            earlyPlantar[ii,:,:] = thisMat[thisHS[ii],:,:]
            midPlantar[ii,:,:] = thisMat[midIdx,:,:]
            latePlantar[ii,:,:] = thisMat[thisTO[ii],:,:]
        earlyPlantarAvg.append(np.mean(earlyPlantar, axis = 0))
        midPlantarAvg.append(np.mean(midPlantar, axis = 0))
        latePlantarAvg.append(np.mean(latePlantar, axis = 0))

    # Dorsal is a single sensor, so it is averaged over the first available
    # foot's stance timing. Note: with two insoles the dorsal column follows
    # that foot alone.
    refHS = feetHS[0]
    refTO = feetTO[0]
    # Preallocate
    earlyDorsal = np.zeros([len(refHS), 18, 10])
    midDorsal = np.zeros([len(refHS), 18, 10])
    lateDorsal = np.zeros([len(refHS), 18, 10])
    for ii in range(len(refHS)):
        midIdx = refHS[ii] + round((refTO[ii]-refHS[ii])/2)
        earlyDorsal[ii,:,:] = inputDC.dorsalMat[refHS[ii],:,:]
        midDorsal[ii,:,:] = inputDC.dorsalMat[midIdx,:,:]
        lateDorsal[ii,:,:] = inputDC.dorsalMat[refTO[ii],:,:]

    earlyDorsalAvg = np.mean(earlyDorsal, axis = 0)
    midDorsalAvg = np.mean(midDorsal, axis = 0)
    lateDorsalAvg = np.mean(lateDorsal, axis = 0)

    # One row per stance phase, one column for dorsal plus one per foot
    nCols = 1 + len(feetNames)
    fig, axes = plt.subplots(3, nCols, squeeze = False)
    rowNames = ['Initial Contact', 'Midstance', 'Toe off']
    dorsalAvgs = [earlyDorsalAvg, midDorsalAvg, lateDorsalAvg]
    plantarAvgs = [earlyPlantarAvg, midPlantarAvg, latePlantarAvg]

    for rr in range(3):
        thisDorsal = dorsalAvgs[rr]
        dorsalAx = axes[rr][0]
        sns.heatmap(thisDorsal, ax = dorsalAx, cmap = 'mako', vmin = 0,
                    vmax = np.max(thisDorsal))
        dorsalAx.set_ylabel(rowNames[rr])
        if rr == 0:
            dorsalAx.set_title('Dorsal Pressure')

        # Scale the feet in a row against each other so an asymmetry is visible
        # by eye. Separate scales would normalise the difference away, which is
        # exactly what the check is looking for.
        rowMax = 0
        for ff in range(len(feetNames)):
            thisMax = np.max(plantarAvgs[rr][ff])
            if thisMax > rowMax:
                rowMax = thisMax

        for ff in range(len(feetNames)):
            thisPlantar = plantarAvgs[rr][ff]
            plantarAx = axes[rr][ff+1]
            sns.heatmap(thisPlantar, ax = plantarAx, cmap = 'mako', vmin = 0,
                        vmax = rowMax)
            if rr == 0:
                plantarAx.set_title(feetNames[ff])

    fig.set_size_inches(2.5*nCols, 10)

    plt.suptitle(inputDC.subject +' '+ inputDC.movement +' '+ inputDC.config)
    plt.tight_layout()
    plt.margins(0.1)

    saveFolder= FilePath + '2DPlots'

    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)

    plt.savefig(saveFolder + '/' + inputDC.subject +' '+ inputDC.movement +' '+ inputDC.config + '.png')
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
    # Clear the previous trial's cyclic detections. They are assigned
    # inside conditionals, so a trial that skips a detection would keep
    # the last trial's events and be processed against them silently.
    LHS = []; LTO = []; LGS = []
    RHS = []; RTO = []; RGS = []

    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    # A trial rejected in a previous run is skipped. The file is still there
    # to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(fName, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + fName)
        continue
    moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
    orderTmp = fName.split(sep = "_")[3][0]
    
    # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
    # if ('skater' in moveTmp) or ('cmj' in moveTmp) or ('run' in moveTmp) or ('walk' in moveTmp): # Note: good stride exclusion might need to be updated for agility movements
    if ('dh' in moveTmp) or ('uh' in moveTmp) or ('walk' in moveTmp):    
        tmpDat = readXSENSORFile(fName,fPath)
        tmpDat = delimitTrial(tmpDat,fName,fPath)
        tmpDat = createTSmat(fName, fPath, tmpDat)
        
        # Estimate the frequency from the time stamps
        freq = 1/np.mean(np.diff(tmpDat.time[1:-1]))
        if freq < 50:
            print('Time off for: ' + fName)
            print('Previous freq est:' + str(freq))
            print('Defaulting to 50 Hz for contact event estimations')
            freq = 50
        
        # Remove strides that are below 0.3 and above 2 seconds    
        timemin = 0.3
        timemax = 2
        
        # Compute cyclic events
        if len(tmpDat.LplantarMat) != 0:
            tmpDat.LForce = zeroInsoleForce(tmpDat.LForce,freq)
            [LHS,LTO] = findGaitEvents(tmpDat.LForce,freq)
            
            LGS = [] # GS indicates good step
            for jj in range(len(LHS)-1):
                if (LHS[jj+1] - LHS[jj]) > timemin*freq and LHS[jj+1] - LHS[jj] < timemax*freq:
                    LGS.append(jj)
                    
        if len(tmpDat.RplantarMat) != 0:
            tmpDat.RForce = zeroInsoleForce(tmpDat.RForce,freq)
            [RHS,RTO] = findGaitEvents(tmpDat.RForce,freq)
            
            RGS = [] # GS indicates good step
            for jj in range(len(RHS)-1):
                if (RHS[jj+1] - RHS[jj]) > timemin*freq and RHS[jj+1] - RHS[jj] < timemax*freq:
                    RGS.append(jj)

        answer = True # if data check is off. 
        if priorDecision == 'good' and data_check == 1:
            print('Previously accepted, not asking again: ' + fName)
        elif data_check >= 1:
            # Draw every foot that was recorded, then the force traces, and only
            # then ask. The question is asked once per trial, so everything the
            # answer depends on has to be on screen before it appears.
            plotAvgGaitPressure(tmpDat, fPath, LHS, LTO, RHS, RTO)
            if len(tmpDat.LplantarMat) != 0 and len(LGS) != 0:
                plt.figure()
                plt.plot(tmpDat.LForce, label = 'Left Foot Total Force')
                for ii in range(len(LGS)):
                    plt.axvspan(LHS[LGS[ii]], LTO[LGS[ii]], color = 'lightgray', alpha = 0.5)
            if len(tmpDat.RplantarMat) != 0 and len(RGS) != 0:
                plt.figure()
                plt.plot(tmpDat.RForce, label = 'Right Foot Total Force')
                for ii in range(len(RGS)):
                    plt.axvspan(RHS[RGS[ii]], RTO[RGS[ii]], color = 'lightgray', alpha = 0.5)
            answer = messagebox.askyesno("Question","Is data clean?")

        plt.close('all')
        # Record the decision straight away, inside the loop, so a run that
        # fails later keeps every decision made so far
        if priorDecision == '' or data_check == 2:
            if answer == False:
                saveProcessingDecision(fPath, scriptName, fName, 'bad',
                                       Subject=subName, Config=ConfigTmp,
                                       reason='rejected at the data check')
            else:
                saveProcessingDecision(fPath, scriptName, fName, 'good',
                                       Subject=subName, Config=ConfigTmp)
        if answer == False:
            print('Adding file to bad file list')
            badFileList.append(fName)
    
        if answer == True:
            print('Estimating point estimates')
            if len(tmpDat.RplantarMat) != 0: 
                for ii in range(len(RGS)):
                    side.append('Right')
                    config.append(tmpDat.config)
                    subject.append(tmpDat.subject)
                    movement.append(moveTmp)
                    oOrder.append(orderTmp)
                    frames = RTO[RGS[ii]] - RHS[RGS[ii]]
                    ct.append(tmpDat.time[RTO[ii]]-tmpDat.time[RHS[ii]])
                    pct40 = RHS[RGS[ii]] + round(frames*.4)
                    pct50 = RHS[RGS[ii]] + round(frames*.5)
                    pct60 = RHS[RGS[ii]] + round(frames*.6)
                    pct90 = RHS[RGS[ii]] + round(frames*.9)
                    
                    maxmaxToes.append(np.max(tmpDat.RplantarToe[RHS[RGS[ii]]:RTO[RGS[ii]]])*6.895)
                    toePmidstance.append(np.mean(tmpDat.RplantarToe[pct40:pct60,:,:])*6.895)
                                       
                    heelAreaLate.append(np.count_nonzero(tmpDat.RplantarHeel[pct50:RTO[RGS[ii]], :, :])/(RTO[RGS[ii]] - pct50)/tmpDat.RplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                    heelPLate.append(np.mean(tmpDat.RplantarHeel[pct90:RTO[RGS[ii]], :, :])*6.895)
    
                    latPmidstance.append(np.mean(tmpDat.RplantarLateral[pct40:pct60, :, :])*6.895)
                    latAreamidstance.append(np.count_nonzero(tmpDat.RplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarLateralSensNo*100)
                    medPmidstance.append(np.mean(tmpDat.RplantarMedial[pct40:pct60, :, :])*6.895)
                    medAreamidstance.append(np.count_nonzero(tmpDat.RplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.RplantarMedialSensNo*100)
                    latPropMid.append(np.sum(tmpDat.RplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
                    medPropMid.append(np.sum(tmpDat.RplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
                    
                    if len(tmpDat.dorsalMat) != 0: 
                        dorsalVar.append(np.std(tmpDat.dorsalMat[RHS[RGS[ii]]:RTO[RGS[ii]], :, :])/np.mean(tmpDat.dorsalMat[RHS[ii]:RTO[ii], :, :]))
                        maxDorsal.append(np.max(tmpDat.dorsalMat[RHS[RGS[ii]]:RTO[RGS[ii]], :, :])*6.895)
                        
                    else:
                        dorsalVar.append('nan')
                        maxDorsal.append('nan')
                        
            if len(tmpDat.LplantarMat) != 0:
                for ii in range(len(LGS)):
                    side.append('Left')
                    config.append(tmpDat.config)
                    subject.append(tmpDat.subject)
                    movement.append(moveTmp)
                    oOrder.append(orderTmp)
                    frames = LTO[LGS[ii]] - LHS[LGS[ii]]
                    ct.append(tmpDat.time(LTO[LGS[ii]])-tmpDat.time(LHS[LGS[ii]]))
                    pct40 = LHS[ii] + round(frames*.4)
                    pct50 = LHS[ii] + round(frames*.5)
                    pct60 = LHS[ii] + round(frames*.6)
                    pct90 = LHS[ii] + round(frames*.9)
                    
                    maxmaxToes.append(np.max(tmpDat.LplantarToe[LHS[LGS[ii]]:LTO[LGS[ii]]])*6.895)
                    toePmidstance.append(np.mean(tmpDat.LplantarToe[pct40:pct60,:,:])*6.895)
                                       
                    heelAreaLate.append(np.count_nonzero(tmpDat.LplantarHeel[pct50:LTO[LGS[ii]], :, :])/(LTO[LGS[ii]] - pct50)/tmpDat.LplantarHeelSensNo*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                    heelPLate.append(np.mean(tmpDat.LplantarHeel[pct90:LTO[LGS[ii]], :, :])*6.895)
    
                    latPmidstance.append(np.mean(tmpDat.LplantarLateral[pct40:pct60, :, :])*6.895)
                    latAreamidstance.append(np.count_nonzero(tmpDat.LplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarLateralSensNo*100)
                    medPmidstance.append(np.mean(tmpDat.LplantarMedial[pct40:pct60, :, :])*6.895)
                    medAreamidstance.append(np.count_nonzero(tmpDat.LplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/tmpDat.LplantarMedialSensNo*100)
                    latPropMid.append(np.sum(tmpDat.LplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
                    medPropMid.append(np.sum(tmpDat.LplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
                    
                    if len(tmpDat.dorsalMat) != 0: 
                        dorsalVar.append(np.std(tmpDat.dorsalMat[LHS[LGS[ii]]:LTO[LGS[ii]], :, :])/np.mean(tmpDat.dorsalMat[LHS[ii]:tmpDat.LTO[ii], :, :]))
                        maxDorsal.append(np.max(tmpDat.dorsalMat[LHS[LGS[ii]]:LTO[LGS[ii]], :, :])*6.895)
                        
                    else:
                        dorsalVar.append('nan')
                        maxDorsal.append('nan')         
            
            
outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(oOrder), 'ContactTime':list(ct),
                         'toeP_mid':list(toePmidstance), 'maxmaxToes':list(maxmaxToes),
                         'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate),  
                         'latP_mid':list(latPmidstance), 'latArea_mid':list(latAreamidstance), 'latPropMid':list(latPropMid),
                         'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medPropMid':list(medPropMid),
                         'dorsalVar':list(dorsalVar), 'maxDorsalP':list(maxDorsal)
                         
                         })

outfileName = fPath + '0_CompiledResults.csv'
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates every
    # row when a study is re-processed after more data has been collected.
    outcomes.to_csv(outfileName, header=True, index = False)
            
            
        


