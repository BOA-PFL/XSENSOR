# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Bethany
"""

"""
This code directly pulls in two dorsal pads used to measure shin/thigh and calf pressure 
and indirectly pulls in dorsal and plantar pressure from XSENSORfuntions. 

Both files get analyzed and combined in the export


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox
import scipy.signal as sig
from XSENSORFunctions import (readXSENSORFile, delimitTrial, createTSmat, zeroInsoleForce, findGaitEvents,
                              loadProcessingDecisions, saveProcessingDecision)




save_on = 1
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about trials with no recorded decision (the normal setting)
# 2 = re-review EVERY trial, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied.
data_check = 1

# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'ShinCalf_v2'


# Read in files
# only read .asc files for this work

fPath = 'C:\\Users\\bethany.kilpatrick\\BOA Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\2025_Mechanistic_KneeSleeveStability_Baurfeind\\Xsensor\\FullTrials\\'
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
entries_FrontBack = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FrontBack' in fName)]
entries_FootDorsum = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FootDorsum' in fName)  ]


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
class FB_avgData:
    
    avgshinMat: np.array 

    
    avgcalfMat: np.array   

    
    config: str
    subject: str
    
   
    dat_FB: pd.DataFrame
   
    
def createAvg_FB_Mat(inputName,FilePath):
    """ 
    Reads in file, creates average matrix data, in the shape of the pressure sensor(s), to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    
    inputName: string
        filename of data static trial you are processing.  
        
     FilePath : str
         file path string
    """
   
   
   

    subj = inputName.split(sep="_")[0]  
    config = inputName.split(sep="_")[1]
    

    dat_FB = pd.read_csv(os.path.join(FilePath, inputName), sep=',', header=1, low_memory=False)
    
   # HX210.10.18.04-L S0003 == the Thigh Low Max dorsal pad 
   # HX210.10.18.04-L S0002 == Calf High Max dorsal pad

    if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0003":  # check to see if Shin/Thigh first
        shinSensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
        calfSensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1']
    
    # if "Sensor" in dat_FB.columns:
    if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0002":  # check to see if Low Max pad was used - Calf first
        calfSensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
        shinSensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1'] 
      
    if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0004":  # check to see if Calf first
        calfSensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
        shinSensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1']
    
    #Shin
    avgshinMat = np.array(np.mean(shinSensel, axis = 0)).reshape((18,10))
   
    avgshinMat = np.flip(avgshinMat, axis = 0) 
    
    avgshinMat[avgshinMat <.1] = 0   
    
    #Calf 
    avgcalfMat = np.array(np.mean(calfSensel, axis = 0)).reshape((18,10))
   
    avgcalfMat = np.flip(avgcalfMat , axis = 0) 
    
    avgcalfMat [avgcalfMat  <.1] = 0   
    
  
    
    
    result = FB_avgData(avgshinMat, avgcalfMat,config, subj, dat_FB)
    
    return(result)  


def plotAvgStaticFDPressure(plantarMat, inputFD, FilePath):
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
    avgDorsal = np.mean(inputFD.dorsalMat, axis = 0)



    fig, ((ax1, ax2)) = plt.subplots(1,2)
    ax1 = sns.heatmap(avgDorsal, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(avgDorsal) * 2)
    ax1.set(xticklabels=[])
    ax1.set_title('Dorsal Pressure') 
    ax2 = sns.heatmap(avgPlantar, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(avgPlantar) * 2)
    ax2.set(xticklabels=[])
    ax2.set_title('Plantar Pressure')  
 
    
    plt.suptitle(inputFD.config)
    plt.tight_layout() 
    
    saveFolder= FilePath + '2DPlots'
    
    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)
        
    plt.savefig(saveFolder + '/' + inputFD.subject + inputFD.config + '.png')
    return fig

def plotAvgStaticFBPressure( inputFB, FilePath):
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
    

    avgShin = inputFB.avgshinMat
    avgCalf = inputFB.avgcalfMat


    fig, ((ax1, ax2)) = plt.subplots(1,2)

    ax1 = sns.heatmap(avgShin, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(avgShin) * 2)
    ax1.set(xticklabels=[])
    ax1.set_title('Shin Pressure')  
    ax2 = sns.heatmap(avgCalf, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(avgCalf) * 2)
    ax2.set(xticklabels=[])
    ax2.set_title('Calf Pressure')  
    
    plt.suptitle(inputFB.config)
    plt.tight_layout() 
    
    saveFolder= FilePath + '2DPlots'
    
    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)
        
    plt.savefig(saveFolder + '/' + inputFB.subject + inputFB.config + '.png')
    return fig



config = []
subject = []
Movement = []
badFileList = []

plantarContact = []
plantarPeakPressure = []
plantarAvgPressure = []
plantarSDPressure = []
plantarTotalPressure = []

heelArea = [] 





if entries_FootDorsum == [0] or not entries_FootDorsum:
    parentList = entries
else:
    parentList = entries_FootDorsum




fd_outcomes = []
 

# check if front back has more data
if len(entries_FrontBack) > len(entries_FootDorsum):    
    for entry in entries_FrontBack: 
        subject = entry.split(sep="_")[0] 
        config = entry.split(sep="_")[1]
        trial = entry.split(sep="_")[3].split(sep='.')[0]   
        counter = 0
        for ii in range(len(entries_FootDorsum)):
            sub2 = entries_FootDorsum[ii].split(sep="_")[0]
            config2 = entries_FootDorsum[ii].split(sep="_")[1]
            trial2 = entries_FootDorsum[ii].split(sep="_")[3].split(sep='.')[0] 
            if subject == sub2 and config == config2 and trial == trial2: 
                counter = 1 
        if counter == 0: 
            parentList.append(entry) 
            
        

# Each trial contributes one row, gathered here
allOutcomes = []

for entry in parentList:
    print(entry )  
    # A trial rejected in a previous run is skipped. The file is still there
    # to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(entry, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + entry)
        continue
    
    
    # entry = parentList[12] 
    
    if "FootDorsum" in entry:  
        fd_found = 1

        subject = entry.split(sep="_")[0] 
        config = entry.split(sep="_")[1]
        trial = entry.split(sep="_")[3].split(sep='.')[0]  
        tmpDat = readXSENSORFile(entry,fPath) 
        plantDorDat = createTSmat(entry,fPath,tmpDat)  
        
        
        fb_found = 0 
            
        # For loop to find corresponding file in other list
        for ii in range(len(entries_FrontBack)):
            sub2 = entries_FrontBack[ii].split(sep="_")[0]
            config2 = entries_FrontBack[ii].split(sep="_")[1]
            trial2 = entries_FrontBack[ii].split(sep="_")[3].split(sep='.')[0]
            

            if (subject == sub2) and (config == config2) and (trial == trial2): 
                print(entries_FrontBack[ii]) 
                print(' ')
                fb_found = 1
                frontBackDat = createAvg_FB_Mat(entries_FrontBack[ii],fPath)  
                
    else:
        # Everything that is not a FootDorsum entry is read as a FrontBack
        # file. The FrontBack test that used to sit above this did the same
        # thing, but called the reader with the file path missing and left
        # the subject, config and trial as the previous trial's.
        subject = entry.split(sep="_")[0] 
        config = entry.split(sep="_")[1]
        trial = entry.split(sep="_")[3].split(sep='.')[0]  
        
        fb_found = 1
        fd_found = 0
        frontBackDat = createAvg_FB_Mat(entry, fPath)

        
                
   
        
    answer = True
    # Show whatever this trial actually has, then ask once. fd_found and
    # fb_found say which of the two sensor sets this entry ended up with - a
    # FootDorsum trial does not always have a matching FrontBack file, and
    # plotting one that is not there shows the previous trial's data.
    if priorDecision == 'good' and data_check == 1:
        print('Previously accepted, not asking again: ' + entry)
    elif data_check >= 1:
        if fd_found == 1:
            if len(plantDorDat.LplantarMat) != 0:
                plotAvgStaticFDPressure(plantDorDat.LplantarMat, plantDorDat, fPath)

            if len(plantDorDat.RplantarMat) != 0:
                plotAvgStaticFDPressure(plantDorDat.RplantarMat, plantDorDat, fPath)

        if fb_found == 1:
            plotAvgStaticFBPressure(frontBackDat, fPath)

        answer = messagebox.askyesno("Question","Is data clean?") # If entire rows of sensels are blank, its not clean!
        plt.close('all')

        if answer == False:
            print('Adding file to bad file list')
        else:
            print('Estimating point estimates')
    
    # Record the decision straight away, inside the loop, so a run that
    # fails later keeps every decision made so far
    if priorDecision == '' or data_check == 2:
        if answer == False:
            saveProcessingDecision(fPath, scriptName, entry, 'bad',
                                   Subject=subject, Config=config,
                                   reason='rejected at the data check')
        else:
            saveProcessingDecision(fPath, scriptName, entry, 'good',
                                   Subject=subject, Config=config)

    # A rejected trial is skipped, not just noted. Until now the answer
    # only printed - the metrics below ran either way, so rejecting a
    # trial still put it in the results file.
    if answer == False:
        continue





    if fd_found == 1:
        avgDorsal = np.mean(plantDorDat.dorsalMat, axis = 0)
        avgffDorsal = np.mean(plantDorDat.dorsalForefoot, axis = 0) 
        avgmfDorsal = np.mean(plantDorDat.dorsalMidfoot, axis = 0) 
        avginstepDorsal = np.mean(plantDorDat.dorsalInstep, axis = 0) 
        
        avgPlantar = np.mean(plantDorDat.LplantarMat, axis = 0) 
        avgToe = np.mean(plantDorDat.LplantarToe, axis = 0) 
        avgFF = np.mean(plantDorDat.LplantarForefoot, axis = 0)  
        avgMF = np.mean(plantDorDat.LplantarMidfoot, axis = 0)  
        avgHeel = np.mean(plantDorDat.LplantarHeel, axis = 0)  
        
       
        meanDorsalPressure = float(np.mean(plantDorDat.dorsalMidfoot) *6.895)
        maxDorsalPressure = float(np.max(avgDorsal)*6.895)
        sdDorsalPressure = float(np.std(avgDorsal)*6.895)
        covDorsalPressure = float(np.std(plantDorDat.dorsalMat)/np.mean(avgDorsal))
        totalDorsalPressure = float(np.sum(avgDorsal)*6.895)
        dorsalContact = float(np.count_nonzero(avgDorsal)/plantDorDat.dorsalSensNo*100)
        
         
        ffDorsalContact = float(np.count_nonzero(avgffDorsal)/plantDorDat.dorsalForefootSensNo*100) 
        ffDorsalPressure = float(np.mean(avgffDorsal)*6.895)
        ffDorsalMaxPressure = float(np.max(avgffDorsal)*6.895)
        
        
        mfDorsalContact = float(np.count_nonzero(avgmfDorsal)/plantDorDat.dorsalMidfootSensNo*100)
        mfDorsalPressure = float(np.mean(avgmfDorsal)*6.895)
        mfDorsalMaxPressure = float(np.max(avgmfDorsal)*6.895)
      
        
        instepDorsalContact = float(np.count_nonzero(avginstepDorsal)/plantDorDat.dorsalInstepSensNo*100)
        instepDorsalPressure = float(np.mean(avginstepDorsal)*6.895)
        instepDorsalMaxPressure = float(np.mean(avginstepDorsal)*6.895)

        
        plantarContact = float(np.count_nonzero(avgPlantar)/plantDorDat.LplantarSensNo*100)
        plantarPeakPressure = float(np.max(avgPlantar)*6.895)
        plantarAvgPressure = float(np.mean(avgPlantar)*6.895)
        plantarSDPressure = float(np.std(avgPlantar)*6.895)
        plantarTotalPressure = float(np.sum(avgPlantar)*6.895)
        
        
        toeContact = float(np.count_nonzero(avgToe)/plantDorDat.LplantarToeSensNo*100)
        toePressure = float(np.mean(avgToe)*6.895) 
        
        
        ffContact = float(np.count_nonzero(avgFF)/plantDorDat.LplantarForefootSensNo*100)
        ffPressure = float(np.mean(avgFF)*6.895)
        
        
        mfContact = float(np.count_nonzero(avgMF)/plantDorDat.LplantarMidfootSensNo*100)
        mfPressure = float(np.mean(avgMF)*6.895)
        
        heelContact = float(np.count_nonzero(avgHeel)/plantDorDat.LplantarHeelSensNo*100)
        heelPressure = float(np.mean(avgHeel)*6.895)
        
        
        
        fd_outcomes = pd.DataFrame([[dorsalContact, meanDorsalPressure, maxDorsalPressure,sdDorsalPressure,covDorsalPressure,totalDorsalPressure,
                       ffDorsalContact, ffDorsalPressure, ffDorsalMaxPressure, mfDorsalContact, mfDorsalPressure,mfDorsalMaxPressure,
                       instepDorsalContact, instepDorsalPressure,instepDorsalMaxPressure,plantarContact, plantarAvgPressure, plantarPeakPressure,
                       plantarSDPressure, plantarTotalPressure,toeContact, toePressure, ffContact, ffPressure, mfContact, mfPressure, heelContact, heelPressure]], 
                        
                                   columns=[
                                'dorsalContact', 'meanDorsalPressure','maxDorsalPressure','sdDorsalPressure','covDorsalPressure','totalDorsalPressure',
                             

                                'ffDorsalContact', 'ffDorsalPressure', 'ffDorsalMaxPressure', 'mfDorsalContact', 'mfDorsalPressure', 
                                'mfDorsalMaxPressure', 'instepDorsalContact', 'instepDorsalPressure','instepDorsalMaxPressure',
                             
                                'plantarContact', 'meanPlantarPressure', 'maxPlantarPressure', 'sdPlantarPressure', 'totalPlantarPressure',
                             
                                'toeContact', 'toePressure', 'ffContact', 'ffPressure',
                                'mfContact', 'mfPressure', 'heelContact', 'heelPressure'])
        

        
    else: 
        fd_outcomes =   pd.DataFrame(
                        
                                   columns=[
                                'dorsalContact', 'meanDorsalPressure','maxDorsalPressure','sdDorsalPressure','covDorsalPressure','totalDorsalPressure',
                             

                                'ffDorsalContact', 'ffDorsalPressure', 'ffDorsalMaxPressure', 'mfDorsalContact', 'mfDorsalPressure', 
                                'mfDorsalMaxPressure', 'instepDorsalContact', 'instepDorsalPressure','instepDorsalMaxPressure',
                             
                                'plantarContact', 'meanPlantarPressure', 'maxPlantarPressure', 'sdPlantarPressure', 'totalPlantarPressure',
                             
                                'toeContact', 'toePressure', 'ffContact', 'ffPressure',
                                'mfContact', 'mfPressure', 'heelContact', 'heelPressure'])

        
    if fb_found == 1: 
        
       
        meanCalf = float(np.mean(frontBackDat.avgcalfMat)*6.895)
        pkCalf = float(np.max(frontBackDat.avgcalfMat) *6.895)
        varCalf = float(np.std(frontBackDat.avgcalfMat)  / np.mean(frontBackDat.avgcalfMat) *6.895)
 
      
        meanShin = float(np.mean(frontBackDat.avgshinMat)*6.895)
        pkShin = float(np.max(frontBackDat.avgshinMat) *6.895) 
        varShin = float(np.std(frontBackDat.avgshinMat)  / np.mean(frontBackDat.avgshinMat) *6.895) 
        
        fb_outcomes =  pd.DataFrame([[subject, config, trial, meanCalf, pkCalf, varCalf, meanShin, pkShin, varShin]], 
                                    columns=['Subject','Config', 'Order', 'avgCalf', 'pkCalf', 'varCalf', 'avgShin','pkShin','varShin'])

    else: 
        fb_outcomes = pd.DataFrame(
                                    columns=['Subject','Config', 'Order', 'avgCalf', 'pkCalf', 'varCalf','avgShin','pkShin','varShin'])

   
   
    

    outcomes = pd.concat([ fb_outcomes,fd_outcomes], axis = 1)   

    
    

                       
          
    # One row per trial, written out after the loop
    allOutcomes.append(outcomes)

# Collected across the loop and written once. Writing from inside the
# loop meant appending, and appending duplicates every row whenever a
# study is re-processed after more data has been collected.
outcomes = pd.concat(allOutcomes, ignore_index = True)

outfileName = fPath + '0_CompiledResults_Static.csv'
if save_on == 1 and data_check < 2:
    outcomes.to_csv(outfileName, header=True, index = False)

            
    
             
    
