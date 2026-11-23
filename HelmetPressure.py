# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:30:31 2023

@author: Eric.Honert

This code analyses the pressure data for STATIC collections of helmet tests. One sensor is placed around the front of the head. One sensor is placed around the back of the head. 
The first half of each trial is collected with the subject standing straight up, the second half of the trial the subject tilts their head backwards. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import addcopyfighandler
import tkinter as tk
from tkinter import messagebox
from XSENSORFunctions import loadProcessingDecisions, saveProcessingDecision

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
scriptName = 'HelmetPressure'

# Read in files
# only read .csv files for this work
fPath = 'C:\\Users\\Eric.Honert\\OneDrive - Boa Technology Inc\\PFL Team - General\\Testing Segments\\Helmets\\2024\\HelmetTest_Sept2024\\Xsensor\\Cropped\\'

fileExt = r".csv" # Don't call all .csv files, only the ones that were collected in lab
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

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and not fName.startswith('0_')]

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


def reformat_sensels(dat_sensels):
    """
    The purpose of this function is to use the headers provided for the sensels
    to transform the data into a square matrix. Note: average sensel pressures
    should be input

    Parameters
    ----------
    dat_sensels : dataframe with one average pressure entry for each sensel
        Average pressure from the selected task

    Returns : 
    -------
    updated_sensels : numpy array
        Average pressure reformated into an array in the shape of the pressure sensor. Note: in cases where there
        is not an entry, the sensel pressure will be zero.

    """
    headers = dat_sensels.index
    store_r = []
    store_c = []
    
    # Store the header names into row and columns. A column that is not a
    # sensel is named here rather than left to int() to complain, because
    # the useful part is which column it was.
    for name in headers:
        parts = name.split(sep = "_")
        if len(parts) < 3:
            raise ValueError('not a sensel column: ' + str(name)
                             + ' - expected the S_row_column naming')
        store_r.append(int(parts[1])-1)
        store_c.append(int(parts[2].split(sep=".")[0])-1)
    
    # Place the data in the affiliated row/column
    updated_sensels = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
    #updated_sensels[updated_sensels==0] = np.nan
    # range(len(headers)), not len(headers)-1: the last sensel was being
    # left at zero in every map and every average taken from one
    for ii in range(len(headers)):
        updated_sensels[store_r[ii],store_c[ii]] = dat_sensels.iloc[ii]

    return(updated_sensels)


def plot_helmet_avgs(Press):
    """
    The purpose of the function is to generate side-by-side pressure maps for
    two pressure sensors. In the helmet case: the pressure at the front & back. 
    Figures are saved into a folder inside the fPath folder called  "PressureMaps"

    Parameters
    ----------
    Press1 : dataframe with one average pressure entry for each sensel
        Average data from pressure sensor 1
    Press2 : dataframe with one average pressure entry for each sensel
        Average data from pressure sensor 2

    Returns
    -------
    fig : matplotlib figure
        Figure with a 1x2 subplot for each pressure sensor

    """
    
    fig, ax1 = plt.subplots(1,1)
    ax1 = sns.heatmap(reformat_sensels(Press), ax = ax1, cmap="mako", vmin = 0, vmax = 15)
    ax1.set(xticklabels=[])
    ax1.set_title('Helmet Pressure') 
   
    
    outFileFolder = fPath + '/PressureMaps/'
    if os.path.isdir(outFileFolder) == False:
        os.mkdir(outFileFolder)
    
    plt.savefig(fPath + '/PressureMaps/' + TMPsubj + TMPconfig + TMPcond + TMPtrial +'.png')
    
    return(fig)


config = []
subject = []
condition = [] # tight or preferred
trial = []
badFileList = []


avgP = []
pP = []
avgP_Front = []
pP_Front = []
avgP_Side = []
pP_Side = []
avgP_Back = []
pP_Back = []


# Index through all of the selected entries
# Every trial, not entries[1:]. That slice was skipping the first file, which
# used to be an output csv sitting in the listing - the 0_ filter above keeps
# those out now, so the slice would drop a real trial.
for entry in entries:
    # load the file
    print(entry)
    # A trial rejected in a previous run is skipped. The file is still there
    # to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(entry, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + entry)
        continue
    
    
    #entry= entries[0]
    dat = pd.read_csv(fPath+entry, sep=',', header = 'infer')
    
    TMPsubj = entry.split(sep="_")[0]
    TMPconfig = entry.split(sep="_")[1]
    TMPcond = entry.split(sep="_")[3]
    TMPtrial = entry.split(sep="_")[4].split('.')[0]
    
    avg = np.mean(dat.iloc[:,18:], axis = 0)*6.895
    
    # Reset the matrices for each loop. F is for Front, B is for Back
   
    
    
        
    answer = True # if data check is off
    # Evaluate pressure with the data check plots
    if priorDecision == 'good' and data_check == 1:
        print('Previously accepted, not asking again: ' + entry)
    elif data_check >= 1:
        plot_helmet_avgs(avg)
        answer = messagebox.askyesno("Question","Is data clean?")
        plt.close()
    
    # Record the decision straight away, inside the loop, so a run that
    # fails later keeps every decision made so far
    if priorDecision == '' or data_check == 2:
        if answer == False:
            saveProcessingDecision(fPath, scriptName, entry, 'bad',
                                   Subject=TMPsubj, Config=TMPconfig,
                                   reason='rejected at the data check')
        else:
            saveProcessingDecision(fPath, scriptName, entry, 'good',
                                   Subject=TMPsubj, Config=TMPconfig)
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(entry)
    
    if answer == True:
        # plt.close('all')
        print('Estimating point estimates')
        
        subject.append(TMPsubj)
        config.append(TMPconfig)
        condition.append(TMPcond)
        trial.append(TMPtrial)
        
        avg = reformat_sensels(avg)
        # Evaluate Level Pressure
        avgP.append(np.mean(avg, axis = None))
        pP.append(np.max(avg, axis = None))
        avgP_Front.append(np.mean(avg[:,26:], axis = None))
        pP_Front.append(np.max(avg[:,26:], axis = None))
        avgP_Side.append(np.mean(avg[:,13:26], axis = None))
        pP_Side.append(np.max(avg[:,13:26], axis = None))
        avgP_Back.append(np.mean(avg[:,:13], axis = None))
        pP_Back.append(np.max(avg[:,:13], axis = None))
        
  


# Save outcomes in a .csv
outcomes = pd.DataFrame({'Subject':list(subject), 'Config': list(config),'Condition': list(condition), 'Trial':list(trial),
                         'PeakPressure': list(pP), 'AvgPressure': list(avgP), 
                         'AvgPressure_Front':list(avgP_Front), 'PeakPressure_Front':list(pP_Front), 
                         'AveragePressure_Side':list(avgP_Side), 'PeakPressure_Side':list(pP_Side),
                         'AveragePressure_Back':list(avgP_Back), 'PeakPressure_Back':list(pP_Back)
                         })

outfileName = fPath + '0_HelmetPressureStatic.csv'

if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    outcomes.to_csv(outfileName, header=True, index = False)
