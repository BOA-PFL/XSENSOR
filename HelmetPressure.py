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
from tkinter import messagebox

save_on = 0
debug = 1

# Read in files
# only read .csv files for this work
fPath = 'C:/Users/kate.harrison/Boa Technology Inc/PFL Team - General/Testing Segments/Helmets/HelmetPressureTest_Aug2023/Xsensor/'

fileExt = r"Static_1.csv" # Don't call all .csv files, only the ones that were collected in lab
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


def reformat_sensels(dat_sensels):
    """
    The purpose of this function is to use the headers provided for the sensels
    to transform the data into a square matrix. Note: average sensel pressures
    should be input

    Parameters
    ----------
    dat_sensels : dataframe with one average pressure entry for each sensel
        Average pressure from the selected task

    Returns : dataframe reformatted into a matrix in the shape of the pressure sensors. 
    -------
    updated_sensels : numpy array
        Average pressure reformated into an array. Note: in cases where there
        is not an entry, the sensel pressure will be zero.

    """
    headers = dat_sensels.index
    store_r = []
    store_c = []
    
    # Store the header names into row and columns
    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    # Place the data in the affiliated row/column
    updated_sensels = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
    updated_sensels[updated_sensels==0] = np.nan
    for ii in range(len(headers)-1):
        updated_sensels[store_r[ii],store_c[ii]] = dat_sensels[ii]

    return(updated_sensels)


def plot_helmet_avgs(Press1,Press2):
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
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1 = sns.heatmap(reformat_sensels(Press1), ax = ax1, cmap="mako", vmin = 0, vmax = 15)
    ax1.set(xticklabels=[])
    ax1.set_title('Front Pressure') 
    ax2 = sns.heatmap(reformat_sensels(Press2), ax = ax2, cmap="mako", vmin = 0, vmax = 15)
    ax2.set(xticklabels=[])
    ax2.set_title('Back Pressure') 
    
    outFileFolder = fPath + '/PressureMaps/'
    if os.path.isdir(outFileFolder) == False:
        os.mkdir(outFileFolder)
    
    plt.savefig(fPath + '/PressureMaps/' + TMPsubj + TMPconfig + '.png')
    
    return(fig)


config = []
subject = []
Position = []
badFileList = []
aspect = []

PeakPressure = []
AvgPressure = []
TotPressure = []
VarPressure = []

CentrePressure = []
CentrePeakPressure = []
SidePressure = []
SidePeakPressure = []


# Index through all of the selected entries
for entry in entries:          
    # load the file
    print(entry)
    
    #entry= entries[0]
    dat = pd.read_csv(fPath+entry, sep=',',skiprows = 1, header = 'infer')
    
    TMPsubj = entry.split(sep="_")[0]
    TMPconfig = entry.split(sep="_")[1]
    
    # Some files only have 1 sensor
    TMPside = entry.split(sep="_")[2]
    
    
    # Reset the matrices for each loop. F is for Front, B is for Back
    avgF_level = []; avgF_tilt = []
    avgB_level = []; avgB_tilt = []
    
    # Separate out trials if there was only one pressure sensor
    if TMPside == 'Static':
        # Create averages from each of the different tasks
        # Assume first 5 seconds in from the level static, last 5 seconds is from the head tilted back
        # If both sensors are in, the left is the front, right is the back
        avgF_level = np.mean(dat.iloc[0:501,18:238],axis=0)*6.895
        F_level = reformat_sensels(avgF_level)
        avgF_tilt = np.mean(dat.iloc[501:,18:238],axis=0)*6.895
        
        avgB_level = np.mean(dat.iloc[0:501,250:],axis=0)*6.895
        avgB_tilt = np.mean(dat.iloc[501:,250:],axis=0)*6.895
        B_level = reformat_sensels(avgB_level)
        
    elif TMPside == 'Front':
        # F is for front
        avgF_level = np.mean(dat.iloc[0:501,18:238],axis=0)*6.895
        avgF_tilt = np.mean(dat.iloc[501:,18:238],axis=0)*6.895
        F_level = reformat_sensels(avgF_level)
        
    elif TMPside == 'Back':
        # F is for front
        avgB_level = np.mean(dat.iloc[0:501,18:238],axis=0)*6.895
        avgB_tilt = np.mean(dat.iloc[501:,18:238],axis=0)*6.895
        B_level = reformat_sensels(avgB_level)
        
    # Evaluate pressure with debugging plots
    if debug == 1: 
        if TMPside == 'Static':
            plot_helmet_avgs(avgF_level,avgB_level)
            #plot_helmet_avgs(avgF_tilt,avgB_tilt)
        elif TMPside == 'Front':
            plot_helmet_avgs(avgF_level,avgF_level*0)
            #plot_helmet_avgs(avgF_tilt,avgF_tilt*0)
        elif TMPside == 'Back':
            plot_helmet_avgs(avgB_level*0,avgB_level)
            #plot_helmet_avgs(avgB_tilt*0,avgB_tilt)
        answer = messagebox.askyesno("Question","Is data clean?")
        plt.close()
    else:
        answer = True
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(entry)
    
    if answer == True:
        # plt.close('all')
        print('Estimating point estimates')
        
        # Evaluate Level Pressure
        if TMPside == 'Static':
            # Level
            aspect.append('Front')
            config.append(TMPconfig)
            subject.append(TMPsubj)
            Position.append('Level')
            PeakPressure.append(np.max(avgF_level))
            AvgPressure.append(np.mean(avgF_level))
            TotPressure.append(np.sum(avgF_level))
            VarPressure.append(np.std(avgF_level)/np.mean(avgF_level))
            CentrePressure.append(np.nanmean(F_level[10:21, :]))
            CentrePeakPressure.append(np.nanmax(F_level[10:21, :]))
            SidePressure.append(np.nanmean(F_level[:10, :]))
            SidePeakPressure.append(np.nanmax(F_level[:10, :]))
            
            aspect.append('Back')
            config.append(TMPconfig)
            subject.append(TMPsubj)
            Position.append('Level')
            PeakPressure.append(np.max(avgB_level))
            AvgPressure.append(np.mean(avgB_level))
            TotPressure.append(np.sum(avgB_level))
            VarPressure.append(np.std(avgB_level)/np.mean(avgB_level))
            CentrePressure.append(np.nanmean(B_level[10:21, :]))
            CentrePeakPressure.append(np.nanmax(B_level[10:21, :]))
            SidePressure.append(np.nanmean(B_level[:10, :]))
            SidePeakPressure.append(np.nanmax(B_level[:10, :]))
            
            # # Tilted
            # aspect.append('Front')
            # config.append(TMPconfig)
            # subject.append(TMPsubj)
            # Position.append('Tilt')
            # PeakPressure.append(np.max(avgF_tilt))
            # AvgPressure.append(np.mean(avgF_tilt))
            # TotPressure.append(np.sum(avgF_tilt))
            # VarPressure.append(np.std(avgF_tilt)/np.mean(avgF_tilt))
            
        if TMPside == 'Front':
            
            aspect.append('Front')
            config.append(TMPconfig)
            subject.append(TMPsubj)
            Position.append('Level')
            PeakPressure.append(np.max(avgF_level))
            AvgPressure.append(np.mean(avgF_level))
            TotPressure.append(np.sum(avgF_level))
            VarPressure.append(np.std(avgF_level)/np.mean(avgF_level))
            CentrePressure.append(np.nanmean(F_level[10:21, :]))
            CentrePeakPressure.append(np.nanmax(F_level[10:21, :]))
            SidePressure.append(np.nanmean(F_level[:10, :]))
            SidePeakPressure.append(np.nanmax(F_level[:10, :]))
        
        if TMPside == 'Static' or TMPside == 'Back':
            # Level
            aspect.append('Back')
            config.append(TMPconfig)
            subject.append(TMPsubj)
            Position.append('Level')
            PeakPressure.append(np.max(avgB_level))
            AvgPressure.append(np.mean(avgB_level))
            TotPressure.append(np.sum(avgB_level))
            VarPressure.append(np.std(avgB_level)/np.mean(avgB_level))
            CentrePressure.append(np.nanmean(B_level[10:21, :]))
            CentrePeakPressure.append(np.nanmax(B_level[10:21, :]))
            SidePressure.append(np.nanmean(B_level[:10, :]))
            SidePeakPressure.append(np.nanmax(B_level[:10, :]))
            
            # # Tilted
            # aspect.append('Back')
            # config.append(TMPconfig)
            # subject.append(TMPsubj)
            # Position.append('Tilt')
            # PeakPressure.append(np.max(avgB_tilt))
            # AvgPressure.append(np.mean(avgB_tilt))
            # TotPressure.append(np.sum(avgB_tilt))
            # VarPressure.append(np.std(avgB_tilt)/np.mean(avgB_tilt))


# Save outcomes in a .csv
outcomes = pd.DataFrame({'Subject':list(subject), 'Config': list(config),'Position': list(Position), 'Aspect':list(aspect),
                         'PeakPressure': list(PeakPressure), 'AvgPressure': list(AvgPressure), 'TotPressure': list(TotPressure), 'VarPressure': list(VarPressure),
                         'CentrePressure':list(CentrePressure), 'CentrePeakPressure':list(CentrePeakPressure), 'SidePressure':list(SidePressure), 'SidePeakPressure':list(SidePeakPressure)})

if save_on == 1:   
    outcomes.to_csv(fPath + '0_HelmetPressureStatic1.csv', header=True, index = False)
