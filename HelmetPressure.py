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

save_on = 1
debug = 1

# Read in files
# only read .csv files for this work
fPath = 'C:\\Users\\Kate.Harrison\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Helmets\\HelmetTest_Sept2024\\Xsensor\\Cropped\\'

fileExt = r".csv" # Don't call all .csv files, only the ones that were collected in lab
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

    Returns : 
    -------
    updated_sensels : numpy array
        Average pressure reformated into an array in the shape of the pressure sensor. Note: in cases where there
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
    #updated_sensels[updated_sensels==0] = np.nan
    for ii in range(len(headers)-1):
        updated_sensels[store_r[ii],store_c[ii]] = dat_sensels[ii]

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
for entry in entries[1:]:          
    # load the file
    #entry = entries[1]
    print(entry)
    
    #entry= entries[0]
    dat = pd.read_csv(fPath+entry, sep=',', header = 'infer')
    
    TMPsubj = entry.split(sep="_")[0]
    TMPconfig = entry.split(sep="_")[1]
    TMPcond = entry.split(sep="_")[3]
    TMPtrial = entry.split(sep="_")[4].split('.')[0]
    
    avg = np.mean(dat.iloc[:,18:], axis = 0)*6.895
    
    # Reset the matrices for each loop. F is for Front, B is for Back
   
    
    
        
    # Evaluate pressure with debugging plots
    if debug == 1: 

        plot_helmet_avgs(avg)
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

if save_on == 1:   
    outcomes.to_csv(fPath + '0_HelmetPressureStatic1.csv', header=True, index = False)
