# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:28:44 2021
Script to process MVA files from cycling pilot test

@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import addcopyfighandler

fwdLook = 30
fThresh = 50
freq = 75 # sampling frequency (intending to change to 150 after this study)
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force, fThreshold):
    ric = []
    for step in range(len(force)-1):
        if force[step] < fThreshold and force[step + 1] >= fThreshold:
            ric.append(step)
    return ric

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force, fThreshold):
    rto = []
    for step in range(len(force)-1):
        if force[step] >= fThreshold and force[step + 1] < fThreshold:
            rto.append(step + 1)
    return rto

def trimTakeoffs(landings, takeoffs):
    if takeoffs[0] < landings[0]:
        del(takeoffs[0])
    return(takeoffs)

def trimLandings(landings, trimmedTakeoffs):
    if landings[len(landings)-1] > trimmedTakeoffs[len(trimmedTakeoffs)-1]:
        del(landings[-1])
    return(landings)

def trimForce(inputDFCol, threshForce):
    forceTot = inputDFCol
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\CuffPressure_Mar2022\\XSENSOR\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

# Initialize Variables
maxpress = []
meanpress = []
contperc = []
Sub = []
Config = []
sensel_avg = []

for fName in entries:
        # try:
            #fName = entries[0] #Load one file at a time
            dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            subName = fName.split(sep = "_")[0]
            ConfigTmp = fName.split(sep="_")[1]
            
            #__________________________________________________________________
            # First examine summary metrics
            maxpress.append(np.mean(dat.iloc[:,12]))
            meanpress.append(np.mean(dat.iloc[:,9]))
            contperc.append(np.mean(dat.iloc[:,14]))
            
            Sub.append(subName)
            Config.append(ConfigTmp)
            #__________________________________________________________________
            # Assign a dataframe with only the sensel data            
            sensel = dat.iloc[:,17:]
            headers = sensel.columns
            
            
            
            store_r = []
            store_c = []
            
            
            
            for name in headers:
                store_r.append(int(name.split(sep = "_")[1])-1)
                store_c.append(int(name.split(sep = "_")[2])-1)
            
            con_press = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
            
            for ii in range(len(headers)-1):
                con_press[store_r[ii],store_c[ii]] = np.mean(sensel.iloc[:,ii])
                
            sensel_avg.append(np.flip(con_press))
            
                
            
            
        # except:
        #     print(fName) 
            
             
# outcomes = pd.DataFrame({'Subject':list(Sub),'Config':list(Config),'MeanPress':list(meanpress),
#                           'MaxPress': list(maxpress),'ContPerc':list(contperc)
#                           }) 
# FileName = fPath + 'StaticPressureData.csv'
# outcomes.to_csv(FileName, header = True)

#______________________________________________________________________________
# Create Pretty Pictures

# ss_scale = [8,12,8,8,8]
# cc = 0

# for ii in range(0,len(entries),4):
#     fig, ( ax1, ax2, ax3 ) = plt.subplots(1,3)
#     g1 = sns.heatmap(sensel_avg[ii+3]-sensel_avg[ii+1], cmap="jet", ax = ax1, vmin = -ss_scale[cc], vmax = ss_scale[cc])
#     # plt.xlabel.set_fontsize(18)
#     # g1.set(xticklabels=['1','5','9'])
#     # g1.set(yticklabels=['1','4','7','10','13','16','19','22','25','28','31'])
#     # g1.set_title('Single Pull vs. Dual Pull')
    
#     g2 = sns.heatmap(sensel_avg[ii+3]-sensel_avg[ii+2], cmap="jet", ax = ax2, vmin = -ss_scale[cc], vmax = ss_scale[cc])
#     # g2.set(xticklabels=['1','5','9'])
#     # g2.set(yticklabels=['1','4','7','10','13','16','19','22','25','28','31'])
#     # g2.set_title('Single Pull vs. Dual Pull 2G')
    
#     g3 = sns.heatmap(sensel_avg[ii+3]-sensel_avg[ii], cmap="jet", ax = ax3, vmin = -ss_scale[cc], vmax = ss_scale[cc])
#     # g3.set(xticklabels=['1','5','9'])
#     # g3.set(yticklabels=['1','4','7','10','13','16','19','22','25','28','31'])
#     # g3.set_title('Single Pull vs. Buckle')
#     cc = cc+1

cc = 0
ss_scale = [10,18,18,15,14]
for ii in range(0,len(entries),4):
    
    fig, ( ax1, ax2, ax3, ax4 ) = plt.subplots(1,4)
    sns.set(font_scale=1.5)
    g1 = sns.heatmap(sensel_avg[ii], cmap="jet", ax = ax1, vmax = ss_scale[cc])
    g1.set(yticklabels=['1','3','5','7','9','11','13','15','17','19','21','23','25','27','29','31'])
    g1.set(xticklabels=['1','5','9'])
    ax1.set_title('Buckle')
    
    g2 = sns.heatmap(sensel_avg[ii+1], cmap="jet", ax = ax2, vmax = ss_scale[cc])
    g2.set(yticklabels=['1','3','5','7','9','11','13','15','17','19','21','23','25','27','29','31'])
    g2.set(xticklabels=['1','5','9'])
    g2.set_title('Dual Pull')
    
    g3 = sns.heatmap(sensel_avg[ii+2], cmap="jet", ax = ax3, vmax = ss_scale[cc])
    g3.set(yticklabels=['1','3','5','7','9','11','13','15','17','19','21','23','25','27','29','31'])
    g3.set(xticklabels=['1','5','9'])
    g3.set_title('Dual Pull 2G')
    
    g4 = sns.heatmap(sensel_avg[ii+3], cmap="jet", ax = ax4, vmax = ss_scale[cc])
    g4.set(yticklabels=['1','3','5','7','9','11','13','15','17','19','21','23','25','27','29','31'])
    g4.set(xticklabels=['1','5','9'])
    g4.set_title('Single Pull')
    
    
    cc = cc+1
    
    
    
# Buckles as the baseline
# cc = 0
# for ii in range(0,len(entries),4):
#     fig, ( ax1, ax2, ax3 ) = plt.subplots(1,3)
#     g1 = sns.heatmap(sensel_avg[ii]-sensel_avg[ii+1], cmap="jet", ax = ax1, vmin = -ss_scale[cc], vmax = ss_scale[cc])
#     g1.set(xticklabels=['1','5','9'])
#     g1.set(yticklabels=['1','4','7','10','13','16','19','22','25','28','31'])
#     g1.set_title('Buckle vs. Dual Pull 2G')
    
#     g2 = sns.heatmap(sensel_avg[ii]-sensel_avg[ii+2], cmap="jet", ax = ax2, vmin = -ss_scale[cc], vmax = ss_scale[cc])
#     g2.set(xticklabels=['1','5','9'])
#     g2.set(yticklabels=[])
#     g2.set_title('Buckle vs. Dual Pull')
    
#     g3 = sns.heatmap(sensel_avg[ii]-sensel_avg[ii+3], cmap="jet", ax = ax3, vmin = -ss_scale[cc], vmax = ss_scale[cc])
#     g3.set(xticklabels=['1','5','9'])
#     g3.set(yticklabels=[])
#     g3.set_title('Buckle vs. Single Pull')
#     cc = cc+1