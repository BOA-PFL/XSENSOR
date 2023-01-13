# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 03:30:09 2022

This code takes in a csv exported using the code PerformanceTest_Agility_Pressure. It plots individual times series data as well as group means and standard deviations..
Figures are saved as PNG. Folders for the figure files must be created and specified correctly in the figure save function before running.

@author: Kate.Harrison
"""


import tkinter as tk
from tkinter import filedialog
import pandas as pd
from matplotlib import pyplot as plt

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

dat = pd.read_csv(file_path)

# subset for a single movment

dat = dat[dat.Movement == 'Skater']

colours = ['g', 'r', 'b']

groupCTcat = []

for sub in set(dat.Subject):

    
    #sub = dat.Subject[87]
    plt.figure()
    c = 0
    subDat = dat[dat.Subject == sub].reset_index(drop = True)
    
    meanCt = subDat['ContactTime'].mean()
    
    ctCat = []
    
    for j in range(subDat.shape[0]):
        #j = 0
        if subDat.ContactTime[j] >= meanCt:
            ctCat.append('Slow')
            groupCTcat.append('Slow')
            
        else:
            ctCat.append('Fast')
            groupCTcat.append('Fast')
            
    subDat['Fast_Slow'] = ctCat
    
    for cond in set(subDat.Fast_Slow):
        
        #config = dat.Config[0]
        
        configDat = subDat[subDat.Fast_Slow == cond]
        meanDat = configDat.iloc[:, 0:101].mean(axis = 0)
        plt.plot(range(101), meanDat, color = colours[c], linewidth = 2.5, label = cond)
        
        
        for i in range(configDat.shape[0]):
            
            #i = 0 
            
            plt.plot(range(101), configDat.iloc[i, 0:101], color = colours[c], linewidth = 0.5)
        
        c = c+1
    
    plt.legend()
    plt.xlabel('% contact time')
    
    if 'Pressure' in file_path.split('/')[-1]:
        plt.ylabel('Pressure (kPa)')
    
    elif 'Contact' in file_path.split('/')[-1]:
        plt.ylabel('Contact Area (%)')
        
    plt.savefig('/'.join(file_path.split('/')[:-1]) + '/FiguresBySpeedSkater/' + file_path.split('/')[-1].split('.')[0] + '_' + sub + '.jpg.')
        
        
#### Find mean data across subjects

dat['Fast_Slow'] = groupCTcat
plt.figure('GroupData')
c = 0
for cond in set(dat.Fast_Slow):
    
    #config = dat.Config[0]
    grpConfigDat = dat[dat.Fast_Slow == cond]
    grpMeanDat = grpConfigDat.iloc[:,0:101].mean(axis = 0)
    grpSdDat = grpConfigDat.iloc[:,0:101].std()
    
    plt.plot(range(101), grpMeanDat, color = colours[c], label = cond)
    plt.fill_between(range(101), grpMeanDat + grpSdDat, grpMeanDat - grpSdDat, facecolor = colours[c], alpha = 0.3)
    
    c = c+1
    
plt.legend()
plt.xlabel('% contact time')

if 'Pressure' in file_path.split('/')[-1]:
    plt.ylabel('Pressure (kPa)')

elif 'Contact' in file_path.split('/')[-1]:
    plt.ylabel('Contact Area (%)')
plt.savefig('/'.join(file_path.split('/')[:-1]) + '/FiguresBySpeedSkater/' + file_path.split('/')[-1].split('.')[0] + '_GROUPDATA.jpg.')





    

    