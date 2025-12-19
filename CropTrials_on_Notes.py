# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:10:41 2023

@author: Eric.Honert

This code crops the XSENSOR data based on the Notes column. The data will be
cropped into smaller trials based on the notes and saved in a different folder

This code searches the existing cropped trial based on the subject name and
configuration. If the current subject/configuration exists in the "saved"
this code skips cropping the file
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/PFL Internal Studies/ML_Pressure_COMPower/Pilot/XSENSOR/raw/' #Where the uncropped data live
fSave = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/PFL Internal Studies/ML_Pressure_COMPower/Pilot/XSENSOR/' #Where to save cropped data
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]
cropped_entries = [fName for fName in os.listdir(fSave) if fName.endswith(fileExt)]

# Index through the raw uncropped data files
for entry in entries:
    subject = entry.split('_')[0]
    tmpConfig = entry.split('_')[1]
    # Don't redo cropping, check if it has already been cropped
    cropped_count = 0
    for sub_entry in cropped_entries:
        if str(subject + '_' + tmpConfig) in sub_entry:
            cropped_count = cropped_count + 1
    
    if cropped_count == 0:
        print(entry)
        tmpSesh = entry.split('_')[2][0]
        dat = pd.read_csv(fPath+entry, sep=',', skiprows = 1, header = 'infer')
        dat.Note = dat['Note'].fillna(0)
        Notes  = np.array(dat.Note)
        idx = np.where(Notes != 0)[0] # Find rows where there was a note indicating end of one activity
        
        prev_note = 0
        for ii in idx:
            if ii-prev_note > 5: #Must have at least 5 frames between activies
                print(Notes[ii])
                portion = dat.iloc[prev_note:ii+1,:]
                if "\r" in Notes[ii]:
                    # If there is a space in the note name, remove it.
                    tmpTrial = Notes[ii].split('\r')[0].replace(" ","")
                else:
                    tmpTrial = Notes[ii].replace(" ","")
                        
                # Save the portion of the data
                portion.to_csv(fSave+subject+'_'+tmpConfig+'_'+tmpTrial+'_'+tmpSesh+'.csv',index=False)
            
            prev_note = ii+1
        
            
        