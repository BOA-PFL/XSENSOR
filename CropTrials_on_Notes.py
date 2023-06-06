# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:10:41 2023

@author: Eric.Honert

This code crops the XSENSOR data based on the Notes column. The data will be
cropped into smaller trials based on the notes and saved in a different folder
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/WorkWear_Performance/EH_Workwear_MidCutStabilityI_CPDMech_June23/Pressure/Raw/'
fSave = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/WorkWear_Performance/EH_Workwear_MidCutStabilityI_CPDMech_June23/Pressure/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]
cropped_entries = [fName for fName in os.listdir(fSave) if fName.endswith(fileExt)]


for entry in entries:
    subject = entry.split('_')[0]
    # Don't redo cropping
    cropped_count = 0
    for sub_entry in cropped_entries:
        if subject in sub_entry:
            cropped_count = cropped_count + 1
    
    if cropped_count == 0:
        print(entry)
        tmpConfig = entry.split('_')[1]
        tmpSesh = entry.split('_')[2][0]
        dat = pd.read_csv(fPath+entry, sep=',', skiprows = 1, header = 'infer')
        dat.Note = dat['Note'].fillna(0)
        Notes  = np.array(dat.Note)
        idx = np.where(Notes != 0)[0]
        
        prev_note = 0
        for ii in idx:
            if ii-prev_note > 5:
                print(Notes[ii])
                portion = dat.iloc[prev_note:ii,:]
                if "\r" in Notes[ii]:
                    # If there is a space in the note name
                    tmpTrial = Notes[ii].split('\r')[0].replace(" ","")
                else:
                    tmpTrial = Notes[ii]
                        
                # Save the portion of the data
                portion.to_csv(fSave+subject+'_'+tmpConfig+'_'+tmpTrial+'_'+tmpSesh+'.csv',index=False)
            
            prev_note = ii
        
            
        