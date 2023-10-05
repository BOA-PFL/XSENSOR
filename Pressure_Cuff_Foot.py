# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:35:44 2022

@author: Dan.Feeney

This code analyzes data collected from the plantar (insole) and dorsal surfaces of the foot when testing boots. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox

save_on = 1
check_data = 0
# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\SkiValidation_Dec2022\\InLabPressure\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

@dataclass
class avgData:
    avgDorsal: np.array
    avgPlantar: np.array
    config: str
    subject: str
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1 = sns.heatmap(self.avgDorsal, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(self.avgDorsal) * 2)
        ax1.set(xticklabels=[])
        ax1.set_title('Posterior Pressure') 
        ax2 = sns.heatmap(self.avgPlantar, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(self.avgPlantar) * 2)
        ax2.set(xticklabels=[])
        ax2.set_title('Anterior Pressure') 
        return fig  
    
    def sortDF(self, colName):
        """ 
        Grabs each individual grouping by location of foot from regions 
        specified in XSENSOR output
        """
        subsetDat = self.fullDat.iloc[:,self.fullDat.columns.get_loc(colName):self.fullDat.columns.get_loc(colName)+12]
        return(subsetDat)

# dd = test.sortDF('Group')
# filter_col = [col for col in dat if col.startswith('Group')]

@dataclass
class footLocData:
    RLHeel: pd.DataFrame
    RMHeel: pd.DataFrame
    RMMidfoot: pd.DataFrame
    RLMidfoot: pd.DataFrame
    RMMets: pd.DataFrame
    RLMets: pd.DataFrame
    RMToes: pd.DataFrame
    RLToes: pd.DataFrame
    

def createAvgMat(inputName):
    """ 
    Reads in file, creates average matrix data to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    
    inputName: string
        filename of the data you are analyzing, extracted from the 'entries' list
        
    result: avgData (see dataclass above)
    """
    dat = pd.read_csv(fPath+inputName, sep=',', skiprows = 1, header = 'infer')
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1].split(sep=".")[0]
    sensel = dat.iloc[:,17:197]
    plantarSensel = dat.iloc[:,214:425]
    
    headers = plantarSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    con_press = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        con_press[store_r[ii],store_c[ii]] = np.mean(plantarSensel.iloc[:,ii])
        
   
    avgDorsalMat = np.array(np.mean(sensel, axis = 0)).reshape((18,10))
    avgPlantarMat = np.array(np.flip(con_press))
    
    result = avgData(avgDorsalMat, avgPlantarMat, config, subj, dat)
    
    return(result)

def subsetMat(inputDF):
    """ 
    Input an instance of class avgData with regions of the foot separated
    with masks from XSENSOR
    
    Outputs an instance of class footLocData with each region separated out 
    """
    
    filter_col = [col for col in inputDF.fullDat if col.startswith('Group')]
   
    if len(filter_col) != 8:
        print('check subsetMat function if correct class names are specified')
        
    RLH = inputDF.sortDF(filter_col[0])
    RMH = inputDF.sortDF(filter_col[1])
    RMM = inputDF.sortDF(filter_col[2])
    RLM = inputDF.sortDF(filter_col[3])
    RMMet = inputDF.sortDF(filter_col[4])
    RLMet = inputDF.sortDF(filter_col[5])
    RMToes = inputDF.sortDF(filter_col[6])
    RLToes = inputDF.sortDF(filter_col[7])
    
    output = footLocData(RLH, RMH, RMM, RLM, RMMet, RLMet, RMToes, RLToes)
    
    return(output)

def calcSummaryStats(inputDF):
    """ 
    Calculates mean, peak, and contact % from an input DF
    4,6,and 9 are mean, peak, and contact% from XSENSOR output
    
    input: avgData (see dataclass above)
    
    output: outcome variables described above
    """
    colsInterest = list((4,6,9))
    outVal = []
    for val in colsInterest:
        outVal.append( np.mean(inputDF.iloc[:,val]) ) 
    [avg, peak, avgCon] = outVal
    
    return(outVal)

meanDorsalPressure = []
maxDorsalPressure = [] 
sdDorsalPressure = []
totalDorsalPressure = []
config = []
subject = []

for entry in entries:
    
    if entry == 'CompiledResults2.csv': # don't try to process compiled data csv if it already exists
        print('Compiled results csv exists & will be added to')
    
    else:
        tmpAvgMat = createAvgMat(entry)
        if check_data == 0:
            answer = True
        else:
            tmpAvgMat.plotAvgPressure()
            answer = messagebox.askyesno("Question","Is data clean?")
        
        if answer == False:
            plt.close('all')
            print('Adding file to bad file list')
            #badFileList.append(fName)
        
        if answer == True:
            plt.close('all')
            print('Estimating point estimates')
    
            config = tmpAvgMat.config
            subject = tmpAvgMat.subject
            DContact = np.count_nonzero(tmpAvgMat.avgDorsal)
            meanDorsalPressure = float(np.mean(tmpAvgMat.avgDorsal))
            maxDorsalPressure = float(np.max(tmpAvgMat.avgDorsal))
            sdDorsalPressure = float(np.std(tmpAvgMat.avgDorsal))
            totalDorsalPressure = float(np.sum(tmpAvgMat.avgDorsal))
            
            footLocations = subsetMat(tmpAvgMat)
            # Calculate average, peak, and contact of masked regions of foot
            [avgMHeel, pkMHeel, conMHeel] = calcSummaryStats(footLocations.RMHeel)
            [avgLHeel, pkLHeel, conLHeel] = calcSummaryStats(footLocations.RLHeel)
            [avgMMid, pkMMid, conMMid] = calcSummaryStats(footLocations.RMMidfoot)
            [avgLMid, pkLMid, conLMid] = calcSummaryStats(footLocations.RLMidfoot)
            [avgMMets, pkMMets, conMMets] = calcSummaryStats(footLocations.RMMets)
            [avgLMets, pkLMets, conLMets] = calcSummaryStats(footLocations.RLMets)
            [avgMToes, pkMToes, conMToes] = calcSummaryStats(footLocations.RMToes)
            [avgLToes, pkLToes, conLToes] = calcSummaryStats(footLocations.RLToes)
    
            
            outcomes = pd.DataFrame([[subject,config,DContact,meanDorsalPressure,maxDorsalPressure,sdDorsalPressure,totalDorsalPressure,
                                    avgMHeel, pkMHeel, conMHeel, avgLHeel, pkLHeel, conLHeel, avgMMid, pkMMid, conMMid,
                                    avgLMid, pkLMid, conLMid, avgMMets, pkMMets, conMMets, avgLMets, pkLMets, conLMets,
                                    avgMToes, pkMToes, conMToes, avgLToes, pkLToes, conLToes]],
                                    columns=['Subject','Config','DorsalContact','meanDorsalPressure','maxDorsalPressure','sdDorsalpressure','totalDorsalPressure',
                                    'avgMHeel', 'pkMHeel', 'conMHeel', 'avgLHeel', 'pkLHeel', 'conLHeel', 'avgMMid', 'pkMMid', 'conMmid',
                                    'avgLMid', 'pkLMid', 'conLMid', 'avgMMets', 'pkMMets', 'conMMets', 'avgLMets', 'pkLMets', 'conLMets',
                                    'avgMToes', 'pkMToes', 'conMToes', 'avgLToes', 'pkLToes', 'conLToes'])
              
            outfileName = fPath + 'CompiledResults2.csv'
            if save_on == 1:
                if os.path.exists(outfileName) == False:
                    
                    outcomes.to_csv(outfileName, header=True, index = False)
                
                else:
                    outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
                
    
    
               