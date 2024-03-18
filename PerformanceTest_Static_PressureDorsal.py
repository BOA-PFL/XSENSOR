# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Kate.Harrison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox

save_on = 1

# Read in files
# only read .asc files for this work

fPath = 'C:\\Users\\bethany.kilpatrick\\Boa Technology Inc\\PFL - General\\Testing Segments\\Cycling Performance Tests\\PP_CyclingUpperStiffness_Performance_March24\\Xsensor\\Static\\'


fileExt = r".csv"
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

@dataclass
class avgData:
    avgDorsal: np.array
    avgPlantar: np.array
    config: str
    subject: str
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis. 
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1 = sns.heatmap(self.avgDorsal, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(self.avgDorsal) * 2)
        ax1.set(xticklabels=[])
        ax1.set_title('Dorsal Pressure') 
        ax2 = sns.heatmap(self.avgPlantar, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(self.avgPlantar) * 2)
        ax2.set(xticklabels=[])
        ax2.set_title('Plantar Pressure') 
        plt.suptitle(self.config)
        plt.tight_layout()
        return fig  
    
    def sortDF(self, colName):
        """ 
        Grabs each individual grouping by location of foot from regions 
        specified in XSENSOR output
        """
        subsetDat = self.fullDat.iloc[:,self.fullDat.columns.get_loc(colName):self.fullDat.columns.get_loc(colName)+12]
        return(subsetDat)

    

def createAvgMat(inputName):
    """ 
    Reads in file, creates average matrix data, in the shape of the pressure sensor(s), to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    
    inputName: string
        filename of data static trial you are processing. 
    """
   
        
    inputName = entries[4]
   
    dat = pd.read_csv(fPath+inputName, sep=',', header = 'infer')
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1].split(sep=".")[0]
    
    insoleSide = dat['Insole Side'][0]
    
    if (insoleSide == 'Left'): 
        
        # Left side
        plantarSensel = dat.iloc[:,18:238]
        dorsalSensel = dat.iloc[:,250:430]
    else:  
        dorsalSensel = dat.iloc[:,18:198]
        plantarSensel = dat.iloc[:,210:430] 
        
    
    
    headers = plantarSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    con_press = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        con_press[store_r[ii],store_c[ii]] = np.mean(plantarSensel.iloc[:,ii])
        
    con_press[con_press < 1] = 0
    
    
    avgDorsalMat = np.array(np.mean(dorsalSensel, axis = 0)).reshape((18,10))
   
    avgDorsalMat = np.flip(avgDorsalMat, axis = 0) 
    
    avgDorsalMat[avgDorsalMat <1] = 0  
    
    avgPlantarMat = np.array(con_press) 
    
    
    
    result = avgData(avgDorsalMat, avgPlantarMat, config, subj, dat)
    
    return(result)


# meanDorsalPressure = []
# maxDorsalPressure = [] 
# sdDorsalPressure = []
# totalDorsalPressure = []
# config = []
# subject = []
# Movement = []

# plantarContact = []
# plantarPeakPressure = []
# plantarAvgPressure = []
# plantarSDPressure = []
# plantarTotalPressure = []

heelArea = [] 
heelAreaUP = []

for entry in entries:
    


    # entry = entries[3]
    if 'tanding' in entry:
        Movement ='Standing'
    if 'tand' in entry:
        Movement ='Standing'
    if 'itting' in entry: 
        Movement ='Sitting'
    if 'it' in entry: 
        Movement ='Sitting'
            


    tmpAvgMat = createAvgMat(entry)
    tmpAvgMat.plotAvgPressure()
    answer = messagebox.askyesno("Question","Is data clean?") # If entire rows of sensels are blank, its not clean!
    
    plantar = tmpAvgMat.avgPlantar
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        #badFileList.append(fName)
    
    if answer == True:
        plt.close('all')
        print('Estimating point estimates')

        config = (tmpAvgMat.config)
        subject = (tmpAvgMat.subject)
        
        meanDorsalPressure = float(np.mean(tmpAvgMat.avgDorsal)*6.895)
        maxDorsalPressure = float(np.max(tmpAvgMat.avgDorsal)*6.895)
        sdDorsalPressure = float(np.std(tmpAvgMat.avgDorsal)*6.895)
        totalDorsalPressure = float(np.sum(tmpAvgMat.avgDorsal)*6.895)
        dorsalContact = float(np.count_nonzero(tmpAvgMat.avgDorsal)/180*100)
        

        ffDorsalContact = float(np.count_nonzero(tmpAvgMat.avgDorsal[:6, :])/60*100)
        ffDorsalPressure = float(np.mean(tmpAvgMat.avgDorsal[:6, :])*6.895)
        ffDorsalMaxPressure = float(np.max(tmpAvgMat.avgDorsal[:6, :])*6.895)
        mfDorsalContact = float(np.count_nonzero(tmpAvgMat.avgDorsal[6:12, :])/60*100)
        mfDorsalPressure = float(np.mean(tmpAvgMat.avgDorsal[6:12, :])*6.895)
        mfDorsalMaxPressure = float(np.max(tmpAvgMat.avgDorsal[6:12, :])*6.895)
        instepDorsalContact = float(np.count_nonzero(tmpAvgMat.avgDorsal[12:, :])/60*100)
        instepDorsalPressure = float(np.mean(tmpAvgMat.avgDorsal[12:, :])*6.895)
        instepDorsalMaxPressure = float(np.mean(tmpAvgMat.avgDorsal[12:, :])*6.895)


        plantarContact = float(np.count_nonzero(tmpAvgMat.avgPlantar)/220*100)
        plantarPeakPressure = float(np.max(tmpAvgMat.avgPlantar)*6.895)
        plantarAvgPressure = float(np.mean(tmpAvgMat.avgPlantar)*6.895)
        plantarSDPressure = float(np.std(tmpAvgMat.avgPlantar)*6.895)
        plantarTotalPressure = float(np.sum(tmpAvgMat.avgPlantar)*6.895)
        
        toeContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[:7, :])/47*100)
        toePressure = float(np.mean(tmpAvgMat.avgPlantar[:7, :])*6.895)
        ffContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[7:15, :])/67*100)
        ffPressure = float(np.mean(tmpAvgMat.avgPlantar[7:15, :])*6.895)
        mfContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[15:25, :])/70*100)
        mfPressure = float(np.mean(tmpAvgMat.avgPlantar[15:25, :])*6.895)
        
        heelContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[25:, :])/35*100)
        heelPressure = float(np.mean(tmpAvgMat.avgPlantar[25:, :])*6.895)

        

        outcomes = pd.DataFrame([[subject,config, Movement,
                                  dorsalContact, meanDorsalPressure, maxDorsalPressure,sdDorsalPressure,totalDorsalPressure,

                                  ffDorsalContact, ffDorsalPressure, ffDorsalMaxPressure, mfDorsalContact, mfDorsalPressure,  mfDorsalMaxPressure, instepDorsalContact, instepDorsalPressure, instepDorsalMaxPressure,
                    
                                  plantarContact, plantarAvgPressure, plantarPeakPressure,  plantarSDPressure, plantarTotalPressure,
                                  
                                  toeContact, toePressure, ffContact, ffPressure, mfContact, mfPressure, heelContact, heelPressure
                                  ]],
                                
                                columns=['Subject','Config', 'Movement', 
                                         'dorsalContact', 'meanDorsalPressure','maxDorsalPressure','sdDorsalPressure','totalDorsalPressure',
                                         

                                         'ffDorsalContact', 'ffDorsalPressure', 'ffDorsalMaxPressure', 'mfDorsalContact', 'mfDorsalPressure', 
                                         'mfDorsalMaxPressure', 'instepDorsalContact', 'instepDorsalPressure','instepDorsalMaxPressure',
                                         
                                         'plantarContact', 'meanPlantarPressure', 'maxPlantarPressure', 'sdPlantarPressure', 'totalPlantarPressure',
                                         
                                         'toeContact', 'toePressure', 'ffContact', 'ffPressure',
                                         'mfContact', 'mfPressure', 'heelContact', 'heelPressure'
                                         
                                         ])

          
        outfileName = fPath + '0_CompiledResults_Static.csv'
        if save_on == 1:
            if os.path.exists(outfileName) == False:
                
                outcomes.to_csv(outfileName, header=True, index = False)
            
            else:
                outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
    
    
               
