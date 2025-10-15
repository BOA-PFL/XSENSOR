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

import scipy.signal as sig

save_on = 1

# Read in files
# only read .asc files for this work

fPath = 'C:\\Users\\bethany.kilpatrick\\BOA Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\2025_Performance_K2Pressure_Taro\\xsensor\\Exported\\'
fileExt = r".csv"

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]
entries_frontBack = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FrontBack' in fName) ]
entries_FootDorsum = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FootDorsum' in fName) ]


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
    plantarSensNo: int
    plantarToeSensNo: int
    plantarForefootSensNo: int
    plantarMidfootSensNo: int
    plantarHeelSensNo: int 
    
    avgshinMat: np.array 

    
    avgcalfMat: np.array   

    
    config: str
    subject: str
    
    dat_FD: pd.DataFrame
    dat_FB: pd.DataFrame
    #entire stored dataframe. 
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
        
        saveFolder= fPath + '2DPlots'
        
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)
            
        plt.savefig(saveFolder + '/' + self.subject + self.config + '.png')
         
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
   
    # entries_frontBack = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FrontBack' in fName) ]
    # entries_FootDorsum = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FootDorsum' in fName) ] 

    
    # inputName = entries_frontBack[1]
    # Front back 
   

    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    trial = inputName.split(sep="_")[3].split(sep='.')[0]
    
    
    frontBack = [fName for fName in  entries_frontBack if (subj in fName and config in fName and trial in fName)  ]
    dat_FB = pd.read_csv(fPath+frontBack[0], sep=',', header =1 , low_memory=False)
    
    
   

    if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0002":  # check to see if right insole used
        shinSensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
        calfSensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1']
    
    # if "Sensor" in dat_FB.columns:
    if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0004":  # check to see if right insole used
        calfSensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
        shinSensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1']
    
    
    #Shin
    avgshinMat = np.array(np.mean(shinSensel, axis = 0)).reshape((18,10))
   
    avgshinMat = np.flip(avgshinMat, axis = 0) 
    
    avgshinMat[avgshinMat <1] = 0   
    
    #Calf 
    avgcalfMat = np.array(np.mean(calfSensel, axis = 0)).reshape((18,10))
   
    avgcalfMat = np.flip(avgcalfMat , axis = 0) 
    
    avgcalfMat [avgcalfMat  <1] = 0   
    
    
    
     #Foot Dorsum 
    footDorsum = [fName for fName in entries_FootDorsum if (subj in fName and config in fName and trial in fName)  ]
    dat_FD = pd.read_csv(fPath + footDorsum[0], sep=',', header = 1, low_memory=False)


    if dat_FD['Insole'][0] != 'Right' and dat_FD['Insole'][0] != 'Left':  # check to see if dorsal pad was used
        dorsalSensel = dat_FD.loc[:,'S_1_1':'S_18_10'] 
    if dat_FD['Insole.1'][0] == 'Right':
            # if dat_FD['Insole.1'][0] == 'Right':  
            plantarSensel = dat_FD.loc[:, 'S_1_2.1':'S_31_7']




    
    headers = plantarSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    con_press = np.zeros((np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        con_press[store_r[ii],store_c[ii]] = np.mean(plantarSensel.iloc[:,ii])
    
    # Sensel Number Computation
    store_r = np.array(store_r)
    plantarSensNo = len(store_r)
    plantarToeSensNo = len(np.where(store_r < 7)[0])
    plantarForefootSensNo = len(np.where((store_r >= 7)*(store_r < 15))[0])
    plantarMidfootSensNo = len(np.where((store_r >= 15)*(store_r < 25))[0])
    plantarHeelSensNo = len(np.where(store_r >= 25)[0])
        
    con_press[con_press < 1] = 0
    
    
    avgDorsalMat = np.array(np.mean(dorsalSensel, axis = 0)).reshape((18,10))
   
    avgDorsalMat = np.flip(avgDorsalMat, axis = 0) 
    
    avgDorsalMat[avgDorsalMat <1] = 0  
    
    avgPlantarMat = np.array(con_press) 
    
   
    
    
 
    
    
    result = avgData(avgDorsalMat, avgPlantarMat, 
                     plantarSensNo, plantarToeSensNo, plantarForefootSensNo, plantarMidfootSensNo, plantarHeelSensNo, avgshinMat, 
                     avgcalfMat,config, subj, dat_FD, dat_FB)
    
    return(result)


meanDorsalPressure = []
maxDorsalPressure = [] 
sdDorsalPressure = []
covDorsalPressure = []
totalDorsalPressure = []
config = []
subject = []
Movement = []

plantarContact = []
plantarPeakPressure = []
plantarAvgPressure = []
plantarSDPressure = []
plantarTotalPressure = []

heelArea = [] 
heelAreaUP = []

for entry in entries_FootDorsum:
    print(entry)
    # Deliniate the static type
    # if 'tanding' in entry:
    #     Movement ='Standing'
    # if 'tand' in entry:
    #     Movement ='Standing'
    # if 'itting' in entry: 
    #     Movement ='Sitting'
    # if 'it' in entry: 
    #     Movement ='Sitting'
    # entry = entries_FootDorsum[37]
    tmpAvgMat = createAvgMat(entry)
    tmpAvgMat.plotAvgPressure()
    answer = messagebox.askyesno("Question","Is data clean?") # If entire rows of sensels are blank, its not clean!
    
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
        covDorsalPressure = float(np.std(tmpAvgMat.avgDorsal)/np.mean(tmpAvgMat.avgDorsal))
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


        plantarContact = float(np.count_nonzero(tmpAvgMat.avgPlantar)/tmpAvgMat.plantarSensNo*100)
        plantarPeakPressure = float(np.max(tmpAvgMat.avgPlantar)*6.895)
        plantarAvgPressure = float(np.mean(tmpAvgMat.avgPlantar)*6.895)
        plantarSDPressure = float(np.std(tmpAvgMat.avgPlantar)*6.895)
        plantarTotalPressure = float(np.sum(tmpAvgMat.avgPlantar)*6.895)
        
        toeContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[:7, :])/tmpAvgMat.plantarToeSensNo*100)
        toePressure = float(np.mean(tmpAvgMat.avgPlantar[:7, :])*6.895)
        ffContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[7:15, :])/tmpAvgMat.plantarForefootSensNo*100)
        ffPressure = float(np.mean(tmpAvgMat.avgPlantar[7:15, :])*6.895)
        mfContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[15:25, :])/tmpAvgMat.plantarMidfootSensNo*100)
        mfPressure = float(np.mean(tmpAvgMat.avgPlantar[15:25, :])*6.895)
        
        heelContact = float(np.count_nonzero(tmpAvgMat.avgPlantar[25:, :])/tmpAvgMat.plantarHeelSensNo*100)
        heelPressure = float(np.mean(tmpAvgMat.avgPlantar[25:, :])*6.895)


        avgCalf= float(np.mean(tmpAvgMat.avgcalfMat)*6.895)
        pkCalf = float(np.max(tmpAvgMat.avgcalfMat) *6.895)
        varCalf = float(np.std(tmpAvgMat.avgcalfMat)  / np.mean(tmpAvgMat.avgcalfMat) *6.895)
         
        avgShin = float(np.mean(tmpAvgMat.avgshinMat)*6.895)
        pkShin = float(np.max(tmpAvgMat.avgshinMat) *6.895)
        varShin = float(np.std(tmpAvgMat.avgshinMat)  / np.mean(tmpAvgMat.avgshinMat) *6.895)


        outcomes = pd.DataFrame([[subject,config, 
                                  dorsalContact, meanDorsalPressure, maxDorsalPressure,sdDorsalPressure,covDorsalPressure,totalDorsalPressure,
        
                                  ffDorsalContact, ffDorsalPressure, ffDorsalMaxPressure, mfDorsalContact, mfDorsalPressure,  mfDorsalMaxPressure, instepDorsalContact, instepDorsalPressure, instepDorsalMaxPressure,
                    
                                  plantarContact, plantarAvgPressure, plantarPeakPressure,  plantarSDPressure, plantarTotalPressure,
                                  
                                  toeContact, toePressure, ffContact, ffPressure, mfContact, mfPressure, heelContact, heelPressure,  avgCalf, pkCalf, varCalf,
                                    avgShin, pkShin, varShin
                                  ]],
                                
                                columns=['Subject','Config', 
                                         'dorsalContact', 'meanDorsalPressure','maxDorsalPressure','sdDorsalPressure','covDorsalPressure','totalDorsalPressure',
                                         
        
                                         'ffDorsalContact', 'ffDorsalPressure', 'ffDorsalMaxPressure', 'mfDorsalContact', 'mfDorsalPressure', 
                                         'mfDorsalMaxPressure', 'instepDorsalContact', 'instepDorsalPressure','instepDorsalMaxPressure',
                                         
                                         'plantarContact', 'meanPlantarPressure', 'maxPlantarPressure', 'sdPlantarPressure', 'totalPlantarPressure',
                                         
                                         'toeContact', 'toePressure', 'ffContact', 'ffPressure',
                                         'mfContact', 'mfPressure', 'heelContact', 'heelPressure', 'avgCalf', 'pkCalf', 'varCalf', 'avgShin','pkShin','varShin'
                                         
                                         ])
        
              
        outfileName = fPath + '0_CompiledResults_Static_6.csv'
        if save_on == 1:
            if os.path.exists(outfileName) == False:
                outcomes.to_csv(outfileName, header=True, index = False)
            else:
                outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
    
    
               
