# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Bethany
"""

"""
This code directly pulls in two dorsal pads used to measure shin/thigh and calf pressure 
and indirectly pulls in dorsal and plantar pressure from XSENSORfuntions. 

Both files get analyzed and combined in the export


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
from tkinter import messagebox
import scipy.signal as sig
from XSENSORFunctions import readXSENSORFile, delimitTrial, createTSmat, zeroInsoleForce, findGaitEvents




save_on = 0
data_check = 0


# Read in files
# only read .asc files for this work

fPath = 'Z:\\Testing Segments\\EndurancePerformance\\2025_Mechanistic_KneeSleeveStability_Baurfeind\\Xsensor\\FullTrials\\'
fileExt = r".csv"

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]
entries_FrontBack = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FrontBack' in fName)]
entries_FootDorsum = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'FootDorsum' in fName)  ]


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
class FB_avgData:
    
    pad1Mat: np.array 

    
    pad2Mat: np.array   

    
    config: str
    subject: str
    
   
    dat_FB: pd.DataFrame
   
    
def createAvg_FB_Mat(inputName,FilePath):
    """ 
    Reads in file, creates average matrix data, in the shape of the pressure sensor(s), to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting
    
    inputName: string
        filename of data static trial you are processing.  
        
     FilePath : str
         file path string
    """
   
   
   

    subj = inputName.split(sep="_")[0]  
    config = inputName.split(sep="_")[1]
    

    dat_FB = pd.read_csv(os.path.join(FilePath, inputName), sep=',', header=1, low_memory=False)
    
   # HX210.10.18.04-L S0003 == the Thigh Low Max dorsal pad 
   # HX210.10.18.04-L S0002 == Calf High Max dorsal pad

    # if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0003":  # check to see if Shin/Thigh first
    pad1Sensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
    pad2Sensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1']
    
    # if "Sensor" in dat_FB.columns:
    # if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0002":  # check to see if Low Max pad was used - Calf first
    #     calfSensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
    #     shinSensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1'] 
      
    # if dat_FB["Sensor"][0] == "HX210.10.18.04-L S0004":  # check to see if Calf first
    #     calfSensel = dat_FB.loc[:, 'S_1_1':'S_18_10']
    #     shinSensel = dat_FB.loc[:, 'S_1_1.1':'S_18_10.1']
    
    #Shin
    # avgshinMat = np.array(np.mean(shinSensel, axis = 0)).reshape((18,10))
   
    
    if 'pad1Sensel' in locals():
        headers = pad1Sensel.columns
        store_r = []
        store_c = []

        for name in headers:
            store_r.append(int(name.split(sep = "_")[1])-1)
            store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
        
        pad1Mat = np.zeros((dat_FB.shape[0], np.max(store_r)+1,np.max(store_c)+1))
        
        for ii in range(len(headers)):
            pad1Mat[:, store_r[ii],store_c[ii]] = pad1Sensel.iloc[:,ii]
       
        
    
    
    pad1Mat = np.flip(pad1Mat , axis = 0) 
    
    pad1Mat[pad1Mat  <.1] = 0   
    

    if 'pad2Sensel' in locals():
        headers = pad2Sensel.columns
        store_r = []
        store_c = []
 
        for name in headers:
            store_r.append(int(name.split(sep = "_")[1])-1)
            store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
        
        pad2Mat = np.zeros((dat_FB.shape[0], np.max(store_r)+1,np.max(store_c)+1))
        
        for ii in range(len(headers)):
            pad2Mat[:, store_r[ii],store_c[ii]] = pad2Sensel.iloc[:,ii]
               
               
    pad2Mat = np.flip(pad2Mat , axis = 0) 
    
    pad2Mat [pad2Mat  <.1] = 0   
    
  
    
    
    result = FB_avgData(pad1Mat, pad2Mat,config, subj, dat_FB)
    
    return(result)  


def plotAvgStaticFDPressure(plantarMat, inputFD, FilePath):
    """
    Plot the average static plantar and dorsal pressure
    Function dependencies: need to use createTSmat for the appropriate array
    shape for the plantar and dorsal pressure

    Parameters
    ----------
    plantarMat : numpy array
        DESCRIPTION.
    inputDC : dataclass
        Created from the function "createTSmat"
    FilePath : str
        file path string

    Returns
    -------
    fig : matplotlib figure
        Figure showing average dorsal and plantar pressure during the static trial

    """
    
    avgPlantar = np.mean(plantarMat, axis = 0)
    avgDorsal = np.mean(inputFD.dorsalMat, axis = 0)



    fig, ((ax1, ax2)) = plt.subplots(1,2)
    ax1 = sns.heatmap(avgDorsal, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(avgDorsal) * 2)
    ax1.set(xticklabels=[])
    ax1.set_title('Dorsal Pressure') 
    ax2 = sns.heatmap(avgPlantar, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(avgPlantar) * 2)
    ax2.set(xticklabels=[])
    ax2.set_title('Plantar Pressure')  
 
    
    plt.suptitle(inputFD.config)
    plt.tight_layout() 
    
    saveFolder= FilePath + '2DPlots'
    
    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)
        
    plt.savefig(saveFolder + '/' + inputFD.subject + inputFD.config + '.png')
    return fig

def plotAvgStaticFBPressure( inputFB, FilePath):
    """
    Plot the average static plantar and dorsal pressure
    Function dependencies: need to use createTSmat for the appropriate array
    shape for the plantar and dorsal pressure

    Parameters
    ----------
    plantarMat : numpy array
        DESCRIPTION.
    inputDC : dataclass
        Created from the function "createTSmat"
    FilePath : str
        file path string

    Returns
    -------
    fig : matplotlib figure
        Figure showing average dorsal and plantar pressure during the static trial

    """
    


    
    avgShin = np.mean(inputFB.pad1Mat, axis = 0)
    avgCalf = np.mean(inputFB.pad2Mat, axis = 0)


    fig, ((ax1, ax2)) = plt.subplots(1,2)

    ax1 = sns.heatmap(avgShin, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(avgShin) * 2)
    ax1.set(xticklabels=[])
    ax1.set_title('Shin Pressure')  
    ax2 = sns.heatmap(avgCalf, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(avgCalf) * 2)
    ax2.set(xticklabels=[])
    ax2.set_title('Calf Pressure')  
    
    plt.suptitle(inputFB.config)
    plt.tight_layout() 
    
    saveFolder= FilePath + '2DPlots'
    
    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)
        
    plt.savefig(saveFolder + '/' + inputFB.subject + inputFB.config + '.png')
    return fig



config = []
subject = []
Movement = []
badFileList = []

plantarContact = []
plantarPeakPressure = []
plantarAvgPressure = []
plantarSDPressure = []
plantarTotalPressure = []

heelArea = [] 





if entries_FootDorsum == [0] or not entries_FootDorsum:
    parentList = entries
else:
    parentList = entries_FootDorsum




fd_outcomes = []
 

# check if front back has more data
if len(entries_FrontBack) > len(entries_FootDorsum):    
    for entry in entries_FrontBack: 
        subject = entry.split(sep="_")[0] 
        config = entry.split(sep="_")[1]
        trial = entry.split(sep="_")[3].split(sep='.')[0]   
        counter = 0
        for ii in range(len(entries_FootDorsum)):
            sub2 = entries_FootDorsum[ii].split(sep="_")[0]
            config2 = entries_FootDorsum[ii].split(sep="_")[1]
            trial2 = entries_FootDorsum[ii].split(sep="_")[3].split(sep='.')[0] 
            if subject == sub2 and config == config2 and trial == trial2: 
                counter = 1 
        if counter == 0: 
            parentList.append(entry) 
            
        

for entry in parentList:
    print(entry )  
    
    
    
    entry = parentList[1] 
    

     
    if "FootDorsum" in entry:  
        fd_found = 1

        subject = entry.split(sep="_")[0] 
        config = entry.split(sep="_")[1]
        trial = entry.split(sep="_")[3].split(sep='.')[0]  
        tmpDat = readXSENSORFile(entry,fPath) 
        plantDorDat = createTSmat(entry,fPath,tmpDat)  
        
        
        fb_found = 0 
            
        # For loop to find corresponding file in other list
        for ii in range(len(entries_FrontBack)):
            sub2 = entries_FrontBack[ii].split(sep="_")[0]
            config2 = entries_FrontBack[ii].split(sep="_")[1]
            trial2 = entries_FrontBack[ii].split(sep="_")[3].split(sep='.')[0]
            

            if (subject == sub2) and (config == config2) and (trial == trial2): 
                print(entries_FrontBack[ii]) 
                print(' ')
                fb_found = 1
                frontBackDat = createAvg_FB_Mat(entries_FrontBack[ii],fPath)  
                
                if frontBackDat.dat_FB["Sensor"][0] == "HX210.10.18.04-L S0003":  # check to see if Shin/Thigh first
                    shin = frontBackDat.pad1Mat 
                    calf = frontBackDat.pad2Mat 
            
                if frontBackDat.dat_FB["Sensor"][0] == "HX210.10.18.04-L S0002":  # check to see if Low Max pad was used - Calf first
                    calf = frontBackDat.pad1Mat 
                    shin = frontBackDat.pad2Mat
                 
                if frontBackDat.dat_FB["Sensor"][0] == "HX210.10.18.04-L S0004":  # check to see if Calf first
                    calf =  frontBackDat.pad1Mat 
                    shin =  frontBackDat.pad2Mat
                
    elif "FrontBack" in entry: 
        fb_found = 1  
        fd_found = 0 
        frontBackDat = createAvg_FB_Mat(entry)   
    else:  
        # Fallback if neither FootDorsum or FrontBack is in entry
        subject = entry.split(sep="_")[0] 
        config = entry.split(sep="_")[1]
        trial = entry.split(sep="_")[3].split(sep='.')[0]  
        
        fb_found = 1
        fd_found = 0
        frontBackDat = createAvg_FB_Mat(entry, fPath)  
        
        if frontBackDat.dat_FB["Sensor"][0] == "HX210.10.18.04-L S0003":  # check to see if Shin/Thigh first
            shin = frontBackDat.pad1Mat 
            calf = frontBackDat.pad2Mat 
    
        if frontBackDat.dat_FB["Sensor"][0] == "HX210.10.18.04-L S0002":  # check to see if Low Max pad was used - Calf first
            calf = frontBackDat.pad1Mat 
            shin = frontBackDat.pad2Mat
         
        if frontBackDat.dat_FB["Sensor"][0] == "HX210.10.18.04-L S0004":  # check to see if Calf first
            calf =  frontBackDat.pad1Mat 
            shin =  frontBackDat.pad2Mat

   
        
    answer = True
    if data_check == 1 and "FootDorsum" in entry:  
        if  len(plantDorDat.LplantarMat) != 0:
            plotAvgStaticFDPressure(plantDorDat.LplantarMat,plantDorDat, fPath) 
       
            
        if  len(plantDorDat.RplantarMat) != 0:
            plotAvgStaticFDPressure(plantDorDat.RplantarMat,plantDorDat,fPath)   

    else:
        plotAvgStaticFBPressure( frontBackDat , fPath)
        

        answer = messagebox.askyesno("Question","Is data clean?") # If entire rows of sensels are blank, its not clean!
    
        if answer == False:
            plt.close('all')
            print('Adding file to bad file list')
            #badFileList.append(fName)
        
        if answer == True:
            plt.close('all')
            print('Estimating point estimates')





    if fd_found == 1:
        avgDorsal = np.mean(plantDorDat.dorsalMat, axis = 0)
        avgffDorsal = np.mean(plantDorDat.dorsalForefoot, axis = 0) 
        avgmfDorsal = np.mean(plantDorDat.dorsalMidfoot, axis = 0) 
        avginstepDorsal = np.mean(plantDorDat.dorsalInstep, axis = 0) 
        
        avgPlantar = np.mean(plantDorDat.LplantarMat, axis = 0) 
        avgToe = np.mean(plantDorDat.LplantarToe, axis = 0) 
        avgFF = np.mean(plantDorDat.LplantarForefoot, axis = 0)  
        avgMF = np.mean(plantDorDat.LplantarMidfoot, axis = 0)  
        avgHeel = np.mean(plantDorDat.LplantarHeel, axis = 0)  
        
       
        meanDorsalPressure = float(np.mean(plantDorDat.dorsalMidfoot) *6.895)
        maxDorsalPressure = float(np.max(avgDorsal)*6.895)
        sdDorsalPressure = float(np.std(avgDorsal)*6.895)
        covDorsalPressure = float(np.std(plantDorDat.dorsalMat)/np.mean(avgDorsal))
        totalDorsalPressure = float(np.sum(avgDorsal)*6.895)
        dorsalContact = float(np.count_nonzero(avgDorsal)/plantDorDat.dorsalSensNo*100)
        
         
        ffDorsalContact = float(np.count_nonzero(avgffDorsal)/plantDorDat.dorsalForefootSensNo*100) 
        ffDorsalPressure = float(np.mean(avgffDorsal)*6.895)
        ffDorsalMaxPressure = float(np.max(avgffDorsal)*6.895)
        
        
        mfDorsalContact = float(np.count_nonzero(avgmfDorsal)/plantDorDat.dorsalMidfootSensNo*100)
        mfDorsalPressure = float(np.mean(avgmfDorsal)*6.895)
        mfDorsalMaxPressure = float(np.max(avgmfDorsal)*6.895)
      
        
        instepDorsalContact = float(np.count_nonzero(avginstepDorsal)/plantDorDat.dorsalInstepSensNo*100)
        instepDorsalPressure = float(np.mean(avginstepDorsal)*6.895)
        instepDorsalMaxPressure = float(np.mean(avginstepDorsal)*6.895)

        
        plantarContact = float(np.count_nonzero(avgPlantar)/plantDorDat.LplantarSensNo*100)
        plantarPeakPressure = float(np.max(avgPlantar)*6.895)
        plantarAvgPressure = float(np.mean(avgPlantar)*6.895)
        plantarSDPressure = float(np.std(avgPlantar)*6.895)
        plantarTotalPressure = float(np.sum(avgPlantar)*6.895)
        
        
        toeContact = float(np.count_nonzero(avgToe)/plantDorDat.LplantarToeSensNo*100)
        toePressure = float(np.mean(avgToe)*6.895) 
        
        
        ffContact = float(np.count_nonzero(avgFF)/plantDorDat.LplantarForefootSensNo*100)
        ffPressure = float(np.mean(avgFF)*6.895)
        
        
        mfContact = float(np.count_nonzero(avgMF)/plantDorDat.LplantarMidfootSensNo*100)
        mfPressure = float(np.mean(avgMF)*6.895)
        
        heelContact = float(np.count_nonzero(avgHeel)/plantDorDat.LplantarHeelSensNo*100)
        heelPressure = float(np.mean(avgHeel)*6.895)
        
        
        
        fd_outcomes = pd.DataFrame([[dorsalContact, meanDorsalPressure, maxDorsalPressure,sdDorsalPressure,covDorsalPressure,totalDorsalPressure,
                       ffDorsalContact, ffDorsalPressure, ffDorsalMaxPressure, mfDorsalContact, mfDorsalPressure,mfDorsalMaxPressure,
                       instepDorsalContact, instepDorsalPressure,instepDorsalMaxPressure,plantarContact, plantarAvgPressure, plantarPeakPressure,
                       plantarSDPressure, plantarTotalPressure,toeContact, toePressure, ffContact, ffPressure, mfContact, mfPressure, heelContact, heelPressure]], 
                        
                                   columns=[
                                'dorsalContact', 'meanDorsalPressure','maxDorsalPressure','sdDorsalPressure','covDorsalPressure','totalDorsalPressure',
                             

                                'ffDorsalContact', 'ffDorsalPressure', 'ffDorsalMaxPressure', 'mfDorsalContact', 'mfDorsalPressure', 
                                'mfDorsalMaxPressure', 'instepDorsalContact', 'instepDorsalPressure','instepDorsalMaxPressure',
                             
                                'plantarContact', 'meanPlantarPressure', 'maxPlantarPressure', 'sdPlantarPressure', 'totalPlantarPressure',
                             
                                'toeContact', 'toePressure', 'ffContact', 'ffPressure',
                                'mfContact', 'mfPressure', 'heelContact', 'heelPressure'])
        

        
    else: 
        fd_outcomes =   pd.DataFrame(
                        
                                   columns=[
                                'dorsalContact', 'meanDorsalPressure','maxDorsalPressure','sdDorsalPressure','covDorsalPressure','totalDorsalPressure',
                             

                                'ffDorsalContact', 'ffDorsalPressure', 'ffDorsalMaxPressure', 'mfDorsalContact', 'mfDorsalPressure', 
                                'mfDorsalMaxPressure', 'instepDorsalContact', 'instepDorsalPressure','instepDorsalMaxPressure',
                             
                                'plantarContact', 'meanPlantarPressure', 'maxPlantarPressure', 'sdPlantarPressure', 'totalPlantarPressure',
                             
                                'toeContact', 'toePressure', 'ffContact', 'ffPressure',
                                'mfContact', 'mfPressure', 'heelContact', 'heelPressure'])

        
    if fb_found == 1: 
        
       
        meanCalf = float(np.mean(calf)*6.895)
        pkCalf = float(np.max(calf) *6.895)
        varCalf = float(np.std(calf)  / np.mean(calf) *6.895)
 
      
        meanShin = float(np.mean(shin)*6.895)
        pkShin = float(np.max(shin) *6.895) 
        varShin = float(np.std(shin)  / np.mean(shin) *6.895) 
        
        fb_outcomes =  pd.DataFrame([[subject, config, trial, meanCalf, pkCalf, varCalf, meanShin, pkShin, varShin]], 
                                    columns=['Subject','Config', 'Order', 'avgCalf', 'pkCalf', 'varCalf', 'avgShin','pkShin','varShin'])

    else: 
        fb_outcomes = pd.DataFrame(
                                    columns=['Subject','Config', 'Order', 'avgCalf', 'pkCalf', 'varCalf','avgShin','pkShin','varShin'])

   
   
    

    outcomes = pd.concat([ fb_outcomes,fd_outcomes], axis = 1)   

    
    

                       
          
    outfileName = fPath + '0_CompiledResults_Static.csv'
    if save_on == 1:
        if os.path.exists(outfileName) == False:
            outcomes.to_csv(outfileName, header=True, index = False)
        else:
            outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
            
    
             
    
