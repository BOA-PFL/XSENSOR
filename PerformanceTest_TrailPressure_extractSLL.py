
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
from XSENSORFunctions import readXSENSORFile, delimitTrial, createTSmat

save_on = 0
data_check = 1

# Read in files
# only read .csv files for this work
fPath = 'Z:\\Testing Segments\\WorkWear_Performance\\2025_Performance_HighCutPFSWorkwearI_TimberlandPro\\Xsensor\\cropped\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]

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

###############################################################################
# Function list specific for this code

# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force, fThreshold):
    """
    The purpose of this function is to determine the landings (foot contacts)
    events when the vertical force exceeds the force threshold
    
    Parameters
    ----------
    force : list
        vertical ground reaction force. 
    
    fThreshold : float
        threshold to detect landings
    
    Returns
    -------
    ric : list
        indices of the landings (foot contacts)

    """
    ric = []
    for step in range(len(force)-1):
        if force[step] < fThreshold and force[step + 1] >= fThreshold:
            ric.append(step)
    return ric


#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force, fThreshold):
    """
    The purpose of this function is to determine the take-off
    events when the vertical force exceeds the force threshold

    Parameters
    ----------
    force : list
        vertical ground reaction force. 
    
    fThreshold : float
        threshold to detect landings
    
    Returns
    -------
    ric : list
        indices of the landings (foot contacts)

    """
    rto = []
    for step in range(len(force)-1):
        if force[step] >= fThreshold and force[step + 1] < fThreshold:
            rto.append(step + 1)
    return rto

def findStabilization(rollavgF, rollsdF, BW): # Include body weight here.
    """
    Using the rolling average and SD values, this calcualtes when the 
    actual stabilized force occurs. 
    
    Parameters
    ----------
    rollavgF : list, calculated using movAvgForce 
        rolling average of pressure.
    rollsdF : list, calcualted using movSDForce above
        rolling SD of pressure.
    BW : float
        Subject body weight calculated from the XSENSOR insole

    Returns
    -------
    floating point number
        Time to stabilize using the heuristic: pressure is within +/- 5% of subject
        mass and the rolling standrd deviation is below 20

    """
    stab = []
    for step in range(len(rollavgF)-1):
        if rollavgF[step] >= (BW - 0.05*BW) and rollavgF[step] <= (BW + 0.05*BW) and rollsdF[step] < 20:
            stab.append(step + 1)
            
    return stab[0] 

def movAvgForce(force, landing, takeoff, length):
    """
    In order to estimate when someone stabilized, we calcualted the moving
    average force and SD of the force signal. This is one of many published 
    methods to calcualte when someone is stationary. 
    
    Parameters
    ----------
    force : Pandas series
        pandas series of force from pressure insoles.
    landing : List
        list of landings calcualted from findLandings.
    takeoff : List
        list of takeoffs from findTakeoffs.
    length : Integer
        length of time in indices to calculate the moving average.

    Returns
    -------
    rollavgF : list
        smoothed average force .

    """
    newForce = np.array(force)
    win_len = length; #window length for steady standing
    rollavgF = []
    for i in range(landing, takeoff):
        rollavgF.append(np.mean(newForce[i : i + win_len]))     
    return rollavgF

#moving SD as calcualted above
def movSDForce(force, landing, takeoff, length):
    """
    This function calculates a rolling standard deviation over an input
    window length
    
    Parameters
    ----------
    force : Pandas series
        pandas series of force from pressure insoles.
    landing : List
        list of landings calcualted from findLandings.
    takeoff : List
        list of takeoffs from findTakeoffs.
    length : Integer
        length of time in indices to calculate the moving average.

    Returns
    -------
    rollavgF : list
        smoothed rolling SD of pressure

    """
    newForce = np.array(force)
    win_len = length; #window length for steady standing
    rollavgF = []
    for i in range(landing, takeoff):
        rollavgF.append(np.std(newForce[i : i + win_len]))     
    return rollavgF

#estimated stability after 200 indices
def findBW(force):
    """
    If you do not have the subject's body weight or want to find from the 
    steady portion of force, this may be used. This is highly conditional on 
    the data and how it was collected. The below assumes quiet standing from
    100 to 200 indices. 
    
    Parameters
    ----------
    force : Pandas series
        DESCRIPTION.

    Returns
    -------
    BW : floating point number
        estimate of body weight of the subject to find stabilized weight
        in Newtons

    """
    BW = np.mean(avgF[100:200])
    return BW

#############################################################################################################################################################

badFileList = []
config = []
subject = []
ct = []
movement = []
order = []


toeClawAvg = []
toeClawPk =[]
toeLatAvg = []
toeLatPk =[]
toeMedAvg = []
toeMedPk =[]

ffAvg = []
ffPk = []
ffConArea = []
ffLatAvg = []
ffLatPk = []
ffMedAvg = []
ffMedPk = []

heelArea = []
heelPres = []

sdFz = []
avgF = []
sdF = []
subBW = [] 

stabilization = []

for fName in entries:
    try: 
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
        torder = fName.split(sep = "_")[3].split(sep = '.')[0]
        
        # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        #if ('SLL' in moveTmp):# or ('SLLT' in moveTmp): # or ('Trail' in moveTmp):
        if (moveTmp == 'sll') or ('sllt' in moveTmp):
            tmpDat = readXSENSORFile(fName,fPath)
            tmpDat = delimitTrial(tmpDat,fName,fPath)
            tmpDat = createTSmat(fName, fPath, tmpDat)
            
            if len(tmpDat.RplantarMat != 0):
                avgFFPress = np.mean(tmpDat.RplantarForefoot, axis = (1,2)) # Alternative detection signal
                pForce = tmpDat.RForce
            
            elif len(tmpDat.LplantarMat != 0):
                avgFFPress = np.mean(tmpDat.LplantarForefoot, axis = (1,2)) # Alternative detection signal
                pForce = tmpDat.LForce    
                
            land = sig.find_peaks(pForce, height = 1000, distance = 600)[0]
            land_ht =  sig.find_peaks(pForce, height = 1000, distance = 600)[1]['peak_heights']
            fft_pk = sig.find_peaks(avgFFPress, height = 10, distance = 500)[0]
            fft_ht = sig.find_peaks(avgFFPress, height = 10, distance = 500)[1]['peak_heights']
            
            htThrsh = np.mean(land_ht) - 400
                        
            true_land = sig.find_peaks(pForce, height = htThrsh, distance = 500)[0]
            land_ht =  sig.find_peaks(pForce, height = htThrsh, distance = 500)[1]['peak_heights']
       
            saveFolder= fPath + 'SLLtDetections'
            answer = True # if data check is off. 
            if data_check == 1:
                plt.figure()
                plt.plot(pForce, label = 'Right Foot Total Force') 
                plt.plot(true_land, land_ht, marker = 'o', linestyle = 'none')
                # plt.plot(range(len(avgFFPress)), avgFFPress)
                # plt.plot(fft_pk, fft_ht, marker = 'v', linestyle = 'none')
                answer = messagebox.askyesno("Question","Is data clean?") 
                saveFolder = fPath + 'SLLtDetections'
                if os.path.exists(saveFolder) == False:
                  os.mkdir(saveFolder) 
      
      
            if answer == False:
                disThrsh = np.mean(land)- 100
                true_land = sig.find_peaks(pForce, height = htThrsh, distance = disThrsh)[0]
                land_ht =  sig.find_peaks(pForce, height = htThrsh, distance = disThrsh)[1]['peak_heights']
                plt.figure()
                plt.plot(pForce, label = 'Right Foot Total Force') 
                plt.plot(land, land_ht, marker = 'o', linestyle = 'none')
                plt.plot(range(len(avgFFPress)), avgFFPress)
                plt.plot(fft_pk, fft_ht, marker = 'v', linestyle = 'none')
                answer = messagebox.askyesno("Question","Is data clean?") 
            
            if answer == False:
                plt.close('all')
                print('Adding file to bad file list')
                badFileList.append(fName)
                
            if answer == True:
                plt.savefig(saveFolder + '/' + fName.split('.csv')[0] +'.png')
                plt.close('all')
                print('Estimating point estimates')
                
                if len(tmpDat.RplantarMat != 0):
                    for ii in range(len(land)):
                        sdFz.append(np.std(pForce [ land[ii] + 100 : land[ii] + 400]))
                        avgF = movAvgForce(pForce,land[ii] , land[ii] + 200 , 10)
                        sdF = movSDForce(pForce, land[ii], land[ii] + 200, 10)
                        subBW = findBW(avgF)
                        try:
                            stabilization.append(tmpDat.time[findStabilization(avgF,sdF,subBW)+land[ii]]-tmpDat.time[land[ii]])
                        except:
                            stabilization.append('NaN')
                            
                        toeClawAvg.append(np.mean(tmpDat.RplantarToe[land[ii]: land[ii]+100]))
                        toeClawPk.append(np.max(tmpDat.RplantarToe[land[ii]: land[ii]+100]))
    
                        ffAvg.append(np.mean(tmpDat.RplantarForefoot[land[ii]: land[ii]+100]))
                        ffPk.append(np.max(tmpDat.RplantarForefoot[land[ii]: land[ii]+100]))
                        ffConArea.append(np.count_nonzero(tmpDat.RplantarForefoot[land[ii]: land[ii]+100])/100/tmpDat.RplantarForefootSensNo*100) # divided by 100 frames
    
                        heelArea.append(np.count_nonzero(tmpDat.RplantarHeel[land[ii]: land[ii]+100])/ 100/tmpDat.RplantarLateralSensNo*100) # divided by 100 frames
                        heelPres.append(np.mean(tmpDat.RplantarHeel[land[ii]: land[ii]+100]))
                        
                        config.append(tmpDat.config)
                        subject.append(tmpDat.subject)
                        movement.append(moveTmp)
                        order.append(torder)
                        
                elif len(tmpDat.LplantarMat != 0):
                        sdFz.append(np.std(pForce [ land[ii] + 100 : land[ii] + 400]))
                        avgF = movAvgForce(pForce,land[ii] , land[ii] + 200 , 10)
                        sdF = movSDForce(pForce, land[ii], land[ii] + 200, 10)
                        subBW = findBW(avgF)
                        try:
                            stabilization.append(tmpDat.time[findStabilization(avgF,sdF,subBW)+land[ii]]-tmpDat.time[land[ii]])
                        except:
                            stabilization.append('NaN')
    
                        for ii in range(len(land)):
                            toeClawAvg.append(np.mean(tmpDat.LplantarToe[land[ii]: land[ii]+100]))
                            toeClawPk.append(np.max(tmpDat.LplantarToe[land[ii]: land[ii]+100]))
    
                            ffAvg.append(np.mean(tmpDat.LplantarForefoot[land[ii]: land[ii]+100]))
                            ffPk.append(np.max(tmpDat.LplantarForefoot[land[ii]: land[ii]+100]))
                            ffConArea.append(np.count_nonzero(tmpDat.LplantarForefoot[land[ii]: land[ii]+100])/100/tmpDat.LplantarForefootSensNo*100) # divided by 100 frames
    
                            heelArea.append(np.count_nonzero(tmpDat.LplantarHeel[land[ii]: land[ii]+100])/ 100/tmpDat.LplantarLateralSensNo*100) # divided by 100 frames
                            heelPres.append(np.mean(tmpDat.LplantarHeel[land[ii]: land[ii]+100]))
                            
                            config.append(tmpDat.config)
                            subject.append(tmpDat.subject)
                            movement.append(moveTmp)
                            order.append(torder)
            
    except:
            print('Not usable data' + fName)             
        
            
            
outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(order), 'StabTime': list(stabilization),
                 'ToeClaw': list(toeClawAvg), 'ToeClawPeak': list(toeClawPk),  'ForefootAvg' : list(ffAvg), 'ForefootPeak' : list(ffPk), 'ForefootContA' : list(ffConArea), 
                 'HeelConArea' : list(heelArea), 'HeelPressure' : list(heelPres)})


                   
outfileName = fPath + '0_CompiledResults_SLL.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
    
        outcomes.to_csv(outfileName, header=True, index = False)

    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 


