
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
import tkinter as tk
from tkinter import messagebox
import scipy.signal as sig
from XSENSORFunctions import (readXSENSORFile, delimitTrial, createTSmat,
                              loadProcessingDecisions, saveProcessingDecision)

save_on = 0
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about trials with no recorded decision (the normal setting)
# 2 = re-review EVERY trial, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied.
data_check = 1

# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'TrailPressure_SLL'

# Read in files
# only read .csv files for this work
fPath = 'Z:\\Testing Segments\\WorkWear\\2025\\2025_Performance_HighCutPFSWorkwearI_TimberlandPro\\Xsensor\\cropped\\'
fileExt = r".csv"
# Data-quality decisions recorded by previous runs of this script. Loading
# these is what stops a trial being judged twice. Rejected files are left
# where they are so they can still be opened and looked at - it is this
# record, not the folder, that decides what gets processed.
decisions = loadProcessingDecisions(fPath, scriptName)

# Results from an out-of-date copy of this script will not match anyone
# else's. Asked only when nothing has been decided here yet: a decision file
# means the question was answered on an earlier run of this dataset.
if len(decisions) == 0:
    # Give this dialog its own window and put it in front. It is the first
    # thing the script does, so unlike the data check further down there is no
    # figure on screen to pull focus, and a borrowed default root can open the
    # box behind Spyder - which just looks like the script has hung.
    dialogRoot = tk.Tk()
    dialogRoot.withdraw()
    dialogRoot.attributes('-topmost', True)
    upToDate = messagebox.askyesno("Question", "Is this script up to date with main?",
                                   parent = dialogRoot)

    # The landings here still come from a peak height threshold, which drops a
    # soft touchdown instead of judging how consistent the landing was. Max's
    # work replaces that detection. Until it is in, results from this script
    # will have to be produced again, so it asks before doing any of it.
    maxDetection = messagebox.askyesno("Question",
                                       "Has this code been updated with new landing detections from Max's work?",
                                       parent = dialogRoot)
    dialogRoot.destroy()

    # Plain errors rather than SystemExit: these scripts get run a cell at a
    # time, and SystemExit can take the console with it. These stop just as
    # firmly, print the reason, and leave the console alone.
    if upToDate == False:
        raise RuntimeError('Pull this script from main before processing')

    if maxDetection == False:
        raise RuntimeError("Bring in the landing detection from Max's work before processing")

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
    # Clear the previous trial's cyclic detections. They are assigned
    # inside conditionals, so a trial that skips a detection would keep
    # the last trial's events and be processed against them silently.
    # A trial rejected in a previous run is skipped. The file is still there
    # to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(fName, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + fName)
        continue
    
    seed_land = []
    seed_ht = []
    land = []
    land_ht = []

    try: 
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
        torder = fName.split(sep = "_")[3].split(sep = '.')[0]
        
        # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        #if ('SLL' in moveTmp):# or ('SLLT' in moveTmp): # or ('Trail' in moveTmp):
        if (moveTmp == 'sll') or ('sllt' in moveTmp):
            print(fName)
            tmpDat = readXSENSORFile(fName,fPath)
            tmpDat = delimitTrial(tmpDat,fName,fPath)
            tmpDat = createTSmat(fName, fPath, tmpDat)
            
            if len(tmpDat.RplantarMat != 0):
                avgFFPress = np.mean(tmpDat.RplantarForefoot, axis = (1,2)) # Alternative detection signal
                pForce = tmpDat.RForce
            
            elif len(tmpDat.LplantarMat != 0):
                avgFFPress = np.mean(tmpDat.LplantarForefoot, axis = (1,2)) # Alternative detection signal
                pForce = tmpDat.LForce    
                
            # A seed pass at a fixed 1000 N. Its only job is to give a
            # height to work from for this subject - these are not the
            # landings and nothing downstream should use them.
            seed_land = sig.find_peaks(pForce, height = 1000, distance = 600)[0]
            seed_ht =  sig.find_peaks(pForce, height = 1000, distance = 600)[1]['peak_heights']
            fft_pk = sig.find_peaks(avgFFPress, height = 10, distance = 500)[0]
            fft_ht = sig.find_peaks(avgFFPress, height = 10, distance = 500)[1]['peak_heights']
            
            htThrsh = np.mean(seed_ht) - 400
                        
            # The landings. The figure you judge and every metric below use
            # these, so what is approved is what is processed.
            land = sig.find_peaks(pForce, height = htThrsh, distance = 500)[0]
            land_ht =  sig.find_peaks(pForce, height = htThrsh, distance = 500)[1]['peak_heights']
       
            answer = True # if data check is off. 
            if priorDecision == 'good' and data_check == 1:
                print('Previously accepted, not asking again: ' + fName)
            elif data_check >= 1:
                plt.figure()
                plt.plot(pForce, label = 'Right Foot Total Force') 
                plt.plot(land, land_ht, marker = 'o', linestyle = 'none')
                for ii in range(len(land)):
                    # Note: do NOT compute sdFz here. It is the list that
                    # collects force variability across every trial, and
                    # assigning to it replaces the list with a single number -
                    # the next sdFz.append() then raises AttributeError, which
                    # the except below reports as 'Not usable data'.
                    avgF = movAvgForce(pForce,land[ii] , land[ii] + 200 , 10)
                    sdF = movSDForce(pForce, land[ii], land[ii] + 200, 10)
                    subBW = findBW(avgF)
                    stabArg = findStabilization(avgF,sdF,subBW)+land[ii]
                    plt.scatter(stabArg, 1000, marker = 'x', color = 'red')
                    
                answer = messagebox.askyesno("Question","Is data clean?") 
                saveFolder = fPath + 'SLLtDetections'
                if os.path.exists(saveFolder) == False:
                  os.mkdir(saveFolder) 
                  if answer == True:
                      plt.savefig(saveFolder + '/' + fName.split('.csv')[0] +'.png')
                      plt.close('all')
            
      
      
            if answer == False:
                # Look again using this trial's own rhythm for the spacing
                # instead of a flat 500 samples. Measured from the seed pass,
                # not from land: land has already had peaks removed, so its
                # gaps overstate the real cadence.
                # Note: np.mean(seed_land) was used here, which is the middle
                # of the trial in samples, not a spacing at all.
                if len(seed_land) > 1:
                    disThrsh = np.mean(np.diff(seed_land)) - 100
                else:
                    disThrsh = 500

                land = sig.find_peaks(pForce, height = htThrsh, distance = disThrsh)[0]
                land_ht =  sig.find_peaks(pForce, height = htThrsh, distance = disThrsh)[1]['peak_heights']
                plt.figure()
                plt.plot(pForce, label = 'Right Foot Total Force') 
                plt.plot(land, land_ht, marker = 'o', linestyle = 'none')
                plt.plot(range(len(avgFFPress)), avgFFPress)
                plt.plot(fft_pk, fft_ht, marker = 'v', linestyle = 'none')
                answer = messagebox.askyesno("Question","Is data clean?") 
                plt.close('all')

            # Recorded after the retry above, not before it: that second look
            # can turn the answer back to yes, and the record has to match what
            # the trial was actually processed as
            # Record the decision straight away, inside the loop, so a run that
            # fails later keeps every decision made so far
            if priorDecision == '' or data_check == 2:
                if answer == False:
                    saveProcessingDecision(fPath, scriptName, fName, 'bad',
                                           Subject=subName, Config=ConfigTmp,
                                           reason='rejected at the data check')
                else:
                    saveProcessingDecision(fPath, scriptName, fName, 'good',
                                           Subject=subName, Config=ConfigTmp)

            if answer == False:
                print('Adding file to bad file list')
                badFileList.append(fName)
                
            if answer == True:
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
            print('Not usable data ' + fName)             
        
            
            
outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(order), 'StabTime': list(stabilization),
                 'ToeClaw': list(toeClawAvg), 'ToeClawPeak': list(toeClawPk),  'ForefootAvg' : list(ffAvg), 'ForefootPeak' : list(ffPk), 'ForefootContA' : list(ffConArea), 
                 'HeelConArea' : list(heelArea), 'HeelPressure' : list(heelPres)})


                   
outfileName = fPath + '0_CompiledResults_SLL.csv'
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    outcomes.to_csv(outfileName, header=True, index = False)


