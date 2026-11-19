# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import scipy
import scipy.signal as sig
from XSENSORFunctions import (readXSENSORFile, createTSmat,
                              loadProcessingDecisions, saveProcessingDecision)


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
scriptName = 'Cycling_XSENSOR_IMU'

# Read in files
# only read .csv files for this work
fPath = 'Z:/Testing Segments/Cycling/2025/2025_Performance_CyclingLacevBOA_Specialized/Xsensor/'
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
    dialogRoot.destroy()
    # A plain error rather than SystemExit: these scripts get run a cell at a
    # time, and SystemExit can take the console with it. This stops just as
    # firmly, prints the reason, and leaves the console alone.
    if upToDate == False:
        raise RuntimeError('Pull this script from main before processing')

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]

# list of functions 
def delimitTrialCycling(inputDF,FName,FilePath):
    """
    This function uses ginput to delimit the start and end of a trial
    You will need to indicate on the plot when to start/end the trial. 
    You must use tkinter or plotting outside the console to use this function
    Parameters
    ----------
    inputDF : Pandas DataFrame
        DF containing all desired output variables.
    fName : str 
        File name
    FilePath : str
        Path of the desired file

    Returns
    -------
    outputDat: Pandas DataFrame
        dataframe subset to the beginning and end of the cycling data
    fsprintStart: int
        start of the sprinting

    """
    
    # generic function to plot and start/end trial #
    if os.path.exists(FilePath+FName+'TrialSeg.npy'):
        trial_segment_old = np.load(fPath+FName+'TrialSeg.npy', allow_pickle =True)
        trialStart = trial_segment_old[0]
        trialEnd = trial_segment_old[1]
        fsprintStart = trial_segment_old[2]
        inputDF = inputDF.iloc[trialStart : trialEnd,:]
        outputDat = inputDF.reset_index(drop = True)
        
    else:         
        # Select the cropped regions of the trial
        insoleSide = inputDF['Insole'][0]
                   
        if (insoleSide == 'Left'): 
            # Left side
            totForce = inputDF['Est. Load (lbf)']*4.44822
        else:  
            totForce = inputDF['Est. Load (lbf).1']*4.44822
         
        fig, ax = plt.subplots()
        print('Select a point on the plot to represent the beginning & end of trial')
        ax.plot(totForce, label = 'Total Force')
        fig.legend()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        outputDat = outputDat.reset_index(drop = True)
        
        # Select the start of the sprinting
        if (insoleSide == 'Left'): 
            # Left side
            totForce = outputDat['Est. Load (lbf)']*4.44822
        else:  
            totForce = outputDat['Est. Load (lbf).1']*4.44822
        print('Select start of sprint')
        plt.plot(totForce)
        plt.ylabel('Total Force')
        fsprintStart = plt.ginput(1)
        fsprintStart = round(fsprintStart[0][0])
        plt.close('all')
        
        trial_segment = np.array([round(pts[0,0]),round(pts[1,0]),fsprintStart], dtype = object)
        np.save(FilePath+FName+'TrialSeg.npy',trial_segment)

    return(outputDat, fsprintStart)

def filtSig(sig_in,cut,t):
    """
    Create a 2nd order filter

    Parameters
    ----------
    sig_in : numpy array
        signal intended for filtering
    cut : float
        cut off frequency
    t : numpy array
        time array

    Returns
    -------
    sig_out : numpy array
        sig_in with a 2nd order filter with the specified cut-off frequency
        applied

    """
    # Set up a 2nd order low pass buttworth filter
    freq = 1/np.mean(np.diff(t))
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    # Filter the IMU signals
    sig_out = sig.filtfilt(b, a, sig_in)  
    return(sig_out)

def findCycleEvents(fAccx,fTime):
    """
    Determine top dead center (TDC) and bottom dead center (BDC) from the
    acceleration sensor within the XSENSOR SPK

    Parameters
    ----------
    fAccx : numpy array
        Acceleration in the X direction
    fTime : numpy array
        Time from the SPK

    Returns
    -------
    fTDC : numpy array
        Top Dead Center Detection
    fBDC : numpy array
        Bottom Dead Center Detection

    """
    # Filter the acceleration with a 2 Hz filter
    detsig = filtSig(fAccx,2,fTime)
    # Top Dead Center (TDC) Detection
    fTDC,_ = sig.find_peaks(detsig, height = -0.5, distance = 30)
    # Bottom Dead Center (BDC) Detection
    fBDC,_ = sig.find_peaks(-detsig, height = 1, distance = 30)
    if fBDC[0] < fTDC[0]:
        fBDC = fBDC[1:]
    [fTDC,fBDC] = EnsureCycleEventsAlternate(detsig,fTDC,fBDC)
    
    return(fTDC,fBDC)

def EnsureCycleEventsAlternate(cycledet,locmax,locmin):
    """
    Alternate between events within a cycle to make sure that local m

    Parameters
    ----------
    cycledet : numpy array
        Signal associated with the detected peaks from locmax
    locmax : numpy array
        Peaks detected from the function "find_peaks" (scipy library) from turndet1
    locmin : numpy array
        Peaks detected from the function "find_peaks" (scipy library) from turndet2

    Returns
    -------
    locmax : numpy array
        Cleaned locmax that oscillates with locmin
    locmin : numpy array
        Cleaned locmin that oscillates with locmax

    """  
    # Ensure that the entire signal oscillates between locmax and locmin
    jj = 0
    it = 0
    while jj <= len(locmin)-2 and it < 1000:
        it = it + 1
        if locmax[jj] < locmin[jj] and locmax[jj+1] > locmin [jj] and locmax[jj+1] < locmin [jj+1]:
            # A normal turn
            jj = jj+1
        else:
            if locmin[jj+1] < locmax[jj+1]:
                # Multiple detected peaks from the following signal before a peak from the leading signal
                idx_multiple = np.where((locmin>locmax[jj])*(locmin<locmax[jj+1])==True)[0]
                # Figure out which of multiple peaks is higher
                idx = np.argmin(cycledet[locmin[idx_multiple]])
                # Remove the other detected peaks
                idx_remove = np.delete(idx_multiple,idx)
                locmin = np.delete(locmin,idx_remove)
            else:
                # Multiple detected peaks from the leading signal before a peak from the following signal
                if jj == 0:
                    idx_multiple = np.where(locmax<locmin[jj])[0]  
                else:
                    idx_multiple = np.where((locmax>locmin[jj-1])*(locmax<locmin[jj])==True)[0]    
                idx = np.argmax(cycledet[locmax[idx_multiple]])
                # Remove the other detected peaks
                idx_remove = np.delete(idx_multiple,idx)
                locmax = np.delete(locmax,idx_remove)
        
    return(locmax,locmin)

def intp_strokes(var,TDC):
    """
    Function to interpolate the variable of interest across a stride
    (from foot contact to subsiquent foot contact) in order to plot the 
    variable of interest over top each other

    Parameters
    ----------
    var : list or numpy array
        Variable of interest. Can be taken from a dataframe or from a numpy array
    landings : list
        Foot contact indicies

    Returns
    -------
    intp_var : numpy array
        Interpolated variable to 101 points with the number of columns dictated
        by the number of strides.

    """
    # Preallocate
    intp_var = np.zeros((101,len(TDC)-1))
    # Index through the strides
    for ii in range(len(TDC)-1):
        dum = var[TDC[ii]:TDC[ii+1]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,ii] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var
   
def plotAvgCyclePressure(inputDC, FilePath, LTDC, LBDC, RTDC, RBDC):
    """
    Plot the average pressure at the down stroke and the up stroke

    Every foot that carries data is drawn: dorsal pressure in the first column,
    then one column per plantar insole. A trial recorded with both insoles gets
    both side by side, because judging a two-insole trial on one foot alone
    hides a failure in the other - which is the thing the data check exists to
    catch.

    Function dependencies: need to use createTSmat for the appropriate array
    shape for the plantar and dorsal pressure

    Parameters
    ----------
    inputDC : dataclass
        Created from the function "createTSmat"
    FilePath : str
        file path string
    LTDC : numpy array or list
        left top dead center detections. Empty when there is no left insole.
    LBDC : numpy array or list
        left bottom dead center detections
    RTDC : numpy array or list
        right top dead center detections. Empty when there is no right insole.
    RBDC : numpy array or list
        right bottom dead center detections

    Returns
    -------
    fig : matplotlib figure
        Figure showing average dorsal and plantar pressure over the stroke.
        None when the trial has no plantar pressure at all.

    """
    # Work out which feet were actually recorded. Both insoles, one, or neither
    # are all possible, and the figure has to cope with each
    feetNames = []
    feetMats = []
    feetTDC = []
    feetBDC = []
    # Fewer than two detections is not a stroke, so that foot has nothing to
    # average - treat it as absent rather than meaning a panel of NaNs
    if len(inputDC.LplantarMat) != 0 and len(LTDC) > 1:
        feetNames.append('Left Plantar')
        feetMats.append(inputDC.LplantarMat)
        feetTDC.append(LTDC)
        feetBDC.append(LBDC)
    if len(inputDC.RplantarMat) != 0 and len(RTDC) > 1:
        feetNames.append('Right Plantar')
        feetMats.append(inputDC.RplantarMat)
        feetTDC.append(RTDC)
        feetBDC.append(RBDC)

    # Say so rather than leaving a blank screen in front of the data check
    # question - a silent empty plot reads as "nothing wrong"
    if len(feetNames) == 0:
        print('No plantar pressure detected for this trial, nothing plotted')
        return None

    # Average each foot over the down stroke and the up stroke. One stroke runs
    # from a top dead center to the next, so there is one fewer stroke than
    # there are detections - preallocating len(TDC) would leave a row of zeros
    # in the mean and pull every average down.
    downPlantarAvg = []
    upPlantarAvg = []
    for ff in range(len(feetNames)):
        thisMat = feetMats[ff]
        thisTDC = feetTDC[ff]
        thisBDC = feetBDC[ff]
        nStrokes = len(thisTDC) - 1
        # Preallocate
        downPlantar = np.zeros([nStrokes, 31, 9])
        upPlantar = np.zeros([nStrokes, 31, 9])
        for ii in range(nStrokes):
            downPlantar[ii,:,:] = np.mean(thisMat[thisTDC[ii]:thisBDC[ii],:,:],axis=0)
            upPlantar[ii,:,:] = np.mean(thisMat[thisBDC[ii]:thisTDC[ii+1],:,:],axis=0)
        downPlantarAvg.append(np.mean(downPlantar, axis = 0))
        upPlantarAvg.append(np.mean(upPlantar, axis = 0))

    # Dorsal is a single sensor, so it is averaged over the first available
    # foot's crank timing. Note: with two insoles the dorsal column follows
    # that foot alone.
    refTDC = feetTDC[0]
    refBDC = feetBDC[0]
    nStrokes = len(refTDC) - 1
    # Preallocate
    downDorsal = np.zeros([nStrokes, 18, 10])
    upDorsal = np.zeros([nStrokes, 18, 10])
    for ii in range(nStrokes):
        downDorsal[ii,:,:] = np.mean(inputDC.dorsalMat[refTDC[ii]:refBDC[ii],:,:],axis=0)
        upDorsal[ii,:,:] = np.mean(inputDC.dorsalMat[refBDC[ii]:refTDC[ii+1],:,:],axis=0)

    downDorsalAvg = np.mean(downDorsal, axis = 0)
    upDorsalAvg = np.mean(upDorsal, axis = 0)

    # One row per stroke, one column for dorsal plus one per foot
    nCols = 1 + len(feetNames)
    fig, axes = plt.subplots(2, nCols, squeeze = False)
    rowNames = ['Average Down Stroke', 'Average Up Stroke']
    dorsalAvgs = [downDorsalAvg, upDorsalAvg]
    plantarAvgs = [downPlantarAvg, upPlantarAvg]

    # Both rows are scaled to the down stroke, so the drop from down to up is
    # visible rather than normalised away. The feet share a scale for the same
    # reason: an asymmetry between them is the thing worth seeing.
    dorsalMax = np.max(downDorsalAvg)
    plantarMax = 0
    for ff in range(len(feetNames)):
        thisMax = np.max(downPlantarAvg[ff])
        if thisMax > plantarMax:
            plantarMax = thisMax

    for rr in range(2):
        thisDorsal = dorsalAvgs[rr]
        dorsalAx = axes[rr][0]
        sns.heatmap(thisDorsal, ax = dorsalAx, cmap = 'mako', vmin = 0,
                    vmax = dorsalMax)
        dorsalAx.set_ylabel(rowNames[rr])
        if rr == 0:
            dorsalAx.set_title('Dorsal Pressure')

        for ff in range(len(feetNames)):
            thisPlantar = plantarAvgs[rr][ff]
            plantarAx = axes[rr][ff+1]
            sns.heatmap(thisPlantar, ax = plantarAx, cmap = 'mako', vmin = 0,
                        vmax = plantarMax)
            if rr == 0:
                plantarAx.set_title(feetNames[ff])

    fig.set_size_inches(2.5*nCols, 8)

    plt.suptitle(inputDC.subject +' '+ inputDC.movement +' '+ inputDC.config)
    plt.tight_layout()
    plt.margins(0.1)

    saveFolder= FilePath + '2DPlots'

    if os.path.exists(saveFolder) == False:
        os.mkdir(saveFolder)

    plt.savefig(saveFolder + '/' + inputDC.subject +' '+ inputDC.movement +' '+ inputDC.config + '.png')
    return fig
#############################################################################################################################################################


badFileList = []
config = []
subject = []
st = []
movement = []
Order = []

heelAreaUp = []
heelAreaDown = []
maxmaxToes = []

dorsalVar = []
maxDorsal = []

ffDorsalMax=[]
mfDorsalMax=[]
instepMax=[]

for fName in entries:
    print(fName)    
    # Clear the previous trial's cyclic detections. They are assigned
    # inside conditionals, so a trial that skips a detection would keep
    # the last trial's events and be processed against them silently.
    LTDC = []
    LBDC = []
    RTDC = []
    RBDC = []

    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    trialTmp = fName.split(sep = "_")[3].split(sep = ".")[0]

    # A trial rejected in a previous run is skipped. The file is still there
    # to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(fName, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + fName)
        continue
    
    tmpDat = readXSENSORFile(fName,fPath)
    [tmpDat,sprintStart] = delimitTrialCycling(tmpDat,fName,fPath)
    tmpDat = createTSmat(fName, fPath, tmpDat)
    
    # Find cycling events
    if len(tmpDat.LplantarMat) != 0:
        [LTDC,LBDC] = findCycleEvents(tmpDat.Lacc[:,0],tmpDat.time)
    
    if len(tmpDat.RplantarMat) != 0:
        [RTDC,RBDC] = findCycleEvents(tmpDat.Racc[:,0],tmpDat.time)

    answer = True # if data check is off. 
    if priorDecision == 'good' and data_check == 1:
        print('Previously accepted, not asking again: ' + fName)
    elif data_check >= 1:
        # Draw every foot that was recorded, then the force traces, and only
        # then ask. The question is asked once per trial, so everything the
        # answer depends on has to be on screen before it appears.
        plotAvgCyclePressure(tmpDat, fPath, LTDC, LBDC, RTDC, RBDC)

        # Save a force trace per foot, not just whichever one came first
        saveFolder= fPath + 'ForcePlots'
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)

        if len(tmpDat.LplantarMat) != 0 and len(LTDC) != 0:
            plt.figure()
            plt.plot(intp_strokes(tmpDat.LForce,LTDC))
            plt.xlabel('% Stroke')
            plt.ylabel('Left Insole Force (N)')
            plt.savefig(saveFolder + '/' + subName + '_' + ConfigTmp + '_' + trialTmp + '_Left.png')

        if len(tmpDat.RplantarMat) != 0 and len(RTDC) != 0:
            plt.figure()
            plt.plot(intp_strokes(tmpDat.RForce,RTDC))
            plt.xlabel('% Stroke')
            plt.ylabel('Right Insole Force (N)')
            plt.savefig(saveFolder + '/' + subName + '_' + ConfigTmp + '_' + trialTmp + '_Right.png')

        answer = messagebox.askyesno("Question","Is data clean?")
        
    plt.close('all')
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
        # Detect if left or right insole was used
        if len(tmpDat.LplantarMat) != 0:
            TDC = LTDC
            BDC = LBDC
            plantarMat = tmpDat.LplantarMat
            plantarHeel = tmpDat.LplantarHeel
            plantarToe = tmpDat.LplantarToe
            HeelSensNo = tmpDat.LplantarHeelSensNo
            ToeSensNo = tmpDat.LplantarToeSensNo
            
        else:
            TDC = RTDC
            BDC = RBDC
            plantarMat = tmpDat.RplantarMat
            plantarHeel = tmpDat.RplantarHeel
            plantarToe = tmpDat.RplantarToe
            HeelSensNo = tmpDat.RplantarHeelSensNo
            ToeSensNo = tmpDat.RplantarToeSensNo 

        for ii in range(len(TDC)-1):            
            # try: 
            heelArea = np.count_nonzero(plantarHeel[BDC[ii]:TDC[ii+1], :, :], axis = (1,2))
            heelAreaUp.append(np.mean(heelArea)/HeelSensNo*100) 
            hAreaDown = np.count_nonzero(plantarHeel[TDC[ii]:BDC[ii], :, :], axis = (1,2))
            heelAreaDown.append(np.mean(hAreaDown)/HeelSensNo*100)
            maxmaxToes.append(np.max(plantarToe[TDC[ii]:BDC[ii]])*6.895)
            # Stroke Time    
            st.append(tmpDat.time[TDC[ii+1]]-tmpDat.time[TDC[ii]])
                
            if len(tmpDat.dorsalMat) != 0:    
                dorsalVar.append(np.std(tmpDat.dorsalMat[TDC[ii]:TDC[ii+1], :, :])/np.mean(tmpDat.dorsalMat[TDC[ii]:TDC[ii+1], :, :]))
                maxDorsal.append(np.max(tmpDat.dorsalMat[TDC[ii]:TDC[ii+1], :, :])*6.895)
                    
                ffDorsalMax.append(np.max(tmpDat.dorsalForefoot[TDC[ii]:TDC[ii+1], :, :])*6.895)
                mfDorsalMax.append(np.max(tmpDat.dorsalMidfoot[TDC[ii]:TDC[ii+1], :, :])*6.895)
                instepMax.append(np.max(tmpDat.dorsalInstep[TDC[ii]:TDC[ii+1], :, :])*6.895)
            else:
                dorsalVar.append('nan')
                maxDorsal.append('nan')
                    
                ffDorsalMax.append('nan')
                mfDorsalMax.append('nan')
                instepMax.append('nan')
                
            if RTDC[ii] < sprintStart:
                movement.append('Steady')
            else:
                movement.append('Sprint')
            config.append(tmpDat.config)
            subject.append(tmpDat.subject)
            Order.append(trialTmp)
                    
            
            # except: print(fName + 'Bad stroke ' + str(ii))
            
outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(Order), 'StrokeTime':list(st),
                          'maxmaxToes':list(maxmaxToes),
                          'heelArea_Up':list(heelAreaUp), 'heelArea_Down':list(heelAreaDown),
                          'dorsalVar':list(dorsalVar), 'maxDorsalP':list(maxDorsal),
                          'ffDorsalMax':list(ffDorsalMax), 'mfDorsalMax':list(mfDorsalMax), 'instepMax':list(instepMax)
                         
                          })

outfileName = fPath + '0_CompiledResults.csv'
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates every
    # row when a study is re-processed after more data has been collected.
    outcomes.to_csv(outfileName, header=True, index = False)
                
        
        
               
            
            
            
            
            
        


