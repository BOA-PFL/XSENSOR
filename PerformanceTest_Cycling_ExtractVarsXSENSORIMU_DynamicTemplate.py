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
from tkinter import messagebox
import scipy
import scipy.signal as sig
from XSENSORFunctions import readXSENSORFile, createTSmat 


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
data_check = 0

# Read in files
# only read .csv files for this work
fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/Cycling Performance Tests/2025_Performance_CyclingLacevBOA_Specialized/Xsensor/'
fileExt = r".csv"
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
   
def plotAvgCyclePressure(plantarMat, inputDC, FilePath, fTDC, fBDC):
    """
    Plot the average pressure across all detections at top dead center (TDC)
    and bottom dead center (BDC). 
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
    fTDC : numpy array
        Top dead center detection
    fBDC : numpy array
        Bottom dead center detection

    Returns
    -------
    fig : matplotlib figure
        Figure showing average dorsal and plantar pressure at TDC and BDC

    """  
    # Define a blank arrays for down & upstrokes
    downPlantar = np.zeros([len(fTDC), 31, 9])
    upPlantar = np.zeros([len(fTDC), 31, 9])
    
    downDorsal = np.zeros([len(fTDC), 18, 10])
    upDorsal = np.zeros([len(fTDC), 18, 10])
    
    # Compute the average downstroke and upstroke pressure for plotting
    for ii in range(len(fTDC)-1):
        downPlantar[ii,:,:] = np.mean(plantarMat[fTDC[ii]:fBDC[ii],:,:],axis=0)
        upPlantar[ii,:,:] = np.mean(plantarMat[fBDC[ii]:fTDC[ii+1],:,:],axis=0)
        downDorsal[ii,:,:] = np.mean(inputDC.dorsalMat[fTDC[ii]:fBDC[ii],:,:],axis = 0)
        upDorsal[ii,:,:] = np.mean(inputDC.dorsalMat[fBDC[ii]:fTDC[ii+1],:,:],axis = 0)
        
    downPlantarAvg = np.mean(downPlantar, axis = 0)
    upPlantarAvg = np.mean(upPlantar, axis = 0)
    downDorsalAvg = np.mean(downDorsal, axis = 0)
    upDorsalAvg = np.mean(upDorsal, axis = 0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1 = sns.heatmap(downDorsalAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(downDorsalAvg))
    ax1.set_title('Dorsal Pressure') 
    ax1.set_ylabel('Average Down Stroke')
    
    
    ax2 = sns.heatmap(downPlantarAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(downPlantarAvg))
    ax2.set_title('Plantar Pressure') 
    
    ax3 = sns.heatmap(upDorsalAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(upDorsalAvg))
    ax3.set_ylabel('Average Up Stroke')
    
    ax4 = sns.heatmap(upPlantarAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(downPlantarAvg)) # Max from the downstroke to see the differences between up and down
    
    fig.set_size_inches(5, 8)
    
    plt.suptitle(inputDC.subject +' '+ inputDC. movement +' '+ inputDC.config)
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
    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    trialTmp = fName.split(sep = "_")[3].split(sep = ".")[0]
    
    tmpDat = readXSENSORFile(fName,fPath)
    [tmpDat,sprintStart] = delimitTrialCycling(tmpDat,fName,fPath)
    tmpDat = createTSmat(fName, fPath, tmpDat)
    
    # Find cycling events
    if len(tmpDat.LplantarMat) != 0:
        [LTDC,LBDC] = findCycleEvents(tmpDat.Lacc[:,0],tmpDat.time)
    
    if len(tmpDat.RplantarMat) != 0:
        [RTDC,RBDC] = findCycleEvents(tmpDat.Racc[:,0],tmpDat.time)

    answer = True # if data check is off. 
    if data_check == 1:
        # Create the heat plot of the plantar/dorsal pressure
        if len(tmpDat.RplantarMat) != 0:
            plotAvgCyclePressure(tmpDat.RplantarMat, tmpDat, fPath, RTDC, RBDC)
        elif len(tmpDat.LplantarMat) != 0:
            plotAvgCyclePressure(tmpDat.LplantarMat, tmpDat, fPath, LTDC, LBDC)
        
        # Create the debugging plot for time continuous estimated force
        plt.figure()
        if len(tmpDat.RplantarMat) != 0:
            plt.plot(intp_strokes(tmpDat.RForce,RTDC))
        elif len(tmpDat.LplantarMat) != 0:
            plt.plot(intp_strokes(tmpDat.LForce,LTDC))
        plt.xlabel('% Stroke')
        plt.ylabel('Insole Force (N)')
        
        # Save the debugging figure
        saveFolder= fPath + 'ForcePlots'
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)    
        plt.savefig(saveFolder + '/' + subName + '_' + ConfigTmp + '_' + trialTmp +'.png')
        
        answer = messagebox.askyesno("Question","Is data clean?")
        
    plt.close('all')
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
if save_on == 1:
    if os.path.exists(outfileName) == False:
    
        outcomes.to_csv(outfileName, header=True, index = False)

    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
                
        
        
               
            
            
            
            
            
        


