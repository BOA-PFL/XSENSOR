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
from dataclasses import dataclass
from tkinter import messagebox
import scipy
import scipy.signal as sig
from datetime import datetime


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

save_on = 1
data_check = 0

# Read in files
# only read .csv files for this work
fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Testing Segments/Cycling Performance Tests/2025_Performance_CyclingLacevBOA_Specialized/Xsensor/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]

# list of functions 
def delimitTrialCycling(inputDF,FName):
    """
    This function uses ginput to delimit the start and end of a trial
    You will need to indicate on the plot when to start/end the trial. 
    You must use tkinter or plotting outside the console to use this function
    Parameters
    ----------
    inputDF : Pandas DataFrame
        DF containing all desired output variables.
    zForce : numpy array 
        of force data from which we will subset the dataframe

    Returns
    -------
    outputDat: Pandas DataFrame
        dataframe subset to the beginning and end of the cycling data
    fsprintStart: int
        start of the sprinting

    """
    
    # generic function to plot and start/end trial #
    if os.path.exists(fPath+FName+'TrialSeg.npy'):
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
        np.save(fPath+FName+'TrialSeg.npy',trial_segment)

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

@dataclass    
class tsData:
    time: np.array
    dorsalMat: np.array
    dorsalForefoot: np.array 
    dorsalMidfoot: np.array 
    dorsalInstep: np.array
    dorsalSensNo: int
    dorsalForefootSensNo: int
    dorsalMidfootSensNo: int
    dorsalInstepSensNo: int
    
    LplantarMat: np.array
    LplantarToe: np.array
    LplantarForefoot: np.array
    LplantarMidfoot: np.array
    LplantarHeel: np.array
    LplantarLateral: np.array
    LplantarMedial: np.array
    LplantarSensNo: int
    LplantarToeSensNo: int
    LplantarForefootSensNo: int
    LplantarMidfootSensNo: int
    LplantarHeelSensNo: int
    LplantarLateralSensNo: int
    LplantarMedialSensNo: int
    
    RplantarMat: np.array
    RplantarToe: np.array
    RplantarForefoot: np.array
    RplantarMidfoot: np.array
    RplantarHeel: np.array
    RplantarLateral: np.array
    RplantarMedial: np.array
    RplantarSensNo: int
    RplantarToeSensNo: int
    RplantarForefootSensNo: int
    RplantarMidfootSensNo: int
    RplantarHeelSensNo: int
    RplantarLateralSensNo: int
    RplantarMedialSensNo: int
    
    LForce: np.array
    LTDC: np.array
    LBDC: np.array
    RForce: np.array
    RTDC: np.array
    RBDC: np.array
    
    LCOP_X: np.array
    LCOP_Y: np.array
    RCOP_X: np.array
    RCOP_Y: np.array
    
    sprintStart: int
    
    config: str
    movement: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    def plotAvgCyclePressure(self):
        """
        Create a 2D plot of the plantar & dorsal pressure during the down & up
        strokes

        Returns
        -------
        fig : matplotlib figure
            Plantar & dorsal pressure sensor plot

        """
        # Detect if left or right insole was used
        if len(self.LTDC) > 0:
            fTDC = self.LTDC
            fBDC = self.LBDC
            fplantarMat = self.LplantarMat
        else:
            fTDC = self.RTDC
            fBDC = self.RBDC
            fplantarMat = self.RplantarMat
        
        # Define a blank arrays for down & upstrokes
        downPlantar = np.zeros([len(fTDC), 31, 9])
        upPlantar = np.zeros([len(fTDC), 31, 9])
        
        downDorsal = np.zeros([len(fTDC), 18, 10])
        upDorsal = np.zeros([len(fTDC), 18, 10])
        
        # Compute the average downstroke and upstroke pressure for plotting
        for ii in range(len(fTDC)-1):
            downPlantar[ii,:,:] = np.mean(fplantarMat[fTDC[ii]:fBDC[ii],:,:],axis=0)
            upPlantar[ii,:,:] = np.mean(fplantarMat[fBDC[ii]:fTDC[ii+1],:,:],axis=0)
            downDorsal[ii,:,:] = np.mean(self.dorsalMat[fTDC[ii]:fBDC[ii],:,:],axis = 0)
            upDorsal[ii,:,:] = np.mean(self.dorsalMat[fBDC[ii]:fTDC[ii+1],:,:],axis = 0)
            
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
        
        plt.suptitle(self.subject +' '+ self. movement +' '+ self.config)
        plt.tight_layout()
        plt.margins(0.1)
        
        saveFolder= fPath + '2DPlots'
        
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)
            
        plt.savefig(saveFolder + '/' + self.subject +' '+ self. movement +' '+ self.config + '.png')
        return fig  



def createTSmat(inputName):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    
    Note: the number of sensels across narrow XSENSOR insoles is invariate
    across insole sizes. The number of sensels per area are as follows:
        
    """

    dat = pd.read_csv(fPath+inputName, sep=',', header = 0, low_memory=False)
    if dat.shape[1] == 2:
        dat = pd.read_csv(fPath+inputName, sep=',', header = 1, low_memory=False)
   
    dat, sprintStart = delimitTrialCycling(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2]
    
    time = np.array([((datetime.strptime(timestr,' %H:%M:%S.%f')-datetime(1900,1,1)).total_seconds()) for timestr in dat.Time])
    time = time - time[0]

    RplantarMat = []
    RplantarToe = []
    RplantarForefoot = []
    RplantarMidfoot = []
    RplantarHeel = []
    RplantarLateral = []
    RplantarMedial = []
    RplantarSensNo = []
    RplantarToeSensNo = []
    RplantarForefootSensNo = []
    RplantarMidfootSensNo = []
    RplantarHeelSensNo = []
    RplantarLateralSensNo = []
    RplantarMedialSensNo = []
    RForce = []
    RTDC = []
    RBDC = []
    RCOP_Y = []
    RCOP_X = []

    LplantarMat = []
    LplantarToe = []
    LplantarForefoot = []
    LplantarMidfoot = []
    LplantarHeel = []
    LplantarLateral = []
    LplantarMedial = []
    LplantarSensNo = []
    LplantarToeSensNo = []
    LplantarForefootSensNo = []
    LplantarMidfootSensNo = []
    LplantarHeelSensNo = []
    LplantarLateralSensNo = []
    LplantarMedialSensNo = []
    LForce = []
    LTDC = []
    LBDC = []
    LCOP_Y = []
    LCOP_X = []
    
    dorsalMat = []
    dorsalForefoot = []
    dorsalMidfoot = []
    dorsalInstep = []
    dorsalSensNo = []
    dorsalForefootSensNo = []
    dorsalMidfootSensNo = []
    dorsalInstepSensNo = []

    if 'Insole' in dat.columns:
        if  dat['Insole'][0] == 'Left':      # check to see if right insole used
           
            LplantarSensel = dat.loc[:,'S_1_5':'S_31_5']
            
            headers = LplantarSensel.columns
            store_r = []
            store_c = []
           
            for name in headers:
                store_r.append(int(name.split(sep = "_")[1])-1)
                store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
            
            LplantarMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
            
            for ii in range(len(headers)):
                LplantarMat[:, store_r[ii],store_c[ii]] = LplantarSensel.iloc[:,ii]
            
            LplantarMat[LplantarMat < 1] = 0
            LplantarToe = LplantarMat[:,:7,:]
            LplantarForefoot = LplantarMat[:,7:15, :]
            LplantarMidfoot = LplantarMat[:,15:25,:]
            LplantarHeel = LplantarMat[:,25:, :]
            LplantarLateral = LplantarMat[:,:,:4:]
            LplantarMedial =LplantarMat[:,:,4]
            
            # Sensel Number Computation: note this needs to match the column/row
            # callouts from the lines above
            store_r = np.array(store_r); store_c = np.array(store_c)
            LplantarSensNo = len(store_r)
            LplantarToeSensNo = len(np.where(store_r < 7)[0])
            LplantarForefootSensNo = len(np.where((store_r >= 7)*(store_r < 15))[0])
            LplantarMidfootSensNo = len(np.where((store_r >= 15)*(store_r < 25))[0])
            LplantarHeelSensNo = len(np.where(store_r >= 25)[0])
            LplantarLateralSensNo = len(np.where(store_c >= 4)[0])
            LplantarMedialSensNo = len(np.where(store_c < 4)[0])
            
            LForce = dat['Est. Load (lbf)']*4.44822
            
            # Base the acceleration detection from the "X" direction
            Lacc_det = filtSig(dat['IMU AX'],2,time)
            
            LTDC,_ = sig.find_peaks(Lacc_det, height = -0.5, distance = 30)
            LBDC,_ = sig.find_peaks(-Lacc_det, height = 1, distance = 30)
            if LBDC[0] < LTDC[0]:
                LBDC = LBDC[1:]
            [LTDC,LBDC] = EnsureCycleEventsAlternate(Lacc_det,LTDC,LBDC)
        
        if dat['Insole'][0] != 'Right' and dat['Insole'][0] != 'Left' :       # check to see if dorsal pad was used
            dorsalSensel = dat.loc[:,'S_1_1':'S_18_10']
            
        elif 'Insole.1' in dat.columns:
            if dat['Insole.1'][0] != 'Right' and dat['Insole.1'][0] != 'Left' :
                dorsalSensel = dat.loc[:,'S_1_1':'S_18_10']
                
        if 'dorsalSensel' in locals():        
            headers = dorsalSensel.columns
            store_r = []
            store_c = []

            for name in headers:
                store_r.append(int(name.split(sep = "_")[1])-1)
                store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
            
            dorsalMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
            
            for ii in range(len(headers)):
                dorsalMat[:, store_r[ii],store_c[ii]] = dorsalSensel.iloc[:,ii]
            
            
            dorsalMat = np.flip(dorsalMat, axis = 0) 
            dorsalMat[dorsalMat <1] = 0  
            
            dorsalForefoot = dorsalMat[:,:6,:]
            dorsalMidfoot = dorsalMat[:,6:12, :]
            dorsalInstep = dorsalMat[:,12:, :]
            # Sensel Number Computation: note this needs to match the column/row
            # callouts from the lines above
            store_r = np.array(store_r)
            dorsalSensNo = len(store_r)
            dorsalForefootSensNo = len(np.where(store_r < 6)[0])
            dorsalMidfootSensNo = len(np.where((store_r >= 6)*(store_r < 12))[0])
            dorsalInstepSensNo = len(np.where(store_r >= 12)[0])
            
        
        if  dat['Insole'][0] == 'Right':  # check to see if left insole used
            
            RplantarSensel = dat.loc[:, 'S_1_2':'S_31_7'] 
        
        elif  'Insole.1' in dat.columns:
            if dat['Insole.1'][0] == 'Right':  
                
                RplantarSensel = dat.loc[:, 'S_1_2.1':'S_31_7']
            
        if 'RplantarSensel' in locals():  
            headers = RplantarSensel.columns
            store_r = []
            store_c = []
              
            for name in headers:
               store_r.append(int(name.split(sep = "_")[1])-1)
               store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
           
            RplantarMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
           
            for ii in range(len(headers)):
               RplantarMat[:, store_r[ii],store_c[ii]] = RplantarSensel.iloc[:,ii]
            
            RplantarMat[RplantarMat < 1] = 0
            RplantarToe = RplantarMat[:,:7,:]
            RplantarForefoot = RplantarMat[:,7:15, :]
            RplantarMidfoot = RplantarMat[:,15:25,:]
            RplantarHeel = RplantarMat[:,25:, :]
            RplantarLateral = RplantarMat[:,:,4:]
            RplantarMedial = RplantarMat[:,:,:4]
            
            # Sensel Number Computation: note this needs to match the column/row
            # callouts from the lines above
            store_r = np.array(store_r); store_c = np.array(store_c)
            RplantarSensNo = len(store_r)
            RplantarToeSensNo = len(np.where(store_r < 7)[0])
            RplantarForefootSensNo = len(np.where((store_r >= 7)*(store_r < 15))[0])
            RplantarMidfootSensNo = len(np.where((store_r >= 15)*(store_r < 25))[0])
            RplantarHeelSensNo = len(np.where(store_r >= 25)[0])
            RplantarLateralSensNo = len(np.where(store_c >= 4)[0])
            RplantarMedialSensNo = len(np.where(store_c < 4)[0])
            
            RForce = dat['Est. Load (lbf).1']*4.44822
            
            # Base the acceleration detection from the "X" direction
            Racc_det = filtSig(dat['IMU AX.1'],2,time)
            
            RTDC,_ = sig.find_peaks(Racc_det, height = -0.5, distance = 30)
            RBDC,_ = sig.find_peaks(-Racc_det, height = 1, distance = 30)
            if RBDC[0] < RTDC[0]:
                RBDC = RBDC[1:]
            [RTDC,RBDC] = EnsureCycleEventsAlternate(Racc_det,RTDC,RBDC)


        if 'COP Row' in dat.columns:  
            
            if dat['Insole'][0] == 'Left':
                
                LCOP_Y = dat['COP Column']
                LCOP_X = dat['COP Row']
                
            if dat['Insole'][0] == 'Right':
                
                RCOP_Y = dat['COP Column']
                RCOP_X = dat['COP Row']
               
            if 'Insole.1' in dat.columns:
                if dat['Insole.1'][0] == 'Right':
                
                    RCOP_Y = dat['COP Column.1']
                    RCOP_X = dat['COP Row.1']
                
    result = tsData(time, dorsalMat, dorsalForefoot, dorsalMidfoot, dorsalInstep,
                     dorsalSensNo, dorsalForefootSensNo, dorsalMidfootSensNo, dorsalInstepSensNo,
                     LplantarMat, LplantarToe, LplantarForefoot, LplantarMidfoot, LplantarHeel, LplantarLateral, LplantarMedial,
                     LplantarSensNo, LplantarToeSensNo, LplantarForefootSensNo, LplantarMidfootSensNo, LplantarHeelSensNo, LplantarLateralSensNo, LplantarMedialSensNo,
                     RplantarMat, RplantarToe, RplantarForefoot, RplantarMidfoot, RplantarHeel,  RplantarLateral, RplantarMedial,
                     RplantarSensNo, RplantarToeSensNo, RplantarForefootSensNo, RplantarMidfootSensNo, RplantarHeelSensNo, RplantarLateralSensNo, RplantarMedialSensNo,
                     LForce, LTDC, LBDC, RForce, RTDC, RBDC, 
                     LCOP_X, LCOP_Y, RCOP_X, RCOP_Y, sprintStart,
                     config, movement, subj, dat)
    
    return(result)



#############################################################################################################################################################


badFileList = []

for fName in entries:
    print(fName)
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
    
    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    trialTmp = fName.split(sep = "_")[3].split(sep = ".")[0]
    
    tmpDat = createTSmat(fName)

    answer = True # if data check is off. 
    if data_check == 1:
        # Create the heat plot of the plantar/dorsal pressure
        tmpDat.plotAvgCyclePressure()
        
        # Create the debugging plot for time continuous estimated force
        plt.figure()
        plt.plot(intp_strokes(tmpDat.RForce,tmpDat.RTDC))
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
        if len(tmpDat.LTDC) > 0:
            TDC = tmpDat.LTDC
            BDC = tmpDat.LBDC
            plantarMat = tmpDat.LplantarMat
            plantarHeel = tmpDat.LplantarHeel
            plantarToe = tmpDat.LplantarToe
            HeelSensNo = tmpDat.LplantarHeelSensNo
            ToeSensNo = tmpDat.LplantarToeSensNo
            
        else:
            TDC = tmpDat.RTDC
            BDC = tmpDat.RBDC
            plantarMat = tmpDat.RplantarMat
            plantarHeel = tmpDat.RplantarHeel
            plantarToe = tmpDat.RplantarToe
            HeelSensNo = tmpDat.RplantarHeelSensNo
            ToeSensNo = tmpDat.RplantarToeSensNo 

        for ii in range(len(TDC)-1):            
            try: 
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
                    
                if tmpDat.RTDC[ii] < tmpDat.sprintStart:
                    movement.append('Steady')
                else:
                    movement.append('Sprint')
                config.append(tmpDat.config)
                subject.append(tmpDat.subject)
                Order.append(trialTmp)
                    
            
            except: print(fName + 'Bad stroke ' + str(ii))
            
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
                
        
        
               
            
            
            
            
            
        


