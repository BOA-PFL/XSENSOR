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
import scipy.interpolate
import scipy


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

fwdLook = 30
fThresh = 50
freq = 100 # sampling frequency
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold

# def delimitTrial(inputDF,FName):
#     """
#      This function uses ginput to delimit the start and end of a trial
#     You will need to indicate on the plot when to start/end the trial. 
#     You must use tkinter or plotting outside the console to use this function
#     Parameters
#     ----------
#     inputDF : Pandas DataFrame
#         DF containing all desired output variables.
#     zForce : numpy array 
#         of force data from which we will subset the dataframe

#     Returns
#     -------
#     outputDat: dataframe subset to the beginning and end of jumps.

#     """

#     # generic function to plot and start/end trial #
#     if os.path.exists(fPath+FName+'TrialSeg.npy'):
#         trial_segment_old = np.load(fPath+FName+'TrialSeg.npy', allow_pickle =True)
#         trialStart = trial_segment_old[1][0,0]
#         trialEnd = trial_segment_old[1][1,0]
#         inputDF = inputDF.iloc[int(np.floor(trialStart)) : int(np.floor(trialEnd)),:]
#         outputDat = inputDF.reset_index(drop = True)
        
#     else: 
        
#         #inputDF = dat
#         fig, ax = plt.subplots()
        
#         insoleSide = inputDF['Insole'][0]
                   
        
#         if (insoleSide == 'Left'): 
            
#             # Left side
#             totForce = np.mean(inputDF.iloc[:,18:238], axis = 1)*6895*0.014699
#         else:  
            
#             totForce = np.mean(inputDF.iloc[:,210:430], axis = 1)*6895*0.014699
#         print('Select a point on the plot to represent the beginning & end of trial')


#         ax.plot(totForce, label = 'Total Force')
#         fig.legend()
#         pts = np.asarray(plt.ginput(2, timeout=-1))
#         plt.close()
#         outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
#         outputDat = outputDat.reset_index(drop = True)
#         trial_segment = np.array([FName, pts], dtype = object)
#         np.save(fPath+FName+'TrialSeg.npy',trial_segment)

#     return(outputDat)



def zeroInsoleForce(vForce,freq):
    """
    Function to detect "foot-off" pressure to zero the insole force during swing
    This is expecially handy when residual pressue on the foot "drops out" 
    during swing

    Parameters
    ----------
    vForce : list
        Vertical force (or approximate vertical force)
    freq : int
        Data collection frequency

    Returns
    -------
    newForce : list
        Updated vertical force
        

    """
    newForce = vForce
    # Quasi-constants
    zcoeff = 0.7
    
    windowSize = round(0.8*freq)
    
    zeroThresh = 50
    
    # Ensure that the minimum vertical force is zero
    vForce = vForce - np.min(vForce)
    
    # Filter the forces
    # Set up a 2nd order 10 Hz low pass buttworth filter
    cut = 10
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    # Filter the force signal
    vForce = sig.filtfilt(b, a, vForce)
        
    # Set the threshold to start to find possible gait events
    thresh = zcoeff*np.mean(vForce)
    
    # Allocate gait variables
    nearHS = []
    # Index through the force    
    for ii in range(len(vForce)-1):
        if vForce[ii] <= thresh and vForce[ii+1] > thresh: 
            nearHS.append(ii+1)
    
    nearHS = np.array(nearHS)
    nearHS = nearHS[(nearHS > windowSize)*(nearHS < len(vForce)-windowSize)]
    
    HS = []
    for idx in nearHS:
        HS_unity_slope = (vForce[idx-windowSize] - vForce[idx])/windowSize*np.array(range(windowSize))
        HS_sig = HS_unity_slope - vForce[idx-windowSize:idx]
        HS.append(idx-windowSize+np.argmax(HS_sig))
    
    newForce = newForce - np.median(newForce[HS])
    newForce[newForce < zeroThresh] = 0
    
    return(newForce)
    
def findGaitEvents(vForce,freq):
    """
    Function to determine foot contacts (here HS or heel strikes) and toe-off (TO)
    events using a gradient decent formula

    Parameters
    ----------
    vForce : list
        Vertical force (or approximate vertical force)
    freq : int
        Data collection frequency

    Returns
    -------
    HS : numpy array
        Heel strike (or foot contact) indicies
    TO : numpy array
        Toe-off indicies

    """
    # Quasi-constants
    zcoeff = 0.9
    
    
        
    # Set the threshold to start to find possible gait events
    thresh = zcoeff*np.mean(vForce)
    
    # Allocate gait variables
    nearHS = []
    nearTO = []
    # Index through the force 
    for ii in range(len(vForce)-1):
        if vForce[ii] <= thresh and vForce[ii+1] > thresh:
            nearHS.append(ii+1)
        if vForce[ii] > thresh and vForce[ii+1] <= thresh:
            nearTO.append(ii)
    
    nearHS = np.array(nearHS); nearTO = np.array(nearTO)
    # # Remove events that are too close to the start/end of the trial
    # nearHS = nearHS[(nearHS > windowSize)*(nearHS < len(vForce)-windowSize)]
    # nearTO = nearTO[(nearTO > windowSize)*(nearTO < len(vForce)-windowSize)]
    
    HS = []
    for idx in nearHS:
        #if idx < len(vForce)-windowSize:
            for ii in range(idx,0,-1):
                if vForce[ii] == 0 :
                    HS.append(ii)
                    break
    
    # Only utilize unique event detections
    HS = np.unique(HS)
    HS = np.array(HS)
    # Used to eliminate dections that could be due to the local minima in the 
    # middle of a walking step 
    HS = HS[vForce[HS]<100]
    
    # Only search for toe-offs that correspond to heel-strikes
    newTO = []
    newHS = []
    for ii in range(len(HS)):
        tmp = np.where(nearTO > HS[ii])[0]
        if len(tmp) > 0:
            idx = nearTO[tmp[0]]
            #if idx < len(vForce)-windowSize:
            for jj in range(idx,len(vForce)):
                if vForce[jj] == 0 :
                    newHS.append(HS[ii])
                    newTO.append(jj)
                    break
                # if (len(TO)-1) < ii:
                #     np.delete(HS,ii)
        else:
            np.delete(HS,ii)
    
    if newHS[-1] > newTO[-1]:
        newHS = newHS[:-1]
    
    
            
    return(newHS,newTO)





   
@dataclass    
class tsData:

    LplantarMat: np.array
    LplantarToe: np.array 
    LplantarToeLat: np.array 
    LplantarToeMed: np.array 
    LplantarForefoot: np.array 
    LplantarForefootLat : np.array 
    LplantarForefootMed: np.array 
    LplantarMidfoot: np.array 
    LplantarMidfootLat: np.array 
    LplantarMidfootMed: np.array
    LplantarHeel: np.array 
    LplantarHeelLat: np.array 
    LplantarHeelMed: np.array 
    
    LplantarLateral: np.array
    LplantarMedial: np.array
    
    RplantarMat: np.array
    RplantarToe: np.array 
    RplantarToeLat: np.array 
    RplantarToeMed: np.array 
    RplantarForefoot: np.array 
    RplantarForefootLat : np.array 
    RplantarForefootMed: np.array 
    RplantarMidfoot: np.array 
    RplantarMidfootLat: np.array 
    RplantarMidfootMed: np.array
    RplantarHeel: np.array 
    RplantarHeelLat: np.array 
    RplantarHeelMed: np.array 
    
    RplantarLateral: np.array
    RplantarMedial: np.array
 
    LForce: np.array 
    LHS: np.array 
    LTO: np.array
    
    RForce: np.array 
    RHS: np.array 
    RTO: np.array

    config: str
    movement: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        
        LearlyPlantar = np.zeros([len(self.LHS), 31, 9])
        LmidPlantar = np.zeros([len(self.LHS), 31, 9])
        LlatePlantar = np.zeros([len(self.LHS), 31, 9])
        
        RearlyPlantar = np.zeros([len(self.RHS), 31, 9])
        RmidPlantar = np.zeros([len(self.RHS), 31, 9])
        RlatePlantar = np.zeros([len(self.RHS), 31, 9])
        
        
        # Note: For Altra trail test - the maximum pressures are plotted
        for i in range(len(self.RHS)):
            RearlyPlantar[i,:,:] = self.RplantarMat[self.RHS[i],:,:]*6.895
            RmidPlantar[i,:,:] = self.RplantarMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]*6.895
            RlatePlantar[i,:,:] = self.RplantarMat[self.RTO[i],:,:]*6.895
        
        for i in range(len(self.LHS)):
            LearlyPlantar[i,:,:] = self.LplantarMat[self.LHS[i],:,:]*6.895
            LmidPlantar[i,:,:] = self.LplantarMat[self.LHS[i] + round((self.LTO[i]-self.LHS[i])/2),:,:]*6.895
            LlatePlantar[i,:,:] = self.LplantarMat[self.LTO[i],:,:]*6.895
            
        LearlyPlantarMax = np.max(LearlyPlantar, axis = 0)
        LmidPlantarMax = np.max(LmidPlantar, axis = 0)
        LlatePlantarMax = np.max(LlatePlantar, axis = 0)
        
        RearlyPlantarMax = np.max(RearlyPlantar, axis = 0)
        RmidPlantarMax = np.max(RmidPlantar, axis = 0)
        RlatePlantarMax = np.max(RlatePlantar, axis = 0)
        
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
        ax1 = sns.heatmap(LearlyPlantarMax, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(LearlyPlantarMax))
        ax1.set_title('Left Plantar Pressure') 
        ax1.set_ylabel('Initial Contact')
        
        
        ax2 = sns.heatmap(RearlyPlantarMax, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(RearlyPlantarMax),cbar_kws={'label': 'Peak Pressure (kPa)'})
        ax2.set_title('Plantar Pressure') 
        
        ax3 = sns.heatmap(LmidPlantarMax, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(LmidPlantarMax))
        ax3.set_ylabel('Midstance')
        
        ax4 = sns.heatmap(RmidPlantarMax, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(RmidPlantarMax),cbar_kws={'label': 'Peak Pressure (kPa)'})
        
        ax5 = sns.heatmap(LlatePlantarMax, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(LlatePlantarMax))
        ax5.set_ylabel('Toe off')
        
        
        ax6 = sns.heatmap(RlatePlantarMax, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(RlatePlantarMax),cbar_kws={'label': 'Peak Pressure (kPa)'})
        
        fig.set_size_inches(8, 15)
        
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
    """
    
    
    #inputName = entries[2]

  
    # dat = pd.read_csv(fPath+inputName, sep=',', usecols=(columns) )  
    #dat = pd.read_csv(fPath+inputName, sep=',', header = 'infer', low_memory=False)
    dat = pd.read_csv(fPath+inputName, sep=',', header = 1 , low_memory=False)
   
    # dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2] 
    
    

    
        
    LplantarSensel = dat.iloc[:,18:238]

    RplantarSensel = dat.iloc[:,250:470] 

        
   
    # left
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
    
    
    
    # right 
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
    
    

    LplantarToe = LplantarMat[:,:7,:] 
    LplantarToeLat = LplantarMat[:,:7, :5]
    LplantarToeMed = LplantarMat[:,:7,5:] 
    LplantarForefoot = LplantarMat[:,7:15, :] 
    LplantarForefootLat = LplantarMat[:,7:15,:5] #Opposite ":," sequence from R side
    LplantarForefootMed = LplantarMat[:,7:15,5:] 
    LplantarMidfoot = LplantarMat[:,15:25,:] 
    LplantarMidfootLat = LplantarMat[:,15:25,:5] #Opposite ":," sequence from R side
    LplantarMidfootMed = LplantarMat[:,15:25,5:] 
    
    LplantarHeel = LplantarMat[:,25:, :] 
    LplantarHeelLat = LplantarMat[:,25:,:5] #Opposite ":," sequence from R side
    LplantarHeelMed = LplantarMat[:,25:, 5:]
    

    
    LplantarLateral = LplantarMat[:,:5]
    LplantarMedial = LplantarMat[:,:,5:]
    
    LForce = np.mean(LplantarMat, axis = (1,2))*6895*0.014699
    LForce = zeroInsoleForce(LForce,freq)
    [LHS,LTO] = findGaitEvents(LForce,freq)
  
    
  
    
    #right
    RplantarToe = RplantarMat[:,:7,:]
    RplantarToeLat = RplantarMat[:,:7,4:]
    RplantarToeMed = RplantarMat[:,:7,:4]
    RplantarForefoot = RplantarMat[:,7:15, :]
    RplantarForefootLat = RplantarMat[:,7:15,4:]
    RplantarForefootMed = RplantarMat[:,7:15,:4]
    RplantarMidfoot = RplantarMat[:,15:25,:]
    RplantarMidfootLat = RplantarMat[:,15:25,4:]
    RplantarMidfootMed = RplantarMat[:,15:25,:4]
    
    RplantarHeel = RplantarMat[:,25:, :]
    RplantarHeelLat = RplantarMat[:,25:,4:]
    RplantarHeelMed = RplantarMat[:,25:, :4]
    
    
    RplantarLateral = RplantarMat[:,:,4:]
    RplantarMedial = RplantarMat[:,:,:4]
    
    
    RForce = np.mean(RplantarMat, axis = (1,2))*6895*0.014699
    RForce = zeroInsoleForce(RForce,freq)
    [RHS,RTO] = findGaitEvents(RForce,freq)
    
    
    
    
    
    result = tsData(  LplantarMat, LplantarToe, LplantarToeLat, LplantarToeMed,
                    LplantarForefoot, LplantarForefootLat, LplantarForefootMed,
                    LplantarMidfoot, LplantarMidfootLat, LplantarMidfootMed,
                    LplantarHeel, LplantarHeelLat, LplantarHeelMed, LplantarLateral, LplantarMedial,
     
                     RplantarMat, RplantarToe, RplantarToeLat, RplantarToeMed,
                     RplantarForefoot, RplantarForefootLat, RplantarForefootMed,
                     RplantarMidfoot, RplantarMidfootLat, RplantarMidfootMed,
                     RplantarHeel, RplantarHeelLat, RplantarHeelMed, RplantarLateral, RplantarMedial, LForce, LHS, LTO, RForce, RHS, RTO,
                     
                     config, movement, subj, dat)
    
    return(result)

def intp_strides(var,landings,takeoffs,GS):
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
    takeoffs : list
        Toe-off indicies

    Returns
    -------
    intp_var : numpy array
        Interpolated variable to 101 points with the number of columns dictated
        by the number of strides.

    """
    # Preallocate
    intp_var = np.zeros((101,len(GS)))
    # Index through the strides
    for count,ii in enumerate(GS):
        dum = var[landings[ii]:takeoffs[ii]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,count] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var

#############################################################################################################################################################

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\EH_Trail_AltraMidsole_Perf_Mar24\\Xsensor\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) ]


GPStiming = pd.read_csv('C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\EH_Trail_AltraMidsole_Perf_Mar24\\GPS\\CombinedGPS.csv')
SeshConf = GPStiming[['Config', 'Sesh']]

badFileList = []

toeF1 = np.zeros((101,len(entries)))
toeF2 = np.zeros((101,len(entries)))
toeF3 = np.zeros((101,len(entries)))

ffF1 = np.zeros((101,len(entries)))
ffF2 = np.zeros((101,len(entries)))
ffF3 = np.zeros((101,len(entries)))

mfF1 = np.zeros((101,len(entries)))
mfF2 = np.zeros((101,len(entries)))
mfF3 = np.zeros((101,len(entries)))

hlF1 = np.zeros((101,len(entries)))
hlF2 = np.zeros((101,len(entries)))
hlF3 = np.zeros((101,len(entries)))


for ii in range(0,len(entries)):
#for ii in range(3,6):
    config = []
    subject = []
    ct = []
    movement = []
    side = []
    sesh = []
    oLabel = np.array([])

    toePmidstance = []
    toeAreamidstance = []
    ffAreaLate = []
    ffPLate = []
    ffPMaxLate = []
    heelAreaLate = []
    heelPLate = []
    heelPmax = []
    maxmaxToes = []
    
    ffAreaMid = []
    ffPMid = []
    
    mfAreaLate = []
    mfPLate = []
    mfAreaMid = []
    mfPMid = []
    mfmax = []

    latPmidstance = []
    latAreamidstance = []
    latPLate = []
    latAreaLate = []
    medPmidstance = []
    medAreamidstance = []
    medPLate = []
    medAreaLate = []
    
    latPropMid = []
    medPropMid = []


    # try: 
    fName = entries[ii]
    print(fName)
    subName = fName.split(sep = "_")[0]
    ConfigTmp = fName.split(sep="_")[1]
    moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0].lower()
    
    # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement


    tmpDat = createTSmat(fName)
    tmpDat.plotAvgPressure()
    
    # Find the correct GPS trial
    #GPStrial = np.array(GPStiming.Subject == subName) * np.array(GPStiming.Config == ConfigTmp) * np.array(GPStiming.Sesh == Sesh)
    GPStrial = np.array(GPStiming.Subject == subName) * np.array(GPStiming.Config == ConfigTmp)

    
    start_LHS = []; start_RHS = []
    if subName == 'ChadPrichard': # change 25 to more or less depending on time standing around before start of run
        checkwnd = 25
        checkless = 20
    else:  
        checkwnd = 15 
        checkless = 15

        
        
    for jj in range(checkwnd): 
        jump_check = np.where(np.abs(tmpDat.LHS[jj] - np.array(tmpDat.RHS[0:checkwnd])) < checkless) 
        if jump_check[0].size > 0:
            print('Jump Found')
            start_LHS = jj+1
            start_RHS = np.argmin(np.abs(tmpDat.LHS[jj] - np.array(tmpDat.RHS[0:checkwnd])))+1
        
    
    LHS = tmpDat.LHS[start_LHS:]
    LTO = tmpDat.LTO[start_LHS:]
    RHS = tmpDat.RHS[start_RHS:]
    RTO = tmpDat.RTO[start_RHS:]
       
    # Remove strides that have a peak GRF below 1000 N or over 1900
    # Remove strides that are below 0.5 and above 1.5 seconds
    # pk = 1000
    # timemin = 0.5
    # timemax = 1.5
    
    pk = 1000
    upper = 1900
    timemin = 0.5
    timemax = 1.25

    LGS = []    
    # May need to exclude poor toe-off dectections here as well
    for jj in range(len(LHS)-1):
        if np.max(tmpDat.LForce[LHS[jj]:LTO[jj]]) > pk and np.max(tmpDat.LForce[LHS[jj]:LTO[jj]]) < upper:
            if (LHS[jj+1] - LHS[jj]) > timemin*freq and LHS[jj+1] - LHS[jj] < timemax*freq:
                LGS.append(jj)
    
    RGS = []
    for jj in range(len(RHS)-1):
        if np.max(tmpDat.RForce[RHS[jj]:RTO[jj]]) > pk and np.max(tmpDat.RForce[RHS[jj]:RTO[jj]]) < upper :
            if (RHS[jj+1] - RHS[jj]) > timemin*freq and RHS[jj+1] - RHS[jj] < timemax*freq:
                RGS.append(jj)
    
    
    
    
    
    answer = True # if data check is off. 
    if data_check == 1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(intp_strides(tmpDat.LForce,LHS,LTO,LGS))
        plt.ylabel('Total Left Insole Force (N)')
        
        plt.subplot(1,2,2)
        plt.plot(intp_strides(tmpDat.RForce,RHS,RTO,RGS))
        plt.ylabel('Total Right Insole Force (N)')
        
        # plt.plot(tmpDat.RForce[RHS[0]:RTO[-1]], label = 'Right Foot Total Force')
        
        # plt.plot(tmpDat.LForce[LHS[0]:LTO[-1]], label = 'Left Foot Total Force')

        #     plt.axvspan(RHS[i], RTO[i], color = 'lightgray', alpha = 0.5)
        answer = messagebox.askyesno("Question","Is data clean?")
    
    
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(fName)

    if answer == True:
        plt.close('all')
        print('Estimating point estimates')
        
        # Create Labels
        Rlabel = np.zeros([len(RGS),1])
        Llabel = np.zeros([len(LGS),1])
        

        for kk in RGS:
            
            #i = 5
            config.append(tmpDat.config)
            subject.append(tmpDat.subject)
            sesh.append(moveTmp)
            #movement.append(moveTmp)
            movement.append('run')
            side.append('R')
            frames = tmpDat.RTO[kk] - tmpDat.RHS[kk]
            ct.append(frames/200)
            pct10 = tmpDat.RHS[kk] + round(frames*.1)
            pct40 = tmpDat.RHS[kk] + round(frames*.4)
            pct50 = tmpDat.RHS[kk] + round(frames*.5)
            pct60 = tmpDat.RHS[kk] + round(frames*.6)
            pct90 = tmpDat.RHS[kk] + round(frames*.9)
            

            
            maxmaxToes.append(np.max(tmpDat.RplantarToe[tmpDat.RHS[kk]:tmpDat.RTO[kk]])*6.895)
            toePmidstance.append(np.mean(tmpDat.RplantarToe[pct40:pct60,:,:])*6.895)
            toeAreamidstance.append(np.count_nonzero(tmpDat.RplantarToe[pct40:pct60,:,:])/(pct60 - pct40)/39*100)
            ffAreaLate.append(np.count_nonzero(tmpDat.RplantarForefoot[pct90:tmpDat.RTO[kk], :,:])/(tmpDat.RTO[kk] - pct90)/68*100)
            ffPLate.append(np.mean(tmpDat.RplantarForefoot[pct90:tmpDat.RTO[kk], :, :])*6.895)
            ffPMaxLate.append(np.max(tmpDat.RplantarForefoot[pct90:tmpDat.RTO[kk], :, :]))
            ffAreaMid.append(np.count_nonzero(tmpDat.RplantarForefoot[pct40:pct60, :,:])/(pct60 - pct40)/68*100)
            ffPMid.append((np.mean(tmpDat.RplantarForefoot[pct40:pct60, :, :]))*6.895)
            
            mfAreaLate.append(np.count_nonzero(tmpDat.RplantarMidfoot[pct90:tmpDat.RTO[kk], :,:])/(tmpDat.RTO[kk] - pct90)/70*100)
            mfPLate.append(np.mean(tmpDat.RplantarMidfoot[pct90:tmpDat.RTO[kk], :, :])*6.895)
            mfAreaMid.append(np.count_nonzero(tmpDat.RplantarMidfoot[pct40:pct60, :,:])/(pct60 - pct40)/70*100)
            mfPMid.append((np.mean(tmpDat.RplantarMidfoot[pct40:pct60, :, :]))*6.895)
            mfmax.append((np.max(tmpDat.RplantarMidfoot[tmpDat.RHS[kk]:tmpDat.RTO[kk], :, :]))*6.895)
            
            heelAreaLate.append(np.count_nonzero(tmpDat.RplantarHeel[pct50:tmpDat.RTO[kk], :, :])/(tmpDat.RTO[kk] - pct50)/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
            heelPLate.append(np.mean(tmpDat.RplantarHeel[pct90:tmpDat.RTO[kk], :, :])*6.895)
            heelPmax.append(np.max(tmpDat.RplantarHeel[tmpDat.RHS[kk]:tmpDat.RTO[kk], :, :])*6.895)

            latPmidstance.append(np.mean(tmpDat.RplantarLateral[pct40:pct60, :, :])*6.895)
            latAreamidstance.append(np.count_nonzero(tmpDat.RplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/138*100)
            latPLate.append(np.mean(tmpDat.RplantarLateral[pct90:tmpDat.RTO[kk], :, :])*6.895)
            latAreaLate.append(np.count_nonzero(tmpDat.RplantarLateral[pct90:tmpDat.RTO[kk], :, :])/(tmpDat.RTO[kk] - pct90)/138*100)
            medPmidstance.append(np.mean(tmpDat.RplantarMedial[pct40:pct60, :, :])*6.895)
            medAreamidstance.append(np.count_nonzero(tmpDat.RplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/82*100)
            medPLate.append(np.mean(tmpDat.RplantarMedial[pct90:tmpDat.RTO[kk], :, :])*6.895)
            medAreaLate.append(np.count_nonzero(tmpDat.RplantarMedial[pct90:tmpDat.RTO[kk], :, :])/(tmpDat.RTO[kk]-pct90)/82*100)
            
            latPropMid.append(np.sum(tmpDat.RplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
            medPropMid.append(np.sum(tmpDat.RplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.RplantarMat[pct40:pct60, :, :]))
            
            
            
        for kk in LGS:
            
            #i = 5
            config.append(tmpDat.config)
            subject.append(tmpDat.subject)
            sesh.append(moveTmp)
            #movement.append(moveTmp)
            movement.append('run')
            side.append('L')
            frames = tmpDat.LTO[kk] - tmpDat.LHS[kk]
            ct.append(frames/200)
            pct10 = tmpDat.LHS[kk] + round(frames*.1)
            pct40 = tmpDat.LHS[kk] + round(frames*.4)
            pct50 = tmpDat.LHS[kk] + round(frames*.5)
            pct60 = tmpDat.LHS[kk] + round(frames*.6)
            pct90 = tmpDat.LHS[kk] + round(frames*.9)
            
        
            
            maxmaxToes.append(np.max(tmpDat.LplantarToe[tmpDat.LHS[kk]:tmpDat.LTO[kk]])*6.895)
            toePmidstance.append(np.mean(tmpDat.LplantarToe[pct40:pct60,:,:])*6.895)
            toeAreamidstance.append(np.count_nonzero(tmpDat.LplantarToe[pct40:pct60,:,:])/(pct60 - pct40)/39*100)
            ffAreaLate.append(np.count_nonzero(tmpDat.LplantarForefoot[pct90:tmpDat.LTO[kk], :,:])/(tmpDat.LTO[kk] - pct90)/68*100)
            ffPLate.append(np.mean(tmpDat.LplantarForefoot[pct90:tmpDat.LTO[kk], :, :])*6.895)
            ffPMaxLate.append(np.max(tmpDat.LplantarForefoot[pct90:tmpDat.LTO[kk], :, :]))
            ffAreaMid.append(np.count_nonzero(tmpDat.LplantarForefoot[pct40:pct60, :,:])/(pct60 - pct40)/68*100)
            ffPMid.append((np.mean(tmpDat.LplantarForefoot[pct40:pct60, :, :]))*6.895)
            
            mfAreaLate.append(np.count_nonzero(tmpDat.LplantarMidfoot[pct90:tmpDat.LTO[kk], :,:])/(tmpDat.LTO[kk] - pct90)/70*100)
            mfPLate.append(np.mean(tmpDat.LplantarMidfoot[pct90:tmpDat.LTO[kk], :, :])*6.895)
            mfAreaMid.append(np.count_nonzero(tmpDat.LplantarMidfoot[pct40:pct60, :,:])/(pct60 - pct40)/70*100)
            mfPMid.append((np.mean(tmpDat.LplantarMidfoot[pct40:pct60, :, :]))*6.895)
            mfmax.append((np.max(tmpDat.RplantarMidfoot[tmpDat.LHS[kk]:tmpDat.LTO[kk], :, :]))*6.895)
            
            heelAreaLate.append(np.count_nonzero(tmpDat.LplantarHeel[pct50:tmpDat.LTO[kk], :, :])/(tmpDat.LTO[kk] - pct50)/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
            heelPLate.append(np.mean(tmpDat.LplantarHeel[pct90:tmpDat.LTO[kk], :, :])*6.895)
            heelPmax.append(np.max(tmpDat.RplantarHeel[tmpDat.LHS[kk]:tmpDat.LTO[kk], :, :])*6.895)
        
            latPmidstance.append(np.mean(tmpDat.LplantarLateral[pct40:pct60, :, :])*6.895)
            latAreamidstance.append(np.count_nonzero(tmpDat.LplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/138*100)
            latPLate.append(np.mean(tmpDat.LplantarLateral[pct90:tmpDat.LTO[kk], :, :])*6.895)
            latAreaLate.append(np.count_nonzero(tmpDat.LplantarLateral[pct90:tmpDat.LTO[kk], :, :])/(tmpDat.LTO[kk] - pct90)/138*100)
            medPmidstance.append(np.mean(tmpDat.LplantarMedial[pct40:pct60, :, :])*6.895)
            medAreamidstance.append(np.count_nonzero(tmpDat.LplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/82*100)
            medPLate.append(np.mean(tmpDat.LplantarMedial[pct90:tmpDat.LTO[kk], :, :])*6.895)
            medAreaLate.append(np.count_nonzero(tmpDat.LplantarMedial[pct90:tmpDat.LTO[kk], :, :])/(tmpDat.LTO[kk]-pct90)/82*100)
            
            latPropMid.append(np.sum(tmpDat.LplantarLateral[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
            medPropMid.append(np.sum(tmpDat.LplantarMedial[pct40:pct60, :, :])/np.sum(tmpDat.LplantarMat[pct40:pct60, :, :]))
            
        toeFtmp = intp_strides(np.mean(tmpDat.RplantarToe, axis = (1,2))*6895*0.014699,RHS,RTO,RGS)
        ffFtmp = intp_strides(np.mean(tmpDat.RplantarForefoot, axis = (1,2))*6895*0.014699,RHS,RTO,RGS)
        mfFtmp = intp_strides(np.mean(tmpDat.RplantarMidfoot, axis = (1,2))*6895*0.014699,RHS,RTO,RGS)
        hlFtmp = intp_strides(np.mean(tmpDat.RplantarHeel, axis = (1,2))*6895*0.014699,RHS,RTO,RGS)
        
        # Create Labels
        Llabel = np.zeros([len(LGS),1])
        LHS = np.array(LHS)
        LGS = np.array(LGS)
        # Uphill label
        idx = LHS[LGS]/freq < float(GPStiming.EndS1[GPStrial])
        Llabel[idx] = 1
        # Top label
        idx = (LHS[LGS]/freq > float(GPStiming.StartS2[GPStrial]))*(LHS[LGS]/freq < float(GPStiming.EndS2[GPStrial]))
        Llabel[idx] = 2
        # Bottom label
        idx = LHS[LGS]/freq > float(GPStiming.StartS3[GPStrial])
        Llabel[idx] = 3
        
        Rlabel = np.zeros([len(RGS),1])
        RHS = np.array(RHS)
        RGS = np.array(RGS)
        # Uphill label
        idx = RHS[RGS]/freq < float(GPStiming.EndS1[GPStrial])
        Rlabel[idx] = 1
        
        toeF1[:,ii] = np.mean(toeFtmp[:,idx],axis = 1)
        ffF1[:,ii] = np.mean(ffFtmp[:,idx],axis = 1)
        mfF1[:,ii] = np.mean(mfFtmp[:,idx],axis = 1)
        hlF1[:,ii] = np.mean(hlFtmp[:,idx],axis = 1)
        
        # Top label
        idx = (RHS[RGS]/freq > float(GPStiming.StartS2[GPStrial]))*(RHS[RGS]/freq < float(GPStiming.EndS2[GPStrial]))
        Rlabel[idx] = 2
        
        toeF2[:,ii] = np.mean(toeFtmp[:,idx],axis = 1)
        ffF2[:,ii] = np.mean(ffFtmp[:,idx],axis = 1)
        mfF2[:,ii] = np.mean(mfFtmp[:,idx],axis = 1)
        hlF2[:,ii] = np.mean(hlFtmp[:,idx],axis = 1)
        # Bottom label
        idx = RHS[RGS]/freq > float(GPStiming.StartS3[GPStrial])
        Rlabel[idx] = 3
        
        toeF3[:,ii] = np.mean(toeFtmp[:,idx],axis = 1)
        ffF3[:,ii] = np.mean(ffFtmp[:,idx],axis = 1)
        mfF3[:,ii] = np.mean(mfFtmp[:,idx],axis = 1)
        hlF3[:,ii] = np.mean(hlFtmp[:,idx],axis = 1)
    
    # # plot first step time series for different regions of foot 
    # plt.figure()
    # plt.plot((np.mean(tmpDat.LplantarToe, axis =(1,2)) *6895*0.014699)[LGS[1]:LGS[-1]])
    # plt.figure()
    # plt.plot((np.mean(tmpDat.LplantarForefoot, axis =(1,2)) *6895*0.014699)[LGS[1]:LGS[-1]])
    # plt.figure()
    # plt.plot((np.mean(tmpDat.LplantarMidfoot, axis =(1,2)) *6895*0.014699)[LGS[1]:LGS[-1]])
    # plt.figure()
    # plt.plot((np.mean(tmpDat.LplantarHeel, axis =(1,2)) *6895*0.014699)[LGS[1]:LGS[-1]])
    # # other  plots
    # # plt.figure()
    # # plt.plot((np.mean(tmpDat.LplantarLateral, axis =(1,2)) *6895*0.014699)[LGS[1]:LGS[-1]])
    # # plt.figure()
    # # plt.plot((np.mean(tmpDat.LplantarMedial, axis =(1,2)) *6895*0.014699)[LGS[1]:LGS[-1]])

    oLabel = np.concatenate((oLabel,Llabel,Rlabel),axis = None)
    

    outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Side':list(side), 'Label':list(oLabel), 'ContactTime':list(ct),
                             'toeP_mid':list(toePmidstance),'toeArea_mid':list(toeAreamidstance), 'maxmaxToes':list(maxmaxToes),
                             'ffP_late':list(ffPLate), 'ffArea_late':list(ffAreaLate), 'ffP_Mid':list(ffPMid), 'ffArea_Mid':list(ffAreaMid), 'ffPMax_late':list(ffPMaxLate),
                             'mfP_late':list(mfPLate), 'mfArea_late':list(mfAreaLate), 'mfP_Mid':list(mfPMid), 'mfArea_Mid':list(mfAreaMid), 'mfMax': list(mfmax),
                             'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate), 'heelMaxP': list(heelPmax), 
                             'latP_mid':list(latPmidstance), 'latArea_mid':list(latAreamidstance), 'latP_late':list(latPLate), 'latArea_late':list(latAreaLate), 'latPropMid':list(latPropMid),
                             'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medP_late':list(medPLate), 'medArea_late':list(medAreaLate), 'medPropMid':list(medPropMid),

                             
                             })

    outfileName = fPath + '0_CompiledResults.csv'
    if save_on == 1:
        if os.path.exists(outfileName) == False:
        
            outcomes.to_csv(outfileName, header=True, index = False)
    
        else:
            outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
        
        
        
    # except:
    #         print('Not usable data')
    #         badFileList.append(fName)             
            
            

