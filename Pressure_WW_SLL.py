# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:47:49 2023

@author: Milena.Singletary
"""
### figuring out pressure template for the SLL/ overground landings 
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

fwdLook = 30
fThresh = 50
freq = 100 # sampling frequency
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold

def delimitTrial(inputDF,FName):
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
    outputDat: dataframe subset to the beginning and end of jumps.

    """

    # generic function to plot and start/end trial #
    if os.path.exists(fPath+FName+'TrialSeg.npy'):
        trial_segment_old = np.load(fPath+FName+'TrialSeg.npy', allow_pickle =True)
        trialStart = trial_segment_old[1][0,0]
        trialEnd = trial_segment_old[1][1,0]
        inputDF = inputDF.iloc[int(np.floor(trialStart)) : int(np.floor(trialEnd)),:]
        outputDat = inputDF.reset_index(drop = True)
        
    else: 
        fig, ax = plt.subplots()

        totForce = np.mean(inputDF.iloc[:,214:425], axis = 1)*6895*0.014699
        # changing the column to 27 and no axis call
        #totForce = np.mean(inputDF.iloc[:,27])*6895*0.014699
        print('Select a point on the plot to represent the beginning & end of trial')


        ax.plot(totForce, label = 'Total Force')
        fig.legend()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        outputDat = outputDat.reset_index(drop = True)
        trial_segment = np.array([FName, pts])
        np.save(fPath+FName+'TrialSeg.npy',trial_segment)

    return(outputDat)



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



   
@dataclass    
class tsData:
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """
    dorsalMat: np.array
    dorsalForefoot: np.array
    dorsalForefootLat: np.array 
    dorsalForefootMed: np.array 
    dorsalMidfoot: np.array
    dorsalMidfootLat: np.array 
    dorsalMidfootMed: np.array 
    dorsalInstep: np.array 
    dorsalInstepLat: np.array 
    dorsalInstepMed: np.array 
    
    plantarMat: np.array
    plantarToe: np.array 
    plantarToeLat: np.array 
    plantarToeMed: np.array 
    plantarForefoot: np.array 
    plantarForefootLat : np.array 
    plantarForefootMed: np.array 
    plantarMidfoot: np.array 
    plantarMidfootLat: np.array 
    plantarMidfootMed: np.array
    plantarHeel: np.array 
    plantarHeelLat: np.array 
    plantarHeelMed: np.array 
    
    plantarLateral: np.array
    plantarMedial: np.array
    
    RForce: np.array 
    
    #Rland : np.array
    #Rtake : np.array
    # RHS: np.array 
    # RTO: np.array
    
    config: str
    movement: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    # def plotAvgPressure(self):
        
    #     earlyPlantar = np.zeros([len(self.RHS), 30, 9])
    #     midPlantar = np.zeros([len(self.RHS), 30, 9])
    #     latePlantar = np.zeros([len(self.RHS), 30, 9])
        
    #     earlyDorsal = np.zeros([len(self.RHS), 18, 10])
    #     midDorsal = np.zeros([len(self.RHS), 18, 10])
    #     lateDorsal = np.zeros([len(self.RHS), 18, 10])
        
    #     for i in range(len(self.RHS)):
            
    #         earlyPlantar[i,:,:] = self.plantarMat[self.RHS[i],:,:]
    #         midPlantar[i,:,:] = self.plantarMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]
    #         latePlantar[i,:,:] = self.plantarMat[self.RTO[i],:,:]
    #         earlyDorsal[i,:,:] = self.dorsalMat[self.RHS[i],:,:]
    #         midDorsal[i,:,:] = self.dorsalMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]
    #         lateDorsal[i,:,:] = self.dorsalMat[self.RTO[i],:,:]
            
    #     earlyPlantarAvg = np.mean(earlyPlantar, axis = 0)
    #     midPlantarAvg = np.mean(midPlantar, axis = 0)
    #     latePlantarAvg = np.mean(latePlantar, axis = 0)
    #     earlyDorsalAvg = np.mean(earlyDorsal, axis = 0)
    #     midDorsalAvg = np.mean(midDorsal, axis = 0)
    #     lateDorsalAvg = np.mean(lateDorsal, axis = 0)
        
    #     fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    #     ax1 = sns.heatmap(earlyDorsalAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(earlyDorsalAvg))
    #     ax1.set_title('Dorsal Pressure') 
    #     ax1.set_ylabel('Initial Contact')
        
        
    #     ax2 = sns.heatmap(earlyPlantarAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(earlyPlantarAvg))
    #     ax2.set_title('Plantar Pressure') 
        
    #     ax3 = sns.heatmap(midDorsalAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(midDorsalAvg))
    #     ax3.set_ylabel('Midstance')
        
    #     ax4 = sns.heatmap(midPlantarAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(midPlantarAvg))
        
    #     ax5 = sns.heatmap(lateDorsalAvg, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(latePlantarAvg))
    #     ax5.set_ylabel('Toe off')
        
        
    #     ax6 = sns.heatmap(latePlantarAvg, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(lateDorsalAvg))
        
    #     fig.set_size_inches(5, 10)
        
    #     plt.suptitle(self.subject +' '+ self. movement +' '+ self.config)
    #     plt.tight_layout()
    #     plt.margins(0.1)
        
    #     saveFolder= fPath + '2DPlots'
        
    #     if os.path.exists(saveFolder) == False:
    #         os.mkdir(saveFolder)
            
    #     plt.savefig(saveFolder + '/' + self.subject +' '+ self. movement +' '+ self.config + '.png')
    #     return fig  



def createTSmat(inputName):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """
    columns = ['Frame', 'Date', 'Time', 'Units', 'Threshold', 'Note', 'Sensor', 'Insole Side', 'Rows', 'Columns', 'Average Pressure (PSI)', 
                 'Minimum Pressure (PSI)', 'Peak Pressure (PSI)', 'Contact Area (cm²)', 'Total Area (cm²)', 'Contact %', 'Est. Load (lbf)', 'Std Dev.', 
                 'S_1_1', 'S_1_2', 'S_1_3', 'S_1_4', 'S_1_5', 'S_1_6', 'S_1_7', 'S_1_8', 'S_1_9', 'S_1_10', 
                 'S_2_1', 'S_2_2', 'S_2_3', 'S_2_4', 'S_2_5', 'S_2_6', 'S_2_7', 'S_2_8', 'S_2_9', 'S_2_10', 
                 'S_3_1', 'S_3_2', 'S_3_3', 'S_3_4', 'S_3_5', 'S_3_6', 'S_3_7', 'S_3_8', 'S_3_9', 'S_3_10', 
                 'S_4_1', 'S_4_2', 'S_4_3', 'S_4_4', 'S_4_5', 'S_4_6', 'S_4_7', 'S_4_8', 'S_4_9', 'S_4_10', 
                 'S_5_1', 'S_5_2', 'S_5_3', 'S_5_4', 'S_5_5', 'S_5_6', 'S_5_7', 'S_5_8', 'S_5_9', 'S_5_10', 
                 'S_6_1', 'S_6_2', 'S_6_3', 'S_6_4', 'S_6_5', 'S_6_6', 'S_6_7', 'S_6_8', 'S_6_9', 'S_6_10', 
                 'S_7_1', 'S_7_2', 'S_7_3', 'S_7_4', 'S_7_5', 'S_7_6', 'S_7_7', 'S_7_8', 'S_7_9', 'S_7_10', 
                 'S_8_1', 'S_8_2', 'S_8_3', 'S_8_4', 'S_8_5', 'S_8_6', 'S_8_7', 'S_8_8', 'S_8_9', 'S_8_10', 
                 'S_9_1', 'S_9_2', 'S_9_3', 'S_9_4', 'S_9_5', 'S_9_6', 'S_9_7', 'S_9_8', 'S_9_9', 'S_9_10', 
                 'S_10_1', 'S_10_2', 'S_10_3', 'S_10_4', 'S_10_5', 'S_10_6', 'S_10_7', 'S_10_8', 'S_10_9', 'S_10_10', 
                 'S_11_1', 'S_11_2', 'S_11_3', 'S_11_4', 'S_11_5', 'S_11_6', 'S_11_7', 'S_11_8', 'S_11_9', 'S_11_10', 
                 'S_12_1', 'S_12_2', 'S_12_3', 'S_12_4', 'S_12_5', 'S_12_6', 'S_12_7', 'S_12_8', 'S_12_9', 'S_12_10', 
                 'S_13_1', 'S_13_2', 'S_13_3', 'S_13_4', 'S_13_5', 'S_13_6', 'S_13_7', 'S_13_8', 'S_13_9', 'S_13_10', 
                 'S_14_1', 'S_14_2', 'S_14_3', 'S_14_4', 'S_14_5', 'S_14_6', 'S_14_7', 'S_14_8', 'S_14_9', 'S_14_10', 
                 'S_15_1', 'S_15_2', 'S_15_3', 'S_15_4', 'S_15_5', 'S_15_6', 'S_15_7', 'S_15_8', 'S_15_9', 'S_15_10', 
                 'S_16_1', 'S_16_2', 'S_16_3', 'S_16_4', 'S_16_5', 'S_16_6', 'S_16_7', 'S_16_8', 'S_16_9', 'S_16_10', 
                 'S_17_1', 'S_17_2', 'S_17_3', 'S_17_4', 'S_17_5', 'S_17_6', 'S_17_7', 'S_17_8', 'S_17_9', 'S_17_10', 
                 'S_18_1', 'S_18_2', 'S_18_3', 'S_18_4', 'S_18_5', 'S_18_6', 'S_18_7', 'S_18_8', 'S_18_9', 'S_18_10', 
                 'Sensor.1', 'Insole Side.1', 'Rows.1', 'Columns.1', 'Average Pressure (PSI).1', 'Minimum Pressure (PSI).1', 'Peak Pressure (PSI).1', 'Contact Area (cm²).1', 
                 'Total Area (cm²).1', 'Contact %.1', 'Est. Load (lbf).1', 'Std Dev..1', 
                 'S_1_2.1', 'S_1_3.1', 'S_1_4.1', 'S_1_5.1', 'S_2_2.1', 'S_2_3.1', 'S_2_4.1', 'S_2_5.1', 'S_2_6.1', 'S_3_1.1', 'S_3_2.1', 'S_3_3.1', 'S_3_4.1', 'S_3_5.1', 
                 'S_3_6.1', 'S_3_7.1', 'S_4_1.1', 'S_4_2.1', 'S_4_3.1', 'S_4_4.1', 'S_4_5.1', 'S_4_6.1', 'S_4_7.1', 'S_5_1.1', 'S_5_2.1', 'S_5_3.1', 'S_5_4.1', 'S_5_5.1', 
                 'S_5_6.1', 'S_5_7.1', 'S_5_8.1', 'S_6_1.1', 'S_6_2.1', 'S_6_3.1', 'S_6_4.1', 'S_6_5.1', 'S_6_6.1', 'S_6_7.1', 'S_6_8.1', 'S_7_1.1', 'S_7_2.1', 'S_7_3.1', 
                 'S_7_4.1', 'S_7_5.1', 'S_7_6.1', 'S_7_7.1', 'S_7_8.1', 'S_8_1.1', 'S_8_2.1', 'S_8_3.1', 'S_8_4.1', 'S_8_5.1', 'S_8_6.1', 'S_8_7.1', 'S_8_8.1', 'S_8_9.1', 
                 'S_9_1.1', 'S_9_2.1', 'S_9_3.1', 'S_9_4.1', 'S_9_5.1', 'S_9_6.1', 'S_9_7.1', 'S_9_8.1', 'S_9_9.1', 'S_10_1.1', 'S_10_2.1', 'S_10_3.1', 'S_10_4.1', 'S_10_5.1', 
                 'S_10_6.1', 'S_10_7.1', 'S_10_8.1', 'S_10_9.1', 'S_11_1.1', 'S_11_2.1', 'S_11_3.1', 'S_11_4.1', 'S_11_5.1', 'S_11_6.1', 'S_11_7.1', 'S_11_8.1', 'S_11_9.1', 
                 'S_12_2.1', 'S_12_3.1', 'S_12_4.1', 'S_12_5.1', 'S_12_6.1', 'S_12_7.1', 'S_12_8.1', 'S_12_9.1', 'S_13_2.1', 'S_13_3.1', 'S_13_4.1', 'S_13_5.1', 'S_13_6.1', 
                 'S_13_7.1', 'S_13_8.1', 'S_13_9.1', 'S_14_2.1', 'S_14_3.1', 'S_14_4.1', 'S_14_5.1', 'S_14_6.1', 'S_14_7.1', 'S_14_8.1', 'S_14_9.1', 'S_15_3.1', 'S_15_4.1', 
                 'S_15_5.1', 'S_15_6.1', 'S_15_7.1', 'S_15_8.1', 'S_15_9.1', 'S_16_3.1', 'S_16_4.1', 'S_16_5.1', 'S_16_6.1', 'S_16_7.1', 'S_16_8.1', 'S_16_9.1', 'S_17_3.1', 
                 'S_17_4.1', 'S_17_5.1', 'S_17_6.1', 'S_17_7.1', 'S_17_8.1', 'S_17_9.1', 'S_18_3.1', 'S_18_4.1', 'S_18_5.1', 'S_18_6.1', 'S_18_7.1', 'S_18_8.1', 'S_18_9.1', 
                 'S_19_3', 'S_19_4', 'S_19_5', 'S_19_6', 'S_19_7', 'S_19_8', 'S_19_9', 'S_20_3', 'S_20_4', 'S_20_5', 'S_20_6', 'S_20_7', 'S_20_8', 'S_20_9', 'S_21_3', 'S_21_4', 
                 'S_21_5', 'S_21_6', 'S_21_7', 'S_21_8', 'S_21_9', 'S_22_3', 'S_22_4', 'S_22_5', 'S_22_6', 'S_22_7', 'S_22_8', 'S_22_9', 'S_23_3', 'S_23_4', 'S_23_5', 'S_23_6', 
                 'S_23_7', 'S_23_8', 'S_23_9', 'S_24_3', 'S_24_4', 'S_24_5', 'S_24_6', 'S_24_7', 'S_24_8', 'S_24_9', 'S_25_3', 'S_25_4', 'S_25_5', 'S_25_6', 'S_25_7', 'S_25_8', 
                 'S_25_9', 'S_26_3', 'S_26_4', 'S_26_5', 'S_26_6', 'S_26_7', 'S_26_8', 'S_26_9', 'S_27_3', 'S_27_4', 'S_27_5', 'S_27_6', 'S_27_7', 'S_27_8', 'S_27_9', 'S_28_3', 
                 'S_28_4', 'S_28_5', 'S_28_6', 'S_28_7', 'S_28_8', 'S_28_9', 'S_29_3', 'S_29_4', 'S_29_5', 'S_29_6', 'S_29_7', 'S_29_8', 'S_29_9', 'S_30_4', 'S_30_5', 'S_30_6', 
                 'S_30_7', 'S_30_8', 'S_31_5', 'S_31_6', 'S_31_7']
    
    #inputName = entries[1]
    #dat = pd.read_csv(fPath+inputName, sep=',', skiprows = 1, header = 'infer')
    dat = pd.read_csv(fPath+inputName, sep=',', usecols = columns)
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2]
    dorsalSensel = dat.iloc[:,18:198]
    plantarSensel = dat.iloc[:,210:430]
    
    headers = plantarSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    plantarMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        plantarMat[:, store_r[ii],store_c[ii]] = plantarSensel.iloc[:,ii]

    plantarMat[plantarMat < 1 ] = 0
    
    headers = dorsalSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    dorsalMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)-1):
        dorsalMat[:, store_r[ii],store_c[ii]] = dorsalSensel.iloc[:,ii]
    
    dorsalMat = np.flip(dorsalMat, axis = 1)
    dorsalMat[dorsalMat < 1 ] = 0


    
    plantarToe = plantarMat[:,:7,:]
    plantarToeLat = plantarMat[:,:7,4:]
    plantarToeMed = plantarMat[:,:7,:4]
    plantarForefoot = plantarMat[:,7:15, :]
    plantarForefootLat = plantarMat[:,7:15,4:]
    plantarForefootMed = plantarMat[:,7:15,:4]
    plantarMidfoot = plantarMat[:,15:25,:]
    plantarMidfootLat = plantarMat[:,15:25,4:]
    plantarMidfootMed = plantarMat[:,15:25,:4]
    plantarHeel = plantarMat[:,25:, :]
    plantarHeelLat = plantarMat[:,25:,4:]
    plantarHeelMed = plantarMat[:,25:, :4]
    
    dorsalForefoot = dorsalMat[:,:6,:]
    dorsalForefootLat = dorsalMat[:,:6,5:]
    dorsalForefootMed = dorsalMat[:,:6,:5]
    dorsalMidfoot = dorsalMat[:,6:12, :]
    dorsalMidfootLat = dorsalMat[:,6:12,:5]
    dorsalMidfootMed = dorsalMat[:, 6:12,5:]
    dorsalInstep = dorsalMat[:,12:, :]
    dorsalInstepLat = dorsalMat[:,12:,5:]
    dorsalInstepMed = dorsalMat[:,12:,:5]
    
    plantarLateral = plantarMat[:,:,4:]
    plantarMedial = plantarMat[:,:,:4]
    
    RForce = np.mean(plantarMat, axis = (1,2))*6895*0.014699
    #RForce = zeroInsoleForce(RForce,freq)
    #[land] = findLandings(RForce, fThresh)
    #[take] = findTakeoffs(RForce, fThresh)
    #[RHS,RTO] = findGaitEvents(RForce,freq)
    
    result = tsData(dorsalMat, dorsalForefoot, dorsalForefootLat, dorsalForefootMed, 
                     dorsalMidfoot, dorsalMidfootLat, dorsalMidfootMed, 
                     dorsalInstep, dorsalInstepLat, dorsalInstepMed, 
                     plantarMat, plantarToe, plantarToeLat, plantarToeMed,
                     plantarForefoot, plantarForefootLat, plantarForefootMed,
                     plantarMidfoot, plantarMidfootLat, plantarMidfootMed,
                     plantarHeel, plantarHeelLat, plantarHeelMed, plantarLateral, plantarMedial, RForce, 
                     config, movement, subj, dat) #land, take,) 
    
    return(result)



#############################################################################################################################################################

# Read in files
# only read .asc files for this work
fPath = 'C:/Users/milena.singletary/Boa Technology Inc/PFL Team - Documents/General/Testing Segments/WorkWear_Performance/EH_Workwear_MidCutStabilityII_CPDMech_Sept23/XSENSOR/cropped/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]




badFileList = []

for fName in entries:
    
    config = []
    subject = []
    ct = []
    movement = []

    
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

    try: 
        #fName = entries[2]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
        
        # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        #if ('SLL' in moveTmp):# or ('SLLT' in moveTmp): # or ('Trail' in moveTmp):
        if (moveTmp == 'SLL'):
            #dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            
  
            tmpDat = createTSmat(fName)
            ffoot = np.mean(tmpDat.plantarForefoot, axis = (1,2)) * 6895 * 0.014699
            
            
            land = sig.find_peaks(tmpDat.RForce, height = 1000, distance = 600)[0]
            land_ht =  sig.find_peaks(tmpDat.RForce, height = 1000, distance = 600)[1]['peak_heights']
            fft_pk = sig.find_peaks(ffoot, height = 1000, distance = 500)[0]
            fft_ht = sig.find_peaks(ffoot, height = 1000, distance = 500)[1]['peak_heights']
            
            htThrsh = np.mean(land_ht) - 400
                        
            true_land = sig.find_peaks(tmpDat.RForce, height = htThrsh, distance = 500)[0]
            land_ht =  sig.find_peaks(tmpDat.RForce, height = htThrsh, distance = 500)[1]['peak_heights']
   
  
            
            answer = True # if data check is off. 
            if data_check == 1:
                plt.figure()
                plt.plot(tmpDat.RForce, label = 'Right Foot Total Force') 
                plt.plot(true_land, land_ht, marker = 'o', linestyle = 'none')
                #plt.plot(range(len(ffoot)), ffoot)
                #plt.plot(fft_pk, fft_ht, marker = 'v', linestyle = 'none')
                
  
                answer = messagebox.askyesno("Question","Is data clean?")    
  
  
  
                
            
            if answer == False:
                disThrsh = np.mean(land)- 100
                true_land = sig.find_peaks(tmpDat.RForce, height = htThrsh, distance = disThrsh)[0]
                land_ht =  sig.find_peaks(tmpDat.RForce, height = htThrsh, distance = disThrsh)[1]['peak_heights']
                plt.figure()
                plt.plot(tmpDat.RForce, label = 'Right Foot Total Force') 
                plt.plot(land, land_ht, marker = 'o', linestyle = 'none')
                plt.plot(range(len(ffoot)), ffoot)
                plt.plot(fft_pk, fft_ht, marker = 'v', linestyle = 'none')
                answer = messagebox.askyesno("Question","Is data clean?") 
            
                            
            if answer == False:
                plt.close('all')
                print('Adding file to bad file list')
                badFileList.append(fName)
                
            if answer == True:
                plt.close('all')
                print('Estimating point estimates')
                
        
        
                
                # toe 39 ; fft 68 ; heel 43
                for ii in range(len(land)):
                    toeClawAvg.append(np.mean(tmpDat.plantarToe[land[ii]: land[ii]+100]))
                    toeClawPk.append(np.max(tmpDat.plantarToe[land[ii]: land[ii]+100]))
                    toeLatAvg.append(np.mean(tmpDat.plantarToeLat[land[ii]: land[ii]+100]))
                    toeLatPk.append(np.max(tmpDat.plantarToeLat[land[ii]: land[ii]+100]))
                    toeMedAvg.append(np.mean(tmpDat.plantarToeMed[land[ii]: land[ii]+100]))
                    toeMedPk.append(np.max(tmpDat.plantarToeMed[land[ii]: land[ii]+100]))
            
                    ffAvg.append(np.mean(tmpDat.plantarForefoot[land[ii]: land[ii]+100]))
                    ffPk.append(np.max(tmpDat.plantarForefoot[land[ii]: land[ii]+100]))
                    ffConArea.append(np.count_nonzero(tmpDat.plantarForefoot[land[ii]: land[ii]+100])/100/68*100) # divided by 100 frames
                    ffLatAvg.append(np.mean(tmpDat.plantarForefootLat[land[ii]: land[ii]+100]))
                    ffLatPk.append(np.max(tmpDat.plantarForefootLat[land[ii]: land[ii]+100]))
                    ffMedAvg.append(np.mean(tmpDat.plantarForefootMed[land[ii]: land[ii]+100]))
                    ffMedPk.append(np.max(tmpDat.plantarForefootMed[land[ii]: land[ii]+100]))
            
                    heelArea.append(np.count_nonzero(tmpDat.plantarHeel[land[ii]: land[ii]+100])/ 100/ 43*100) # divided by 100 frames
                    heelPres.append(np.mean(tmpDat.plantarHeel[land[ii]: land[ii]+100]))
                    
                    config.append(tmpDat.config)
                    subject.append(tmpDat.subject)
                    movement.append(moveTmp)
            
                outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config),
                                 'ToeClaw': list(toeClawAvg), 'ToeClawPeak': list(toeClawPk), 'ToeLat': list(toeLatAvg), 'ToeLatPeak' : list(toeLatPk),
                                 'ToeMed' : list(toeMedAvg), 'ToeMedPeak' : list(toeMedPk), 'ForefootAvg' : list(ffAvg), 'ForefootPeak' : list(ffPk), 'ForefootContA' : list(ffConArea),
                                 'ForefootLat': list(ffLatAvg), 'ForefootLatPk': list(ffLatPk), 'ForefootMed': list(ffMedAvg), 'ForefootMedPk': list(ffMedPk),
                                 'HeelConArea' : list(heelArea), 'HeelPressure' : list(heelPres)})
       
        
                                   
                outfileName = fPath + '1_CompiledResults_SLL.csv'
                if save_on == 1:
                    if os.path.exists(outfileName) == False:
                    
                        outcomes.to_csv(outfileName, header=True, index = False)
                
                    else:
                        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
                       
                        
        
        
    except:
            print('Not usable data')             
        
            
            



