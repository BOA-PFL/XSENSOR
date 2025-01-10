# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 08:20:04 2024

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


# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\Kate.Harrison\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Snow Performance\\EH_Snowboard_BurtonWrap_Perf_Dec2024\\InLabPressure\\'
fileExt = r".csv"
entries_back = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'back' in fName) ]
entries_shin = [fName for fName in os.listdir(fPath) if (fName.endswith(fileExt) and 'Shin' in fName) ]

# turn_det_nob = 0.75 # filter frequency for turn detection: may need to be different for different subjects

save_on = 1
data_check = 0


freq = 100 # sampling frequency
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold



@dataclass    
class tsData:

    
    DorsalMat: np.array
    Forefoot: np.array 
    Midfoot: np.array 
    Instep: np.array
    ShinMat: np.array
    BackMat: np.array 
    
    
    DorsalForce: np.array
    ShinForce: np.array
    CalfForce: np.array
    
    
    config: str
    subject: str
    trial: str
    
    backDat: pd.DataFrame #entire stored dataframe. 
    shinDat: pd.DataFrame
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    # def plotAvgPressure(self):
        
    #     earlyTurnLeft = np.zeros([len(self.RTurns), 31, 9])
    #     midTurnLeft = np.zeros([len(self.RTurns), 31, 9])
    #     lateTurnLeft = np.zeros([len(self.RTurns), 31, 9])
        
    #     earlyTurnRight = np.zeros([len(self.RTurns), 31, 9])
    #     midTurnRight = np.zeros([len(self.RTurns), 31, 9])
    #     lateTurnRight = np.zeros([len(self.RTurns), 31, 9])
        
       
        
    #     for i in range(len(self.RTurns)):
            
    #         earlyTurnLeft[i,:,:] = self.LMat[self.RTurns[i],:,:]
    #         midTurnLeft[i,:,:] = self.LMat[self.RTurns[i] + round((self.LTurns[i]-self.RTurns[i])/2),:,:]
    #         lateTurnLeft[i,:,:] = self.LMat[self.LTurns[i],:,:]
            
    #         earlyTurnRight[i,:,:] = self.RMat[self.RTurns[i],:,:]
    #         midTurnRight[i,:,:] = self.RMat[self.RTurns[i] + round((self.LTurns[i]-self.RTurns[i])/2),:,:]
    #         lateTurnRight[i,:,:] = self.RMat[self.LTurns[i],:,:]
           
            
    #     earlyTurnLeftAvg = np.mean(earlyTurnLeft, axis = 0)
    #     midTurnLeftAvg = np.mean(midTurnLeft, axis = 0)
    #     lateTurnLeftAvg = np.mean(lateTurnLeft, axis = 0)
        
    #     earlyTurnRightAvg = np.mean(earlyTurnRight, axis = 0)
    #     midTurnRightAvg = np.mean(midTurnRight, axis = 0)
    #     lateTurnRightAvg = np.mean(lateTurnRight, axis = 0)
       
        
    #     fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    #     ax1 = sns.heatmap(earlyTurnLeftAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax1.set_title('Downhill Ski Pressure') 
    #     ax1.set_ylabel('Early Turn')
        
    #     ax2 = sns.heatmap(earlyTurnRightAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax2.set_title('Uphill Ski Pressure') 
        
    #     ax3 = sns.heatmap(midTurnLeftAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax3.set_ylabel('Mid Turn')
        
    #     ax4 = sns.heatmap(midTurnRightAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        
    #     ax5 = sns.heatmap(lateTurnLeftAvg, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
    #     ax5.set_ylabel('Late Turn')
    
    #     ax6 = sns.heatmap(lateTurnRightAvg, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        
    #     fig.set_size_inches(5, 10)
        
    #     plt.suptitle(self.subject +' '+ self.config)
    #     plt.tight_layout()
    #     plt.margins(0.1)
        
    #     saveFolder= fPath + '2DPlots'
        
    #     if os.path.exists(saveFolder) == False:
    #         os.mkdir(saveFolder)
            
    #     plt.savefig(saveFolder + '/' + self.subject +' '+ self.config + '.png')
    #     return fig  



def createTSmat(inputName):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """

    #inputName = entries_back[0]

    # dat = pd.read_csv(fPath+inputName, sep=',', usecols=(columns) )  
    
    
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[2]
    trial = inputName.split(sep="_")[3].split(sep='.')[0]
    
    
    shin = [fName for fName in entries_shin if (subj in fName and config in fName and trial in fName)  ]
    dat_shin = pd.read_csv(fPath+shin[0], sep=',', header = 0 , low_memory=False)
    if dat_shin.shape[1] <= 2:    
        dat_shin = pd.read_csv(fPath+shin[0], sep=',', header = 1 , low_memory=False)
        
    dorsalSensel = dat_shin.iloc[:, 18:198]
    shinSensel = dat_shin.iloc[:, 210:]
    
   

    dat_back = pd.read_csv(fPath+inputName, sep=',', header = 0 , low_memory=False)
    #dat_back = delimitTrial(dat_back, inputName)
    if dat_back.shape[1] <= 2:    
        dat_back= pd.read_csv(fPath+inputName, sep=',', header = 1 , low_memory=False)
   

    backSensel = dat_back.iloc[:,18:]


    # Set up filters: Note - 6 Hz is the cut-off
    w1 = 6 / (100 / 2)
    b1, a1 = sig.butter(2, w1, 'low')

    headers = dorsalSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)

    DorsalMat_raw = np.zeros((dat_shin.shape[0], np.max(store_r)+1,np.max(store_c)+1))

    for ii in range(len(headers)):
        DorsalMat_raw[:, store_r[ii],store_c[ii]] = dorsalSensel.iloc[:,ii]


################### Need to filter the signals here.
    DorsalMat = sig.filtfilt(b1, a1, DorsalMat_raw,axis=0)
    DorsalMat[DorsalMat < 1] = 0


    headers = shinSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)

    ShinMat_raw = np.zeros((dat_shin.shape[0], np.max(store_r)+1,np.max(store_c)+1))

    for ii in range(len(headers)):
        ShinMat_raw[:, store_r[ii],store_c[ii]] = shinSensel.iloc[:,ii]
    

    ################### Need to filter the signals here.  
    ShinMat = sig.filtfilt(b1, a1, ShinMat_raw,axis=0)
    ShinMat[ShinMat <1] = 0
    
    
    headers = backSensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)

    BackMat_raw = np.zeros((dat_back.shape[0], np.max(store_r)+1,np.max(store_c)+1))

    for ii in range(len(headers)):
        BackMat_raw[:, store_r[ii],store_c[ii]] = backSensel.iloc[:,ii]
    

    ################### Need to filter the signals here.  
    BackMat = sig.filtfilt(b1, a1, BackMat_raw,axis=0)
    BackMat[BackMat <1] = 0


    Forefoot = DorsalMat[:,:6, :] 
    Midfoot = DorsalMat[:,6:12,:] 
    Instep = DorsalMat[:,12:, :] 

    

           
    DorsalForce = (np.mean(DorsalMat, axis = (1,2))) * 6895 * 0.014699
    ShinForce = (np.mean(ShinMat, axis = (1,2))) * 6895 * 0.014699
    CalfForce = (np.mean(BackMat, axis = (1,2))) * 6895 * 0.014699

# The indices were garbled on the output. Disabling feature.
# [RT,LT] = findTurns(RForce, LForce, freq)

    result = tsData(DorsalMat, Forefoot, Midfoot, Instep,
                     ShinMat, BackMat,
                     DorsalForce, ShinForce, CalfForce,
                     config, subj, trial, dat_back, dat_shin
                    )
    
    
    return(result)



avgCalf = [] 
avgShin = []
avgDorsal = []
pkCalf = []
pkShin = []
pkDorsal = []
varCalf = []
varShin = []
varDorsal = []
avgFF = []
avgMF = []
avgIns = []

config  =[]
subject = []
trial = []

for entry in entries_back:
    
            
    #entry = entries_back[0]
    try:
        tmpAvgMat = createTSmat(entry)
      

        config.append(tmpAvgMat.config)
        subject.append(tmpAvgMat.subject)
        trial.append(tmpAvgMat.trial)
        
        avgDorsal.append(float(np.mean(tmpAvgMat.DorsalMat)*6.895))
        pkDorsal.append( float(np.max(tmpAvgMat.DorsalMat)*6.895))
        varDorsal.append( float(np.std(tmpAvgMat.DorsalMat)/np.mean(tmpAvgMat.DorsalMat)*6.895))
        
        avgCalf.append( float(np.mean(tmpAvgMat.BackMat[:, :15, :])*6.895)) #only using data from the front half of sensor and the back half sticks out of the boot
        pkCalf.append( float(np.max(tmpAvgMat.BackMat[:, :15, :])*6.895))
        varCalf.append( float(np.std(tmpAvgMat.BackMat[:, :15, :])*6.895))
        
        avgShin.append(float(np.mean(tmpAvgMat.ShinMat)*6.895) )
        pkShin.append(float(np.max(tmpAvgMat.ShinMat)*6.895))
        varShin.append( float(np.std(tmpAvgMat.ShinMat)*6.895))
        
        avgFF.append( float(np.mean(tmpAvgMat.DorsalMat[:,:6, :])*6.895))
        avgMF.append(float(np.mean(tmpAvgMat.DorsalMat[:, 6:12, :])*6.895))
        avgIns.append(float(np.mean(tmpAvgMat.DorsalMat[:, 12:, :]) *6.895))
    
    except:
          print('Missing Data ' + entry)


outcomes = pd.DataFrame({'Subject': list(subject), 'Config':list(config),'trialNo':list(trial), 
                           'avgDorsal':list(avgDorsal), 'pkDorsal':list(pkDorsal), 'varDorsal':list(varDorsal),
                           'avgCalf':list(avgCalf), 'pkCalf':list(pkCalf), 'varCalf':list(varCalf),
                           'avgShin':list(avgShin), 'pkShin':list(pkShin), 'varShin':list(varShin),
                           'avgFF':list(avgFF), 'avgMF':list(avgMF), 'avgIns':list(avgIns)
                           })

outfileName = fPath + '0_CompiledResults.csv'
if save_on == 1:
     if os.path.exists(outfileName) == False:
     
         outcomes.to_csv(outfileName, header=True, index = False)
 
     else:
         outcomes.to_csv(outfileName, mode='a', header=False, index = False)