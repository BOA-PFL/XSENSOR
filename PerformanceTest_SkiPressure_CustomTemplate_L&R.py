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
        
        #inputDF = dat
        fig, ax = plt.subplots()
        
        LForce = np.mean(inputDF.iloc[:,18:238], axis = 1)*6895*0.014699
        RForce = np.mean(inputDF.iloc[:,250:470], axis = 1)*6895*0.014699
       
        print('Select a point on the plot to represent the beginning & end of trial')


        ax.plot(LForce, label = 'Left Force')
        ax.plot(RForce, label = 'Rigth Force')
        fig.legend()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        outputDat = outputDat.reset_index(drop = True)
        trial_segment = np.array([FName, pts], dtype = object)
        np.save(fPath+FName+'TrialSeg.npy',trial_segment)

    return(outputDat)




    
def findTurns(RForce, LForce, freq):
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
    
   
    
    b, a = sig.butter(4, .5, fs = freq)
    RForce = sig.filtfilt(b, a, RForce)
    LForce = sig.filtfilt(b, a, LForce)
    # plt.figure()
    # plt.plot(RForce)
    # plt.plot(LForce)
    
    RTurns = []
    LTurns = []
    
    for step in range(len(RForce)-1):
        if LForce[step] <= RForce[step] and LForce[step + 1] > RForce[step + 1] :
            RTurns.append(step)
    
    LTurns = []
    for step in range(len(RForce)-1):
        if RForce[step] <= LForce[step] and RForce[step + 1] > LForce[step + 1] :
            LTurns.append(step)
    
    if RTurns[0] > LTurns[0]:
        LTurns = LTurns[1:]
        
    if LTurns[-1] < RTurns[-1]:
        RTurns = RTurns[:-1]
            
    return(RTurns,LTurns)





   
@dataclass    
class tsData:

    
    LMat: np.array
    LToe: np.array 
    LToeLat: np.array 
    LToeMed: np.array 
    LForefoot: np.array 
    LForefootLat : np.array 
    LForefootMed: np.array 
    LffToe: np.array
    LffToeLat: np.array
    LffToeMed:np.array
    LMidfoot: np.array 
    LMidfootLat: np.array 
    LMidfootMed: np.array
    LHeel: np.array 
    LHeelLat: np.array 
    LHeelMed: np.array 
    LLateral: np.array
    LMedial: np.array
    
    RMat: np.array
    RToe: np.array 
    RToeLat: np.array 
    RToeMed: np.array 
    RForefoot: np.array 
    RForefootLat : np.array 
    RForefootMed: np.array 
    RffToe: np.array
    RffToeLat: np.array
    RffToeMed: np.array
    RMidfoot: np.array 
    RMidfootLat: np.array 
    RMidfootMed: np.array
    RHeel: np.array 
    RHeelLat: np.array 
    RHeelMed: np.array 
    
    RLateral: np.array
    RMedial: np.array
    
    LForce: np.array
    RForce: np.array
    
    RTurns: np.array 
    LTurns: np.array
    
    config: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    # below is a method of the dataclass
    def plotAvgPressure(self):
        
        earlyTurnLeft = np.zeros([len(self.RTurns), 31, 9])
        midTurnLeft = np.zeros([len(self.RTurns), 31, 9])
        lateTurnLeft = np.zeros([len(self.RTurns), 31, 9])
        
        earlyTurnRight = np.zeros([len(self.RTurns), 31, 9])
        midTurnRight = np.zeros([len(self.RTurns), 31, 9])
        lateTurnRight = np.zeros([len(self.RTurns), 31, 9])
        
       
        
        for i in range(len(self.RTurns)):
            
            earlyTurnLeft[i,:,:] = self.LMat[self.RTurns[i],:,:]
            midTurnLeft[i,:,:] = self.LMat[self.RTurns[i] + round((self.LTurns[i]-self.RTurns[i])/2),:,:]
            lateTurnLeft[i,:,:] = self.LMat[self.LTurns[i],:,:]
            
            earlyTurnRight[i,:,:] = self.RMat[self.RTurns[i],:,:]
            midTurnRight[i,:,:] = self.RMat[self.RTurns[i] + round((self.LTurns[i]-self.RTurns[i])/2),:,:]
            lateTurnRight[i,:,:] = self.RMat[self.LTurns[i],:,:]
           
            
        earlyTurnLeftAvg = np.mean(earlyTurnLeft, axis = 0)
        midTurnLeftAvg = np.mean(midTurnLeft, axis = 0)
        lateTurnLeftAvg = np.mean(lateTurnLeft, axis = 0)
        
        earlyTurnRightAvg = np.mean(earlyTurnRight, axis = 0)
        midTurnRightAvg = np.mean(midTurnRight, axis = 0)
        lateTurnRightAvg = np.mean(lateTurnRight, axis = 0)
       
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
        ax1 = sns.heatmap(earlyTurnLeftAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(midTurnLeftAvg))
        ax1.set_title('Downhill Ski Pressure') 
        ax1.set_ylabel('Early Turn')
        
        ax2 = sns.heatmap(earlyTurnRightAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(midTurnLeftAvg))
        ax2.set_title('Uphill Ski Pressure') 
        
        ax3 = sns.heatmap(midTurnLeftAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        ax3.set_ylabel('Mid Turn')
        
        ax4 = sns.heatmap(midTurnRightAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        
        ax5 = sns.heatmap(lateTurnLeftAvg, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        ax5.set_ylabel('Late Turn')
    
        ax6 = sns.heatmap(lateTurnRightAvg, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(midTurnLeftAvg))
        
        fig.set_size_inches(5, 10)
        
        plt.suptitle(self.subject +' '+ self.config)
        plt.tight_layout()
        plt.margins(0.1)
        
        saveFolder= fPath + '2DPlots'
        
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)
            
        plt.savefig(saveFolder + '/' + self.subject +' '+ self.config + '.png')
        return fig  



def createTSmat(inputName):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """

    #inputName = entries[0]

    # dat = pd.read_csv(fPath+inputName, sep=',', usecols=(columns) )  
    dat = pd.read_csv(fPath+inputName, sep=',', header = 0 , low_memory=False)
    
    if dat.shape[1] <= 2:    
        dat = pd.read_csv(fPath+inputName, sep=',', header = 1 , low_memory=False)
   
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]

    Lsensel = dat.iloc[:,18:238]
    Rsensel = dat.iloc[:,250:470]        
    
    headers = Lsensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    LMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)):
        LMat[:, store_r[ii],store_c[ii]] = Lsensel.iloc[:,ii]
    
    LMat[LMat < 1] = 0
    
    
    headers = Rsensel.columns
    store_r = []
    store_c = []

    for name in headers:
        store_r.append(int(name.split(sep = "_")[1])-1)
        store_c.append(int(name.split(sep = "_")[2].split(sep=".")[0])-1)
    
    RMat = np.zeros((dat.shape[0], np.max(store_r)+1,np.max(store_c)+1))
    
    for ii in range(len(headers)):
        RMat[:, store_r[ii],store_c[ii]] = Rsensel.iloc[:,ii]
    
    RMat[RMat <1] = 0  
    
    
    LToe = LMat[:,:7,:]  
    LToeLat = LMat[:,:7, :5]
    LToeMed = LMat[:,:7,5:] 
    LForefoot = LMat[:,7:15, :] 
    LForefootLat = LMat[:,7:15,:5] #Opposite ":," sequence from R side
    LForefootMed = LMat[:,7:15,5:] 
    LffToe = LMat[:,:15,:]
    LffToeLat = LMat[:, :15, :5]
    LffToeMed = LMat[:, :15, 5:]
    LMidfoot = LMat[:,15:25,:] 
    LMidfootLat = LMat[:,15:25,:5] #Opposite ":," sequence from R side
    LMidfootMed = LMat[:,15:25,5:] 
    LHeel = LMat[:,25:, :] 
    LHeelLat = LMat[:,25:,:5] #Opposite ":," sequence from R side
    LHeelMed = LMat[:,25:, 5:]
    LLateral = LMat[:,:5]
    LMedial = LMat[:,:,5:]
           
    RToe = RMat[:,:7,:]
    RToeLat = RMat[:,:7,4:]
    RToeMed = RMat[:,:7,:4]
    RForefoot = RMat[:,7:15, :]
    RForefootLat = RMat[:,7:15,4:]
    RffToe = RMat[:,:15, :]
    RffToeLat = RMat[:, :15, 4:]
    RffToeMed = RMat[:, :15, :4]
    RForefootMed = RMat[:,7:15,:4]
    RMidfoot = RMat[:,15:25,:]
    RMidfootLat = RMat[:,15:25,4:]
    RMidfootMed = RMat[:,15:25,:4]
    RHeel = RMat[:,25:, :]
    RHeelLat = RMat[:,25:,4:]
    RHeelMed = RMat[:,25:, :4]
    RLateral = RMat[:,:,4:]
    RMedial = RMat[:,:,:4]
    
    RForce = np.mean(RMat, axis = (1,2))*6895*0.014699
    LForce = np.mean(LMat, axis = (1,2))*6895*0.014699
    
    [RT,LT] = findTurns(RForce, LForce, freq)
    
    result = tsData(LMat, LToe, LToeLat, LToeMed, 
                     LForefoot, LForefootLat, LForefootMed, LffToe, LffToeLat, LffToeMed, 
                     LMidfoot, LMidfootLat, LMidfootMed, 
                     LHeel, LHeelLat, LHeelMed, LLateral, LMedial,
                     RMat, RToe, RToeLat, RToeMed,
                     RForefoot, RForefootLat, RForefootMed, RffToe, RffToeLat, RffToeMed,
                     RMidfoot, RMidfootLat, RMidfootMed,
                     RHeel, RHeelLat, RHeelMed, RLateral, RMedial, RForce, LForce, RT, LT,
                     config, subj, dat
                     
                    )
    
    return(result)



#############################################################################################################################################################

# Read in files
# only read .asc files for this work
fPath = 'C:/Users/Kate.Harrison/Boa Technology Inc/PFL Team - General/Testing Segments/Snow Performance/EH_Alpine_FullBootvsShell_Mech_Jan2024/XSENSOR/Cropped/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) ]

badFileList = []

for fName in entries:
    
    
    config = []
    subject = []
    ct = []

    # initialize outcome variables
    maxmaxToes = []
    ffPmidstance = []
    ffAreaMid = []
    heelAreaLate = []
    heelPLate = []
    medPmidstance = []
    medAreamidstance = []
    medPropMid = []
    uphillMax = []
    downhillMax = []
    RFD = []
    TurnDirection = []
    

    try: 
        #fName = entries[0]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        
        
        # Make sure the files are named FirstLast_Config_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        
        #dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
    
        tmpDat = createTSmat(fName)
        tmpDat.plotAvgPressure()
        
        answer = True # if data check is off. 
        if data_check == 1:
            plt.figure()
            plt.plot(tmpDat.RForce, label = 'Right Foot Total Force')
            for i in range(len(tmpDat.RHS)):

                plt.axvspan(tmpDat.RT[i], tmpDat.LT[i], color = 'lightgray', alpha = 0.5)
                answer = messagebox.askyesno("Question","Is data clean?")
        
        if answer == False:
            plt.close('all')
            print('Adding file to bad file list')
            badFileList.append(fName)
    
        if answer == True:
            plt.close('all')
            print('Estimating point estimates')
            

            for i in range(len(tmpDat.RTurns)):
                
                #i = 1
                config.append(tmpDat.config)
                subject.append(tmpDat.subject)
               
                frames = tmpDat.LTurns[i] - tmpDat.RTurns[i]
                
                
                ct.append(frames/freq)
                pct10 = tmpDat.RTurns[i] + round(frames*.1)
                pct40 = tmpDat.RTurns[i] + round(frames*.4)
                pct50 = tmpDat.RTurns[i] + round(frames*.5)
                pct60 = tmpDat.RTurns[i] + round(frames*.6)
                pct90 = tmpDat.RTurns[i] + round(frames*.9)
                
                maxmaxToes.append(np.max(tmpDat.LToe[tmpDat.RTurns[i]:tmpDat.LTurns[i]])*6.895)
                ffPmidstance.append(np.mean(tmpDat.LffToe[pct40:pct60,:,:])*6.895)
                ffAreaMid.append(np.count_nonzero(tmpDat.LffToe[pct40:pct60, :,:])/(pct60 - pct40)/68*100)
                heelAreaLate.append(np.count_nonzero(tmpDat.LHeel[pct50:tmpDat.LTurns[i], :, :])/(tmpDat.LTurns[i] - pct50)/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                heelPLate.append(np.mean(tmpDat.LHeel[pct90:tmpDat.LTurns[i], :, :])*6.895)
                medPmidstance.append(np.mean(tmpDat.LMedial[pct40:pct60, :, :])*6.895)
                medAreamidstance.append(np.count_nonzero(tmpDat.LMedial[pct40:pct60, :, :])/(pct60-pct40)/82*100)
                medPropMid.append(np.sum(tmpDat.LMedial[pct40:pct60, :, :])/np.sum(tmpDat.LMat[pct40:pct60, :, :]))
                uphillMax.append(np.max(tmpDat.RForce[tmpDat.RTurns[i]:tmpDat.LTurns[i]]))
                downhillMax.append(np.max(tmpDat.LForce[tmpDat.RTurns[i]:tmpDat.LTurns[i]]))
                RFD.append(np.max(np.gradient(tmpDat.LForce[tmpDat.RTurns[i]:tmpDat.LTurns[i]], .001)))
                TurnDirection.append('Right')
                    
            for i in range(len(tmpDat.LTurns)-1):
                
                #i = 0
                config.append(tmpDat.config)
                subject.append(tmpDat.subject)
               
                frames = tmpDat.RTurns[i+1] - tmpDat.LTurns[i]
                
                
                ct.append(frames/freq)
                pct10 = tmpDat.LTurns[i] + round(frames*.1)
                pct40 = tmpDat.LTurns[i] + round(frames*.4)
                pct50 = tmpDat.LTurns[i] + round(frames*.5)
                pct60 = tmpDat.LTurns[i] + round(frames*.6)
                pct90 = tmpDat.LTurns[i] + round(frames*.9)
                
                maxmaxToes.append(np.max(tmpDat.RToe[tmpDat.LTurns[i]:tmpDat.RTurns[i+1]])*6.895)
                ffPmidstance.append(np.mean(tmpDat.RffToe[pct40:pct60,:,:])*6.895)
                ffAreaMid.append(np.count_nonzero(tmpDat.RffToe[pct40:pct60, :,:])/(pct60 - pct40)/68*100)
                heelAreaLate.append(np.count_nonzero(tmpDat.RHeel[pct50:tmpDat.RTurns[i+1], :, :])/(tmpDat.RTurns[i+1] - pct50)/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
                heelPLate.append(np.mean(tmpDat.RHeel[pct90:tmpDat.RTurns[i+1], :, :])*6.895)
                medPmidstance.append(np.mean(tmpDat.RMedial[pct40:pct60, :, :])*6.895)
                medAreamidstance.append(np.count_nonzero(tmpDat.RMedial[pct40:pct60, :, :])/(pct60-pct40)/82*100)
                medPropMid.append(np.sum(tmpDat.RMedial[pct40:pct60, :, :])/np.sum(tmpDat.RMat[pct40:pct60, :, :]))
                uphillMax.append(np.max(tmpDat.LForce[tmpDat.LTurns[i]:tmpDat.RTurns[i+1]]))
                downhillMax.append(np.max(tmpDat.RForce[tmpDat.LTurns[i]:tmpDat.RTurns[i+1]]))
                RFD.append(np.max(np.gradient(tmpDat.RForce[tmpDat.LTurns[i]:tmpDat.RTurns[i+1]], .001)))
                TurnDirection.append('Left')

        outcomes = pd.DataFrame({'Subject': list(subject), 'Config':list(config), 'TurnDirection':list(TurnDirection), 'TurnTime':list(ct),
                                 'maxmaxToes':list(maxmaxToes),
                                 'ffP_Mid':list(ffPmidstance), 'ffArea_Mid':list(ffAreaMid),
                                 'heelPressure_late':list(heelPLate), 'heelAreaP':list(heelAreaLate),  
                                 'medP_mid':list(medPmidstance), 'medArea_mid':list(medAreamidstance), 'medPropMid':list(medPropMid),
                                 'uphillMaxF':list(uphillMax), 'downhillMax':list(downhillMax), 'RFD':list(RFD)
                                 })

        outfileName = fPath + '0_CompiledResults_4.csv'
        if save_on == 1:
            if os.path.exists(outfileName) == False:
            
                outcomes.to_csv(outfileName, header=True, index = False)
        
            else:
                outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
        
        
        
    except:
            print('Not usable data')
            badFileList.append(fName)             
            
            
            
            
            
        


