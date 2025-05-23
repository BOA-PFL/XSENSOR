# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

@author: Kate.Harrison
Last Updated: Eric Honert
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
from XSENSORFunctions import readXSENSORFile, delimitTrial, createTSmat, zeroInsoleForce, findGaitEvents


# Read in files
# only read .csv files for this work
fPath = 'Z:\\Testing Segments\\EndurancePerformance\\2024\\EH_Trail_AltraMidsole_Perf_Mar24\\Xsensor\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) ]


GPStiming = pd.read_csv('Z:\\Testing Segments\\EndurancePerformance\\2024\\EH_Trail_AltraMidsole_Perf_Mar24\\GPS\\CombinedGPS.csv')
SeshConf = GPStiming[['Config', 'Sesh']]

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

###############################################################################
# Function list specific for this code
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
    
    # Find the correct GPS trial
    #GPStrial = np.array(GPStiming.Subject == subName) * np.array(GPStiming.Config == ConfigTmp) * np.array(GPStiming.Sesh == Sesh)
    GPStrial = np.array(GPStiming.Subject == subName) * np.array(GPStiming.Config == ConfigTmp)
    
    tmpDat = readXSENSORFile(fName,fPath)
    # tmpDat = delimitTrial(tmpDat,fName,fPath) make one specfic for trail running
    tmpDat = createTSmat(fName, fPath, tmpDat)
    
    # Estimate the frequency from the time stamps
    freq = 1/np.mean(np.diff(tmpDat.time))
    if freq < 50:
        print('Time off for: ' + fName)
        print('Previous freq est:' + str(freq))
        print('Defaulting to 50 Hz for contact event estimations')
        freq = 50
    
    # Compute foot contact events
    tmpDat.LForce = zeroInsoleForce(tmpDat.LForce,freq)
    [LHS,LTO] = findGaitEvents(tmpDat.LForce,freq)
    
    tmpDat.RForce = zeroInsoleForce(tmpDat.RForce,freq)
    [RHS,RTO] = findGaitEvents(tmpDat.RForce,freq)

    
    start_LHS = []; start_RHS = []
    if subName == 'ChadPrichard': # change 25 to more or less depending on time standing around before start of run
        checkwnd = 25
        checkless = 20
    else:  
        checkwnd = 15 
        checkless = 15

        
        
    for jj in range(checkwnd): 
        jump_check = np.where(np.abs(LHS[jj] - np.array(RHS[0:checkwnd])) < checkless) 
        if jump_check[0].size > 0:
            print('Jump Found')
            start_LHS = jj+1
            start_RHS = np.argmin(np.abs(LHS[jj] - np.array(RHS[0:checkwnd])))+1
        
    
    LHS = LHS[start_LHS:]
    LTO = LTO[start_LHS:]
    RHS = RHS[start_RHS:]
    RTO = RTO[start_RHS:]
       
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
            frames = RTO[kk] - RHS[kk]
            ct.append(tmpDat.time[RTO[ii]]-tmpDat.time[RHS[ii]])
            pct10 = RHS[kk] + round(frames*.1)
            pct40 = RHS[kk] + round(frames*.4)
            pct50 = RHS[kk] + round(frames*.5)
            pct60 = RHS[kk] + round(frames*.6)
            pct90 = RHS[kk] + round(frames*.9)
            

            
            maxmaxToes.append(np.max(tmpDat.RplantarToe[RHS[kk]:RTO[kk]])*6.895)
            toePmidstance.append(np.mean(tmpDat.RplantarToe[pct40:pct60,:,:])*6.895)
            toeAreamidstance.append(np.count_nonzero(tmpDat.RplantarToe[pct40:pct60,:,:])/(pct60 - pct40)/39*100)
            ffAreaLate.append(np.count_nonzero(tmpDat.RplantarForefoot[pct90:RTO[kk], :,:])/(RTO[kk] - pct90)/68*100)
            ffPLate.append(np.mean(tmpDat.RplantarForefoot[pct90:RTO[kk], :, :])*6.895)
            ffPMaxLate.append(np.max(tmpDat.RplantarForefoot[pct90:RTO[kk], :, :]))
            ffAreaMid.append(np.count_nonzero(tmpDat.RplantarForefoot[pct40:pct60, :,:])/(pct60 - pct40)/68*100)
            ffPMid.append((np.mean(tmpDat.RplantarForefoot[pct40:pct60, :, :]))*6.895)
            
            mfAreaLate.append(np.count_nonzero(tmpDat.RplantarMidfoot[pct90:RTO[kk], :,:])/(RTO[kk] - pct90)/70*100)
            mfPLate.append(np.mean(tmpDat.RplantarMidfoot[pct90:RTO[kk], :, :])*6.895)
            mfAreaMid.append(np.count_nonzero(tmpDat.RplantarMidfoot[pct40:pct60, :,:])/(pct60 - pct40)/70*100)
            mfPMid.append((np.mean(tmpDat.RplantarMidfoot[pct40:pct60, :, :]))*6.895)
            mfmax.append((np.max(tmpDat.RplantarMidfoot[RHS[kk]:RTO[kk], :, :]))*6.895)
            
            heelAreaLate.append(np.count_nonzero(tmpDat.RplantarHeel[pct50:RTO[kk], :, :])/(RTO[kk] - pct50)/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
            heelPLate.append(np.mean(tmpDat.RplantarHeel[pct90:RTO[kk], :, :])*6.895)
            heelPmax.append(np.max(tmpDat.RplantarHeel[RHS[kk]:RTO[kk], :, :])*6.895)

            latPmidstance.append(np.mean(tmpDat.RplantarLateral[pct40:pct60, :, :])*6.895)
            latAreamidstance.append(np.count_nonzero(tmpDat.RplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/138*100)
            latPLate.append(np.mean(tmpDat.RplantarLateral[pct90:RTO[kk], :, :])*6.895)
            latAreaLate.append(np.count_nonzero(tmpDat.RplantarLateral[pct90:RTO[kk], :, :])/(RTO[kk] - pct90)/138*100)
            medPmidstance.append(np.mean(tmpDat.RplantarMedial[pct40:pct60, :, :])*6.895)
            medAreamidstance.append(np.count_nonzero(tmpDat.RplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/82*100)
            medPLate.append(np.mean(tmpDat.RplantarMedial[pct90:RTO[kk], :, :])*6.895)
            medAreaLate.append(np.count_nonzero(tmpDat.RplantarMedial[pct90:RTO[kk], :, :])/(RTO[kk]-pct90)/82*100)
            
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
            frames = LTO[kk] - LHS[kk]
            ct.append(tmpDat.time[LTO[ii]]-tmpDat.time[LHS[ii]])
            pct10 = LHS[kk] + round(frames*.1)
            pct40 = LHS[kk] + round(frames*.4)
            pct50 = LHS[kk] + round(frames*.5)
            pct60 = LHS[kk] + round(frames*.6)
            pct90 = LHS[kk] + round(frames*.9)
            
        
            
            maxmaxToes.append(np.max(tmpDat.LplantarToe[LHS[kk]:LTO[kk]])*6.895)
            toePmidstance.append(np.mean(tmpDat.LplantarToe[pct40:pct60,:,:])*6.895)
            toeAreamidstance.append(np.count_nonzero(tmpDat.LplantarToe[pct40:pct60,:,:])/(pct60 - pct40)/39*100)
            ffAreaLate.append(np.count_nonzero(tmpDat.LplantarForefoot[pct90:LTO[kk], :,:])/(LTO[kk] - pct90)/68*100)
            ffPLate.append(np.mean(tmpDat.LplantarForefoot[pct90:LTO[kk], :, :])*6.895)
            ffPMaxLate.append(np.max(tmpDat.LplantarForefoot[pct90:LTO[kk], :, :]))
            ffAreaMid.append(np.count_nonzero(tmpDat.LplantarForefoot[pct40:pct60, :,:])/(pct60 - pct40)/68*100)
            ffPMid.append((np.mean(tmpDat.LplantarForefoot[pct40:pct60, :, :]))*6.895)
            
            mfAreaLate.append(np.count_nonzero(tmpDat.LplantarMidfoot[pct90:LTO[kk], :,:])/(LTO[kk] - pct90)/70*100)
            mfPLate.append(np.mean(tmpDat.LplantarMidfoot[pct90:LTO[kk], :, :])*6.895)
            mfAreaMid.append(np.count_nonzero(tmpDat.LplantarMidfoot[pct40:pct60, :,:])/(pct60 - pct40)/70*100)
            mfPMid.append((np.mean(tmpDat.LplantarMidfoot[pct40:pct60, :, :]))*6.895)
            mfmax.append((np.max(tmpDat.RplantarMidfoot[LHS[kk]:LTO[kk], :, :]))*6.895)
            
            heelAreaLate.append(np.count_nonzero(tmpDat.LplantarHeel[pct50:LTO[kk], :, :])/(LTO[kk] - pct50)/43*100) # making this from 50% stance time to toe off to match big data. Consider switing to 90% to toe off?
            heelPLate.append(np.mean(tmpDat.LplantarHeel[pct90:LTO[kk], :, :])*6.895)
            heelPmax.append(np.max(tmpDat.RplantarHeel[LHS[kk]:LTO[kk], :, :])*6.895)
        
            latPmidstance.append(np.mean(tmpDat.LplantarLateral[pct40:pct60, :, :])*6.895)
            latAreamidstance.append(np.count_nonzero(tmpDat.LplantarLateral[pct40:pct60, :, :])/(pct60-pct40)/138*100)
            latPLate.append(np.mean(tmpDat.LplantarLateral[pct90:LTO[kk], :, :])*6.895)
            latAreaLate.append(np.count_nonzero(tmpDat.LplantarLateral[pct90:LTO[kk], :, :])/(LTO[kk] - pct90)/138*100)
            medPmidstance.append(np.mean(tmpDat.LplantarMedial[pct40:pct60, :, :])*6.895)
            medAreamidstance.append(np.count_nonzero(tmpDat.LplantarMedial[pct40:pct60, :, :])/(pct60-pct40)/82*100)
            medPLate.append(np.mean(tmpDat.LplantarMedial[pct90:LTO[kk], :, :])*6.895)
            medAreaLate.append(np.count_nonzero(tmpDat.LplantarMedial[pct90:LTO[kk], :, :])/(LTO[kk]-pct90)/82*100)
            
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
    

    outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Sesh' : list(sesh), 'Side':list(side), 'Label':list(oLabel), 'ContactTime':list(ct),
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
            
            

