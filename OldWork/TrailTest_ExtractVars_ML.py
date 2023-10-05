# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:44:05 2022
Script to process trail running plantar pressure

@author: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import addcopyfighandler
import scipy.interpolate
import scipy
import scipy.signal as sig

fwdLook = 30
fThresh = 50
freq = 100 # sampling frequency (intending to change to 150 after this study)
save_on = 1
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force, fThreshold):
    ric = []
    for step in range(len(force)-1):
        if force[step] < fThreshold and force[step + 1] >= fThreshold:
            ric.append(step)
    return ric

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force, fThreshold):
    rto = []
    for step in range(len(force)-1):
        if force[step] >= fThreshold and force[step + 1] < fThreshold:
            rto.append(step + 1)
    return rto

def trimTakeoffs(landings, takeoffs):
    if takeoffs[0] < landings[0]:
        del(takeoffs[0])
    return(takeoffs)

def trimLandings(landings, trimmedTakeoffs):
    if landings[len(landings)-1] > trimmedTakeoffs[len(trimmedTakeoffs)-1]:
        del(landings[-1])
    return(landings)

def trimForce(inputDFCol, threshForce):
    forceTot = inputDFCol
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

def zeroInsoleForce(vForce,freq):
    newForce = vForce
    # Quasi-constants
    zcoeff = 0.7
    
    windowSize = round(0.8*freq)
    
    zeroThresh = 100
    
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
    # Quasi-constants
    zcoeff = 0.9
    
    windowSize = round(0.8*freq)
    
    n = 5
    
    # Filter the forces
    # Set up a 2nd order 10 Hz low pass buttworth filter
    # cut = 10
    # w = cut / (freq / 2) # Normalize the frequency
    # b, a = sig.butter(2, w, 'low')
    # # Filter the force signal
    # vForce = sig.filtfilt(b, a, vForce)
        
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
    # Remove events that are too close to the start/end of the trial
    nearHS = nearHS[(nearHS > windowSize)*(nearHS < len(vForce)-windowSize)]
    nearTO = nearTO[(nearTO > windowSize)*(nearTO < len(vForce)-windowSize)]
    
    HS = []
    for idx in nearHS:
        for ii in range(idx,idx-windowSize,-1):
            if vForce[ii] == 0 or vForce[ii] < vForce[ii-n] or ii == idx-windowSize+1:
                HS.append(ii)
                break
    # Eliminate detections above 100 N
    HS = np.unique(HS)
    HS = np.array(HS)
    HS = HS[vForce[HS]<200]
    
    # Only search for toe-offs that correspond to heel-strikes
    TO = []
    for ii in range(len(HS)):
        tmp = np.where(nearTO > HS[ii])[0]
        if len(tmp) > 0:
            idx = nearTO[tmp[0]]
            for jj in range(idx,idx+windowSize):
                if vForce[jj] == 0 or vForce[jj] < vForce[jj+n] or jj == idx+windowSize-1:
                    TO.append(jj)
                    break
        else:
            np.delete(HS,ii)
            
            
            
            
    #     1
    
    
    # for idx in nearTO:       
    #     for ii in range(idx,idx+windowSize):
    #         if vForce[ii] == 0 or vForce[ii] < vForce[ii+n] or ii == idx+windowSize-1:
    #             TO.append(ii)
    #             break
    # # Eliminate detections above 100 N
    # TO = np.array(TO)
    # TO = TO[vForce[TO]<200]
    
    # # Only take unique events
    
    # TO = np.unique(TO)
      
    # # Remove extra HS/TO
    # if TO[0] < HS[0]:
    #     TO = TO[1:]
    # if HS[-1] > TO[-1]:
    #     HS = HS[0:-1]
    return(HS,TO)
    


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

def filt_press_sig(in_sig,cut):
    # Set up a 2nd order 50 Hz low pass buttworth filter
    freq = 100
    w = cut / (freq / 2) # Normalize the frequency
    b, a = sig.butter(2, w, 'low')
    
    # Filter the IMU signals
    sig_filt = sig.filtfilt(b, a, in_sig)
    
    return(sig_filt)

def ZeroPressSig(signal,HS,TO):
    
    swingsig = []
    
    for ii in range(len(HS)-1):
        swingsig.extend(signal[TO[ii]:HS[ii+1]])
    
    signal_up = signal-np.median(swingsig)
    
    return signal_up
    
    
    

# Grab the GPS data for the timing of segments
GPStiming = pd.read_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\CombinedGPS.csv')
# Define the path: This is the way
fPath = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\PressureData\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

# Preallocate
oSubject = []
oConfig = []
oLabel = np.array([])
oSesh = []


heel_con = []
heel_con_1 = [] # 1st half of stance
heel_con_2 = [] # 2nd half of stance

m_heel_ratio = []
m_mid_ratio = []
m_met_ratio = []
m_toe_ratio = []

m_heelPP_lat = []
m_midPP_lat = []
m_metPP_lat = []
m_toePP_lat = []

avg_toe_force = []


poorR = ['S15','S16','S17','S19','S20','S23','S24','S26','S29','S31']

for ii in range(0,len(entries)):
    print(entries[ii])
    dat = pd.read_csv(fPath+entries[ii], sep=',', header = 'infer')
    Subject = entries[ii].split(sep = "_")[0]
    Config = entries[ii].split(sep="_")[1]
    Sesh = int(entries[ii].split(sep = "_")[2][0])
    
    # Find the correct GPS trial
    GPStrial = np.array(GPStiming.Subject == Subject) * np.array(GPStiming.Config == Config) * np.array(GPStiming.Sesh == Sesh)
    
    # Total left force: 16
    # Total right force: 28
    # left heel force: 40
    # right heel force: 52
    # left midfoot force: 64
    # right midfoot force: 76
    # left met force: 88
    # right met force: 100
    # left toe force: 112
    # right toe force: 124
    
    # Convert the force into newtons, pressures into kPa
    L_tot_force = np.array(dat.iloc[:,16])*4.44822
    L_tot_force = zeroInsoleForce(L_tot_force,100)
    [LHS,LTO] = findGaitEvents(L_tot_force,100)
    
    R_tot_force = np.array(dat.iloc[:,28])*4.44822
    R_tot_force = zeroInsoleForce(R_tot_force,100)
    [RHS,RTO] = findGaitEvents(R_tot_force,100)
    
    
    L_heel_con = np.array((dat.iloc[:,37]+dat.iloc[:,49])/(dat.iloc[:,38]+dat.iloc[:,50])*100)
    R_heel_con = np.array((dat.iloc[:,61]+dat.iloc[:,73])/(dat.iloc[:,62]+dat.iloc[:,74])*100)
    
    L_heelPP_lat = np.array(dat.iloc[:,36]-np.min(dat.iloc[:,36]))*6.89476
    R_heelPP_lat = np.array(dat.iloc[:,60]-np.min(dat.iloc[:,60]))*6.89476
    
    L_midPP_lat = np.array(dat.iloc[:,84]-np.min(dat.iloc[:,84]))*6.89476
    R_midPP_lat = np.array(dat.iloc[:,108]-np.min(dat.iloc[:,108]))*6.89476
    
    L_metPP_lat = np.array(dat.iloc[:,132]-np.min(dat.iloc[:,132]))*6.89476
    R_metPP_lat = np.array(dat.iloc[:,156]-np.min(dat.iloc[:,156]))*6.89476
    
    L_toePP_lat = np.array(dat.iloc[:,180]-np.min(dat.iloc[:,180]))*6.89476
    R_toePP_lat = np.array(dat.iloc[:,204]-np.min(dat.iloc[:,204]))*6.89476
    
    
    L_heel_ratio = np.array((dat.iloc[:,40]-np.min(dat.iloc[:,40]))/(dat.iloc[:,52]-np.min(dat.iloc[:,52])))
    R_heel_ratio = np.array((dat.iloc[:,64]-np.min(dat.iloc[:,64]))/(dat.iloc[:,76]-np.min(dat.iloc[:,76])))
    
    L_mid_ratio = np.array((dat.iloc[:,88]-np.min(dat.iloc[:,88]))/(dat.iloc[:,100]-np.min(dat.iloc[:,100])))
    R_mid_ratio = np.array((dat.iloc[:,112]-np.min(dat.iloc[:,112]))/(dat.iloc[:,124]-np.min(dat.iloc[:,124])))
    
    L_met_ratio = np.array((dat.iloc[:,136]-np.min(dat.iloc[:,136]))/(dat.iloc[:,148]-np.min(dat.iloc[:,148])))
    R_met_ratio = np.array((dat.iloc[:,160]-np.min(dat.iloc[:,160]))/(dat.iloc[:,172]-np.min(dat.iloc[:,172])))
    
    L_toe_ratio = np.array((dat.iloc[:,184]-np.min(dat.iloc[:,184]))/(dat.iloc[:,196]-np.min(dat.iloc[:,196])))
    R_toe_ratio = np.array((dat.iloc[:,208]-np.min(dat.iloc[:,208]))/(dat.iloc[:,220]-np.min(dat.iloc[:,220])))
    
    
    L_toe_force = dat.iloc[:,196]-np.min(dat.iloc[:,196]) # Toe force
    R_toe_force = dat.iloc[:,220]-np.min(dat.iloc[:,220]) # Toe force
    
    # There should HS that are close to one another: from the 3 hops.
    # This indicates the start of the trial
    # Let's look at the first n HS to see if there are similarities
    
    start_LHS = []; start_RHS = []
    for jj in range(15):
        jump_check = np.where(np.abs(LHS[jj] - np.array(RHS[0:15])) < 10)
        if jump_check[0].size > 0:
            print('Jump Found')
            start_LHS = jj+1
            start_RHS = np.argmin(np.abs(LHS[jj] - np.array(RHS[0:15])))+1
        
    
    LHS = LHS[start_LHS:]
    LTO = LTO[start_LHS:]
    RHS = RHS[start_RHS:]
    RTO = RTO[start_RHS:]
       
    # Remove strides that have a peak GRF below 1000 N
    # Remove strides that are below 0.5 and above 1.5 seconds
    LGS = []    
    # May need to exclude poor toe-off dectections here as well
    for jj in range(len(LHS)-1):
        if np.max(L_tot_force[LHS[jj]:LTO[jj]]) > 1000:
            if (LHS[jj+1] - LHS[jj]) > 0.5*freq and LHS[jj+1] - LHS[jj] < 1.5*freq:
                LGS.append(jj)
    
    RGS = []
    for jj in range(len(RHS)-1):
        if np.max(R_tot_force[RHS[jj]:RTO[jj]]) > 1000:
            if (RHS[jj+1] - RHS[jj]) > 0.5*freq and RHS[jj+1] - RHS[jj] < 1.5*freq:
                RGS.append(jj)
    
        
    # Compute metrics of interest
    for jj in LGS:
        heel_con.append(np.mean(L_heel_con[LHS[jj]:LTO[jj]]))
        # Estimate for 1st and 2nd half of stance
        heel_con_1.append(np.mean(L_heel_con[LHS[jj]:LHS[jj]+int(.5*(LTO[jj]-LHS[jj]))]))
        heel_con_2.append(np.mean(L_heel_con[LHS[jj]+int(.5*(LTO[jj]-LHS[jj])):LTO[jj]]))
        
        # Examine ratio of forces
        m_heel_ratio.append(np.max(L_heel_ratio[LHS[jj]:LTO[jj]]))
        m_mid_ratio.append(np.max(L_mid_ratio[LHS[jj]:LTO[jj]]))
        m_met_ratio.append(np.max(L_met_ratio[LHS[jj]:LTO[jj]]))
        m_toe_ratio.append(np.max(L_toe_ratio[LHS[jj]:LTO[jj]]))
        
        # Examine the maximum lateral pressures
        m_heelPP_lat.append(np.max(L_heelPP_lat[LHS[jj]:LTO[jj]]))
        m_midPP_lat.append(np.max(L_midPP_lat[LHS[jj]:LTO[jj]]))
        m_metPP_lat.append(np.max(L_metPP_lat[LHS[jj]:LTO[jj]]))
        m_toePP_lat.append(np.max(L_toePP_lat[LHS[jj]:LTO[jj]]))
        
        # Average toe force
        avg_toe_force.append(np.mean(L_toe_force[LHS[jj]:LTO[jj]]))
        

    
    if Subject not in poorR:
        for jj in RGS:
            heel_con.append(np.mean(R_heel_con[RHS[jj]:RTO[jj]]))
            # Estimate for 1st and 2nd half of stance
            heel_con_1.append(np.mean(R_heel_con[RHS[jj]:RHS[jj]+int(.5*(RTO[jj]-RHS[jj]))]))
            heel_con_2.append(np.mean(R_heel_con[RHS[jj]+int(.5*(RTO[jj]-RHS[jj])):RTO[jj]]))
            
            # Examine ratio of forces
            m_heel_ratio.append(np.max(R_heel_ratio[RHS[jj]:RTO[jj]]))
            m_mid_ratio.append(np.max(R_mid_ratio[RHS[jj]:RTO[jj]]))
            m_met_ratio.append(np.max(R_met_ratio[RHS[jj]:RTO[jj]]))
            m_toe_ratio.append(np.max(R_toe_ratio[RHS[jj]:RTO[jj]]))
            
            # Examine the maximum lateral pressures
            m_heelPP_lat.append(np.max(R_heelPP_lat[RHS[jj]:RTO[jj]]))
            m_midPP_lat.append(np.max(R_midPP_lat[RHS[jj]:RTO[jj]]))
            m_metPP_lat.append(np.max(R_metPP_lat[RHS[jj]:RTO[jj]]))
            m_toePP_lat.append(np.max(R_toePP_lat[RHS[jj]:RTO[jj]]))
            
            # Average toe force
            avg_toe_force.append(np.mean(R_toe_force[RHS[jj]:RTO[jj]]))
            
            
    
    
    
    # Create Labels
    Llabel = np.zeros([len(LGS),1])
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
    # Uphill label
    idx = RHS[RGS]/freq < float(GPStiming.EndS1[GPStrial])
    Rlabel[idx] = 1
    # Top label
    idx = (RHS[RGS]/freq > float(GPStiming.StartS2[GPStrial]))*(RHS[RGS]/freq < float(GPStiming.EndS2[GPStrial]))
    Rlabel[idx] = 2
    # Bottom label
    idx = RHS[RGS]/freq > float(GPStiming.StartS3[GPStrial])
    Rlabel[idx] = 3
    
    if Subject in poorR:
        oSubject = oSubject + [Subject]*len(LGS)
        oConfig = oConfig + [Config]*len(LGS)
        oSesh = oSesh + [Sesh]*len(LGS)
        oLabel = np.concatenate((oLabel,Llabel),axis = None)
    else:
        oSubject = oSubject + [Subject]*len(LGS) + [Subject]*len(RGS)
        oConfig = oConfig + [Config]*len(LGS) + [Config]*len(RGS)
        oSesh = oSesh + [Sesh]*len(LGS) + [Sesh]*len(RGS)
        oLabel = np.concatenate((oLabel,Llabel,Rlabel),axis = None)
    
    # plt.figure(ii)
    # plt.subplot(2,1,1)
    # plt.plot(intp_strides(L_tot_force,LHS,LTO,LGS))
    # plt.ylabel('Insole Force [N]')
    
    # plt.subplot(2,1,2)
    # plt.plot(intp_strides(R_tot_force,RHS,RTO,RGS))
    # plt.ylabel('Insole Force [N]')
    1



outcomes = pd.DataFrame({'Subject':list(oSubject), 'Config': list(oConfig),'Sesh': list(oSesh),
                         'Label':list(oLabel), 'HeelCon':list(heel_con),'HeelCon1':list(heel_con_1), 'HeelCon2':list(heel_con_2),
                         'm_heel_ratio':list(m_heel_ratio),'m_mid_ratio':list(m_mid_ratio),'m_met_ratio':list(m_met_ratio),'m_toe_ratio':list(m_toe_ratio),
                         'm_heelPP_lat':list(m_heelPP_lat),'m_midPP_lat':list(m_midPP_lat),'m_metPP_lat':list(m_metPP_lat),'m_toePP_lat':list(m_toePP_lat),
                         'avg_toe_med':list(avg_toe_force)})

if save_on == 1:
    outcomes.to_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\PressureOutcomes.csv',header=True)
elif save_on == 2: 
    outcomes.to_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\PressureOutcomes.csv',mode = 'a', header=False)


# c = ['r','r','g','g']
# fig, ax = plt.subplots()
# ax.bar(np.array(range(1,len(L_heel_var_avg)+1)),L_heel_var_avg,yerr=L_heel_var_std,color=c)
# ax.set_ylabel('Heel Variation')
# ax.set_xticks(np.array(range(1,len(L_heel_var_avg)+1)))
# ax.set_xticklabels(Config)
    
# fig, ax = plt.subplots()
# ax.bar(np.array(range(1,len(L_toe_max_avg[4:])+1)),L_toe_max_avg[4:],yerr=L_toe_max_std[4:],color=c)
# ax.set_ylabel('Maximum Toe Pressure [Psi]')
# ax.set_xticks(np.array(range(1,len(L_toe_max_avg[4:])+1)))
# ax.set_xticklabels(Config[4:])

# # fig, ax = plt.subplots()
# # ax.bar(np.array(range(1,len(L_toe_max_avg[4:])+1)),L_heel_con_avg[4:],yerr=L_heel_con_std[4:],color=c)
# # ax.set_ylabel('Average Heel Contact [%]')
# # ax.set_xticks(np.array(range(1,len(L_toe_max_avg[4:])+1)))
# # ax.set_xticklabels(Config[4:])

# fig, ax = plt.subplots()
# ax.bar(np.array(range(1,len(L_toe_max_avg[4:])+1)),L_met_max_avg[4:],yerr=L_met_max_std[4:],color=c)
# ax.set_ylabel('Maximum Metatarsal Pressure [Psi]')
# ax.set_xticks(np.array(range(1,len(L_met_max_avg[4:])+1)))
# ax.set_xticklabels(Config[4:])
