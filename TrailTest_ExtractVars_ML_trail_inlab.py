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
freq = 100
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

def trimgaitevents(HS,TO):
    if HS[0] > TO[0]:
        TO = TO[1:]
        
    if HS[-1] > TO[-1]:
        HS = HS[0:-1]
    return(HS,TO)

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
    # Quasi-constants
    zcoeff = 1.6
    
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

def threehopstart(HS,TO,freq):
    # Identify the start of the trial
    # Counters
    jc = 0  # jump counter
    stc = 0 # start trial counter
    approx_CT = np.diff(HS)
    
    for jj in range(0,10):
        if approx_CT[jj] < 75:
            jc = jc+1
        if jc >= 2 and approx_CT[jj] > 150:
            lasthop = HS[jj]
            idx = (HS > lasthop + 10*freq)*(HS < lasthop + 55*freq)
            HS = HS[idx]
            idx = (TO > lasthop + 10*freq)*(TO < lasthop + 55*freq)
            TO = TO[idx] 
            stc = 1
    
    if stc == 0:
        print('Warning: 3 hops not found')
        
    if HS[0] > TO[0]:
        TO = TO[1:]
        
    if HS[-1] > TO[-1]:
        HS = HS[0:-1]
    
    return(HS,TO)
    
# Define the path: This is the way
fPath = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabData\\Pressure\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

save_on = 2
debug = 0

# Preallocate
oSubject = []
oConfig = []
oSpeed = []
oLabel = []
oSesh = []
oSide = []


heel_con = []
heel_con_1 = [] # 1st half of stance
heel_con_2 = [] # 2nd half of stance
toe_con = []

m_heel_ratio = []
m_mid_ratio = []
m_met_ratio = []
m_toe_ratio = []

m_heelPP_lat = []
m_heelPP = []
m_midPP_lat = []
m_metPP_lat = []
m_toePP_lat = []
m_toePP = []

avg_toe_force = []

poorL = ['S07','S10']

for ii in range(270,len(entries)):
    print(entries[ii])
    dat = pd.read_csv(fPath+entries[ii], sep=',',skiprows = 1, header = 'infer')
    Subject = entries[ii].split(sep = "_")[0]
    Config = entries[ii].split(sep="_")[1]
    Speed = entries[ii].split(sep="_")[2]
    Slope = entries[ii].split(sep="_")[3]
    Sesh = entries[ii].split(sep="_")[4][0]
    
    if Slope == 'p4':
        Label = '1'
    elif Slope == 'p2':
        Label = '2'
    elif Slope == 'n6':
        Label = '3'
    
    # As the insoles currently have a temporal shift, analyze the last 40 sec of data
    dat = dat.iloc[len(dat)-4000:,:]
    
    # Convert the force into newtons, pressures into kPa
    L_tot_force = np.array(dat.iloc[:,15])*4.44822
    L_tot_force = zeroInsoleForce(L_tot_force,100)
    LHS = np.array(findLandings(L_tot_force, 50))
    LTO = np.array(findTakeoffs(L_tot_force, 50))
    
    R_tot_force = np.array(dat.iloc[:,27])*4.44822
    R_tot_force = zeroInsoleForce(R_tot_force,100)
    RHS = np.array(findLandings(R_tot_force, 50))
    RTO = np.array(findTakeoffs(R_tot_force, 50))
    
    [LHS,LTO] = trimgaitevents(LHS,LTO)
    [RHS,RTO] = trimgaitevents(RHS,RTO)
    
    # # Attempt to find the jumps: 
    # jump_found = 0
    # start_LHS = []; start_RHS = []
    # for jj in range(15):
    #     jump_check = np.where(np.abs(LHS[jj] - np.array(RHS[0:15])) < 10)
    #     if jump_check[0].size > 0:
    #         print('Jump Found')
    #         Llastjumpi = jj
    #         Rlastjumpi = np.argmin(np.abs(LHS[jj] - np.array(RHS[0:15])))
    #         jump_found = 1
            
    # if jump_found == 1:
    #     idx1 = (LHS > LHS[Llastjumpi] + 10*freq)*(LHS < LHS[Llastjumpi] + 55*freq)
    #     idx2 = (LTO > LHS[Llastjumpi] + 10*freq)*(LTO < LHS[Llastjumpi] + 55*freq)
    #     LHS = LHS[idx1]
    #     LTO = LTO[idx2]
    #     [LHS,LTO] = trimgaitevents(LHS,LTO)
        
    #     idx1 = (RHS > RHS[Rlastjumpi] + 10*freq)*(RHS < RHS[Rlastjumpi] + 55*freq)
    #     idx2 = (RTO > RHS[Rlastjumpi] + 10*freq)*(RTO < RHS[Rlastjumpi] + 55*freq)
    #     RHS = RHS[idx1]
    #     RTO = RTO[idx2]
        
    #     [RHS,RTO] = trimgaitevents(RHS,RTO)
        
    
    # if jump_found == 0:
    #     print('Temporal Shift may have occured')
    #     [LHS,LTO] = threehopstart(LHS,LTO,freq)
    #     [RHS,RTO] = threehopstart(RHS,RTO,freq)
    
        
    L_heel_con = np.array((dat.iloc[:,36]+dat.iloc[:,60])/(dat.iloc[:,37]+dat.iloc[:,61])*100)
    R_heel_con = np.array((dat.iloc[:,48]+dat.iloc[:,72])/(dat.iloc[:,49]+dat.iloc[:,73])*100)
    
    L_toe_con = np.array((dat.iloc[:,180]+dat.iloc[:,204])/(dat.iloc[:,181]+dat.iloc[:,205])*100)
    R_toe_con = np.array((dat.iloc[:,192]+dat.iloc[:,216])/(dat.iloc[:,193]+dat.iloc[:,217])*100)
        
    L_heelPP_lat = np.array(dat.iloc[:,37]-np.min(dat.iloc[:,37]))*6.89476
    L_heelPP_med = np.array(dat.iloc[:,59]-np.min(dat.iloc[:,59]))*6.89476
    L_heelPP = np.maximum(L_heelPP_lat,L_heelPP_med)
    
    R_heelPP_lat = np.array(dat.iloc[:,47]-np.min(dat.iloc[:,47]))*6.89476
    R_heelPP_med = np.array(dat.iloc[:,71]-np.min(dat.iloc[:,71]))*6.89476
    R_heelPP = np.maximum(R_heelPP_lat,R_heelPP_med)
    
    L_midPP_lat = np.array(dat['Peak Pressure.6']-np.min(dat['Peak Pressure.6']))*6.89476
    R_midPP_lat = np.array(dat['Peak Pressure.7']-np.min(dat['Peak Pressure.7']))*6.89476
    
    L_metPP_lat = np.array(dat['Peak Pressure.10']-np.min(dat['Peak Pressure.10']))*6.89476
    R_metPP_lat = np.array(dat['Peak Pressure.11']-np.min(dat['Peak Pressure.11']))*6.89476
    
    L_toePP_lat = np.array(dat['Peak Pressure.14']-np.min(dat['Peak Pressure.14']))*6.89476
    R_toePP_lat = np.array(dat['Peak Pressure.15']-np.min(dat['Peak Pressure.15']))*6.89476
    
    L_toePP_lat = np.array(dat['Peak Pressure.14']-np.min(dat['Peak Pressure.14']))*6.89476
    L_toePP_med = np.array(dat['Peak Pressure.16']-np.min(dat['Peak Pressure.16']))*6.89476
    L_toePP = np.maximum(L_toePP_lat,L_toePP_med)
    
    R_toePP_lat = np.array(dat['Peak Pressure.15']-np.min(dat['Peak Pressure.15']))*6.89476
    R_toePP_med = np.array(dat['Peak Pressure.17']-np.min(dat['Peak Pressure.17']))*6.89476
    R_toePP = np.maximum(R_toePP_lat,R_toePP_med)
          
       
    if Subject not in poorL:
        # Remove strides that have a peak GRF below 1000 N
        # Remove strides that are below 0.5 and above 1.5 seconds
        LGS = []    
        # May need to exclude poor toe-off dectections here as well
        for jj in range(len(LHS)-1):
            if np.max(L_tot_force[LHS[jj]:LTO[jj]]) > 800:
                if (LHS[jj+1] - LHS[jj]) > 0.5*freq and LHS[jj+1] - LHS[jj] < 1.5*freq:
                    LGS.append(jj)
                    
        # Compute metrics of interest
        for jj in LGS:
            heel_con.append(np.mean(L_heel_con[LHS[jj]:LTO[jj]]))
            # Estimate for 1st and 2nd half of stance
            heel_con_1.append(np.mean(L_heel_con[LHS[jj]:LHS[jj]+int(.5*(LTO[jj]-LHS[jj]))]))
            heel_con_2.append(np.mean(L_heel_con[LHS[jj]+int(.5*(LTO[jj]-LHS[jj])):LTO[jj]]))
            
            toe_con.append(np.mean(L_toe_con[LHS[jj]:LTO[jj]]))
            
            # Examine the maximum lateral pressures
            m_heelPP_lat.append(np.max(L_heelPP_lat[LHS[jj]:LTO[jj]]))
            m_heelPP.append(np.max(L_heelPP[LHS[jj]:LTO[jj]]))
            m_midPP_lat.append(np.max(L_midPP_lat[LHS[jj]:LTO[jj]]))
            m_metPP_lat.append(np.max(L_metPP_lat[LHS[jj]:LTO[jj]]))
            m_toePP_lat.append(np.max(L_toePP_lat[LHS[jj]:LTO[jj]]))
            m_toePP.append(np.max(L_toePP[LHS[jj]:LTO[jj]]))
        
        oSubject = oSubject + [Subject]*len(LGS)
        oConfig = oConfig + [Config]*len(LGS)
        oSpeed = oSpeed + [Speed]*len(LGS)
        oSesh = oSesh + [Sesh]*len(LGS)
        oLabel = oLabel + [Label]*len(LGS)
        oSide = oSide + ['L']*len(LGS)
    
    RGS = []
    for jj in range(len(RHS)-1):
        if np.max(R_tot_force[RHS[jj]:RTO[jj]]) > 800:
            if (RTO[jj] - RHS[jj]) > 0.15*freq and RHS[jj+1] - RHS[jj] < 1.5*freq:
                RGS.append(jj)
    
    for jj in RGS:
        heel_con.append(np.mean(R_heel_con[RHS[jj]:RTO[jj]]))
        # Estimate for 1st and 2nd half of stance
        heel_con_1.append(np.mean(R_heel_con[RHS[jj]:RHS[jj]+int(.5*(RTO[jj]-RHS[jj]))]))
        heel_con_2.append(np.mean(R_heel_con[RHS[jj]+int(.5*(RTO[jj]-RHS[jj])):RTO[jj]]))
        
        toe_con.append(np.mean(R_toe_con[RHS[jj]:RTO[jj]]))
        
        # Examine the maximum lateral pressures
        m_heelPP_lat.append(np.max(R_heelPP_lat[RHS[jj]:RTO[jj]]))
        m_heelPP.append(np.max(R_heelPP[RHS[jj]:RTO[jj]]))
        m_midPP_lat.append(np.max(R_midPP_lat[RHS[jj]:RTO[jj]]))
        m_metPP_lat.append(np.max(R_metPP_lat[RHS[jj]:RTO[jj]]))
        m_toePP_lat.append(np.max(R_toePP_lat[RHS[jj]:RTO[jj]]))
        m_toePP.append(np.max(R_toePP[RHS[jj]:RTO[jj]]))
        
    oSubject = oSubject + [Subject]*len(RGS)
    oConfig = oConfig + [Config]*len(RGS)
    oSpeed = oSpeed + [Speed]*len(RGS)
    oSesh = oSesh + [Sesh]*len(RGS)
    oLabel = oLabel + [Label]*len(RGS)
    oSide = oSide + ['R']*len(RGS)
    
    if debug == 1:
        plt.figure(ii)
        plt.subplot(2,2,1)
        plt.plot(L_tot_force)
        plt.plot(LHS[LGS],L_tot_force[LHS[LGS]],'ro')
        plt.plot(LTO[LGS],L_tot_force[LTO[LGS]],'ko')
        plt.ylabel('Left Insole Force [N]')
        
        plt.subplot(2,2,2)
        plt.plot(R_tot_force)
        plt.plot(RHS[RGS],R_tot_force[RHS[RGS]],'ro')
        plt.plot(RTO[RGS],R_tot_force[RTO[RGS]],'ko')
        plt.ylabel('Right Insole Force [N]')
        
        plt.subplot(2,2,3)
        plt.plot(intp_strides(L_tot_force,LHS,LTO,LGS))
        plt.ylabel('Insole Force [N]')
        
        plt.subplot(2,2,4)
        plt.plot(intp_strides(R_tot_force,RHS,RTO,RGS))
        plt.ylabel('Insole Force [N]')
        plt.close()



outcomes = pd.DataFrame({'Subject':list(oSubject), 'Config': list(oConfig),'Speed': list(oSpeed),'Sesh': list(oSesh),
                          'Label':list(oLabel), 'Side':list(oSide), 'HeelCon':list(heel_con),'HeelCon1':list(heel_con_1), 'HeelCon2':list(heel_con_2),
                          'ToeCon':list(toe_con),'m_heelPP':list(m_heelPP),'m_toePP':list(m_toePP)})

if save_on == 1:
    outcomes.to_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabData\\InLabPressureOutcomes.csv',header=True)
elif save_on == 2: 
    outcomes.to_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabData\\InLabPressureOutcomes.csv',mode = 'a', header=False)


