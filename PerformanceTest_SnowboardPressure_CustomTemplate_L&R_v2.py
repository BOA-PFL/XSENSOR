# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:23:50 2023

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import messagebox
from XSENSORFunctions import (readXSENSORFile, delimitTrial, createTSmat,
                              EnsureTurnsAlternate, loadProcessingDecisions,
                              saveProcessingDecision)
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
fPath = 'Z:\\Testing Segments\\SnowSports\\Snowboard\\2026_Mechanistic_SnowboardWrapMechII_K2\\Xsensor\\Cropped\\'
fileExt = r".csv"
# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'SnowboardPressure_LR'

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

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and not fName.startswith('0_')]


save_on = 0
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about trials with no recorded decision (the normal setting)
# 2 = re-review EVERY trial, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied.
data_check = 1

# Which signal finds the turns.
# 'IMU'   = the Z gyro, which reads board rotation directly (the default)
# 'force' = the toe/heel force split, which only infers the edge change from
#           where the load sits. Kept as a fallback for trials where an
#           insole IMU is missing, saturated or noisy.
turn_detection = 'IMU'

# Turn detection knobs. These are the defaults for the cohort. Either can be
# overridden per athlete in the block at the top of the trial loop - riders
# differ in how tightly they link turns, and in how much slow drift sits in
# the gyro, so neither of these generalises as well as it looks.
turn_det_nob = 0.5  # gyro cut-off (Hz)
turn_dist_nob = 2.0 # shortest time (s) allowed between two turns on the same edge
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
    
def findSnowboardTurns(dat, cut_hz, min_turn_s, rideDir):
    """
    Determine snowboard turn detection from the Z gyroscope onboard the
    XSENSOR SPK insoles.

    Parameters
    ----------
    dat : dataclass
        Created from the function "createTSmat"
    cut_hz : float
        Cut-off frequency for the gyroscope used to find turning indices
    min_turn_s : float
        Shortest time, in seconds, allowed between two turns on the same
        edge. Converted to samples here from the measured rate.
    rideDir : str
        'Regular' or 'Goofy'. Sets which direction of board rotation is a
        toe-edge turn.

    Returns
    -------
    Heelpeaks : array
        Indices of the heel-edge turns
    Toepeaks : array
        Indices of the toe-edge turns
    heel_sig : array
        Detection signal the heel-edge peaks were found in
    toe_sig : array
        Detection signal the toe-edge peaks were found in

    """
    # Set up the filter from the measured sample rate rather than a constant,
    # the same way the ski detection does
    fs = 1.0 / np.mean(np.diff(dat.time))
    w = cut_hz / (fs / 2.0) # normalized cutoff
    b, a = sig.butter(2, w, 'low') # 2nd order filter

    # Both insoles are bolted to the same board, so they turn together and
    # see the same yaw rate. Averaging the two is steadier than either on its
    # own. Unlike skiing there is no independent inside/outside foot here.
    Lgyro_det = sig.filtfilt(b, a, dat.Lgyr[:,2])
    Rgyro_det = sig.filtfilt(b, a, dat.Rgyr[:,2])
    gyro_det = (Lgyro_det + Rgyro_det) / 2.0

    # Take the resting offset out before anything is compared across zero.
    # Both insoles sit at a binding angle, so their Z axis is not vertical and
    # the averaged signal rests well off zero - measured at -3 deg/s on one
    # rider and +9 on another, consistent across both of that rider's insoles.
    # Over a two minute run that is several hundred degrees of rotation that
    # never happened, and left in it makes an excursion one way and an
    # excursion the other way incomparable. The median is used rather than the
    # mean so the turns themselves do not drag the estimate.
    gyro_det = gyro_det - np.median(gyro_det)

    # Yaw runs one way through a toe-edge turn and the other way through a
    # heel-edge turn. Which sign is which depends on which foot leads, so the
    # rider's stance picks it rather than it being baked in.
    if rideDir == 'Goofy':
        toe_sig = gyro_det * -1
    else:
        toe_sig = gyro_det
    heel_sig = toe_sig * -1

    # Both edges are scaled by the SAME number, so a toe turn and a heel turn
    # are judged on one yardstick. A fixed threshold in deg/s does not carry
    # across riders - one rider's toe-edge yaw is routinely smaller than their
    # heel-edge yaw, so a single absolute cut-off keeps most of the strong
    # edge and about half of the weak one. Scaling each edge by its own
    # maximum fixes that and introduces something worse: it divides one edge
    # by the signal's peak and the other by whatever the opposite excursion
    # reached, which on MattAnderson_Regular_Instep_5 was 43 against 7 deg/s
    # and amplified the heel side roughly six fold.
    scale = np.max(np.abs(gyro_det))
    toe_norm = toe_sig / scale
    heel_norm = heel_sig / scale

    # find_peaks wants the spacing in samples, so it comes from the measured
    # rate rather than a fixed 200. Those agree at 100 Hz and quietly stop
    # agreeing the moment a collection is run at anything else.
    minGap = round(min_turn_s * fs)

    Toepeaks, _ = sig.find_peaks(toe_norm, prominence = 0.3, distance = minGap)
    Heelpeaks, _ = sig.find_peaks(heel_norm, prominence = 0.3, distance = minGap)

    # Nothing to alternate if an edge went undetected. Returning early keeps
    # the caller's plots working so the trial can be looked at and rejected.
    if len(Toepeaks) == 0 or len(Heelpeaks) == 0:
        return Heelpeaks, Toepeaks, heel_norm, toe_norm

    # Turns alternate toe, heel, toe, heel. Where two detections land on the
    # same edge in a row, the taller one is kept - same clean-up the force
    # detection applies.
    if Toepeaks[0] < Heelpeaks[0]:
        Toepeaks, Heelpeaks = EnsureTurnsAlternate(toe_norm, heel_norm, Toepeaks, Heelpeaks)

    elif Toepeaks[0] > Heelpeaks[0]:
        Heelpeaks, Toepeaks = EnsureTurnsAlternate(heel_norm, toe_norm, Heelpeaks, Toepeaks)

    return Heelpeaks, Toepeaks, heel_norm, toe_norm


def findTurns(ToeForce, HeelForce, fs, fc):
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
    Rpeaks:

    """
    
    # ToeForce = ToeForce
    # HeelForce = HeelForce
    # fs = 100
    # fc = 0.5
    
    w = fc / (fs / 2)
    b, a = sig.butter(2, w, 'low')
    
    Toeturn_detect = sig.filtfilt(b, a, ToeForce)
    Toeturn_detect = Toeturn_detect/np.max(Toeturn_detect)
    Heelturn_detect = sig.filtfilt(b, a, HeelForce)
    Heelturn_detect = Heelturn_detect/np.max(Heelturn_detect)
    
    
    
    Toepeaks,_ = sig.find_peaks(Toeturn_detect, prominence=0.3)
    Heelpeaks,_ = sig.find_peaks(Heelturn_detect, prominence=0.3)
    
    # Remove right detections if the peak is lower than the left signal at that
    # time, and vice versa
    to_delete = []
    for ii, peak in enumerate(Toepeaks):
        if Toeturn_detect[peak] < Heelturn_detect[peak]:
            to_delete.append(ii)
    Toepeaks = np.delete(Toepeaks,to_delete)
    
    to_delete = []
    for ii, peak in enumerate(Heelpeaks):
        if Heelturn_detect[peak] < Toeturn_detect[peak]:
            to_delete.append(ii)
    
    Heelpeaks = np.delete(Heelpeaks,to_delete)
    
            
    # Clean up the turn detection: ensure they oscillate
    if Toepeaks[0] < Heelpeaks[0]:
        Toepeaks, Heelpeaks = EnsureTurnsAlternate(Toeturn_detect,Heelturn_detect,Toepeaks,Heelpeaks)
    
    elif Toepeaks[0] > Heelpeaks[0]:
        Heelpeaks, Toepeaks = EnsureTurnsAlternate(Heelturn_detect,Toeturn_detect,Heelpeaks,Toepeaks)
        
    # plt.figure()
    # plt.plot(Toeturn_detect)
    # plt.plot(Toepeaks, Toeturn_detect[Toepeaks], 'gx')
    # plt.plot(Heelturn_detect)
    # plt.plot(Heelpeaks, Heelturn_detect[Heelpeaks], 'rx')
            
    return(Heelpeaks,Toepeaks)

#############################################################################################################################################################

badFileList = []

# One row per turn, across every trial. These have to be built up outside the
# loop: the results are written once after the loop, so re-initialising them
# per trial would leave only the last trial's turns to save.
# FrontTotMaxForce, BackTotMaxForce and TurnDirection were initialised here
# too but never appended to and never reached the output, so they are gone.
subject = []
config = []
trialNo = []
TurnDir = []
TurnTime = []
detector = []
ToeTotMaxForce = []
HeelTotMaxForce = []
maxmaxToes = []
heelAreaP = []
RFD = []
RFDtime = []

for fName in entries:
    print(fName)
    # A trial rejected in a previous run is skipped. The file is still there
    # to open and look at: the recorded decision governs, not the folder
    priorDecision = decisions.get(fName, '')
    if priorDecision == 'bad' and data_check != 2:
        print('Previously excluded, skipping: ' + fName)
        continue
    
    # Clear the previous trial's cyclic detections. They are assigned
    # inside conditionals, so a trial that skips a detection would keep
    # the last trial's events and be processed against them silently.
    Heelmax = []
    Toemax = []
    Heelturn_sig = []
    Toeturn_sig = []

    subName = fName.split(sep = "_")[0]

    # Turn detection settings for this athlete. Reset from the defaults every
    # trial, so an override set for one rider cannot carry into the next one,
    # then name the exceptions underneath. Someone who links turns quickly
    # needs a shorter turn_dist than someone who traverses between them.
    turn_det = turn_det_nob
    turn_dist = turn_dist_nob

    # if subName == 'AndrewRendell':
    #     turn_dist = 1.5

    configTmp = fName.split(sep="_")[2]
    rideDir = fName.split(sep ="_")[1]
    # The gyro detection flips sign on this, so a typo in a file name would
    # silently swap every toe turn for a heel turn
    if rideDir != 'Regular' and rideDir != 'Goofy':
        raise ValueError('Expected Regular or Goofy in field 2 of ' + fName
                         + ', found ' + rideDir)
    trialTmp = fName.split(sep="_")[3].split(sep ='.')[0]

    tmpDat = readXSENSORFile(fName,fPath)
    tmpDat = delimitTrial(tmpDat,fName,fPath)
    tmpDat = createTSmat(fName, fPath, tmpDat)
    
    # Estimate the frequency from the time stamps
    freq = 1/np.mean(np.diff(tmpDat.time))
    
    # Toe-edge and heel-edge force, summed over both insoles. Rows 0-14 are
    # toe and forefoot, rows 15 and up midfoot and heel. The constant turns
    # mean sensel pressure into a force over the sensel area.
    LffToe = tmpDat.LplantarMat[:,:15,:]
    LmfHeel = tmpDat.LplantarMat[:,15:,:]
    RffToe = tmpDat.RplantarMat[:,:15,:]
    RmfHeel = tmpDat.RplantarMat[:,15:,:]
    
    ToeForce_raw = (np.mean(LffToe, axis = (1,2)) + np.mean(RffToe, axis = (1,2))) * 6895 * 0.014699
    HeelForce_raw = (np.mean(LmfHeel, axis = (1,2)) + np.mean(RmfHeel, axis = (1,2))) * 6895 * 0.014699
    
    # Smoothed for turn detection. createTSmat returns the sensel matrices
    # unfiltered on purpose - the 1 kPa cut it applies is the sensor noise
    # floor, and zeroing after filtering would smear that noise into the
    # signal first - so the smoothing the turn detection needs happens here.
    w1 = 6 / (freq / 2)
    b1, a1 = sig.butter(2, w1, 'low')
    ToeForce = sig.filtfilt(b1, a1, ToeForce_raw)
    HeelForce = sig.filtfilt(b1, a1, HeelForce_raw)
    
    # Turn detection. Not every trial has an IMU - in this collection nine of
    # them carry no gyro columns at all - so the force detection stays in as
    # the fallback. Which detector ran is written out with the results: two
    # trials found with different detectors are not directly comparable, and
    # that has to be visible in the data rather than only in the console.
    gyroPresent = len(tmpDat.Lgyr) > 0 and len(tmpDat.Rgyr) > 0

    # The plots below show whichever signal found the turns, so the markers
    # always sit on the signal they came from.
    if turn_detection == 'IMU' and gyroPresent == True:
        [Heelmax,Toemax,Heelturn_sig,Toeturn_sig] = findSnowboardTurns(tmpDat, turn_det, turn_dist, rideDir)
        detectorUsed = 'IMU'
    else:
        if turn_detection == 'IMU':
            print('   no gyro in this trial - falling back to force detection')
        [Heelmax,Toemax] = findTurns(ToeForce, HeelForce, freq, 0.75)
        Heelturn_sig = HeelForce
        Toeturn_sig = ToeForce
        detectorUsed = 'force'
    
    
    
    answer = True # if data check is off. 
    if priorDecision == 'good' and data_check == 1:
        print('Previously accepted, not asking again: ' + fName)
    elif data_check >= 1:
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(Heelturn_sig)
        plt.plot(Heelmax,Heelturn_sig[Heelmax],'rs')
        plt.ylabel('Heel Edge')
        plt.title(fName + '  -  ' + turn_detection + ' detection')
        
      
        plt.subplot(2,1,2)
        plt.plot(Toeturn_sig)
        plt.plot(Toemax,Toeturn_sig[Toemax],'rs')
        plt.ylabel('Toe Edge')
        
        
            
        answer = messagebox.askyesno("Question","Is data clean?")
    
    # Record the decision straight away, inside the loop, so a run that
    # fails later keeps every decision made so far
    if priorDecision == '' or data_check == 2:
        if answer == False:
            saveProcessingDecision(fPath, scriptName, fName, 'bad',
                                   Subject=subName, Config=configTmp,
                                   reason='rejected at the data check')
        else:
            saveProcessingDecision(fPath, scriptName, fName, 'good',
                                   Subject=subName, Config=configTmp)
    
    if answer == False:
        plt.close('all')
        print('Adding file to bad file list')
        badFileList.append(fName)

    if answer == True:
        plt.close('all')
        print('Estimating point estimates')
        
        for jj, value in enumerate(Toemax):
            # Loop through all cleaned turns to calculate discrete outcome measures.
            # using right side only (left turns). No longer assuming left to right
            # transitions are constant and only using data from one side. 
            # Extend this to Left Turns as well
            # variables of interest: outside force (downhill foot peak force) higher is better
            # Outside medial peak force (higher is better)
            # Outside heel force lower is better with an average higher force on forefoot Tricky**

            try:
                if jj>0:
                    
                    # jj = 1
                    # value = Toemax[jj]
                    
                    # +/- 0.5 s around the turn peak to find the true maximum.
                    # This was a fixed 50 frames guarded by "value < 100" - two
                    # numbers that only agree at 100 Hz, and only agree with
                    # each other by luck.
                    window = round(0.5 * freq)
                    if value < window:
                        # Keep the window inside the trial. A negative start
                        # would slice from the end of the signal instead.
                        window = value

                    # +/- 0.1 s around the turn peak for the heel contact area.
                    # The divisor below is the window actually taken, not a
                    # constant: at the start of a trial the window is clipped,
                    # and a fixed frame count would divide by frames that were
                    # never in it.
                    areaStart = value - round(0.1 * freq)
                    if areaStart < 0:
                        areaStart = 0
                    areaEnd = value + round(0.1 * freq)
                    areaFrames = len(tmpDat.LplantarHeel[areaStart:areaEnd])
                    
                    # Toe edge force
                    ToeTotMaxForce.append(np.nanmax(ToeForce[value-window:value+window]))
                    
                    # Look at the Inside (or uphill) ski
                    HeelTotMaxForce.append(np.nanmax(HeelForce[value-window:value+window]))
                    
                    # Peak toe pressure
                    if rideDir == 'Regular':
                        maxmaxToes.append(np.nanmax(tmpDat.LplantarToe[value-window:value+window])*6.895)
                    elif rideDir == 'Goofy':
                        maxmaxToes.append(np.nanmax(tmpDat.RplantarToe[value-window:value+window])*6.895)
                    
                    # Examine a 0.25 second window around the peak of the turn
                    # for fraction of force applied medial/lateral
                    # Medial Fraction
                    
                    # Rate of force development
                    # Find the index of the true local maxima
                    idx_max = np.argmax(ToeForce[value-window:value+window])+value-window
                    idx_min = np.argmin(ToeForce[Toemax[jj-1]:idx_max-1])+Toemax[jj-1]
                    RFD.append(np.mean(np.gradient(ToeForce[idx_min:idx_max],1/freq)))
                    RFDtime.append((idx_max-idx_min)/freq)
                    
                    # Heel Hold
                    if rideDir == 'Regular':
                        heelAreaP.append(np.count_nonzero(tmpDat.LplantarHeel[areaStart:areaEnd, :, :])/areaFrames/tmpDat.LplantarHeelSensNo*100)
                    elif rideDir == 'Goofy':
                        heelAreaP.append(np.count_nonzero(tmpDat.RplantarHeel[areaStart:areaEnd, :, :])/areaFrames/tmpDat.RplantarHeelSensNo*100)
                    # turntime
                    TurnTime.append((value-Toemax[jj-1])/freq)
                    
                    
                    TurnDir.append('Toe')
                    detector.append(detectorUsed)
                    subject.append(subName)
                    config.append(configTmp)
                    trialNo.append(trialTmp)
                
                
            except Exception as e: print(e)
        
        
        for jj, value in enumerate(Heelmax):
            # Loop through all cleaned turns to calculate discrete outcome measures.
            # using right side only (left turns). No longer assuming left to right
            # transitions are constant and only using data from one side. 
            # Extend this to Left Turns as well
            # variables of interest: outside force (downhill foot peak force) higher is better
            # Outside medial peak force (higher is better)
            # Outside heel force lower is better with an average higher force on forefoot Tricky**

            try:
                if jj>0:
                    # +/- 0.5 s around the turn peak to find the true maximum.
                    # This was a fixed 50 frames guarded by "value < 100" - two
                    # numbers that only agree at 100 Hz, and only agree with
                    # each other by luck.
                    window = round(0.5 * freq)
                    if value < window:
                        # Keep the window inside the trial. A negative start
                        # would slice from the end of the signal instead.
                        window = value

                    # +/- 0.1 s around the turn peak for the heel contact area.
                    # The divisor below is the window actually taken, not a
                    # constant: at the start of a trial the window is clipped,
                    # and a fixed frame count would divide by frames that were
                    # never in it.
                    areaStart = value - round(0.1 * freq)
                    if areaStart < 0:
                        areaStart = 0
                    areaEnd = value + round(0.1 * freq)
                    areaFrames = len(tmpDat.LplantarHeel[areaStart:areaEnd])
                    
                    # Heel edge force
                    HeelTotMaxForce.append(np.nanmax(HeelForce[value-window:value+window]))
                    
                    # Look at the Inside (or uphill) ski
                    ToeTotMaxForce.append(np.nanmax(ToeForce[value-window:value+window]))
                    
                    # Peak toe pressure
                    if rideDir == 'Regular':
                        maxmaxToes.append(np.nanmax(tmpDat.LplantarToe[value-window:value+window])*6.895)
                        
                    elif rideDir == 'Goofy':
                        maxmaxToes.append(np.nanmax(tmpDat.RplantarToe[value-window:value+window])*6.895)
                        
                    
                    # Examine a 0.25 second window around the peak of the turn
                    # for fraction of force applied medial/lateral
                    # Medial Fraction
                    
                    # Rate of force development
                    # Find the index of the true local maxima
                    idx_max = np.argmax(HeelForce[value-window:value+window])+value-window
                    idx_min = np.argmin(HeelForce[Heelmax[jj-1]:idx_max])+Heelmax[jj-1]
                    RFD.append(np.mean(np.gradient(HeelForce[idx_min:idx_max],1/freq)))
                    RFDtime.append((idx_max-idx_min)/freq)
                    
                    # Heel Hold
                    if rideDir =='Regular':
                        heelAreaP.append(np.count_nonzero(tmpDat.LplantarHeel[areaStart:areaEnd, :, :])/areaFrames/tmpDat.LplantarHeelSensNo*100)
                    
                    elif rideDir == 'Goofy' :
                        heelAreaP.append(np.count_nonzero(tmpDat.RplantarHeel[areaStart:areaEnd, :, :])/areaFrames/tmpDat.RplantarHeelSensNo*100)
                    
                    # turntime
                    TurnTime.append((value-Heelmax[jj-1])/freq)
                    
                    
                    TurnDir.append('Heel')
                    detector.append(detectorUsed)
                    subject.append(subName)
                    config.append(configTmp)
                    trialNo.append(trialTmp)
                
                
            except Exception as e: print(e)


# Every accepted turn from every trial, written once, after the loop.
outcomes = pd.DataFrame({'Subject': list(subject), 'Config':list(config),'trialNo':list(trialNo) , 'TurnDirection':list(TurnDir), 'TurnTime':list(TurnTime),
                         'ToeTotMaxForce':list(ToeTotMaxForce), 'HeelTotMaxForce':list(HeelTotMaxForce),
                         'maxmaxToes':list(maxmaxToes), 'heelAreaP':list(heelAreaP), 
                         'RFD':list(RFD), 'RFDtime':list(RFDtime),
                         'TurnDetector':list(detector)
                         })

outfileName = fPath + '0_CompiledResults_8.csv'
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    outcomes.to_csv(outfileName, header=True, index = False)     
            
            
            
            
            
        


