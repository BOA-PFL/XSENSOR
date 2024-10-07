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

data_check = 1

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
        
        insoleSide = inputDF['Insole'][0]
                   
        
        if (insoleSide == 'Left'): 
            
            # Left side
            totForce = np.mean(inputDF.iloc[:,18:238], axis = 1)*6895*0.014699
        else:  
            
            totForce = np.mean(inputDF.iloc[:,210:430], axis = 1)*6895*0.014699
        print('Select a point on the plot to represent the beginning & end of trial')


        ax.plot(totForce, label = 'Total Force')
        fig.legend()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        outputDat = outputDat.reset_index(drop = True)
        trial_segment = np.array([FName, pts], dtype = object)
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
    
    
 
    LForce: np.array 

    config: str
    movement: str
    subject: str
    
    fullDat: pd.DataFrame #entire stored dataframe. 
    #this class is useful for plotting and subsequent analysis
    
    


def createTSmat(inputName):
    """ 
    Reads in file, creates 3D time series matrix (foot length, foot width, time) to be plotted and features
    are extracted. The result is a dataclass which can be used for further plotting. Requires findGaitEvents function.
    """
    
    
    # inputName = entries[10]

  
    # dat = pd.read_csv(fPath+inputName, sep=',', usecols=(columns) )  
    #dat = pd.read_csv(fPath+inputName, sep=',', header = 'infer', low_memory=False)
    # dat = pd.read_csv(fPath+inputName, sep=',', header = 1 , low_memory=False)
    dat = pd.read_csv(fPath+inputName, sep=',', header = 'infer', low_memory=False)
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2] 
    
    

    
        
    LplantarSensel = dat.iloc[:,18:238]


   
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
    # # [LHS,LTO] = findGaitEvents(LForce,freq)
  
    

    
    
    
    
    result = tsData(  LplantarMat, LplantarToe, LplantarToeLat, LplantarToeMed,
                    LplantarForefoot, LplantarForefootLat, LplantarForefootMed,
                    LplantarMidfoot, LplantarMidfootLat, LplantarMidfootMed,
                    LplantarHeel, LplantarHeelLat, LplantarHeelMed, LplantarLateral, LplantarMedial, LForce,
                     config, movement, subj, dat)
    
    return(result)

def findStabilization(avgF, sdF):
    """
    Using the rolling average and SD values, this calcualtes when the 
    actual stabilized force occurs. 
    
    Parameters
    ----------
    avgF : list, calculated using movAvgForce 
        rolling average of force.
    sdF : list, calcualted using movSDForce above
        rolling SD of force.

    Returns
    -------
    floating point number
        Time to stabilize using the heuristic: force is within +/- 5% of subject
        mass and the rolling standrd deviation is below 20

    """
    stab = []
    for step in range(len(avgF)-1):
        if avgF[step] >= (subBW - 0.05*subBW) and avgF[step] <= (subBW + 0.05*subBW) and sdF[step] < 20:
            stab.append(step + 1) 
            
    return stab[0] 


def movAvgForce(force, landing, takeoff, length):
    """
    In order to estimate when someone stabilized, we calcualted the moving
    average force and SD of the force signal. This is one of many published 
    methods to calcualte when someone is stationary. 
    
    Parameters
    ----------
    force : Pandas series
        pandas series of force from force plate.
    landing : List
        list of landings calcualted from findLandings.
    takeoff : List
        list of takeoffs from findTakeoffs.
    length : Integer
        length of time in indices to calculate the moving average.

    Returns
    -------
    avgF : list
        smoothed average force .

    """
    newForce = np.array(force)
    win_len = length; #window length for steady standing
    avgF = []
    for i in range(landing, takeoff):
        avgF.append(np.mean(newForce[i : i + win_len]))     
    return avgF

#moving SD as calcualted above
def movSDForce(force, landing, takeoff, length):
    """
    This function calculates a rolling standard deviation over an input
    window length
    
    Parameters
    ----------
    force : Pandas series
        pandas series of force from force plate.
    landing : List
        list of landings calcualted from findLandings.
    takeoff : List
        list of takeoffs from findTakeoffs.
    length : Integer
        length of time in indices to calculate the moving average.

    Returns
    -------
    avgF : list
        smoothed rolling SD of forces

    """
    newForce = np.array(force)
    win_len = length; #window length for steady standing
    avgF = []
    for i in range(landing, takeoff):
        avgF.append(np.std(newForce[i : i + win_len]))     
    return avgF

#estimated stability after 200 indices
def findBW(force):
    """
    If you do not have the subject's body weight or want to find from the 
    steady portion of force, this may be used. This is highly conditional on 
    the data and how it was collected. The below assumes quiet standing from
    100 to 200 indices. 
    
    Parameters
    ----------
    force : Pandas series
        DESCRIPTION.

    Returns
    -------
    BW : floating point number
        estimate of body weight of the subject to find stabilized weight
        in Newtons

    """
    BW = np.mean(avgF[100:200])
    return BW
   
#############################################################################################################################################################

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\bethany.kilpatrick\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\WorkWear_Performance\\EH_Workwear_MidcutStabilityFinal_Mech_Aug24\\Xsensor\\Cropped\\SLLt\\'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]




badFileList = []

for fName in entries:
    
    config = []
    subject = []
    ct = []
    movement = []
    order = []

    
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
    
    sdFz = []
    avgF = []
    sdF = []
    subBW = [] 
    
    stabilization = []

    try: 
        # fName = entries[6]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
        torder = fName.split(sep = "_")[3].split(sep = '.')[0]
        
        # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        #if ('SLL' in moveTmp):# or ('SLLT' in moveTmp): # or ('Trail' in moveTmp):
        if (moveTmp == 'SLLt'):
    
            
  
            tmpDat = createTSmat(fName)
            ffoot = np.mean(tmpDat.LplantarForefoot, axis = (1,2)) * 6895 * 0.014699
            
         
            land = sig.find_peaks(tmpDat.LForce, height = 1000, distance = 600)[0]
            land_ht =  sig.find_peaks(tmpDat.LForce, height = 1000, distance = 600)[1]['peak_heights']
            fft_pk = sig.find_peaks(ffoot, height = 1000, distance = 500)[0]
            fft_ht = sig.find_peaks(ffoot, height = 1000, distance = 500)[1]['peak_heights']
            
            htThrsh = np.mean(land_ht) - 300
                        
            true_land = sig.find_peaks(tmpDat.LForce, height = htThrsh, distance = 500)[0]
            land_ht =  sig.find_peaks(tmpDat.LForce, height = htThrsh, distance = 500)[1]['peak_heights']
   
            forceZ = tmpDat.LForce 
            
           
            # answer = True # if data check is off. 
          
            
            if data_check == 1:
                  plt.figure()
                  plt.plot(tmpDat.LForce, label = 'Left Foot Total Force') 
                  plt.plot(true_land, land_ht, marker = 'o', linestyle = 'none')
                  #plt.plot(range(len(ffoot)), ffoot)
                  #plt.plot(fft_pk, fft_ht, marker = 'v', linestyle = 'none')
                  saveFolder = fPath + 'SLLt'
                  if os.path.exists(saveFolder) == False:
                    os.mkdir(saveFolder) 
                    plt.savefig(saveFolder + '/' + fName.split('.csv')[0] +'.png')
                 
        
  
            answer = messagebox.askyesno("Question","Is data clean?")    
  
            
  
                
            
            if answer == False:
                disThrsh = np.mean(land)- 100
                true_land = sig.find_peaks(tmpDat.LForce, height = htThrsh, distance = disThrsh)[0]
                land_ht =  sig.find_peaks(tmpDat.LForce, height = htThrsh, distance = disThrsh)[1]['peak_heights']
                plt.figure()
                plt.plot(tmpDat.LForce, label = 'Left Foot Total Force') 
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
                    sdFz.append(np.std(forceZ [ land[ii] + 100 : land[ii] + 400]))
                    avgF = movAvgForce(forceZ,land[ii] , land[ii] + 200 , 10)
                    sdF = movSDForce(forceZ, land[ii], land[ii] + 200, 10)
                    subBW = findBW(avgF)
                    try:
                        stabilization.append(findStabilization(avgF, sdF)/100)
                    except:
                        stabilization.append('NaN')
                    
                   
                    toeClawAvg.append(np.mean(tmpDat.LplantarToe[land[ii]: land[ii]+100]))
                    toeClawPk.append(np.max(tmpDat.LplantarToe[land[ii]: land[ii]+100]))
                    toeLatAvg.append(np.mean(tmpDat.LplantarToeLat[land[ii]: land[ii]+100]))
                    toeLatPk.append(np.max(tmpDat.LplantarToeLat[land[ii]: land[ii]+100]))
                    toeMedAvg.append(np.mean(tmpDat.LplantarToeMed[land[ii]: land[ii]+100]))
                    toeMedPk.append(np.max(tmpDat.LplantarToeMed[land[ii]: land[ii]+100]))
            
                    ffAvg.append(np.mean(tmpDat.LplantarForefoot[land[ii]: land[ii]+100]))
                    ffPk.append(np.max(tmpDat.LplantarForefoot[land[ii]: land[ii]+100]))
                    ffConArea.append(np.count_nonzero(tmpDat.LplantarForefoot[land[ii]: land[ii]+100])/100/68*100) # divided by 100 frames
                    ffLatAvg.append(np.mean(tmpDat.LplantarForefootLat[land[ii]: land[ii]+100]))
                    ffLatPk.append(np.max(tmpDat.LplantarForefootLat[land[ii]: land[ii]+100]))
                    ffMedAvg.append(np.mean(tmpDat.LplantarForefootMed[land[ii]: land[ii]+100]))
                    ffMedPk.append(np.max(tmpDat.LplantarForefootMed[land[ii]: land[ii]+100]))
            
                    heelArea.append(np.count_nonzero(tmpDat.LplantarHeel[land[ii]: land[ii]+100])/ 100/ 43*100) # divided by 100 frames
                    heelPres.append(np.mean(tmpDat.LplantarHeel[land[ii]: land[ii]+100]))
                   
                    config.append(tmpDat.config)
                    subject.append(tmpDat.subject)
                    movement.append(moveTmp)
                    order.append(torder)
            
                outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(order), 'StabTime': list(stabilization),
                                 'ToeClaw': list(toeClawAvg), 'ToeClawPeak': list(toeClawPk), 'ToeLat': list(toeLatAvg), 'ToeLatPeak' : list(toeLatPk),
                                 'ToeMed' : list(toeMedAvg), 'ToeMedPeak' : list(toeMedPk), 'ForefootAvg' : list(ffAvg), 'ForefootPeak' : list(ffPk), 
                                 'ForefootContA' : list(ffConArea),'ForefootLat': list(ffLatAvg), 'ForefootLatPk': list(ffLatPk), 'ForefootMed': list(ffMedAvg),
                                 'ForefootMedPk': list(ffMedPk),'HeelConArea' : list(heelArea), 'HeelPressure' : list(heelPres)
                                 })
       
        
                                   
                outfileName = fPath + '0_StabTime_CompiledResults_SLL_Trail.csv'
                if save_on == 1:
                    if os.path.exists(outfileName) == False:
                    
                        outcomes.to_csv(outfileName, header=True, index = False)
                
                    else:
                        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
                       
                        
        
        
    except:
            print(fName)             
        
            
            



