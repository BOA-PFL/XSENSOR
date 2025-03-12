
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





 ## setting up data classes for 6 possible combos: DorsalRightCOP, DorsalLeftCOP, RightLeftCOP, DorsalRightnoCOP, DorsalLeftnoCOP, RightLeftnoCOP 
@dataclass    
class tsData:
     dorsalMat: np.array
     dorsalForefoot: np.array
     dorsalMidfoot: np.array
     dorsalInstep: np.array 
     
     
     LplantarMat: np.array
     LplantarToe: np.array 
     LplantarForefoot: np.array 
     LplantarMidfoot: np.array 
     LplantarHeel: np.array 
     LplantarLateral: np.array
     LplantarMedial: np.array
     
     RplantarMat: np.array
     RplantarToe: np.array 
     RplantarForefoot: np.array 
     RplantarMidfoot: np.array 
     RplantarHeel: np.array 
     RplantarLateral: np.array
     RplantarMedial: np.array
  
     LForce: np.array 
     LHS: np.array 
     LTO: np.array
     
     RForce: np.array 
     RHS: np.array 
     RTO: np.array
     
     LCOP_X: np.array 
     LCOP_Y: np.array 

     RCOP_X: np.array 
     RCOP_Y: np.array 

     config: str
     movement: str
     subject: str
     
     fullDat: pd.DataFrame #entire stored dataframe. 
     #this class is useful for plotting and subsequent analysis
     
     # below is a method of the dataclass
     def plotAvgPressure(self):
         
         earlyPlantar = np.zeros([len(self.RHS), 31, 9])
         midPlantar = np.zeros([len(self.RHS), 31, 9])
         latePlantar = np.zeros([len(self.RHS), 31, 9])
         
         earlyDorsal = np.zeros([len(self.RHS), 18, 10])
         midDorsal = np.zeros([len(self.RHS), 18, 10])
         lateDorsal = np.zeros([len(self.RHS), 18, 10])
         
         for i in range(len(self.RHS)):
             
             earlyPlantar[i,:,:] = self.RplantarMat[self.RHS[i],:,:]
             midPlantar[i,:,:] = self.RplantarMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]
             latePlantar[i,:,:] = self.RplantarMat[self.RTO[i],:,:]
             earlyDorsal[i,:,:] = self.dorsalMat[self.RHS[i],:,:]
             midDorsal[i,:,:] = self.dorsalMat[self.RHS[i] + round((self.RTO[i]-self.RHS[i])/2),:,:]
             lateDorsal[i,:,:] = self.dorsalMat[self.RTO[i],:,:]
             
         earlyPlantarAvg = np.mean(earlyPlantar, axis = 0)
         midPlantarAvg = np.mean(midPlantar, axis = 0)
         latePlantarAvg = np.mean(latePlantar, axis = 0)
         earlyDorsalAvg = np.mean(earlyDorsal, axis = 0)
         midDorsalAvg = np.mean(midDorsal, axis = 0)
         lateDorsalAvg = np.mean(lateDorsal, axis = 0)
         
         fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
         ax1 = sns.heatmap(earlyDorsalAvg, ax = ax1, cmap="mako", vmin = 0, vmax = np.max(earlyDorsalAvg))
         ax1.set_title('Dorsal Pressure') 
         ax1.set_ylabel('Initial Contact')
         
         
         ax2 = sns.heatmap(earlyPlantarAvg, ax = ax2, cmap="mako", vmin = 0, vmax = np.max(earlyPlantarAvg))
         ax2.set_title('Plantar Pressure') 
         
         ax3 = sns.heatmap(midDorsalAvg, ax = ax3, cmap = 'mako', vmin = 0, vmax = np.max(midDorsalAvg))
         ax3.set_ylabel('Midstance')
         
         ax4 = sns.heatmap(midPlantarAvg, ax = ax4, cmap = 'mako', vmin = 0, vmax = np.max(midPlantarAvg))
         
         ax5 = sns.heatmap(lateDorsalAvg, ax = ax5, cmap = 'mako', vmin = 0, vmax = np.max(latePlantarAvg))
         ax5.set_ylabel('Toe off')
         
         
         ax6 = sns.heatmap(latePlantarAvg, ax = ax6, cmap = 'mako', vmin = 0, vmax = np.max(lateDorsalAvg))
         
         fig.set_size_inches(5, 10)
         
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

    #inputName = entries[3]
    
    freq = 100
    dat = pd.read_csv(fPath+inputName, sep=',', header = 0, low_memory=False)
    if dat.shape[1] == 2:
        dat = pd.read_csv(fPath+inputName, sep=',', header = 1, low_memory=False)
   
    dat = delimitTrial(dat, inputName)
    subj = inputName.split(sep="_")[0]
    config = inputName.split(sep="_")[1]
    movement = inputName.split(sep = '_')[2] 

    RplantarMat = []
    RplantarToe = []
    RplantarForefoot = []
    RplantarMidfoot = []
    RplantarHeel = []
    RplantarLateral = []
    RplantarMedial = []
    RForce = []
    RHS = []
    RTO = []
    RCOP_Y = []
    RCOP_X = []

    LplantarMat = []
    LplantarToe = []
    LplantarForefoot = []
    LplantarMidfoot = []
    LplantarHeel = []
    LplantarLateral = []
    LplantarMedial = []
    LForce = []
    LHS = []
    LTO = []
    LCOP_Y = []
    LCOP_X = []
    
    dorsalMat = []
    dorsalForefoot = []
    dorsalMidfoot = []
    dorsalInstep = []

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
            
            LForce = np.mean(LplantarMat, axis = (1,2))*6895*0.014699
            LForce = zeroInsoleForce(LForce,freq)

        
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
          
            RForce = np.mean(RplantarMat, axis = (1,2))*6895*0.014699
            RForce = zeroInsoleForce(RForce,freq)



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
                
    result = tsData(dorsalMat, dorsalForefoot, dorsalMidfoot, dorsalInstep, 
                     LplantarMat, LplantarToe, LplantarForefoot, LplantarMidfoot, LplantarHeel, LplantarLateral, LplantarMedial,
                     RplantarMat, RplantarToe, RplantarForefoot, RplantarMidfoot, RplantarHeel,  RplantarLateral, RplantarMedial,
                     LForce, LHS, LTO,  RForce, RHS, RTO,
                     LCOP_X, LCOP_Y, RCOP_X, RCOP_Y,
                     config, movement, subj, dat)
    
    return(result)





def findStabilization(avgF, sdF):
    """
    Using the rolling average and SD values, this calcualtes when the 
    actual stabilized force occurs. 
    
    Parameters
    ----------
    avgF : list, calculated using movAvgForce 
        rolling average of pressure.
    sdF : list, calcualted using movSDForce above
        rolling SD of pressure.

    Returns
    -------
    floating point number
        Time to stabilize using the heuristic: pressure is within +/- 5% of subject
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
        pandas series of force from pressure insoles.
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
        pandas series of force from pressure insoles.
    landing : List
        list of landings calcualted from findLandings.
    takeoff : List
        list of takeoffs from findTakeoffs.
    length : Integer
        length of time in indices to calculate the moving average.

    Returns
    -------
    avgF : list
        smoothed rolling SD of pressure

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
fPath = 'C:\\Users\\milena.singletary\\OneDrive - BOA Technology Inc\\General - PFL Team\\Testing Segments\\WorkWear_Performance\\2025_Performance_HighCutPFSWorkwearI_TimberlandPro\\Xsensor\\cropped\\'
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
        #fName = entries[2]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
        torder = fName.split(sep = "_")[3].split(sep = '.')[0]
        
        # Make sure the files are named FirstLast_Config_Movement_Trial# - The "if" statement won't work if there isn't a trial number next to the movement
        #if ('SLL' in moveTmp):# or ('SLLT' in moveTmp): # or ('Trail' in moveTmp):
        if (moveTmp == 'SLL') or ('SLLT' in moveTmp):
            #dat = pd.read_csv(fPath+fName, sep=',', skiprows = 1, header = 'infer')
            
  
            tmpDat = createTSmat(fName)
            
            if len(tmpDat.RplantarMat != 0):
                ffoot = np.mean(tmpDat.RplantarForefoot, axis = (1,2)) * 6895 * 0.014699
                pForce = tmpDat.RForce
            
            elif len(tmpDat.LplantarMat != 0):
                ffoot = np.mean(tmpDat.LplantarForefoot, axis = (1,2)) * 6895 * 0.014699
                pForce = tmpDat.LForce    
                
               
                
            land = sig.find_peaks(pForce, height = 1000, distance = 600)[0]
            land_ht =  sig.find_peaks(pForce, height = 1000, distance = 600)[1]['peak_heights']
            fft_pk = sig.find_peaks(ffoot, height = 1000, distance = 500)[0]
            fft_ht = sig.find_peaks(ffoot, height = 1000, distance = 500)[1]['peak_heights']
            
            htThrsh = np.mean(land_ht) - 400
                        
            true_land = sig.find_peaks(pForce, height = htThrsh, distance = 500)[0]
            land_ht =  sig.find_peaks(pForce, height = htThrsh, distance = 500)[1]['peak_heights']
   

            
            answer = True # if data check is off. 
            if data_check == 1:
                plt.figure()
                plt.plot(pForce, label = 'Right Foot Total Force') 
                plt.plot(true_land, land_ht, marker = 'o', linestyle = 'none')
                #plt.plot(range(len(ffoot)), ffoot)
                #plt.plot(fft_pk, fft_ht, marker = 'v', linestyle = 'none')
                answer = messagebox.askyesno("Question","Is data clean?")    
  
  
            if answer == False:
                disThrsh = np.mean(land)- 100
                true_land = sig.find_peaks(pForce, height = htThrsh, distance = disThrsh)[0]
                land_ht =  sig.find_peaks(pForce, height = htThrsh, distance = disThrsh)[1]['peak_heights']
                plt.figure()
                plt.plot(pForce, label = 'Right Foot Total Force') 
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
                
        
        
                if len(tmpDat.RplantarMat != 0):
                
                    # toe 39 ; fft 68 ; heel 43
                    for ii in range(len(land)):
                        sdFz.append(np.std(pForce [ land[ii] + 100 : land[ii] + 400]))
                        avgF = movAvgForce(pForce,land[ii] , land[ii] + 200 , 10)
                        sdF = movSDForce(pForce, land[ii], land[ii] + 200, 10)
                        subBW = findBW(avgF)
                        try:
                            stabilization.append(findStabilization(avgF, sdF)/100)
                        except:
                            stabilization.append('NaN')
                            
                        toeClawAvg.append(np.mean(tmpDat.RplantarToe[land[ii]: land[ii]+100]))
                        toeClawPk.append(np.max(tmpDat.RplantarToe[land[ii]: land[ii]+100]))

                
                        ffAvg.append(np.mean(tmpDat.RplantarForefoot[land[ii]: land[ii]+100]))
                        ffPk.append(np.max(tmpDat.RplantarForefoot[land[ii]: land[ii]+100]))
                        ffConArea.append(np.count_nonzero(tmpDat.RplantarForefoot[land[ii]: land[ii]+100])/100/68*100) # divided by 100 frames

                
                        heelArea.append(np.count_nonzero(tmpDat.RplantarHeel[land[ii]: land[ii]+100])/ 100/ 43*100) # divided by 100 frames
                        heelPres.append(np.mean(tmpDat.RplantarHeel[land[ii]: land[ii]+100]))
                        
                        config.append(tmpDat.config)
                        subject.append(tmpDat.subject)
                        movement.append(moveTmp)
                        order.append(torder)
                        
                elif len(tmpDat.LplantarMat != 0):
                        sdFz.append(np.std(pForce [ land[ii] + 100 : land[ii] + 400]))
                        avgF = movAvgForce(pForce,land[ii] , land[ii] + 200 , 10)
                        sdF = movSDForce(pForce, land[ii], land[ii] + 200, 10)
                        subBW = findBW(avgF)
                        try:
                            stabilization.append(findStabilization(avgF, sdF)/100)
                        except:
                            stabilization.append('NaN')

                    # toe 39 ; fft 68 ; heel 43
                        for ii in range(len(land)):
                            toeClawAvg.append(np.mean(tmpDat.LplantarToe[land[ii]: land[ii]+100]))
                            toeClawPk.append(np.max(tmpDat.LplantarToe[land[ii]: land[ii]+100]))

                            ffAvg.append(np.mean(tmpDat.LplantarForefoot[land[ii]: land[ii]+100]))
                            ffPk.append(np.max(tmpDat.LplantarForefoot[land[ii]: land[ii]+100]))
                            ffConArea.append(np.count_nonzero(tmpDat.LplantarForefoot[land[ii]: land[ii]+100])/100/68*100) # divided by 100 frames

                        
                            heelArea.append(np.count_nonzero(tmpDat.LplantarHeel[land[ii]: land[ii]+100])/ 100/ 43*100) # divided by 100 frames
                            heelPres.append(np.mean(tmpDat.LplantarHeel[land[ii]: land[ii]+100]))
                            
                            config.append(tmpDat.config)
                            subject.append(tmpDat.subject)
                            movement.append(moveTmp)
                            order.append(torder)
                        
                        
                        
            
                outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(order), 'StabTime': list(stabilization),
                                 'ToeClaw': list(toeClawAvg), 'ToeClawPeak': list(toeClawPk),  'ForefootAvg' : list(ffAvg), 'ForefootPeak' : list(ffPk), 'ForefootContA' : list(ffConArea), 
                                 'HeelConArea' : list(heelArea), 'HeelPressure' : list(heelPres)})
       
        
                                   
                outfileName = fPath + '1_CompiledResults_SLL.csv'
                if save_on == 1:
                    if os.path.exists(outfileName) == False:
                    
                        outcomes.to_csv(outfileName, header=True, index = False)
                
                    else:
                        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
                       
                        
        
        
    except:
            print('Not usable data')             
        
            
            



