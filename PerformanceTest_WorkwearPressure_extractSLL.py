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
    RForce = []
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
                     LForce, RForce, 
                     LCOP_X, LCOP_Y, RCOP_X, RCOP_Y,
                     config, movement, subj, dat)
    
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

    try: 
        #fName = entries[2]
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep="_")[1]
        moveTmp = fName.split(sep = "_")[2].split(sep = '.')[0]
        torder = fName.split(sep = "_")[3].split(sep = '.')[0]
        
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
                    order.append(torder)
            
                outcomes = pd.DataFrame({'Subject': list(subject), 'Movement':list(movement), 'Config':list(config), 'Order':list(order),
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
        
            
            



