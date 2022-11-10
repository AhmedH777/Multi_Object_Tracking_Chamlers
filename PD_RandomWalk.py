# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:02:10 2022

@author: Kotb
"""
import numpy as np
import Lib
from MOT_FilterLib import Kalman
from nMOT import nObjectTracker
import MOT_Lib

class Ped:
    def __init__(self,initPosX,initPosY,endPosX,endPosY,noiseX,noiseY,steps,clutterNoiseX,clutterNoiseY,maxClutter,maxMisDetect,motionType = "Linear"):
        self.InitPos = [initPosX,initPosY]
        self.EndPos  = [endPosX,endPosY]
        self.NoiseX  = noiseX
        self.NoiseY  = noiseY
        self.Motion  = motionType
        self.NumSteps = steps
        self.MaxMisDetect  = maxMisDetect
        self.MaxClutter    = maxClutter
        self.ClutterNoiseX = clutterNoiseX
        self.ClutterNoiseY = clutterNoiseY
        self.StepsGT  = []
        self.Steps    = []
        self.Clutter  = []
        
        if(motionType == "Linear"):
           self.StepsGT =  self.calcLinearModel()
           self.Steps   =  self.calcNoisySteps(self.StepsGT)
           self.Steps   =  self.calcMisDetections(self.Steps)
           self.Clutter =  self.calcClutter(self.StepsGT)

    def calcNoisySteps(self,GT):
        StepsNoisy = []
        for x,y in GT:
            x_noise = np.random.normal(0, self.NoiseX, 1)
            y_noise = np.random.normal(0, self.NoiseY, 1)
            
            xN = x + x_noise[0]
            yN = y + y_noise[0]
            
            StepsNoisy.append([xN,yN])
        
        return StepsNoisy
    
    def calcMisDetections(self,Steps):
        import random
        n = len(Steps)
        
        for i in range(self.MaxMisDetect):
            idx = random.randint(1,n-1)
            
            Steps[idx] = []
        
        return Steps
    
    def calcClutter(self,GTsteps):
        import random
        
        ClutterSteps = []
        for x,y in GTsteps:
            num = random.randint(0,self.MaxClutter)
            if(num == 0):
                ClutterSteps.append([])
            else:
                tempClutter = []
                for j in range(num):
                    x_noise = np.random.normal(0, self.ClutterNoiseX, 1)
                    y_noise = np.random.normal(0, self.ClutterNoiseY, 1)
                    xC = x + x_noise[0]
                    yC = y + y_noise[0]
                    tempClutter.append([xC,yC])
                    
                ClutterSteps.append(tempClutter)
        return ClutterSteps
             
                
        
    def calcLinearModel(self):
        Trace = []
        slope     = (self.EndPos[1] - self.InitPos[1]) / (self.EndPos[0] - self.InitPos[0])
        intercept =  self.EndPos[1] - ( slope * self.EndPos[0] )
        
        delta_x = abs(self.EndPos[0] - self.InitPos[0])
        step    = delta_x/self.NumSteps
        
        Trace.append([self.InitPos[0],self.InitPos[1]])
        initX = self.InitPos[0]
        for i in range(1,self.NumSteps):
            X = initX + (i * step)
            Y = (slope * X) + intercept
            
            Trace.append([X,Y])
        
        Trace.append([self.EndPos[0],self.EndPos[1]])
        
        
        return Trace

    def plotGT(self,ax,color ='k'):
        pltColor = color + 'o'
        for x,y in self.StepsGT:
            ax.plot(x,y,pltColor)
    
    def plotSteps(self,ax,color ='b'):
        for i in range(len(self.Steps)):
            Pt = self.Steps[i]
            if(len(Pt) > 0):
                x = Pt[0]
                y = Pt[1]
                state = np.matrix([x,y]).T
                cov   = np.matrix([[self.NoiseX *self.NoiseX  , 0.0],[0.0 , self.NoiseY * self.NoiseY]])
                Lib.PlotPoseWithEllipse(ax, state, cov, color)
    
    def plotClutter(self,ax,color='r'):
        pltColor = color + 'o'
        for i in range(len(self.Clutter)):
            Pts = self.Clutter[i]
            if(len(Pts) > 0):
                for pt in Pts: 
                    x = pt[0]
                    y = pt[1]
                    ax.plot(x,y,pltColor)      
    
    def plotStep(self,ax,step,colorClutter = 'r',colorMeas = 'b', colorGt = 'k', pltGt = False):
        # Meas
        Pt = self.Steps[step]
        if(len(Pt) > 0):
            x = Pt[0]
            y = Pt[1]
            state = np.matrix([x,y]).T
            cov   = np.matrix([[self.NoiseX *self.NoiseX  , 0.0],[0.0 , self.NoiseY * self.NoiseY]])
            Lib.PlotPoseWithEllipse(ax, state, cov, colorMeas)
        
        # Clutter
        pltColor = colorClutter + 'o'
        Pts = self.Clutter[step]
        if(len(Pts) > 0):
            for pt in Pts: 
                x = pt[0]
                y = pt[1]
                ax.plot(x,y,pltColor)   
                
        # GT
        if(pltGt == True):
            pltColor = colorGt + 'o'
            pt = self.StepsGT[step]
            x = pt[0]
            y = pt[1]
            ax.plot(x,y,pltColor)
                
##############################################################################

class Trace:
    def __init__(self,pd_list):
        self.Pd_list = pd_list
        self.Trace   = []
        self.CreateTrace()
        self.nSteps  = len(self.Trace)
        
    def CreateTrace(self):
        #Sanity Check
        numSteps = self.Pd_list[0].NumSteps
        for pd in self.Pd_list:
            if(pd.NumSteps != numSteps):
                print("StepsMisMatch")
                return([])
        
        # Append Measurments:
        for i in range(numSteps):
            Step = []
            for pd in self.Pd_list:
                Meas = pd.Steps[i]
                if(len(Meas) > 0):
                    Step.append(np.matrix(Meas).T)
                
                Clutter = pd.Clutter[i]
                nClutter = len(Clutter)
                if(nClutter > 0 ):
                    for c in Clutter:
                        pass
                        Step.append(np.matrix(c).T)
            self.Trace.append(Step)

    def PlotTrace(self,ax,plotGt = False):
        for pd in self.Pd_list:
            pd.plotSteps(ax)
            pd.plotClutter(ax)
            if(plotGt ==True):
                pd.plotGT(ax)


    def PlotStep(self,ax,Step,plotGt = False):
        for pd in self.Pd_list:        
            pd.plotStep(ax,Step,pltGt = plotGt)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 1.0
    STEPS = 15

    MEAS_NOISE_X = 0.1
    MEAS_NOISE_Y = 0.1
    
    VAR_MEAS_X = MEAS_NOISE_X * MEAS_NOISE_X
    VAR_MEAS_Y = MEAS_NOISE_Y * MEAS_NOISE_Y
    
    PROCESS_NOISE_X = 0.5
    PROCESS_NOISE_Y = 0.5
    
    VAR_PROCESS_X = PROCESS_NOISE_X * PROCESS_NOISE_X
    VAR_PROCESS_Y = PROCESS_NOISE_Y * PROCESS_NOISE_Y
    
    PROCESS_NOISE_VX = 0.5
    PROCESS_NOISE_VY = 0.5
    
    VAR_PROCESS_VX = PROCESS_NOISE_VX * PROCESS_NOISE_VX
    VAR_PROCESS_VY = PROCESS_NOISE_VY * PROCESS_NOISE_VY
    
    CLUTTER_NOISE_X = 1.0
    CLUTTER_NOISE_Y = 2.0
    
    
    MAX_CLUTTER   = 2
    MAX_MISDETECT = 0
    

    pd1 = Ped(1.0,1.0,30.0,30.0,MEAS_NOISE_X,MEAS_NOISE_Y,STEPS,CLUTTER_NOISE_X,CLUTTER_NOISE_Y,MAX_CLUTTER,MAX_MISDETECT,"Linear")
    pd2 = Ped(2.0,15.0,25.0,1.0,MEAS_NOISE_X,MEAS_NOISE_Y,STEPS,CLUTTER_NOISE_X,CLUTTER_NOISE_Y,MAX_CLUTTER,MAX_MISDETECT,"Linear")
    Steps = Trace([pd1,pd2])


    InitialState = np.matrix(np.zeros(4)).T
    InitialState[0] = 1.0
    InitialState[1] = 1.0
    InitialState[2] = 1.0
    InitialState[3] = 1.0    
    InitialState2 = np.matrix(np.zeros(4)).T
    InitialState2[0] = 2.0
    InitialState2[1] = 15.0
    InitialState2[2] = 1.0
    InitialState2[3] = -1.0  
    
    InitialCov   = np.matrix(np.eye(4)) * 0.001
        
    R_mat        = np.matrix([[VAR_MEAS_X , 0.0],[0.0, VAR_MEAS_Y]])
    
    Q_mat        = np.matrix([[VAR_PROCESS_X , 0.0          , 0.0            , 0.0           ],
                              [0.0           , VAR_PROCESS_Y, 0.0            , 0.0           ],
                              [0.0           , 0.0          , VAR_PROCESS_VX , 0.0           ],
                              [0.0           , 0.0          , 0.0            , VAR_PROCESS_VY]])
    
    A_mat        = np.matrix([[1.0           , 0.0          , T              , 0.0           ],
                              [0.0           , 1.0          , 0.0            , T             ],
                              [0.0           , 0.0          , 1.0            , 0.0           ],
                              [0.0           , 0.0          , 0.0            , 1.0           ]])
    
    H_mat        = np.matrix([[1.0           , 0.0          , 0.0            , 0.0           ],
                              [0.0           , 1.0          , 0.0            , 0.0           ]])
     
    P_D          = 0.85
    lambda_c     = 0.3
    range_c      = np.matrix([1,20])

    KF1       = Kalman(InitialState ,InitialCov)   
    KF2       = Kalman(InitialState2,InitialCov) 
    
    KF3       = Kalman(InitialState ,InitialCov)   
    KF4       = Kalman(InitialState2,InitialCov) 
    Clutter  = MOT_Lib.UniClutter(P_D,lambda_c,range_c)    

    FilterObjects   = [KF1,KF2]
    FilterObjects2  = [KF3,KF4]
    
    GNN  = nObjectTracker(FilterObjects,Clutter,0,0,0,0,0)
    JPDA = nObjectTracker(FilterObjects2,Clutter,0,0,0,0,10)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')

    Steps.PlotTrace(ax,True)
    
    # Initial State plot
    Lib.PlotPoseWithEllipse(ax,InitialState,InitialCov,'y')
    Lib.PlotPoseWithEllipse(ax,InitialState2,InitialCov,'y')


    for i in range(Steps.nSteps):
        Step = Steps.Trace[i]
        GNN.GNNfilter(A_mat, Q_mat, Step, H_mat, R_mat)
        JPDA.JPDAfilter(A_mat, Q_mat, Step, H_mat, R_mat)
        
        # Plotting
        FilterObj = GNN.FilterObjList[0]
        pose = FilterObj.State
        cov  = FilterObj.Cov
        Lib.PlotPoseWithEllipse(ax,pose,cov,'g')
        
        FilterObj = GNN.FilterObjList[1]
        pose = FilterObj.State
        cov  = FilterObj.Cov
        Lib.PlotPoseWithEllipse(ax,pose,cov,'m')
        
        # Plotting
        FilterObj = JPDA.FilterObjList[0]
        pose = FilterObj.State
        cov  = FilterObj.Cov
        Lib.PlotPoseWithEllipse(ax,pose,cov,'y')
        
        FilterObj = JPDA.FilterObjList[1]
        pose = FilterObj.State
        cov  = FilterObj.Cov
        Lib.PlotPoseWithEllipse(ax,pose,cov,'c')
    
    #Lib.PlotAssocs(ax,GNN.Assocs)
    '''    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    
    Steps.PlotTrace(ax2,7)
    '''