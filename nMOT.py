# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:04:35 2022

@author: Kotb
"""
from oct2py import octave
import numpy as np
from MOT_FilterLib import Kalman
import MOT_Lib
import Lib


PATH_HUNGARIAN   = r"D:\Projects\Multi_Object_Tracking_Chamlers\PythonCode\HA2\assign2D"
PATH_MURTY       = r"D:\Projects\Multi_Object_Tracking_Chamlers\PythonCode\HA2\kBest2DAssign"
PATH_NORM_LOG_Ws = r"D:\Projects\Multi_Object_Tracking_Chamlers\PythonCode\HA2\normalizeLogWeights"
    
class nObjectTracker:
    def __init__(self,FilterObjList,clutterObj,P_gating,meas_dim,W_min,mergingThres,MaxHyp):
        self.FilterObjList  = FilterObjList
        self.clutterObj     = clutterObj
        self.P_gating       = P_gating
        self.meas_dim       = meas_dim
        self.W_min          = W_min
        self.mergeThres     = mergingThres
        self.MaxHyp         = MaxHyp
        
        self.Assocs         = []
        
        self.N_states       = len(FilterObjList) 
        
    def GNNfilter(self,A_mat, Q_mat, Z_list, H_mat, R_mat): 
        # Weight of No Detection
        l_0_log = np.log(1.0 - self.clutterObj.P_D)
        # Weight of Detection
        l_clut_log = np.log(self.clutterObj.P_D / self.clutterObj.intensity)
        # Size of Meas k
        Mk         = len(Z_list)
        #################### Set L Mat ############################
        # Init
        L_Mat      = np.matrix(np.ones((self.N_states,( self.N_states + Mk ) )) * 100000)
        
        # Fill Mat with Log Likelihoods
        for i in range(self.N_states):
            obj = self.FilterObjList[i]
            S       = (H_mat * obj.Cov * H_mat.T) + R_mat
            l_i_log = l_clut_log - ( 0.5 * np.log( np.linalg.det( 2 * np.pi * S ) ) ) 
            
            for j in range(Mk):
                z              = Z_list[j]
                V              = z - ( H_mat *  obj.State )
                log_likelihood = l_i_log - ( 0.5 * V.T * S.I * V )
                
                L_Mat[i,j] = ( -1.0 * log_likelihood )
            
            L_Mat[i,(i + Mk)] = (-1.0 * l_0_log)
            
        BestAssoc = octave.feval(PATH_MURTY,L_Mat,1) - 1
        
        for i in range(self.N_states):
            obj = self.FilterObjList[i]
            zBestAssoc = int(BestAssoc[i,0])
            if(zBestAssoc < Mk):
                z = Z_list[zBestAssoc]
                obj.Update(z ,H_mat, R_mat)
                self.Assocs.append([obj.State[0:2],z])
            
            obj.Predict(A_mat,Q_mat)
               

    def JPDAfilter(self,A_mat, Q_mat, Z_list, H_mat, R_mat): 
        # Weight of No Detection
        l_0_log = np.log(1.0 - self.clutterObj.P_D)
        # Weight of Detection
        l_clut_log = np.log(self.clutterObj.P_D / self.clutterObj.intensity)
        # Size of Meas k
        Mk         = len(Z_list)
        # Dimension of Meas
        mDim       = len(Z_list[0]) 
        #################### Set L Mat ############################
        # Init
        L_Mat      = np.matrix(np.ones((self.N_states,( self.N_states + Mk ) )) * 100000)
        
        # Fill Mat with Log Likelihoods
        for i in range(self.N_states):
            obj = self.FilterObjList[i]
            S       = (H_mat * obj.Cov * H_mat.T) + R_mat
            l_i_log = l_clut_log - ( 0.5 * np.log( np.linalg.det( 2 * np.pi * S ) ) ) 
            
            for j in range(Mk):
                z              = Z_list[j]
                V              = z - ( H_mat *  obj.State )
                log_likelihood = l_i_log - ( 0.5 * V.T * S.I * V )
                
                L_Mat[i,j] = ( -1.0 * log_likelihood )
            
            L_Mat[i,(i + Mk)] = (-1.0 * l_0_log)
            
        BestAssocs = octave.feval(PATH_MURTY,L_Mat,self.MaxHyp) - 1
        
        # Calculate Log Weights
        LogWeights = []
        
        for i in range(len(BestAssocs[0])):
            A_Mat = np.matrix(np.zeros((self.N_states,( self.N_states + Mk ) )) )
            currentAssoc = BestAssocs[:,i]
            
            # Create A Matrix
            for ObjIdx in range(self.N_states):
                currentMeasIdx = int(currentAssoc[ObjIdx])
                A_Mat[ObjIdx,currentMeasIdx] = 1.0
                      
            logWeight = np.trace(-1.0 * A_Mat.T * L_Mat)
                 
            LogWeights.append(logWeight)
            
        NormLogWeights = octave.feval(PATH_NORM_LOG_Ws,LogWeights)[0]
        
        # Create Local Hypothesis
        Beta  = np.matrix(np.zeros((self.N_states,( self.N_states + Mk ) )) )
        nTheta = len(NormLogWeights) 
        
        for nState in range(self.N_states):
            for h in range(nTheta):
                nAssoc = int(BestAssocs[nState,h])
                Beta[nState,nAssoc] = Beta[nState,nAssoc] + np.exp(NormLogWeights[h])
        
        # Merge Local Hypothesis
        for nState in range(self.N_states):
            V_Mixed_Sq_Sum = np.matrix(np.zeros([mDim,mDim]))
            obj  = self.FilterObjList[nState]
            
            S    = (H_mat * obj.Cov * H_mat.T) + R_mat
            K    = obj.Cov * H_mat.T * S.I
            
            if(Mk > 0):
                V_Mixed = np.matrix(np.zeros(mDim)).T
                # Update State
                for nMeas in range(Mk):
                    z         = Z_list[nMeas]
                    V         = z - ( H_mat * obj.State )
                    V_Mixed   = V_Mixed + ( Beta[nState,nMeas] * V )
                    V_Mixed_Sq_Sum = V_Mixed_Sq_Sum +  ( Beta[nState,nMeas] * V * V.T )

                obj.State = obj.State + (K * V_Mixed)
            
                # Update Cov
                P_bar   = obj.Cov - ( K * S * K.T )
                P_tilde = K * (V_Mixed_Sq_Sum - ( V_Mixed * V_Mixed.T ) ) * K.T
                beta0   = Beta[nState, (Mk + nState)]
                obj.Cov = ( beta0 * obj.Cov ) + ( (1.0 - beta0) * P_bar ) + P_tilde

            obj.Predict(A_mat,Q_mat)


###############################################################################
############################# Unit Tests ######################################
###############################################################################
'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    InitialState  = np.matrix([2.5]).T
    InitialState2 = np.matrix([-2.5]).T 
    InitialCov    = np.matrix([0.36])
    
    A_mat        = np.matrix([1.0])
    H_mat        = np.matrix([1.0])
    
    R_mat        = np.matrix([0.2])
    Q_mat        = np.matrix([0.35])
    
    P_D          = 0.85
    lambda_c     = 0.3
    range_c      = np.matrix([-5,5])

    KF1       = Kalman(InitialState ,InitialCov)   
    KF2       = Kalman(InitialState2,InitialCov) 
    
    KF3       = Kalman(InitialState ,InitialCov)   
    KF4       = Kalman(InitialState2,InitialCov)
    Clutter  = MOT_Lib.UniClutter(P_D,lambda_c,range_c)    
    
    FilterObjects  = [KF1,KF2]
    FilterObjects2 = [KF3,KF4]
    
    GNN  = nObjectTracker(FilterObjects,Clutter,0,0,0,0,0)
    JPDA = nObjectTracker(FilterObjects2,Clutter,0,0,0,0,10)

    Z_List = [[-1.6,1.0],
              [-2.0,3.0],
              [-2.3,0.6],
              [-2.0],
              [],
              [2.8,-2.0]]
    
    for i in range(len(Z_List)):
        #########################
        ###### Plot Init ########
        #########################
        print("############")
        print("Step = ",i)
        print("############")
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Step ' + str(i))
        minRange = -10
        maxRange = 10
        
        ##########################
        ########### Run ##########
        ##########################
        z = Z_List[i]
        GNN.GNNfilter(A_mat, Q_mat, z, H_mat, R_mat)
        JPDA.JPDAfilter(A_mat, Q_mat, z, H_mat, R_mat)
        

        ##########################
        ######## Plot Meas #######
        ##########################
        for Point in z:
            ax.plot(Point,0,'r*')
        ##########################
        ####### Plot Prior #######
        ##########################
        mean1 = InitialState[0,0]
        std1  = np.sqrt(InitialCov[0,0])
        Lib.PlotGaussian1D(ax,mean1,std1,minRange,maxRange,'b')
        
        mean2 = InitialState2[0,0]
        std2  = np.sqrt(InitialCov[0,0])
        Lib.PlotGaussian1D(ax,mean2,std2,minRange,maxRange,'b')
        ##########################
        ##### Plot Posterior #####
        ##########################  
        # GNN
        FilterObj = GNN.FilterObjList[0]
        mean1 = FilterObj.State[0,0]
        std1  = np.sqrt(FilterObj.Cov[0,0])
        Lib.PlotGaussian1D(ax,mean1,std1,minRange,maxRange,'g')
        
        FilterObj = GNN.FilterObjList[1]
        mean2 = FilterObj.State[0,0]
        std2  = np.sqrt(FilterObj.Cov[0,0])
        Lib.PlotGaussian1D(ax,mean2,std2,minRange,maxRange,'g')
        
        print("### GNN ###")
        print("Mean1 = ",mean1)
        print("Std1  = ",std1)
        print("Mean2 = ",mean2)
        print("Std2  = ",std2)
        
        #JPDA
        FilterObj = JPDA.FilterObjList[0]
        mean1 = FilterObj.State[0,0]
        std1  = np.sqrt(FilterObj.Cov[0,0])
        Lib.PlotGaussian1D(ax,mean1,std1,minRange,maxRange,'m')
        
        FilterObj = JPDA.FilterObjList[1]
        mean2 = FilterObj.State[0,0]
        std2  = np.sqrt(FilterObj.Cov[0,0])
        Lib.PlotGaussian1D(ax,mean2,std2,minRange,maxRange,'m')
        
        print("### JPDA ###")
        print("Mean1 = ",mean1)
        print("Std1  = ",std1)
        print("Mean2 = ",mean2)
        print("Std2  = ",std2)
'''       
        