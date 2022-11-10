# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:05:48 2022

@author: Kotb
"""

import numpy as np
from MOT_FilterLib import Kalman
import MOT_Lib
import Lib

class SingleObjectTracker:
    def __init__(self,FilterObj,clutterObj,P_gating,meas_dim,W_min,mergingThres,MaxHyp):
        self.FilterObj  = FilterObj
        self.clutterObj = clutterObj
        self.P_gating   = P_gating
        self.meas_dim   = meas_dim
        self.W_min      = W_min
        self.mergeThres = mergingThres
        self.MaxHyp     = MaxHyp
        
        self.ConceptalHyps = [FilterObj]
        self.ConceptalWs   = [0.0]
        
    def ConceptualFilter(self,A_mat, Q_mat, Z_list, H_mat, R_mat):        
        NumMeas = len(Z_list)
                
        PrevHyps = self.ConceptalHyps.copy()
        self.ConceptalHyps = []        
        NumHyp   = len(PrevHyps)
        
        PrevWs   = self.ConceptalWs.copy()
        self.ConceptalWs = []
        
        # Weight of No Detection
        w_theta0 = np.log(1.0 - self.clutterObj.P_D)
        # Weight of Detection
        w_thetaM = np.log(self.clutterObj.P_D / self.clutterObj.intensity)
        
        # Update All Hyposeses with the new mesaurments
        # Loop over Hyposeses
        #print("##### Concept #####")

        for theta in range(NumHyp):
            Hyp_theta = PrevHyps[theta]
            W_theta   = PrevWs[theta]
            
            hyp0 = Hyp_theta
            w0   = W_theta + w_theta0
                        
            # Loop over Measaurments
            for z_id in range(NumMeas):
                
                currentHyp = Kalman(Hyp_theta.State,Hyp_theta.Cov)
                #print("HypPredictState = ",currentHyp.State[0,0]," HypPredictCov = ",currentHyp.Cov[0,0])
                predictedLikelihood = currentHyp.PredictedLikelihood(Z_list[z_id], H_mat, R_mat)
                
                currentW = W_theta + w_thetaM + predictedLikelihood
                
                currentHyp.Update(Z_list[z_id],H_mat,R_mat)
                
                self.ConceptalHyps.append(currentHyp)
                self.ConceptalWs.append(currentW)
                #print("Meas = ",Z_list[z_id]," HypState = ",currentHyp.State[0,0]," HypCov = ",currentHyp.Cov[0,0]," HypW = ",np.exp(currentW))

            self.ConceptalHyps.append(hyp0)
            self.ConceptalWs.append(w0)
        # Predict for Next Cycle
        for Hyp in self.ConceptalHyps:
           Hyp.Predict(A_mat,Q_mat)


    def NearestNeighbourFilter(self,A_mat, Q_mat, Z_list, H_mat, R_mat):    
        # Weight of No Detection
        w_theta0 = np.log(1.0 - self.clutterObj.P_D)
        # Weight of Detection
        w_thetaM = np.log(self.clutterObj.P_D / self.clutterObj.intensity)
        
        weights  = []
        Meas     = Z_list.copy()
        for z in Z_list:
            predictedLikelihood = self.FilterObj.PredictedLikelihood(z, H_mat, R_mat)
            weight              = predictedLikelihood + w_thetaM
            weights.append(weight)
        
        MaxW,MaxMeas = max(zip(weights, Meas))
        #print("##### NN #####")
        if(MaxW > w_theta0):
            self.FilterObj.Update(MaxMeas,H_mat,R_mat)
            #print("Meas = ",MaxMeas," MergedCov = ",self.FilterObj.Cov)
            
        self.FilterObj.Predict(A_mat,Q_mat)


    def ProbDataAssocFilter(self,A_mat, Q_mat, Z_list, H_mat, R_mat):    
        # Weight of No Detection
        w_theta0 = np.log(1.0 - self.clutterObj.P_D)
        hyp0     = Kalman(self.FilterObj.State,self.FilterObj.Cov)
        # Weight of Detection
        w_thetaM = np.log(self.clutterObj.P_D / self.clutterObj.intensity)
        
        Weights   = [w_theta0]
        Hyps      = [hyp0]
        HypsMean  = [hyp0.State]
        HypsCov   = [hyp0.Cov]
        
        #print("###### PDA #######")
        for z in Z_list:
            currentHyp          = Kalman(self.FilterObj.State,self.FilterObj.Cov)
            
            predictedLikelihood = currentHyp.PredictedLikelihood(z, H_mat, R_mat)
            
            weight              = predictedLikelihood + w_thetaM
            
            currentHyp.Update(z,H_mat,R_mat)
            
            
            Weights.append(weight)            
            Hyps.append(currentHyp)
            HypsMean.append(currentHyp.State)
            HypsCov.append(currentHyp.Cov)
            
            #print("Meas = ",z," ObjCov = ",currentHyp.Cov)
        WeightsNorm = MOT_Lib.normalizeLogWeights(Weights)

        self.FilterObj.MomentMatching(WeightsNorm,HypsMean,HypsCov)
        #WeightExp = np.exp(WeightsNorm)
        #print("Weights  = ",WeightExp)
        #print("HypCovs  = ",HypsCov)
        #print("MergedCov= ", self.FilterObj.Cov)

        
        self.FilterObj.Predict(A_mat,Q_mat)
        #print("##############")


###############################################################################
############################# Unit Tests ######################################
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    InitialState = np.matrix([0.5]).T
    InitialCov   = np.matrix([0.2])
    
    A_mat        = np.matrix([1.0])
    H_mat        = np.matrix([1.0])
    
    R_mat        = np.matrix([0.2])
    Q_mat        = np.matrix([0.35])
    
    P_D          = 0.9
    lambda_c     = 0.4
    range_c      = np.matrix([-4,4])
    
    
    KF1       = Kalman(InitialState,InitialCov)   
    KF2       = Kalman(InitialState,InitialCov) 
    KF3       = Kalman(InitialState,InitialCov) 
    Clutter  = MOT_Lib.UniClutter(P_D,lambda_c,range_c)
    
    SOT_Concept = SingleObjectTracker(KF1,Clutter,0,0,0,0,0)
    SOT_NN      = SingleObjectTracker(KF2,Clutter,0,0,0,0,0)
    SOT_PDA     = SingleObjectTracker(KF3,Clutter,0,0,0,0,0)
    
    Z_List = [[-1.3,1.7],
              [1.3],
              [-0.3,2.3],
              [-2.0,3.0],
              [2.6],
              [-3.5,2.8]]
    
    Z_ListH = [[-1.3,1.7],
              [1.3],
              [-0.3,2.3],
              [-0.7,3.0],
              [-1.0],
              [-1.3]]
    
    for i in range(4):
        print("Step = ",i)
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Step ' + str(i))
        
        # Run
        z = Z_List[i]
        SOT_Concept.ConceptualFilter(A_mat, Q_mat, z, H_mat, R_mat)
        SOT_NN.NearestNeighbourFilter(A_mat, Q_mat, z, H_mat, R_mat)
        SOT_PDA.ProbDataAssocFilter(A_mat, Q_mat, z, H_mat, R_mat)
        ################################# Generic Plot ########################
        for Point in z:
            ax.plot(Point,0,'r*')
        ################################# NN Filter Plot ######################
        mean = SOT_NN.FilterObj.State[0,0]
        std  = np.sqrt(SOT_NN.FilterObj.Cov[0,0])
        Lib.PlotGaussian1D(ax,mean,std,-4,4,'g')
        ############################## PDA Filter Plot ########################
        mean = SOT_PDA.FilterObj.State[0,0]
        std  = np.sqrt(SOT_PDA.FilterObj.Cov[0,0])
        Lib.PlotGaussian1D(ax,mean,std,-4,4,'m')
        ########################### Conceptual Filter Plot ####################
        means = []
        stds  = []
        weights = []
        
        WeightsNorm = MOT_Lib.normalizeLogWeights(SOT_Concept.ConceptalWs)

        for i in range(len(WeightsNorm)):
            Point = SOT_Concept.ConceptalHyps[i].State[0,0]
            Var   = SOT_Concept.ConceptalHyps[i].Cov[0,0]
            W     = np.exp(WeightsNorm[i])
            
            means.append(Point)
            stds.append(np.sqrt(Var))
            weights.append(W)
            
            '''
            # Print
            print("#####")
            print("X   = ",Point)
            print("Var = ",Var)
            print("W   = ",W)
            print("#####")
            
            # Plot Points Only
            if(i == 0):
                ax.plot(Point,0,'ro')
            else:
                ax.plot(Point,0,'r*')
            '''
        # Plot Mixed Distribution
        Lib.PlotGaussianMixture1D(ax,weights,means,stds,-4.0,4.0)