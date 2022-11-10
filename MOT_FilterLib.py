# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:40:56 2022

@author: Kotb
"""
import scipy.stats 
import numpy as np
###############################################################################
############################# Kalman Filter ###################################
###############################################################################
class Kalman:
    def __init__(self,state,cov):
        self.State = state
        self.Cov   = cov
        #self.predictedState = state
        #self.predictedCov   = cov
        self.predictedLikelihood = []
        self.S = []
        self.K = []
        self.V = []
        self.KV = []
        
    def Predict(self,A_mat,Q_mat):
        self.State = ( A_mat * self.State )
        self.Cov   = A_mat * self.Cov * A_mat.T +  Q_mat
        
    def Update(self,z ,H_mat, R_mat):
        self.V = z - ( H_mat * self.State )
        self.S = (H_mat * self.Cov * H_mat.T) + R_mat
        #Ensure S Mat is Positive Definite
        #self.S = ( self.S + self.S.T ) / 2.0
        self.K = self.Cov * H_mat.T * self.S.I
        
        self.KV = self.K * self.V
        
        self.State = self.State + self.KV
        self.Cov   = self.Cov - (self.K * self.S * self.K.T)
    
    """
    Predicted Likelihood in Logarithmic scale
    """
    def PredictedLikelihood(self,z, H_mat, R_mat):
        zBar =  H_mat * self.State
        S_mat = (H_mat * self.Cov * H_mat.T) + R_mat
        #Ensure S Mat is Positive Definite
        #S_mat = ( S_mat + S_mat.T ) / 2.0
        
        self.predictedLikelihood = scipy.stats.multivariate_normal.logpdf(z,zBar,S_mat)
        
        return self.predictedLikelihood
   
    """
    Mahalanobis Distance Gating
    """
    def EllipsoidalGating(self,z, H_mat, R_mat, gatingSize):
        zBar =  H_mat * self.State
        S_mat = (H_mat * self.Cov * H_mat.T) + R_mat
        #Ensure S Mat is Positive Definite
        #S_mat = ( S_mat + S_mat.T ) / 2.0
        
        deltaZ = z - zBar
        
        MahalDist = deltaZ * S_mat.I * deltaZ
        
        if(MahalDist < gatingSize):
            return True
        else:
            return False
        
    def MomentMatching(self,w, states, covs):
        retState = 0.0
        retCov   = 0.0
        
        num = len(w)
        
        if num == 1:
            retState = states[0]
            retCov   = covs[0]
            self.State = retState
            self.Cov   = retCov
            return retState,retCov
    
        # Convert weights to exponentioal domain
        weights = np.exp(w)
        #weights = w
        
        # Calculate accumlated Mean and Cov
        if(type(states[0]) == float):
            meanSum = 0
            covSum  = 0
        else:
            meanSum = np.matrix(np.zeros(len(states[0])))
            covSum  = np.zeros((len(states[0]),len(states[0][0])))
        
        for i in range(num):
            meanSum = meanSum + ( weights[i] * states[i] )
    
        for i in range(num):
            d = meanSum - states[i]
            
            if(type(d) == float):
                covSum  = covSum  + ( weights[i] * covs[i] ) + ( ( d * d ) * weights[i] )
            else:
                covSum  = covSum  + ( weights[i] * covs[i] ) + ( ( d * d.T ) * weights[i] )
    
        self.State = meanSum
        self.Cov   = covSum
        return meanSum,covSum
