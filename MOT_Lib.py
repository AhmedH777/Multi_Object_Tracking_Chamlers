# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:08:55 2022

@author: Kotb
"""
import numpy as np
import Lib
###############################################################################
############################### Funcs #########################################
###############################################################################
def normalizeLogWeights(log_ws_list):
    exp_ws      = [ np.exp(x) for x in log_ws_list]
    ws_sum      = sum(exp_ws)
    norm_exp_ws = [ x/ws_sum for x in exp_ws]
    ret_log_ws  = [np.log(x) for x in norm_exp_ws]
    
    return ret_log_ws
'''
Approximate a Gaussian mixture density as a single gaussian using moment matching

w      = normalized weights of Gaussian compenents in logarithm domain (vector)
states = List of states (list of vectors)
covs   = List of Covs   (list of matrcies)

Return:
    retState = approximated state (vector)
    retCov   = approximated cov   (matrix)
'''
def MomentMatching(w, states, covs):
    retState = 0.0
    retCov   = 0.0
    
    num = len(w)
    
    if num == 1:
        retState = states[0]
        retCov   = covs[0]
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

    return meanSum,covSum
###############################################################################
############################## Models #########################################
###############################################################################
"""
Uniform Clutter Model

Rate         = Lambda * Volume
SpatialPdf   = 1.0 / Volume    (uniform Clutter Intensity)
"""
class UniClutter:
    "range_c is Range Area for 2D = [[xMin,xMax],[yMin,yMax]]"
    def __init__(self,P_D,lambda_c,range_c):
        self.P_D        = P_D
        self.lambda_c   = lambda_c
        self.range     = range_c
        self.rangeVol  = np.prod(np.diff(self.range.T, axis = 0))
        self.pdf       = 1.0 / self.rangeVol
        self.intensity = self.lambda_c * self.pdf
        
        
class HypothesisReduction:
    '''
    weights     = the weights of different hypotheses in logarithmic scale
    hypotheses  = List of hypotheses structs
    '''
    def __init__(self,weights,hypotheses):
        self.weights    = weights
        self.hypotheses = hypotheses

    '''
    Prunes hypotheses with small weights smaller than threshold
    '''
    def Prune(self,threshold):
        validHyp = []
        validHypWeights = []
        for (w,hyp) in zip(self.weights,self.hypotheses):
            if(w > threshold):
                validHypWeights.append(w)
                validHyp.append(hyp)
        
        return validHypWeights,validHyp
    
    '''
    Keeps M hypotheses with the highest weights and discard the rest
    '''
    def Cap(self, M):
        if (M > len(self.weights)):
            return self.weights,self.hypotheses
        else:
            sortedWs,sortedHyp = zip(*sorted(zip(self.weights, self.hypotheses),reverse = True))
            sortedWs  = list(sortedWs)
            sortedHyp = list(sortedHyp)
            return sortedWs[0:M],sortedHyp[0:M]
###############################################################################
############################# Unit Tests ######################################
###############################################################################
def ClutterTest():
    Result = "Passed"
    P_D = 0.5
    lambda_c = 2.0
    range_c  = np.matrix([[1,2],[3,5]])
    
    obj = UniClutter(P_D,lambda_c,range_c)
    if( ( obj.P_D       != P_D ) or \
        ( obj.pdf       != 0.5 ) or \
        ( obj.intensity != 1.0 ) ):
        Result = "Failed"
        
    print("ClutterTest : " + Result)
    
def MomentMatchingTest():
    import matplotlib.pyplot as plt

    Result = "Failed"
    
    weights = [0.5,0.5]
    means   = [-3.0,3.0]
    stds    = [2,2]
    
    logweights = np.log(weights)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    Lib.PlotGaussianMixture1D(ax,weights,means,stds,-6.0,6.0)
    
    mixedMean,mixedStd = MomentMatching(logweights, means, stds)
    
    Lib.PlotGaussian1D(ax,mixedMean,mixedStd,-6.0,6.0)

    if( ( mixedMean < 0.01 ) and ( ( mixedStd - 11.0 ) < 0.01 ) ):
        Result = "Passed"
        
    print("MomentMatchingTest : " + Result)


def HypothesisReductionTest():
    from random import seed
    from random import random

    seed(1)

    Hyps    = np.linspace(1,100,100)
    Weights = []
    for i in range(100):
        Weights.append(random())
        
    HypObj = HypothesisReduction(Weights,Hyps)
    
    wPrune,hypPrune = HypObj.Prune(0.8)
    wCap,hypCap     = HypObj.Cap(10)
    
    #print(wPrune,hypPrune)
    #print(wCap,hypCap)
################################ Run Tests ####################################

if __name__ == "__main__":
    ClutterTest()
    MomentMatchingTest()
    HypothesisReductionTest()
