# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:50:09 2022

@author: Kotb
"""
import numpy as np
import matplotlib.patches as patches
import scipy.stats as stats
import math
#################################################################################
################################# Plots #########################################
#################################################################################
def Create_ellipse(Pos,Cov,color):
    
    vals, vecs = np.linalg.eigh(Cov)
    order = vals.argsort()[::-1]
    vals=vals[order]
    vecs=vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
    return patches.Ellipse(Pos, 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1]),theta,fill=False, color=color)

def PlotEllipse(ax, mean, cov, color):
    
    ax.add_patch(Create_ellipse(mean, cov, color))
    
def PlotPoseWithEllipse(ax, pose, cov_pose, color, label=None):
    
    PlotEllipse(ax, pose, cov_pose[0:2, 0:2], color)
    return ax.plot(pose[0], pose[1], color + 'o', label=label)

def PlotGaussian1D(ax,mean,std,start,end,color = 'g'):
    delta = abs(start - end )
    X = np.linspace(start,end,1000)
    Y = []
    
    for x in X:
        Y.append(stats.norm.pdf(x,mean,std))
        
    ax.plot(X,Y,color)
    
def PlotGaussianMixture1D(ax,weights,means,stds,start,end):
    delta = abs(start - end )
    X = np.linspace(start,end,1000)
    Y = []
    
    for x in X:
        y = 0.0
        for i in range(len(weights)):
            y = y + (stats.norm.pdf(x,means[i],stds[i]) * weights[i])
        Y.append(y)
    
    #maxY = max(Y)
    #Y = [y/maxY for y in Y]
    
    ax.plot(X,Y,'k')

def PlotAssocs(ax,AssocList):
    for Assoc in AssocList:
        point1 = Assoc[0]
        point2 = Assoc[1]
        x_values = [point1[0,0], point2[0,0]]
        y_values = [point1[1,0], point2[1,0]]
        ax.plot(x_values, y_values, 'r')
    
#################################################################################
##################################### Math ######################################
#################################################################################
def normal_dist(x , mean , sd):
    #prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    prob_density =  (1.0 / (sd * math.sqrt(2*math.pi))) * math.exp(-0.5*((x - mean) / sd) ** 2)
    return prob_density


def linear_dist(x, slope, intercept):
    if(x <= (-intercept/slope)):
        return 0
    y = (slope * x) + intercept
    return y

def CalcNormaDistIntegral(Range,mean,std):
    sum_val = 0
    vals    = []
    for i in Range:
        val = normal_dist(i , mean , std)
        sum_val += val
        vals.append(val)
    
    return sum_val,vals

def CalcLinearDistIntegral(Range,slope,intercept):
    sum_val = 0
    vals    = []
    for i in Range:
        val = linear_dist(i , slope , intercept)
        sum_val += val
        vals.append(val)
    
    return sum_val,vals

def CalcSqauredDist(x1,x2):
    deltaX = x1 - x2
    
    deltaXsq = deltaX * deltaX
    
    return deltaXsq