#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 18:55:57 2018

@author: thomasbazeille


"""

import numpy as np
import sklearn
from sklearn import decomposition
import scipy
from scipy import linalg

"""Input : observed variables 
           if known, number of components to search for
   Output: components array
           scores of each component
           the matrix of noise variance"""
def factor_analysis(observed,n_components=-1):

  # Number of components to compute
    
  fa = decomposition.FactorAnalysis()
    
  # Initialization
  if n_components >0:
      fa.n_components = n_components
  fa.copy = True
  # Compute components and covariance
  fa.fit(observed)
  return fa.components_, fa.score, fa.noise_variance_ ,fa.transform(observed)

"""Input : observed variables  (y)
           estimated hidden states (x)
           hidden states covariance matrix (Q)
           pi1 
   Output: components array
           scores of each component
           the matrix of noise variance"""
def lds(observed, latent, loading_matrix, noise_variance):

    # Initialization
    # Calculate R : covariance matrix of noise in output R should be diagonal
    # Calculate C : |yt=Cxt + vt
    # A recover xt+1 = Axt+wt et pi1???
    y = observed
    T = np.shape(y)[0]
    x = latent
    C = loading_matrix
    R = np.diag(noise_variance)
    Phi = np.diag(1/noise_variance)
    temp1 = Phi.dot(C)
    temp2 = Phi-temp1.dot(linalg.inv(np.identity(x.shape[1])+np.transpose(C).dot(temp1))).dot(np.transpose(temp1))
    temp1 = y.dot(temp2).dot(C)
    Q = np.cov(np.transpose(temp1))
    P0 = Q
    t1 = temp1[0:T-1,:]
    t2 = temp1[1:T,:]
    A = np.linalg.pinv((np.transpose(t1).dot(t1)+Q).dot(np.transpose(t1).dot(t2)))
    
    
    #R = #calculate
    threshold = 0.01
    delta = 1
    #nextx[1] = pi1  !!!
    nextx = np.zeros((x.shape))
    nextx[0] = np.transpose(np.mean(t2))
    nextV = np.zeros((T,Q.shape[0],Q.shape[1]))
    x = np.zeros((x.shape))
    x[0] = np.transpose(np.mean(t1))
    V = np.zeros((T,Q.shape[0],Q.shape[1]))
    K = np.zeros((T,x.shape[1],y.shape[1]))
    VT = np.zeros((T,Q.shape[0],Q.shape[1]))
    xT = np.zeros((x.shape))
    Vlink = np.zeros((T,Q.shape[0],Q.shape[1]))
    P = np.zeros((T,Q.shape[0],Q.shape[1]))
    Plink = np.zeros((T,Q.shape[0],Q.shape[1]))
    J = np.zeros((T,Q.shape[0],Q.shape[1]))
    
    while delta>threshold:
        
      # Expectation part,
      
      #forwardrecursion
      #x[t] = x(t|y:1...t) nextx[t] = x(t|y:1...t-1)
      #V[t] = V(t|y:1...t) nextV[t] = V(t|y:1...t-1)
      
      for t in range(1,T):
          nextx[t] = np.dot(A,x[t-1])
          nextV[t] = A.dot(V[t-1]).dot(np.transpose(A)) + Q
          K[t] = nextV[t].dot(np.transpose(C)).dot(linalg.pinv(C.dot(nextV[t]).dot(np.transpose(C))+R))
          x[t] = nextx[t] + K[t].dot(y[t]-C.dot(nextx[t]))
          V[t] = nextV[t] - K[t].dot(C).dot(nextV[t])
      
      Vlink[T-1] = (np.identity(x.shape[1])-K[T-1].dot(C)).dot(A).dot(V[T-2])
      
      #backwardrecursion
      # nextP[t] = P(t|y:1...t-1) 
      # xT[t] = x(t|y:1...T)
      # VT[t] = V(t|y:1...T)
      # Vlink
      
      for t in range(T-1,0,-1):
        
          J[t-1] = V[t-1].dot(np.transpose(A)).dot(linalg.pinv(nextV[T-1]))
          xT[t-1] = x[t-1]+J[t-1].dot(xT[t]-nextx[t])
          VT[t-1] = V[t-1]+J[t-1].dot(VT[t]-nextV[t]).dot(np.transpose(J[t-1]))
          P[t] = VT[t]+xT[t].dot(np.transpose(xT[t]))
          Vlink[t-1] = V[t-1].dot(np.transpose(J[t-2]))+J[t-1].dot(Vlink[t]-A.dot(V[t-1])).dot(np.transpose(J[t-2]))
          Plink[t] = Vlink[t]+xT[t].dot(np.transpose(xT[t-1]))
          
          
   # Maximisation Part 
      Cnew = np.zeros((C.shape))
      Anew = np.zeros((A.shape))
      Rnew = np.zeros((R.shape))
      Qnew = np.zeros((Q.shape))
      V1new = np.zeros((Q.shape))
      Psum = np.zeros((Q.shape))
      Expsum = np.zeros((C.shape))
      
      for t in range(0,T):
          Psum += P[t]
          Expsum += y[t].reshape(-1,1).dot(np.transpose(xT[t].reshape(-1,1)))
      Cnew = Expsum.dot(linalg.pinv(Psum))
      C = Cnew
      
      for t in range(0,T):
          Rnew += 1/T*y[t].dot(np.transpose(y[t]))-Cnew.dot(xT[t]).dot(np.transpose(y[t]))
      R = Rnew
      
      
      Plinksum = 0
      Psum = 0
      for t in range(1,T):
          Plinksum += Plink[t]
          Psum += P[t]
      
      
      Anew = Plinksum.dot(np.transpose(Psum))
      A = Anew
      Qnew = 1/(T-1)*(Psum-Anew.dot(Plinksum))
      
      pi1new = xT[1]
      V1new = P[1]-xT[1].dot(np.transpose(xT[1]))
      V[1]= V1new
      x[1] = pi1new
      
      delta = np.linalg.norm(Qnew-Q)
      Q = Qnew
    
    return A,C,Qnew,R,xT[1]