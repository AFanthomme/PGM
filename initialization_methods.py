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
  return fa.components_, fa.score, fa.noise_variance_

"""Input : observed variables  (y)
           estimated hidden states (x)
           hidden states covariance matrix (Q)
           pi1 
   Output: components array
           scores of each component
           the matrix of noise variance"""
def lds(observed, hidden, noise_variance):

    # Initialization
    # Calculate R : covariance matrix of noise in output R should be diagonal
    # Calculate C : |yt=Cxt + vt
    # A recover xt+1 = Axt+wt et pi1???
    T = len(y)
    y = observed
    x = hidden
    Q = noise_variance
    #C = #calculate
    #R = #calculate
    tol = 0.01
    delta = 1
    nextx[1] = pi1
    
    while delta>tol:
      # Expectation part,
      
      #forwardrecursion
      #xnextnext = x(t|y:1...t) xnext = x(t|y:1...t-1)
      #Vnextnext = x(t|y:1...t) xnext = x(t|y:1...t-1)
      
      for t in range(2,T):
          nextx[t] = np.dot(A,x[t-1])
          nextV[t] = A.dot(V[t-1]).dot(np.transpose(A)) + Q
          K[t] = nextV[t].dot(np.transpose(C)).dot(linalg.inv(C.dot(nextV[t]).dot(np.transpose(C))+R))
          x[t] = nextx[t] + K[t].dot(y[t]-C.dot(nextx[t]))
          V[t] = nextV[t] - K[t].dot(C).dot(nextV[t])
      
      Vlink[T] = (np.identity(3)-K[T].dot(C)).dot(A).dot(V[T-1])
      
      #backwardrecursion
      # nextP[t] = P(t|y:1...t-1) 
      # xT = x(t|y:1...T)
      # Vlink
          
          
      for t in range(T,1): #
          
          J[t-1] = V[t-1].dot(np.transpose(A)).dot(linalg.inv(nextV[T-1]))
          xT[t-1] = x[t-1]+J[t-1].dot(xT[t]-nextx[t])
          VT[t-1] = V[t-1]+J[t-1].dot(VT[t]-nextV[t]).dot(np.transpose(J[t-1]))
          P[t] = VT[t]+xT[t].dot(np.transpose(xT[t]))
          Vlink[t-1] = V[t-1].dot(np.transpose(J[t-2]))+J[t-1].dot(Vlink[t]-A.dot(V[t-1])).dot(np.transpose(J[t-2]))
          Plink[t] = Vlink[t]+xT[t].dot(np.transpose(xT[t-1]))
          
          
   # Maximisation Part 
      Cnew = 0
      Ptsum = 0
      for t in range(1,T):
          Psum += P[t]
          Expsum += y[t].dot(xT[t])
      Cnew = Othsum.dot(linalg.inv(Ptsum))
      C = Cnew
      Rnew = 0
      for t in range(1,T):
          Rnew += 1/T*y[t].dot(np.transpose(y[t]))-Cnew.dot(xT[t]).dot(np.transpose(y[t]))
      R = Rnew
      Plinksum = 0
      Psum = 0
      for t in range(2,T):
          Plinksum += Plink[t]
          Psum += P[t]
      
      Anew = Plinksum.dot(np.transpose(Psum))
      A = Anew
      Qnew = 1/(T-1)*(Psum-Anew.dot(Plinksum))
      
      pi1new = xT[1]
      V1new = P[1]-xT[1].dot(np.transpose(xT[1]))
      V[1]= V1new
      x[1] = pi1new
      delta = Qnew-Q
      Q = Qnew