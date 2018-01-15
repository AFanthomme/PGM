#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:22:54 2018

@author: thomasbazeille
"""

import dynamical_systems
import numpy as np
import sklearn
from sklearn import decomposition
from initialization_methods import factor_analysis, lds

examptle_2 = dynamical_system(n_hidden=2, n_inputs=0, n_outputs=3, flow_function=sin_flow, driving_function=no_drive(1),
                             output_function=dummy_obs)

observed, traj_hidden = example_1.generate_trajectory()

print("True trajectory".format(traj_hidden))

n_comp= 2
components, scores, noise = factor_analysis(observed,n_comp)

hidden_estimate = observed.dot(np.transpose(components))
#print(components)