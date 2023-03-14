#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:08:28 2022

@author: max
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import dimensionality_reduction_methods as drm

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% 

with open('data/pcpsnp.pickle', 'rb') as handle:
    pcpsnp = pickle.load(handle)

with open('data/titles.pickle', 'rb') as handle:
    titles = pickle.load(handle)

with open('data/tonalities.pickle', 'rb') as handle:
    tonalities = pickle.load(handle)

X_embedded = drm.reduce_dimensions(pcpsnp, method='TSNE')

# %% 

plt.clf()
plt.plot( X_embedded[:,0], X_embedded[:,1], '.' )
for i,lab in enumerate(tonalities):
    plt.text( X_embedded[i,0], X_embedded[i,1], lab )