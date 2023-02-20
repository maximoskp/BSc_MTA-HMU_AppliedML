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

with open('data/featuresnp.pickle', 'rb') as handle:
    featuresnp = pickle.load(handle)

with open('data/names.pickle', 'rb') as handle:
    names = pickle.load(handle)

with open('data/categories.pickle', 'rb') as handle:
    categories = pickle.load(handle)

# %% PCA

X_PCA = drm.reduce_dimensions(featuresnp, method='PCA')
with open('data/' + os.sep + 'X_PCA.pickle', 'wb') as handle:
    pickle.dump(X_PCA, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% MDS

X_MDS = drm.reduce_dimensions(featuresnp, method='MDS')
with open('data/' + os.sep + 'X_MDS.pickle', 'wb') as handle:
    pickle.dump(X_MDS, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% TSNE

X_TSNE = drm.reduce_dimensions(featuresnp, method='TSNE')
with open('data/' + os.sep + 'X_TSNE.pickle', 'wb') as handle:
    pickle.dump(X_TSNE, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% 

plt.clf()
plt.scatter( X_TSNE[:,0], X_TSNE[:,1], alpha=0.5, s=9 )
# for i,lab in enumerate(names):
#     plt.text( X_embedded[i,0], X_embedded[i,1], lab )