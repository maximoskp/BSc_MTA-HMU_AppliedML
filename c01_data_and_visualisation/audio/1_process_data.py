#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:01:02 2022

@author: max
"""

import numpy as np
import os

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% 

with open('data/audio_structs.pickle', 'rb') as handle:
    audio_structs = pickle.load(handle)

# prepare array of pcps

features = []
names = []
categories = []
audios = []
for p in audio_structs:
    print(p)
    features.append( p.features )
    names.append( p.name )
    categories.append( p.category )
    if hasattr(p, 'audio'):
        audios.append(p.audio)

featuresnp = np.array(features)
featuresnp = np.reshape( featuresnp, (featuresnp.shape[0], featuresnp.shape[1]) )
# remove nan values
idx = np.argwhere(np.isnan(featuresnp))
while idx.size > 0:
    featuresnp = np.delete( featuresnp , (idx[0,0]), axis=0 )
    del names[idx[0,0]]
    del categories[idx[0,0]]
    del audios[idx[0,0]]
    idx = np.argwhere(np.isnan(featuresnp))

# save pickle

with open('data/' + os.sep + 'featuresnp.pickle', 'wb') as handle:
    pickle.dump(featuresnp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'names.pickle', 'wb') as handle:
    pickle.dump(names, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'categories.pickle', 'wb') as handle:
    pickle.dump(categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

if len(audios) > 0:
    with open('data/' + os.sep + 'audios.pickle', 'wb') as handle:
        pickle.dump(audios, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print( 'size of data: ' + str(len(pickle.dumps(featuresnp, -1))) )