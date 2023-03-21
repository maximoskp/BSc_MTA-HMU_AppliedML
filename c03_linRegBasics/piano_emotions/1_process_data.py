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

with open('data/emo_structs.pickle', 'rb') as handle:
    emo_structs = pickle.load(handle)

# prepare array of pcps

features = []
valences = []
arousals = []
happyflags = []
energeticflags = []
domiantQs = []
for p in emo_structs:
    print(p)
    features.append( p.features )
    valences.append( p.valence )
    arousals.append( p.arousal )
    happyflags.append( p.isHappy )
    energeticflags.append( p.isEnergetic )
    domiantQs.append( p.dominantQ )

featuresnp = np.array(features)
valencesnp = np.array(valences)
arousalsnp = np.array(arousals)
happyflagsnp = np.array(happyflags).astype(np.int8)
energeticflagsnp = np.array(energeticflags).astype(np.int8)
domiantQsnp = np.array(domiantQs).astype(np.int8)

# save pickle

with open('data/' + os.sep + 'featuresnp.pickle', 'wb') as handle:
    pickle.dump(featuresnp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'valencesnp.pickle', 'wb') as handle:
    pickle.dump(valencesnp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'arousalsnp.pickle', 'wb') as handle:
    pickle.dump(arousalsnp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'happyflagsnp.pickle', 'wb') as handle:
    pickle.dump(happyflagsnp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'energeticflagsnp.pickle', 'wb') as handle:
    pickle.dump(energeticflagsnp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'domiantQsnp.pickle', 'wb') as handle:
    pickle.dump(domiantQsnp, handle, protocol=pickle.HIGHEST_PROTOCOL)
