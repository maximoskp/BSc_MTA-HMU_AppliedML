#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 07:12:13 2022

@author: max
"""

import numpy as np
import os
import pandas as pd

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

from symbolic_data_processing import SymbolicEmotionInfo

if not os.path.exists('data'):
    os.makedirs('data')

# %%

folder = '../../data/EMOPIA_1.0/midis/'
metapath = '../../data/EMOPIA_1.0/metadata_by_song.csv'
files = os.listdir( folder )

emo_structs = []

for i,f in enumerate(files):
    print('trying ', f)
    if f.endswith('.mid'):
        print('processing... ' + str(i) + '/' + str(len(files)))
        emo_structs.append( SymbolicEmotionInfo(folder + f, metadatafile=metapath ) )

with open('data/' + os.sep + 'emo_structs.pickle', 'wb') as handle:
    pickle.dump(emo_structs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print( 'size of data: ' + str(len(pickle.dumps(emo_structs, -1))) )