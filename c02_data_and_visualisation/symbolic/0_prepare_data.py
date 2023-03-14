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

from symbolic_data_processing import SymbolicInfo

if not os.path.exists('data'):
    os.makedirs('data')

# %%

folder = '../../data/WTC_I/'
files = os.listdir( folder )

wtc1_structs = []

for f in files:
    print('trying ', f)
    if f.endswith('.mxl'):
        print('processing...')
        wtc1_structs.append( SymbolicInfo(folder + f, metadatafile=folder+'metadata.csv' ) )

with open('data/' + os.sep + 'wtc1_structs.pickle', 'wb') as handle:
    pickle.dump(wtc1_structs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print( 'size of data: ' + str(len(pickle.dumps(wtc1_structs, -1))) )