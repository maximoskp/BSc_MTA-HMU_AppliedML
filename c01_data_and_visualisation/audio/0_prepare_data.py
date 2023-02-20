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

from audio_data_processing import AudioInfo

if not os.path.exists('data'):
    os.makedirs('data')

# %%

# CAUTION: need to remove MaxV - Guiro.wav from Roland JV 1080
folder = '../../data/drumMachines200/'
subfolders = os.listdir( folder )

audio_structs = []

for s in subfolders:
    if os.path.isdir( folder + s):
        subfolder = os.listdir( folder + s )
        for f in subfolder:
            print('trying ', f)
            if f.endswith('.wav'):
                print('processing...')
                audio_structs.append( AudioInfo(folder + s + os.sep + f, keep_audio=True ) )

with open('data/' + os.sep + 'audio_structs.pickle', 'wb') as handle:
    pickle.dump(audio_structs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print( 'size of data: ' + str(len(pickle.dumps(audio_structs, -1))) )