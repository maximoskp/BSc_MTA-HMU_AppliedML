#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:00:52 2023

@author: max
"""

import os
import pandas as pd
import librosa
import sys
import numpy as np
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %%

with open('data/tiny_SOL_all.pickle', 'rb') as f:
    tiny_SOL_all = pickle.load( f )

with open('data/tiny_SOL_cqt_all.pickle', 'rb') as f:
    tiny_SOL_cqt_all = pickle.load( f )

# %% test/view results

import matplotlib.pyplot as plt

# %%
test_category = 'Brass'
sample_number = 12

c = tiny_SOL_cqt_all[test_category][sample_number]
x = tiny_SOL_all[test_category][sample_number]

print(x.shape)
plt.imshow( c, cmap='gray_r', origin='lower' )

# %% test inverse

import sounddevice as sd

# %%

s = librosa.griffinlim_cqt( c, sr=8000, bins_per_octave=12*3, hop_length=256 )

# %% 

print('original')
sd.play(x/np.min(x), 8000)
# %%

print('griffin lim')
sd.play(s/np.min(s), 8000)
