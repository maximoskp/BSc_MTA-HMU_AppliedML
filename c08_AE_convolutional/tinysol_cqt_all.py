# -*- coding: utf-8 -*-

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


# %%

# prepare data
tiny_SOL_cqt_all = {}
for k in list(tiny_SOL_all.keys()):
    tiny_SOL_cqt_all[k] = []

# %%

for k in tiny_SOL_all.keys():
    print('len( tiny_SOL_all[' + k + '] ): ', len( tiny_SOL_all[k] ))
    x_cqt = []
    for i,s in enumerate(tiny_SOL_all[k]):
        print('\r' + str(i), end='')
        tmp_cqt = librosa.cqt( y=s , sr=8000, bins_per_octave=12*3, hop_length=256, n_bins=3*64 )
        x_cqt.append( np.abs( tmp_cqt ) )
    print(' ')
    tiny_SOL_cqt_all[k] = x_cqt

with open('data/' + os.sep + 'tiny_SOL_cqt_all.pickle', 'wb') as handle:
    pickle.dump(tiny_SOL_cqt_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% test/view results

import matplotlib.pyplot as plt

# %%
x = tiny_SOL_cqt_all['Winds'][2]

print(x.shape)
plt.imshow( x, cmap='gray_r', origin='lower' )

# %% test inverse

import sounddevice as sd

# %%

s = librosa.griffinlim_cqt( x, sr=8000, bins_per_octave=12*3, hop_length=256 )

# %% 

sd.play(s/np.min(s), 8000)