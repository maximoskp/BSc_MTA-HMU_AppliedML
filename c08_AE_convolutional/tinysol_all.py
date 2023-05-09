# -*- coding: utf-8 -*-

import os
import pandas as pd
import librosa
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

folder = '../data/TinySOL2020/'
samples = {}
if not os.path.exists('data'):
    os.makedirs('data')

df = pd.read_csv( folder + 'TinySOL_metadata.csv' )

categories = []

# get categories and instruments
for d in os.listdir(folder):
    if os.path.isdir( folder + d ):
        print('category: ', d)
        categories.append( d )

# %%

# load data
tiny_SOL_all = {}
for c in categories:
    tiny_SOL_all[c] = []
    print('extracting: ' + c)
    paths = df[ (df['Family'] == c) ]['Path']
    for p in paths:
        print( p )
        s, _ = librosa.load( folder + p, sr=8000 )
        # keep 1 sec
        tiny_SOL_all[c].append( s[:8000] )

with open('data/' + os.sep + 'tiny_SOL_all.pickle', 'wb') as handle:
    pickle.dump(tiny_SOL_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
