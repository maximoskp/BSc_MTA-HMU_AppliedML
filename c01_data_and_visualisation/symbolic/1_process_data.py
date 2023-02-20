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

with open('data/wtc1_structs.pickle', 'rb') as handle:
    wtc1_structs = pickle.load(handle)

# prepare array of pcps

pcps = []
tonalities = []
titles = []
for p in wtc1_structs:
    print(p)
    pcps.append( p.pcp )
    tonalities.append( str(p.estimated_tonality) )
    titles.append( str(p.title) )

pcpsnp = np.array(pcps)

# save pickle

with open('data/' + os.sep + 'pcpsnp.pickle', 'wb') as handle:
    pickle.dump(pcpsnp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'tonalities.pickle', 'wb') as handle:
    pickle.dump(tonalities, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'titles.pickle', 'wb') as handle:
    pickle.dump(titles, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print( 'size of data: ' + str(len(pickle.dumps(pcpsnp, -1))) )