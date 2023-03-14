# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:23:43 2021

@author: user
"""

import sys
# import os
import audio_file_representation as afr
import pickle

# get dataset from:
# https://imisathena-my.sharepoint.com/:f:/g/personal/maximos_athenarc_gr/EnFYTi0o5bJMjYPo1IPbJkMB8w55AS43mBQjMuObjitF2Q?e=mFFkk8

file_path = '/Users/max/Documents/datasets/speech_sentiment_part/Actor_02/03-01-01-01-01-01-02.wav'

# %% load object

test_obj = afr.AudioFileRepresentation( file_path , keep_audio=True ) # actually, we're calling __init__

# %% check size
print( 'size of object: ' + str( sys.getsizeof(test_obj) ) )
print( 'more accurate size of object: ' + str(len(pickle.dumps(test_obj, -1))) )