#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:38:54 2022

@author: max
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% load data

with open('data/featuresnp.pickle', 'rb') as handle:
    featuresnp = pickle.load(handle)

with open('data/valencesnp.pickle', 'rb') as handle:
    valencesnp = pickle.load(handle)

# %% keep only feature at index 14
featuresnp = np.c_[ featuresnp[:,:12] , featuresnp[:,14] ]

# %% visualise arousal values

# specifically, plot a sorted version to see overall trends
plt.clf()
plt.plot( np.sort( valencesnp ) )

# %% spit data to training and test set

# formulate a "functional" classification just for splitting the data
fake_classes = np.zeros( valencesnp.shape )
fake_classes[ valencesnp == -1 ] = 1
fake_classes[ valencesnp == 1 ] = 2

# visualse result
plt.clf()
sorted_idxs = np.argsort( valencesnp )
plt.plot( np.sort( valencesnp ) )
plt.plot( fake_classes[sorted_idxs] )

# %% stratified split
from sklearn.model_selection import StratifiedShuffleSplit

stratsplit = StratifiedShuffleSplit( n_splits = 1, test_size = 0.2, random_state = 1)
idxs_generator = stratsplit.split( featuresnp, fake_classes )
idxs_list = list (idxs_generator)
train_idxs = idxs_list[0][0]
test_idxs = idxs_list[0][1]

# check result
plt.clf()
plt.subplot(2,1,1)
plt.plot( np.sort( valencesnp[train_idxs] ) )
plt.subplot(2,1,2)
plt.plot( np.sort( valencesnp[test_idxs] ) )

# %% check inputs

plt.clf()
plt.boxplot( featuresnp )

# %% scale inputs

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False, with_std=True)
scaled_features = scaler.fit_transform( featuresnp )

plt.clf()
plt.subplot(2,2,1)
plt.boxplot( featuresnp )
plt.subplot(2,2,2)
plt.boxplot( scaled_features )
plt.subplot(2,2,3)
plt.boxplot( featuresnp[:,:12] )
plt.subplot(2,2,4)
plt.boxplot( scaled_features[:,:12] )

train_input = scaled_features[ train_idxs , : ]
train_output = valencesnp[ train_idxs ]

test_input = scaled_features[ test_idxs , : ]
test_output = valencesnp[ test_idxs ]

# %% linear regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

lin_reg = LinearRegression()
lin_reg.fit( train_input , train_output )
# make predictions from training data
preds = lin_reg.predict( test_input )
rmse = mean_absolute_error( test_output , preds )

# %% plot results

plt.clf()
sorted_idxs = np.argsort( test_output )
plt.plot( np.sort( test_output ) )
plt.plot( preds[sorted_idxs] )
plt.title('RMSE: ' + str(rmse))

# %% feature importance

plt.clf()
plt.bar(np.arange(lin_reg.coef_.size), lin_reg.coef_)
plt.savefig( 'figs/valence_lin_regression_coef.png' , dpi=300 )

# %% redundancy through correlation

feat_corrs = np.corrcoef(featuresnp.T)
plt.imshow( feat_corrs, cmap='gray_r' )
plt.colorbar()

