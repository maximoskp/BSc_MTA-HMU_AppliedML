#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:57:41 2022

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
    
with open('data/energeticflagsnp.pickle', 'rb') as handle:
    energeticflagsnp = pickle.load(handle)

# %% stratified split
from sklearn.model_selection import StratifiedShuffleSplit

stratsplit = StratifiedShuffleSplit( n_splits = 1, test_size = 0.2, random_state = 1)
idxs_generator = stratsplit.split( featuresnp, energeticflagsnp )
idxs_list = list (idxs_generator)
train_idxs = idxs_list[0][0]
test_idxs = idxs_list[0][1]

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
train_output = energeticflagsnp[ train_idxs ]

test_input = scaled_features[ test_idxs , : ]
test_output = energeticflagsnp[ test_idxs ]

# %% linear regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

lin_reg = LinearRegression()
lin_reg.fit( train_input , train_output )
# make predictions from training data
preds = lin_reg.predict( test_input )
preds_binary = np.array( preds >= 0.5 ).astype(int)
comparison_check = np.c_[ preds , preds_binary , test_output ]
accuracy_linear = np.sum( test_output == preds_binary ) / preds.size

# %% random forest

from sklearn.ensemble import RandomForestClassifier

forest_class = RandomForestClassifier()
forest_class.fit( train_input , train_output )
# make predictions from training data
preds_binary = forest_class.predict( test_input )
comparison_check = np.c_[ preds_binary , test_output ]
accuracy_forest = np.sum( test_output == preds_binary ) / preds.size

# %% SVM

from sklearn.svm import SVC

svm_class = SVC()
svm_class.fit( train_input , train_output )
# make predictions from training data
preds_binary = svm_class.predict( test_input )
comparison_check = np.c_[ preds_binary , test_output ]
accuracy_svm = np.sum( test_output == preds_binary ) / preds.size

# %% logistic regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit( train_input , train_output )
# make predictions from training data
preds = log_reg.predict( test_input )
preds_binary = np.array( preds >= 0.5 ).astype(int)
comparison_check = np.c_[ preds , preds_binary , test_output ]
accuracy_linear = np.sum( test_output == preds_binary ) / preds.size

# %% cross validation - custom accuracy metric

from sklearn.metrics import make_scorer

def binary_accuracy( y_true , y_pred ):
    bin_pred = np.array( y_pred >= 0.5 ).astype(int)
    return np.sum( y_true == bin_pred ) / y_true.size

my_scorer = make_scorer(binary_accuracy, greater_is_better=True)

# %% stratified for cross validation

strat10split = StratifiedShuffleSplit( n_splits = 10, test_size = 0.2, random_state = 1)

# %% cross validation

from sklearn.model_selection import cross_val_score

scores_lin = cross_val_score( lin_reg, featuresnp, energeticflagsnp,
                         scoring=my_scorer, cv=strat10split )

scores_forest = cross_val_score( forest_class, featuresnp, energeticflagsnp.ravel(),
                         scoring=my_scorer, cv=strat10split )

scores_svm = cross_val_score( svm_class, featuresnp, energeticflagsnp.ravel(),
                         scoring=my_scorer, cv=strat10split )

scores_log = cross_val_score( log_reg, featuresnp, energeticflagsnp.ravel(),
                         scoring=my_scorer, cv=strat10split )

def present_scores( s , algorithm='method' ):
    print(30*'-')
    print( algorithm + ' accuracy in 10-fold stratified split validation:' )
    print('mean: ' + str( np.mean(s) ))
    print('std: ' + str( np.std(s) ))
    print('median: ' + str( np.median(s) ))

present_scores( scores_lin , algorithm='linear regression' )
present_scores( scores_forest , algorithm='random forest' )
present_scores( scores_svm , algorithm='SVM' )
present_scores( scores_log , algorithm='logistic regression' )

