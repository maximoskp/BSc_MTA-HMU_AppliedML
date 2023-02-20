#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:50:06 2022

@author: max
"""

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

def reduce_dimensions(x, n_components=2, method='PCA', logging=True):
    if method == 'PCA':
        method_runner = PCA(n_components=n_components)
    elif method == 'MDS':
        method_runner = MDS(n_components=n_components, verbose=2)
    else:
        method_runner = TSNE(n_components=n_components, learning_rate='auto', init='pca', verbose=2, n_iter=3000)
    X_embedded = method_runner.fit_transform(x)
    if logging:
        # how much of the variance is explained in each dimension
        if method == 'PCA':
            print('explained variance: ', method_runner.explained_variance_ratio_)
        elif method == 'MDS':
            print('stress: ', method_runner.stress_)
    return X_embedded
# end reduce_dimensions