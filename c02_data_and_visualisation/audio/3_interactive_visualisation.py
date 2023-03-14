#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:23:19 2022

@author: max
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:08:28 2022

@author: max
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import dimensionality_reduction_methods as drm
import sounddevice as sd

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% 

with open('data/featuresnp.pickle', 'rb') as handle:
    featuresnp = pickle.load(handle)

with open('data/names.pickle', 'rb') as handle:
    names = pickle.load(handle)

with open('data/categories.pickle', 'rb') as handle:
    categories = pickle.load(handle)

with open('data/audios.pickle', 'rb') as handle:
    audios = pickle.load(handle)

with open('data/X_PCA.pickle', 'rb') as handle:
    X_PCA = pickle.load(handle)

with open('data/X_MDS.pickle', 'rb') as handle:
    X_MDS = pickle.load(handle)

with open('data/X_TSNE.pickle', 'rb') as handle:
    X_TSNE = pickle.load(handle)

# %% 

ucat = np.unique(categories)
catmap = {}
for i in range(len(ucat)):
    catmap[ucat[i]] = i
catnum = []
for c in categories:
    catnum.append( catmap[c] )

# %% 

X_embedded = X_TSNE

fig,ax = plt.subplots()
sc = plt.scatter( X_embedded[:,0], X_embedded[:,1], cmap='hsv', c=catnum, alpha=0.5, s=7 )

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))

idx_prev = -1

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    # text = 'lala' + str(ind)
    idxs = ind["ind"]
    text = categories[idxs[0]]
    global idx_prev
    if idx_prev != idxs[0]:
        sd.stop()
        sd.play(audios[idxs[0]], samplerate=44100)
    idx_prev = idxs[0]
    # text = ''
    # for i in idxs:
    #     text += metadata[metakeys[i]]['all'] + '\n'
    # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
    #                        " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_facecolor('red')
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()