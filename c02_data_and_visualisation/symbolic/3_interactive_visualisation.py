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

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% 

with open('data/pcpsnp.pickle', 'rb') as handle:
    pcpsnp = pickle.load(handle)

with open('data/titles.pickle', 'rb') as handle:
    titles = pickle.load(handle)

with open('data/tonalities.pickle', 'rb') as handle:
    tonalities = pickle.load(handle)

X_embedded = drm.reduce_dimensions(pcpsnp, method='TSNE')

fig,ax = plt.subplots()
sc = plt.scatter( X_embedded[:,0], X_embedded[:,1], alpha=1, s=13 )

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    # text = 'lala' + str(ind)
    idxs = ind["ind"]
    text = titles[idxs[0]] + '\n' + tonalities[idxs[0]]
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