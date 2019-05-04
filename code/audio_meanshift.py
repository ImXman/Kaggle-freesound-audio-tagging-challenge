# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:10:53 2019
audio challenge MeanShift clustering
@author: Quan Zhou
"""

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd

#%% Import data from .csv file
features = pd.read_csv('../data/audio_embedding_10s.csv', index_col=0)
labels = pd.read_csv('../data/train.csv', index_col=0)
labels.index=labels.index.str.replace('.wav', '', regex=True) # remove audio file postfix
## merge training data with labels
data = (features.transpose()).join(labels, how="left")
# Train data sample index of manually verified ones
verified_idx = np.array(data[data["manually_verified"] == 1].index)
#get the verified data
data_veri = data.loc[verified_idx]
data_verif = data_veri.drop(columns=['manually_verified'])

X = data_verif.drop(columns=['label']).values
y, uniques = pd.factorize(data_verif["label"])

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

#%% #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=7)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()