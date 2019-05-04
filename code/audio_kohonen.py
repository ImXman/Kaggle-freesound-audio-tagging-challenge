# -*- coding: utf-8 -*-
"""
Created on Tue. April  30 16:48:30 2019
Audio tagging challenge Kohonen Map clustering
@author: Quan Zhou

"""

import sys
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA as PCA
from matplotlib.gridspec import GridSpec

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

#%% PCA reduced data
# X after PCA dimensionality reduction
PCAobject = PCA().fit(X)
k_components = next(x[0] for x in enumerate(np.cumsum(PCAobject.explained_variance_ratio_)) if x[1] > 0.90) + 1
PCAobject = PCA(n_components=k_components).fit(X)
pX = PCAobject.transform(X)
#%% SOM initialization and training
print('training...')
n_feature = X.shape[1]
# 1x41 node for SOM structure, 128 input nodes are same as input features
som = MiniSom(20, 20, n_feature, sigma=1,
              learning_rate=0.5, neighborhood_function='gaussian', random_seed=10)  # 20x20 = 400 network nodes
som.pca_weights_init(X)
som.train_batch(X, 5000, verbose=True)  # random training
plt.figure(figsize=(20, 20))
# Plotting the response for each pattern in the training dataset
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
#plt.colorbar()

# plot the Activations frequencies
plt.figure(figsize=(8, 7))
frequencies = np.zeros((20, 20))
for position, values in som.win_map(X).items():
    frequencies[position[0], position[1]] = len(values)
plt.pcolor(frequencies, cmap='Blues')
plt.colorbar()
plt.show()

# plot the class pies
label = np.array(data_verif['label'])
labels_map = som.labels_map(X, label)
label_names = np.unique(label)

plt.figure(figsize=(20, 20))
the_grid = GridSpec(20, 20)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names]
    plt.subplot(the_grid[6-position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)
plt.legend(patches, label_names, bbox_to_anchor=(0, 3), ncol=3)
plt.savefig('som_audio_pies.png')
plt.show()

#%% compute the quantization error
som = MiniSom(20, 20, n_feature, sigma=1,
              learning_rate=0.5, neighborhood_function='gaussian', random_seed=10)

som.pca_weights_init(X)
max_iter = 10000
q_error_pca_init = []

iter_x = []
for i in range(max_iter):
    percent = 100*(i+1)/max_iter
    rand_i = np.random.randint(len(X))
    som.update(X[rand_i], som.winner(X[rand_i]), i, max_iter)
    if (i+1) % 100 == 0:
        error = som.quantization_error(X)
        q_error_pca_init.append(error)
        iter_x.append(i)
        sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')
        
plt.plot(iter_x, q_error_pca_init)
plt.ylabel('quantization error')
plt.xlabel('iteration index')
#%%
#som.random_weights_init(X)
#starting_weights = som.get_weights().copy()  # saving the starting weights
#som.train_random(X, 1000, verbose=True) # maximum iteration = 1000
#
#qnt = som.quantization(X)  # quantize each element of the train dataset
#wm = som.labels_map(X, y)
#weights = som._weights # get the som net weights
#winner = som.winner(y)


"""reference: https://github.com/JustGlowing/minisom/blob/master/examples/Iris.ipynb"""