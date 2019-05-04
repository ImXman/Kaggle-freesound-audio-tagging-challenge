# -*- coding: utf-8 -*-
"""
Created on Tue. April  30 11:44:48 2019
Audio tagging challenge kMeans clustering
@author: Quan Zhou

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics

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

# X after PCA dimensionality reduction
PCAobject = PCA().fit(X)
k_components = next(x[0] for x in enumerate(np.cumsum(PCAobject.explained_variance_ratio_)) if x[1] > 0.90) + 1
PCAobject = PCA(n_components=k_components).fit(X)
pX = PCAobject.transform(X)

# plot the original dataset using first two features
plt.scatter(X[:, 0], X[:, 1], s=1)
# plot the PCA-reduced data with first two principal components
plt.scatter(pX[:, 0], pX[:, 1], s=0.5)

n_class = len(np.unique(y))
kmeans = KMeans(n_clusters=n_class)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
# plot the clustered 
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=0.1, cmap='rainbow')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=10, alpha=0.4);

#accuracy = accuracy_score(y_kmeans,y)
#confusion_matrix = get_confusion_matrix(y_kmeans,y)
#print('kmeans accuracy:',accuracy)
#print('kmeans confusion matrix:',confusion_matrix)
