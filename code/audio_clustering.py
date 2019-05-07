# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:18:22 2019

@author: Yang Xu and Quan Zhou
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import mutual_info_score,v_measure_score
##
def PCA(data,error=0.1):
    
    ##calculate n dimension means
    pmu = np.mean(data,axis=0)
    pmu = pmu.reshape(1,len(pmu))
    ##scatter matrix
    sca_mat = np.zeros((pmu.shape[1],pmu.shape[1]))
    for i in range(data.shape[0]):
        a = data[i,:].reshape(1,pmu.shape[1])
        sca_mat += (a-pmu)*(a-pmu).T
    ##compute eigenvalues and eigenvectors
    w,v = np.linalg.eig(sca_mat)
    pX = np.dot(v.T,data.T)
    ##sort eigenvector by eigenvalues in decreasing order
    orders =np.argsort(w).tolist()
    orders.reverse()
    pX = pX[orders,:]
    ##drop the dimensions and keep error (cumulative variance) under 0.1
    w=np.sort(w)
    errs={}
    for i in range(w.shape[0]):
        errs[(i+1)]=w[:i+1,].sum()/w.sum()
        if w[:i+1,].sum()/w.sum()>0.1:
            pX = pX[:(w.shape[0]-i),:]
            break
    
    return pX.T

def kohonen_map(img,k=256,l=0.01,epoch=1000,batch=256):
    
    net_size=int(np.sqrt(k))
    m,n = img.shape
    ##initialize the net
    minv = np.min(img)
    maxv = np.max(img)
    net  = np.random.randint(low=minv,high=maxv, size=(net_size,net_size,n))
    net = net.astype(np.float32)
    ##in this project, we use 5X5 gaussian kernal with sigma 1 
    ##to define the influence. Since it's hard coding here, if you want
    ##change influence, you may change those numbers (5,2,1,4) below
    x, y =5,5
    x, y = np.mgrid[-(x//2):(x//2)+1, -(y//2):(y//2)+1]
    g = np.exp(-((x/1)**2+(y/1)**2)/2)
    g = g[2:,2:]
    ##find BMU and update neuron weights
    for i in range(epoch):
        index=np.random.randint(img.shape[0],size=batch)
        samples = img[index,:]
        for j in range(samples.shape[0]):
            ##find BMU
            d=net-samples[j,:]
            d=np.linalg.norm(d, axis=2)  
            bmu = np.where(d==np.min(d))
            bmu = [bmu[0][0],bmu[1][0]]
            ##update neuron weight
            for a in range(net.shape[0]):
                for b in range(net.shape[1]):
                    if abs(a-bmu[0])<=2 and abs(b-bmu[1])<= 2:##4 is also hard coding
                        net[a,b,:]+=l*g[abs(a-bmu[0]),abs(b-bmu[1])]*\
                        (samples[j,:]-net[a,b,:])
    
    return net

##-----------------------------------------------------------------------------
##load data
tr = pd.read_table('Downloads/freesound-audio-tagging/audio_embedding_10s.csv',\
                   sep=',',header=0,index_col=0)

tr = pd.DataFrame.transpose(tr)

labels = pd.read_table('Downloads/freesound-audio-tagging/train.csv',\
                       sep=',',header=0,index_col=0)
labels.index=labels.index.str.replace('.wav', '', regex=True)

##merge features with labels
trm=tr.merge(labels, left_on=tr.index, right_on=labels.index)
trm=trm.iloc[:,1:-1]

x=trm.iloc[:,:-1].values

xpca = PCA(x)

y,uniques =pd.factorize(trm.iloc[:,-1])

x_tr, x_te, y_tr, y_te = train_test_split(xpca, y, test_size=0.25, \
                                          random_state=111)

##-----------------------------------------------------------------------------
##Kmeans
kmeans = KMeans(n_clusters=41)
kmeans.fit(x_tr)

km_pred = kmeans.predict(x_te)
mutual_info_score(y_te, km_pred)
v_measure_score(y_te, km_pred)

km_cg = pd.DataFrame(contingency_matrix(y_te, km_pred))
km_cg = km_cg/km_cg.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(km_cg,cmap=cmap)
cm_heat.get_figure().savefig("knn_cm.jpeg",dpi=2400)

km_preds = np.max(km_cg, axis=1)
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': km_preds}
km_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=km_preds)

##-----------------------------------------------------------------------------
##Konohen map
net=kohonen_map(x_tr,k=49,l=0.01,epoch=1000,batch=256)

##assign organized neuron weights to each sample
##first flatten the net
centroid=[]
for i in range(net.shape[2]):
    centroid.append(net[:,:,i].flatten())
centroid= np.asarray(centroid).T
    
dist=np.empty((x_te.shape[0],0))
for i in range(centroid.shape[0]):
    d=x_te-centroid[i,:]
    d=np.linalg.norm(d, axis=1)
    dist=np.column_stack((dist,d))
cluster=np.argmin(dist,axis=1).tolist()

mutual_info_score(y_te, cluster)
v_measure_score(y_te, cluster)

som_cg = pd.DataFrame(contingency_matrix(y_te, cluster))
som_cg = som_cg/som_cg.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(som_cg,cmap=cmap)
cm_heat.get_figure().savefig("knn_cm.jpeg",dpi=2400)

som_preds = np.max(som_cg, axis=1)
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': som_preds}
som_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=som_preds)

##-----------------------------------------------------------------------------
##mean shift
from sklearn.cluster import MeanShift

ms = MeanShift(bandwidth=0.01, bin_seeding=True)
ms.fit(x_tr)

ms_pred = ms.predict(x_te)
mutual_info_score(y_te, ms_pred)
v_measure_score(y_te, ms_pred)

ms_cg = pd.DataFrame(contingency_matrix(y_te, ms_pred))
ms_cg = ms_cg/ms_cg.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(ms_cg,cmap=cmap)
cm_heat.get_figure().savefig("ms_cm.jpeg",dpi=2400)

ms_preds = np.max(ms_cg, axis=1)
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': ms_preds}
km_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=ms_preds)
