# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:10:25 2019

@author: Yang Xu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

##-----------------------------------------------------------------------------
def NB_fusion(cm_array,pred_list):
    x,y = cm_array.shape
    n,m = pred_list.shape
    wprob = np.zeros((y,m))
    for i in range(m):
        p = pred_list[:,i]
        #u, indices = np.unique(p, return_index=True)
        p = p.tolist()
        
        for j in range(len(p)):
            wprob[p[j],i]+=cm_array[j,p[j]]
            
    return wprob/x

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

def para_est(x,y):
    
    l = np.unique(y)
    mu=[]
    #covs=[]
    cova = np.cov(x.T)
    for i in l:
        mu.append(x[y==i].mean(axis=0))
        ##estimate parameters by assuming
        ##The covariance matrices are different from each category, and 
        ##decision boundary would be hyperquadratic for 2-D Gaussian
        #covs.append(np.cov(x[y==i].T))
        
    return [mu,cova]

def bayes_des_rule(para,test,prior=None):
    test=np.array(test)
    mu = para[0]
    covs=para[1]
    b=np.linalg.inv(covs)
    c=np.linalg.det(covs)
    if str(c)=='inf':
        c=1000
    gs=[]
    for i in range(len(mu)):
        
        a = test-mu[i]

        g = (-1/2)*((a.T*b*a).sum()+\
          np.log(c))+np.log(prior[i])
        
        gs.append(g)
    
    return gs.index(max(gs))

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

##split into training and testing sets
x_tr, x_te, y_tr, y_te = train_test_split(xpca, y, test_size=0.25, \
                                          random_state=111)

##-----------------------------------------------------------------------------
##data visualization
import umap
embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(x)
plt.scatter(embedding[:,0], embedding[:,1], c=y)
dt = pd.DataFrame(np.column_stack((embedding,y)))
dt.to_csv("umap.csv",index=False)

##-----------------------------------------------------------------------------
##kNN with 10-fold cross-validation
kf = KFold(n_splits=10)
kf.get_n_splits(x_tr)
error={}
accuracy={}
for k in range(1,16):
    
    kerr=[]
    kacc=[]
    for tr_index,te_index in kf.split(x_tr):
        X_train, X_test =  x_tr[tr_index], x_tr[te_index]
        y_train, y_test = y_tr[tr_index], y_tr[te_index]
        
        knn = KNeighborsClassifier(n_neighbors=k)  
        knn.fit(X_train, y_train) 
        
        y_pred = knn.predict(X_test)
        
        kerr.append(np.mean(y_pred != y_test))
        kacc.append(np.mean(y_pred == y_test))
    error[k]=sum(kerr)/len(kerr)
    accuracy[k]=sum(kacc)/len(kacc)

##plot overall accuracy
accuracy=pd.DataFrame.from_dict(accuracy,orient='index')
plt.plot(accuracy.index,accuracy.iloc[:,0])
plt.ylabel('Accuracy')
plt.xlabel('k')
plt.show()
        
##the optimal k
knn = KNeighborsClassifier(n_neighbors=7)  
knn.fit(x_tr, y_tr)

knn_pred = knn.predict(x_te)
np.mean(knn_pred == y_te)

knn_cm = pd.DataFrame(confusion_matrix(y_te, knn_pred))
knn_cm = knn_cm/knn_cm.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(knn_cm,cmap=cmap)
cm_heat.get_figure().savefig("knn_cm.jpeg",dpi=2400)

knn_preds = knn_cm.values.diagonal()
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': knn_preds}
knn_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=knn_preds)

##-----------------------------------------------------------------------------
##gaussion with 10-fold cross-validation
kf = KFold(n_splits=10)
kf.get_n_splits(x_tr)
accuracy={}
#l = np.unique(y_tr)
#prior = [1/len(l) for i in l]
iters=0
old_accu=0
for tr_index,te_index in kf.split(x_tr):
        iters+=1
        X_train, X_test =  x_tr[tr_index], x_tr[te_index]
        y_train, y_test = y_tr[tr_index], y_tr[te_index]
        
        l = np.unique(y_train)
        y_list = y_train.tolist()
        prior = [y_list.count(i)/len(y_list) for i in l]
        
        paras = para_est(X_train,y_train)
        y_pred=[]
        for i in X_test:
            
            g = bayes_des_rule(para=paras,test=i,prior=prior)
            y_pred.append(g)
        
        if np.mean(y_pred == y_test)>= old_accu:
            new_paras=paras.copy()
            opt_prior = prior.copy()
            old_accu=np.mean(y_pred == y_test)
        accuracy[iters]=np.mean(y_pred == y_test)

##plot overall accuracy
accuracy=pd.DataFrame.from_dict(accuracy,orient='index')
plt.plot(accuracy.index,accuracy.iloc[:,0])
plt.ylabel('Accuracy')
plt.xlabel('iteration')
plt.show()

gk_pred=[]
for i in x_te:
            
    g = bayes_des_rule(para=new_paras,test=i,prior=opt_prior)
    gk_pred.append(g)
    
np.mean(gk_pred == y_te)

gk_cm = pd.DataFrame(confusion_matrix(y_te, gk_pred))
gk_cm = gk_cm/gk_cm.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(gk_cm,cmap=cmap)

gk_preds = gk_cm.values.diagonal()
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': gk_preds}
gk_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=gk_preds)

##-----------------------------------------------------------------------------
##Neural network
from keras.layers import Dense
from keras import regularizers
from keras.models import Sequential
from keras import backend as K
from copy import deepcopy
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

y_tr_o = np_utils.to_categorical(y_tr)

K.clear_session()

# create model
model = Sequential()
model.add(Dense(100, input_dim=59, activation='relu',\
                kernel_initializer='glorot_normal',\
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(100, activation='relu',\
                kernel_initializer='glorot_normal',\
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(100, activation='relu',\
                kernel_initializer='glorot_normal',\
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(41, activation='softmax'))
        
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
                 
model.compile(loss='categorical_crossentropy', optimizer='adam', \
              metrics=['accuracy'])

model.fit(x_tr, y_tr_o, epochs=100, batch_size=256,callbacks=callbacks,\
          validation_split=0.1)

opt_nn = deepcopy(model)
nn_pred=opt_nn.predict(x_te)
nn_pred = np.argmax(nn_pred, axis=1)

np.mean(nn_pred == y_te)

nn_cm = pd.DataFrame(confusion_matrix(y_te, nn_pred))
nn_cm = nn_cm/nn_cm.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(nn_cm,cmap=cmap)

nn_preds = nn_cm.values.diagonal()
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': nn_preds}
nn_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=nn_preds)

##-----------------------------------------------------------------------------
##SVM
from sklearn.svm import LinearSVC

kf = KFold(n_splits=10)
kf.get_n_splits(x_tr)
accuracy={}
#l = np.unique(y_tr)
#prior = [1/len(l) for i in l]
iters=0
old_accu=0
for tr_index,te_index in kf.split(x_tr):
        iters+=1
        X_train, X_test =  x_tr[tr_index], x_tr[te_index]
        y_train, y_test = y_tr[tr_index], y_tr[te_index]
        
        svm = LinearSVC(C=0.0001)
        svm.fit(X_train, y_train)
        
        if np.mean(y_pred == y_test)>= old_accu:
            opt_svm=deepcopy(svm)
            old_accu=np.mean(y_pred == y_test)
        accuracy[iters]=np.mean(y_pred == y_test)
        
svm_pred = opt_svm.predict(x_te)
np.mean(svm_pred == y_te)

svm_cm = pd.DataFrame(confusion_matrix(y_te, svm_pred))
svm_cm = svm_cm/svm_cm.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(svm_cm,cmap=cmap)
cm_heat.get_figure().savefig("svm_cm.jpeg",dpi=2400)

svm_preds = svm_cm.values.diagonal()
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': svm_preds}
svm_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=svm_preds)

##-----------------------------------------------------------------------------
##ensemble
knn_preds = knn_cm.values.diagonal()
gk_preds = gk_cm.values.diagonal()
nn_preds = nn_cm.values.diagonal()
svm_preds = svm_cm.values.diagonal()

cm_array = np.vstack((knn_preds,gk_preds,nn_preds,svm_preds))
pred_list = np.vstack((knn_pred,gk_pred,nn_pred,svm_pred))

wp = NB_fusion(cm_array,pred_list)
y_pred = np.argmax(wp,axis=0)
np.mean(y_pred == y_te)

all_cm = pd.DataFrame(confusion_matrix(y_te, y_pred))
all_cm = all_cm/all_cm.sum(axis = 0)
np.mean(all_cm)

cm_heat=sns.heatmap(all_cm,cmap=cmap)

all_preds = all_cm.values.diagonal()
audio = ["C"+str(i) for i in range(41)]
d={'Category': audio, 'Accuaracy': all_preds}
all_preds = pd.DataFrame(d)
sns.barplot(x="Category", y="Accuaracy", data=all_preds)