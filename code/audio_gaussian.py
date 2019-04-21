# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:53:20 2019

@author: Yang Xu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

##-----------------------------------------------------------------------------
def para_est(x,y):
    
    l = np.unique(y)
    mu=[]
    cova = np.cov(x.T)
    for i in l:
        mu.append(x[y==i].mean(axis=0))
        ##estimate parameters by assuming
        ##The covariance matrices are different from each category, and 
        ##decision boundary would be hyperquadratic for 2-D Gaussian
        #cov.append(np.cov(x[y==i].T))
        
    return [mu,cova]

def bayes_des_rule(para,test,prior=None):
    test=np.array(test)
    mu = para[0]
    b=np.linalg.inv(para[1])
    c=np.linalg.det(para[1])
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
y,uniques =pd.factorize(trm.iloc[:,-1])

##split into training and testing sets
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, \
                                          random_state=111)

##10-fold cross-validation
kf = KFold(n_splits=10)
kf.get_n_splits(x_tr)
accuracy={}
l = np.unique(y_tr)
prior = [1/len(l) for i in l]
iters=0
old_accu=0
for tr_index,te_index in kf.split(x_tr):
        iters+=1
        X_train, X_test =  x_tr[tr_index], x_tr[te_index]
        y_train, y_test = y_tr[tr_index], y_tr[te_index]
        
        paras = para_est(X_train,y_train)
        y_pred=[]
        for i in X_test:
            
            g = bayes_des_rule(para=paras,test=i,prior=prior)
            y_pred.append(g)
        
        if np.mean(y_pred == y_test)>= old_accu:
            new_paras=paras.copy()
            old_accu=np.mean(y_pred == y_test)
        accuracy[iters]=np.mean(y_pred == y_test)

##plot overall accuracy
accuracy=pd.DataFrame.from_dict(accuracy,orient='index')
plt.plot(accuracy.index,accuracy.iloc[:,0])
plt.ylabel('Accuracy')
plt.xlabel('iteration')
plt.show()

y_pred=[]
for i in x_te:
            
    g = bayes_des_rule(para=new_paras,test=i,prior=prior)
    y_pred.append(g)
    
np.mean(y_pred == y_te)

cm = pd.DataFrame(confusion_matrix(y_te, y_pred))
cm = cm/cm.sum(axis = 0)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cm_heat=sns.heatmap(cm,cmap=cmap)
cm_heat.get_figure().savefig("confusion_maxtrix_gaussian.jpeg",dpi=2400)