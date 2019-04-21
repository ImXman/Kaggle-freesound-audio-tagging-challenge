# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:18:58 2019

@author: Yang Xu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

##-----------------------------------------------------------------------------

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

y_pred = knn.predict(x_te)
np.mean(y_pred == y_te)

print(confusion_matrix(y_te, y_pred))