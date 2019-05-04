# -*- coding: utf-8 -*-
"""
Created on Thu April  25 21:12:24 2019
Audio tagging challenge Neural Network
@author: Quan Zhou

"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns; sns.set()  # for plot styling
#import performance
from sklearn.decomposition import PCA as PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
#from DrawingTools import plot_confusion_matrix

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

##split into training and testing sets, 75%-25% split
X_train, X_test, y_train, y_test = train_test_split(pX, y, test_size=0.25, \
                                          random_state=111)

#%% fine tune the best parameters using GridSearchCV
param_grid ={
                    'activation' : ['identity', 'logistic', 'tanh'],
                    'alpha' : [0.0001],
                    'batch_size': ['auto'],
                    'learning_rate_init': [0.001],
                    'max_iter' : [2000], #maximum iterations
                    'hidden_layer_sizes': [ (5,),(10,),(20,),(5,5),(10,10), (50,50), 
                                           (20,20,20),(50,50,50),(20,20,20,20)]
                }
# obtain best model after fine tuning with 10-fold cross validation
model = GridSearchCV(MLPClassifier(), param_grid, cv=10,n_jobs=-1,scoring='accuracy')
model.fit(X_train,y_train) # pretty time consuming
print(model.best_params_)
predicted_labels = model.best_estimator_.predict(X_test)
accuracy = accuracy_score(predicted_labels,y_test);
confusion_matrix = get_confusion_matrix(predicted_labels,y_test)
print('best model accuracy:',accuracy)
print('best model confusion matrix:',confusion_matrix)
##----------Plot confusion matrix---------------------
classes = np.unique(y)
#plot_confusion_matrix(confusion_matrix, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues,
#                          save_path='../data/',
#                          filename='CMBPNN.png',
#                          show_plots=True)
fig = plt.figure(figsize=(11, 9), dpi=100)
sns.heatmap(confusion_matrix.T, square=True, annot=True, cbar=False,cmap="BuPu",
            xticklabels=classes,
            yticklabels=classes)
plt.xlabel('true label')
plt.ylabel('predicted label');


# best parameter is
# {'activation': 'identity', 'alpha': 0.0001, 'batch_size': 'auto', 'hidden_layer_sizes': (20,20,20,20),
# 'learning_rate_init': 0.001, 'max_iter': 2000}
#%%
""" when set as optimal parameter, then plot the "accuracy over epochs" figure
    optimal hidden layer number is 20."""
mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20), max_iter=2000, alpha=1e-4,
                    solver='adam', verbose=0, tol=1e-8, random_state=1,
                    learning_rate_init=.001)

"""mini-batch learning"""
N_TRAIN_SAMPLES = X_train.shape[0]
N_EPOCHS = 200
N_BATCH = 100
N_CLASSES = np.unique(y_train)

scores_train = []
scores_test = []

# EPOCH
epoch = 0
while epoch < N_EPOCHS:
    print('epoch: ', epoch)
    # SHUFFLING
    random_perm = np.random.permutation(X_train.shape[0])
    mini_batch_index = 0
    while True:
        # MINI-BATCH
        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TRAIN_SAMPLES:
            break

    # SCORE TRAIN
    scores_train.append(mlp.score(X_train, y_train))

    # SCORE TEST
    scores_test.append(mlp.score(X_test, y_test))

    epoch += 1

""" Plot """
fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(scores_train)
ax[0].set_title('Train')
ax[1].plot(scores_test)
ax[1].set_title('Test')
fig.suptitle("Accuracy over epochs", fontsize=14)
plt.show()
