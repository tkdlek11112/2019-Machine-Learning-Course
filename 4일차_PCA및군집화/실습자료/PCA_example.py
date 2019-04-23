# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 23:50:09 2019

@author: HQ
"""

# Import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

sns.set_style('whitegrid')

# Import data
data = pd.read_csv("BreastCancerWisconsin.csv")
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))

# Split dataset into X and y
X = data.drop(['diagnosis'], axis=1)
X = X.iloc[:, :10]
y = data['diagnosis']

# Split trian / test data set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Standardize data onto unit scale (mean=0 and variance=1)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Perform PCA
pca = PCA(n_components=None)
pca.fit(X_train)

Z_train = pca.transform(X_train)
print("- Shape of transformed data: ", Z_train.shape)

Z_test = pca.transform(X_test)
print("- Shape of transformed data: ", Z_test.shape)

# Explained variance ratio of principal components
num_components = pca.n_components_
exp_var = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)


# Plot explained variance ratio and cumulative sums
plt.figure(num=1, figsize=(7, 7))
plt.bar(range(num_components), exp_var, alpha=0.5, label='individual explained variance')
plt.step(range(num_components), cum_exp_var, label='cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.show()


# Plot the transformed data (Z) with 2 PCs
plt.figure(num=2, figsize=(7, 7))
for label, color, marker in zip(('B', 'M'), ('blue', 'red'), ('o', '^')):
    plt.scatter(Z_train[y_train == label, 0], Z_train[y_train == label, 1],
                label=label, color=color, marker=marker, alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Plot the transformed data (Z) with 3 PCs
fig = plt.figure(num=3, figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
for label, color, marker in zip(('B', 'M'), ('blue', 'red'), ('o', '^')):
    ax.scatter(Z_train[y_train == label, 0], Z_train[y_train == label, 1],
               Z_train[y_train == label, 2], label=label, color=color,
               marker=marker, alpha=0.5)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend(loc='best')
plt.show(fig)

###########################################################################
## Label Encoding 

print("- sample of 'y': ", y_train[:5])

le = LabelEncoder()
le.fit(y_train)

print(" - encoding label of 'Y': ", le.classes_)
print(" - encoding label of 'Y': [0, 1]")

Y_train = le.transform(y_train)
Y_test = le.transform(y_test)

## Select sub-features 
Z_sub_train = pd.DataFrame(Z_train[:,:3])
Z_sub_test = pd.DataFrame(Z_test[:,:3])

## Build LR model 
log_Z = LogisticRegression()
log_Z.fit(Z_sub_train, Y_train)

## Predict & calculate score 
log_Z.score(Z_sub_test, Y_test)
pred_Z = log_Z.predict(Z_sub_test)
confusion_matrix(Y_test, pred_Z)

############################################################################

## Build model and prediction (Original Data)
log_ori = LogisticRegression()
log_ori.fit(X_train, Y_train)
log_ori.score(X_test, Y_test)
pred_ori = log_ori.predict(X_test)
confusion_matrix(Y_test, pred_ori)





