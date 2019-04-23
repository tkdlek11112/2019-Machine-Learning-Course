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
from mpl_toolkits.mplot3d import Axes3D


from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA


sns.set_style('white')


# Import data
data = pd.read_csv("BreastCancerWisconsin.csv")
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))


# Check class label distribution
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.countplot(data['diagnosis'], palette='Set1')
plt.tight_layout()
plt.show(fig)


# For clustering, we will only be using the X variables
X = data.drop(['diagnosis'], axis=1)
X = X.iloc[:, :10]
y = data['diagnosis']


# Standardize dataset columnwise, to have zero mean and unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Perform hierarchical clustering
''' single / complete / average / weighted / centroid / median / ward '''
method = 'ward' 

''' euclidean / minkowski / cosine / correlation / jaccard ... '''
metric = 'euclidean'
D = linkage(X, method=method, metric=metric)

# Draw dendrogram
fig, ax = plt.subplots(1, 1, figsize=(15, 6))
dendrogram(Z=D,
           p=100,
           truncate_mode='lastp',
           orientation='top',
           show_leaf_counts=True,
           no_labels=False,
           leaf_font_size=12.,
           leaf_rotation=90.,
           ax=ax,
           above_threshold_color='k')
ax.set_xlabel('Observations')
ax.set_ylabel('Distance')
ax.set_title('Method: {}, Metric: {}'.format(method, metric))
plt.show(fig)


# Get cluster labels, by predefining the number of clusters
num_clusters = 5
label_hc = fcluster(D, t=num_clusters, criterion='maxclust')

# Plot on 2D space, using PCA components
# Perform PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X)

# Plot the transformed data (Z) with 2 PCs
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
axes = axes.ravel()
for label, color in zip(('B', 'M'), ('blue', 'red')):
    axes[0].scatter(Z[y == label, 0], Z[y == label, 1],
                    label=label, color=color, marker='o', alpha=0.5)
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(loc='best')
    axes[0].set_title('Original class labels')
for i in range(num_clusters):
    axes[1].scatter(Z[label_hc == i + 1, 0], Z[label_hc == i + 1, 1],
                    label='cluster {}'.format(i + 1), marker='^', alpha=0.5)
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend(loc='best')
    axes[1].set_title('Labels from Hierarchical clustering')
plt.show(fig)



# Get cluster labels, by defining the upper threshold on distance
threshold = 10
label_hc = fcluster(D, t=threshold, criterion='distance')

# Plot on 3D space, using PCA components
# Perform PCA
pca = PCA(n_components=3)
Z = pca.fit_transform(X)

# Plot the transformed data (Z) with 3 PCs
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9), subplot_kw={'projection': '3d'})
axes = axes.ravel()
for label, color in zip(('B', 'M'), ('blue', 'red')):
    axes[0].scatter(Z[y == label, 0], Z[y == label, 1], Z[y == label, 2],
                    label=label, color=color, marker='o', alpha=0.5)
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].set_zlabel('Principal Component 3')
    axes[0].legend(loc='best')
    axes[0].set_title('Original class labels')
for i in range(max(label_hc) + 1):
    axes[1].scatter(Z[label_hc == i, 0], Z[label_hc == i, 1], Z[label_hc == i, 2],
                    label='cluster {}'.format(i + 1), marker='^', alpha=0.5)
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].set_zlabel('Principal Component 3')
    axes[1].legend(loc='best')
    axes[1].set_title('Labels from Hierarchical clustering')
plt.show(fig)
