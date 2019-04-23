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
import matplotlib.cm as cm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

sns.set_style('white')


# Import data
data = pd.read_csv("BreastCancerWisconsin.csv")
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))


# For clustering, we will only be using the X variables
X = data.drop(['diagnosis'], axis=1)
X = X.iloc[:, :10]
y = data['diagnosis']


# Standardize dataset columnwise, to have zero mean and unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Perform K means clustering
n_clusters = 2
km = KMeans(n_clusters=n_clusters, random_state=20190420)
km.fit(X)


# Get predicted labels from K means clustering
labels_km = km.predict(X)


# Visualize to see the result
pca = PCA(n_components=2)
Z = pca.fit_transform(X)
plt.figure(1)
for i in range(max(labels_km) + 1):
    plt.scatter(Z[labels_km == i, 0], Z[labels_km == i, 1],
                label='Cluster {}'.format(i + 1), alpha=.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Number of Clusters: {}'.format(n_clusters))
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Silhouette scores
silhouette_avg = silhouette_score(X, labels_km)
print("For n_clusters =", n_clusters,
      "| The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, labels_km)

# Visualize
plt.figure(2)
y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[labels_km == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.Spectral(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
plt.vlines(x=silhouette_avg, ymin=0, ymax=X.shape[0], color="red", linestyle="--")
plt.title("The silhouette plot for K={}".format(n_clusters))
plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster labels")
plt.yticks([])
plt.show()


# Find the best K
sil_scores = []
list_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in list_n_clusters:
    km = KMeans(n_clusters=n_clusters)
    labels_km = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels_km))

plt.figure(3)
plt.plot(range(1, len(sil_scores) + 1), sil_scores)
plt.title('Silhouette scores')
plt.show()