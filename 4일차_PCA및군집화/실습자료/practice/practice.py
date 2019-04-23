# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:49:01 2019

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
# file path 
data = pd.read_csv( ... )
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))

# Split dataset into X and y
# target variable name 
X = data.drop([ ... ], axis=1)
y = data[ ... ]

# Split trian / test data set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 20190424)