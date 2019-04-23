# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:51:34 2019

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

from graphviz import Source
from IPython.display import Image

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")


# Import data
# file path 
data = pd.read_csv( ... )
print("- Data has {} rows and {} columns.".format(*data.shape))
print("- Column names: ", list(data.columns))


# Split data into X and y; the 'diagnosis' column is the class label
# target variable name 
X = data.drop([ ... ], axis=1)
y = data[ ... ]