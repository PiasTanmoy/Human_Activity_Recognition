# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:02:18 2019

@author: Pias Tanmoy
"""
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')
dataset2 = shuffle(dataset)

dataset2.to_csv('Data2.csv', sep='\t')

dataset2 = np.array(dataset2)

X = dataset2[:,0:561]
y = dataset2[:, 561]
y = y.reshape(y.shape[0],1)'
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=0)
#numpy.savetxt("foo.csv", a, delimiter=",")
np.savetxt("X_train2.csv", X_train, delimiter=",")
np.savetxt("y_train2.csv", y_train, delimiter=",")
np.savetxt("X_test2.csv", X_test, delimiter=",")
np.savetxt("y_test2.csv", y_test, delimiter=",")
