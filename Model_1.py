# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

N_EPOCH = 20
BATCH_SIZE = 10
VERBOSE = 1
N_CLASS = 10
OPTIMIZER = SGD()
N_HIDDEN_1 = 128
VALIDATION_SPLIT = 0.2
RESHAPE = 784
DROPOUT = 0.01


#(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = pd.read_csv('Train/X_train.csv')
y_train = pd.read_csv('Train/y_train.csv')
X_test = pd.read_csv('Test/X_test.csv')
y_test = pd.read_csv('Test/y_test.csv')

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

INPUT_DIM = X_train.shape[1]
OUTPUT_DIM = np.unique(y_train).shape[0]
N_CLASS = OUTPUT_DIM

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()

y_train = y_train.astype('int')
y_test = y_test.astype('int')


#X_train, Y_train = shuffle(X_train, y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = Sequential()
classifier.add(Dense(units = 6, activation='relu', kernel_initializer='glorot_uniform', input_dim=INPUT_DIM))
classifier.add(Dropout(DROPOUT))
classifier.add(Dense(units = 20, activation='relu', kernel_initializer = 'glorot_uniform'))
classifier.add(Dropout(DROPOUT))
classifier.add(Dense(units = 30, activation='relu', kernel_initializer = 'glorot_uniform'))
classifier.add(Dropout(DROPOUT))
classifier.add(Dense(units = OUTPUT_DIM, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model = classifier

history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, 
                    epochs = N_EPOCH, verbose = VERBOSE, 
                    validation_split=VALIDATION_SPLIT)


scores = model.evaluate(X_test, y_test, verbose=1)
print("Test Score: ", scores[0])
print("Accuracy: " , scores[1])


from keras.models import load_model
# Creates a HDF5 file 'my_model.h5'
model.save('HUMAN_ACTIVITY_4.h5')
# Deletes the existing model
#del model  
# Returns a compiled model identical to the previous one
model = load_model('MNIST.h5')


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



scores = model.evaluate(X_test, y_test, verbose=1)
print("Test Score: ", scores[0])
print("Accuracy: " , scores[1])





# Predicting the Test set results
y_pred = model.predict(X_test)


y_pred = (y_pred > 0.5)
y_test2 = (y_test > 0.5)


y_test_argmax = y_test.argmax(axis=1)
y_pred_argmax = y_pred.argmax(axis=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


# =============================================================================
# Code of Confusion matrix
# =============================================================================


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




















