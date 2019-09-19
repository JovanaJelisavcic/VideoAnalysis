from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
import warnings
import itertools
from mpl_toolkits import mplot3d
from matplotlib import cm
from keras import models
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import plot_model
warnings.filterwarnings("ignore")
K.set_image_data_format('channels_last')
np.random.seed(0)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

"""
x_train = X_train.reshape(-1,28, 28, 1)   #Reshape for CNN -  should work!!
x_test = X_test.reshape(-1,28, 28, 1)
"""

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255.
X_test /= 255.

print('X_train:\t{}' .format(X_train.shape),' -reshaped')
print('X_test: \t{}' .format(X_test.shape),' -reshaped')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print('num_classes:\t', num_classes)

# Hyperparameters
training_epochs = 10 # Total number of training epochs
learning_rate = 0.03 # The learning rate


def create_model():
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu',input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile a model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.adadelta(), metrics=['accuracy'])
    return model


model = create_model()
model.summary()

results = model.fit(
 X_train, y_train,
 epochs= training_epochs,
 batch_size = 128,
 validation_data = (X_test, y_test),
 verbose = 2
)

prediction_values = model.predict_classes(X_test)

