#Followed instructions on this guide: https://tinyurl.com/455tnnv2

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# load dataset
mnistFashion = tf.keras.datasets.fashion_mnist
(trainX, trainY), (testX, testY) = mnistFashion.load_data()

# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# one hot encode target values
trainY = to_categorical(trainY)

# convert from integers to floats
trainX = trainX.astype('float32')

# normalize images
trainX = trainX / 255.0

# initialize the model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

# compile model
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(trainX, trainY, epochs=10, batch_size=32)

# save model
model.save('model.h5')