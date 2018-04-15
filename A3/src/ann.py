import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 60

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
# y_train = y_train.reshape(y_train.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# y_test = y_test.reshape(y_test.shape[0], 1)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x=X_train, y=y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs)

model.evaluate(x=X_test, y=y_test, verbose=1)