import pandas as pd
import numpy as np
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 60

(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

model = Sequential()
model.add(Conv2D(32, (5,5), padding='valid')

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x=X_train, y=y_train, batch_size=batch_size,validation_split=0.2, epochs=epochs)

model.evaluate(x=X_test, y=y_test)
