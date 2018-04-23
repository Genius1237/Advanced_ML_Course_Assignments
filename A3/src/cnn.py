#import pandas as pd
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 50


(X_train, y_train), (X_test, y_test) = mnist.load_data()
rows, cols = X_train.shape[1], X_train.shape[2]
input_shape = (rows, cols, 1)
X_train = X_train.reshape(X_train.shape[0], rows, cols, 1).astype(float)
X_test = X_test.reshape(X_test.shape[0], rows, cols, 1).astype(float)

#Convert pixels to [0, 1] range as per assignment requirements
X_train/=255.0
X_test/=255.0

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5,5), padding='valid', activation='sigmoid', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Reduces to a 4x4 dimension matrix after this

model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dense(128, activation='sigmoid'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(x=X_train, y=y_train, batch_size=batch_size,validation_split=0.2, epochs=epochs)

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
plt.savefig('Accuracy.png')
plt.close()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')
plt.legend(['Train Loss', 'Test Loss'], loc='upper left')
plt.savefig('Losses.png')
plt.close()

score = model.evaluate(x=X_test, y=y_test)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])
