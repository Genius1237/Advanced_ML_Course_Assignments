import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras import backend as K

#EPOCHS = 4000 Change number of epochs here

batch_size = 128
epochs = EPOCHS