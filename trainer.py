# Libreria para usar/movernos en directorios del SO
import sys
import os
# Preprocesar imagenes
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# Optimizador de entrenamiento
from tensorflow.python.keras import optimizers
# Libreria para hacer NN en orden (sequencia)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
# Capas en las que se hará convoluciones
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
# Manejo de background en el ordenador
from tensorflow.python.keras import backend as K