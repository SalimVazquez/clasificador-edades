# Libreria para usar/movernos en directorios del SO
import sys
import os
# Preprocesar imagenes
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# Optimizador de entrenamiento
from tensorflow.python.keras import optimizers
# Libreria para hacer NN en orden (sequencial)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
# Capas en las que se hará convoluciones
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
# Manejo de background en el ordenador
from tensorflow.python.keras import backend as K

# Limpiamos los procesos que estan en background
K.clear_session()

# path de dataset
path_training = './data/training'
path_validate = './data/validate'

# Params para la NN
# iteraciones en todo el proceso de entrenamiento
epochs = 20
# Re dimensionando las imagenes del dataset a 100px
height, width = 100, 100
# No de imagenes (lote) a enviar por cada paso 
batch_size = 32
# No de pasos a iterar en cada epoca
steps = 1000
# No de pasos a iterar de validacion
steps_validate = 200
# Filtros a usar en la convolucion
# Capa 1 = 32 profundidad
# Capa 2 = 64 profundidad
filter_conv1 = 32
filter_conv2 = 64
# Tamaño de filtros
# Capa 1 = 3 de altura, 3 de longitud
# Capa 2 = 2 de altura, 2 de longitud
size_filter1 = (3,3)
size_filter2 = (2,2)
# Tamaño de filtro para MaxPooling
size_pool = (2,2)
# No de clases en dataset
classes = 4
# Tasa de aprendizaje (lambda)
lr = 0.005

# pre-procesamiento de imagenes
datagen_training = ImageDataGenerator(
    rescale = 1./255, # Re escala las imagenes (0-1 px)
    shear_range = 0.3, # Inclinar imagenes para mejor entrenamiento
    zoom_range = 0.3, # Aumentar/Alejar imagenes para mejor entrenamiento
    horizontal_flip = True # Invertir imagen para mejor entrenamiento
)

datagen_validate = ImageDataGenerator(rescale=1./255)

img_training = datagen_training.flow_from_directory(
    path_training, # path de imagenes para entrenamiento
    target_size = (height, width), # Lee y procesa las imagenes a $height, $width
    batch_size = batch_size, # Crea lotes de imagenes
    class_mode = 'categorical'
)

img_validate = datagen_validate.flow_from_directory(
    path_validate, # path de imagenes para validación
    target_size = (height, width), # Lee y procesa las imagenes a $height, $width
    batch_size = batch_size, # Crea lotes de imagenes
    class_mode = 'categorical'
)