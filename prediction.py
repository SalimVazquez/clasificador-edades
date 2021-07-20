import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# altura y ancho de imagenes
height, width = 100, 100

# rutas del modelo entrenado
path_model = './model/model.h5'
path_weights = './model/weights.h5'