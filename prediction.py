import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# altura y ancho de imagenes
height, width = 100, 100

# rutas del modelo entrenado
path_model = './model/model.h5'
path_weights = './model/weights.h5'

# Cargar modelo entrenado
cnn = load_model(path_model)
cnn.load_weights(path_weights)

def predict(file):
    # Cargamos una imagen
    x = load_img(file, target_size=(width, height))
    # Convertimos la imagen en un arreglo
    x = img_to_array(x)    
    # Agregando una dimension extra en el eje 0 del arreglo
    x = np.expand_dims(x, axis=0)
    # Hacer una predicción, contiene un arreglo en base al No de clases
    # [[0, 0, 0, 1]]
    data = cnn.predict(x)