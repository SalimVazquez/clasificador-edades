import numpy as np
import tensorflow as tf

# altura y ancho de imagenes
height, width = 1000, 800

# rutas del modelo entrenado
path_model = './model/model.h5'
path_weights = './model/weights.h5'

# Cargar modelo entrenado
cnn = tf.keras.models.load_model(path_model)
cnn.load_weights(path_weights)

def predict(file):
    # Cargamos una imagen
    x = tf.keras.preprocessing.image.load_img(file, target_size=(height, width))
    # Convertimos la imagen en un arreglo
    x = tf.keras.preprocessing.image.img_to_array(x)
    # Agregando una dimension extra en el eje 0 del arreglo
    x = np.expand_dims(x, axis=0)
    # Hacer una predicción, contiene un arreglo en base al No de clases
    # [[0, 0, 0, 1]]
    data = cnn.predict(x)
    # Obtener la posición del arreglo con el valor más alto
    # En este caso retornara la posición de la clase con mas coincidencia
    prediction = np.argmax(data[0])
    # Hacemos validaciones en base al número de clases
    if prediction == 0:
        print('Infancia')
    elif prediction == 1:
        print('Adolescentes')
    elif prediction == 2:
        print('Adulto')
    elif prediction == 3:
        print('3ra edad')

# Llamando la función y
# enviando la imagen a evaluar
# para una prediccion
predict('filename.jpg')