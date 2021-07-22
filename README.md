# Clasificador-edades
Sistema de [red neuronal](https://www.xeridia.com/blog/redes-neuronales-artificiales-que-son-y-como-se-entrenan-parte-i) multicapa con analisis/procesamiento de imagenes.
Proyecto de **Inteligencia Artificial**, capaz de predecir la *edad* o *etapa de
desarrollo humano* a través del análisis de la imagen de **la palma de una mano** proporcionada para pruebas.

### Dataset
Se estan recolectando imagenes, para completar un *dataset*
que contenga **4 clases y 40 observaciones en cada clase**.

### Clases
- Infancia (4-15 años de edad)
- Adolescencia (16-30 años de edad)
- Adulto (31-50 años de edad)
- 3ra edad (+50 años de edad)

### Requerimientos
- [TensorFlow](https://www.tensorflow.org/install?hl=es-419)
- [Keras](https://www.tutorialspoint.com/keras/keras_installation.htm)
- [Dataset](https://drive.google.com/file/d/1mW_IsRvS_dDiFaP85i0yDylFeZyEwfWe/view)

### Configuración
- Necesita descargar el *dataset*, y descomprimir las carpetas que contienen las imagenes dentro de las carpetas:
    - [training](/data/training)
    - [validate](/data/validate)

Con esto el programa al momento de entrenar, leera y procesara las imagenes.
Y así podrá entrenarse y generar el modelo de la red.

### Herramientas
Usamos el siguiente sitio para redimensionar el dataset a un tamaño de pixeles, en este caso
500 px de ancho y largo.
- [Redimensionar imagenes](https://www.iloveimg.com/es/redimensionar-imagen)
