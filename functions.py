import os, h5py
import numpy as np
from keras.preprocessing import image

def is_file(folder, file, extension="h5"):
    # Ruta al directorio de modelos
    directorio_modelos = f"{folder}"
    # Nombre del archivo .h5 que quieres verificar
    nombre_archivo = f"{file}.{extension}"
    # Ruta completa al archivo .h5
    ruta_completa = os.path.join(directorio_modelos, nombre_archivo)
    # Verificar si el archivo existe
    if os.path.exists(ruta_completa):
        return True
    else:
        return False

def check_if_H5(filepath):

    with h5py.File(filepath, 'r') as f:
        # Verificar la estructura del archivo
        print("Estructura del archivo:")
        print(list(f.keys()))  # Lista de grupos en el archivo

        # Acceder a un grupo específico y verificar sus claves
        grupo = f['model_weights']
        print("Claves del grupo 'model_weights':")
        print(list(grupo.keys()))

        # Obtener algunos valores específicos (esto depende de la estructura de tu archivo)
        valor = grupo['conv2d']['conv2d']['kernel:0'][()]
        print("Valor del kernel de conv2d:")
        print(valor)

def load_and_preprocess_image(image_path):
    # Cargar la imagen desde la ruta de archivo
    img = image.load_img(image_path, target_size=(64, 64))  # Ajusta el tamaño según las dimensiones de entrada de tu modelo
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Agregar una dimensión para que sea una muestra única

    # Preprocesar la imagen
    img = img / 255.0  # Normalizar los valores de píxeles al rango [0, 1]

    return img
