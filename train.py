import os
from PIL import UnidentifiedImageError
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback
# Funciones creadas
import functions

class NeuronalNetwork:
    def __init__(self):
        self.classifier = Sequential()
        self.classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(64, 64, 3) ))
        self.classifier.add(MaxPooling2D(pool_size=(2,2)))
        self.classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(64, 64, 3) ))
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=128, activation="relu"))
        self.classifier.add(Dense(units=1, activation="sigmoid"))
        self.classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        # Preprocesar las imágenes y cargar el conjunto de datos
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.test_datagen = ImageDataGenerator(
            rescale=1./255
        )
        # Train
        self.training_dataset = self.train_datagen.flow_from_directory(
            'train_images',  # Directorio que contiene las imágenes
            target_size=(64, 64),  # Redimensiona las imágenes a 150x150 píxeles
            batch_size=32,
            class_mode='binary')  # Para problemas de clasificación con etiquetas numéricas
        # Test
        self.testing_dataset = self.test_datagen.flow_from_directory(
            'train_images',  # Directorio que contiene las imágenes
            target_size=(64, 64),  # Redimensiona las imágenes a 150x150 píxeles
            batch_size=32,
            class_mode='binary')  # Para problemas de clasificación con etiquetas numéricas

    # Función personalizada para mostrar el progreso
    @staticmethod
    def print_training_progress(epoch, logs):
        print(f"Época {epoch+1} - Pérdida: {logs['loss']:.4f} - Precisión: {logs['accuracy']:.4f}")

    def reload_training_ds(self):
        self.training_dataset = self.train_datagen.flow_from_directory(
            'train_images',  # Directorio que contiene las imágenes
            target_size=(64, 64),  # Redimensiona las imágenes a 150x150 píxeles
            batch_size=32,
            class_mode='binary'
        )


# Clase de entrenamiento
class Train:
    red_n = NeuronalNetwork()

    def __init__(self):
        self.modelo_nombre = "modelo_gato_perro"
        self.extension = "h5"
        self.folder = "model"
        self.images_folder = "train_images"
        self.epochs = 24
        self.steps = 500
        modelo_file_path = os.path.join(self.folder, self.modelo_nombre)
        self.modelo_file_path = os.path.abspath(modelo_file_path)
        modelo_path = os.path.join(self.folder, self.modelo_nombre)
        self.modelo_path = os.path.abspath(modelo_path)

    def train(self, force=False, disabled_force=False):
        if force and not disabled_force:
            for epoch in range(self.epochs):
                for step in range(self.steps):
                    try:
                        batch_x, batch_y = next(self.red_n.training_dataset)
                        self.red_n.classifier.train_on_batch(batch_x, batch_y)
                        if step % 25 == 0 or step == 0 or step == self.steps-1:
                            print(f"Epoch: {epoch+1} - Step: {step if step < self.steps-1 else self.steps}.")
                    except UnidentifiedImageError as ex:
                        image_path = self.red_n.training_dataset.filenames[step]
                        # Construir la ruta completa a la imagen dañada
                        image_path = os.path.join(self.images_folder, image_path)
                        # Obtener la ruta absoluta de la imagen dañada
                        absolute_image_path = os.path.abspath(image_path)
                        # Eliminar la imagen dañada
                        if os.path.exists(absolute_image_path):
                            os.remove(absolute_image_path)
                            print(f"Imagen dañada eliminada: {absolute_image_path}")
                            self.red_n.reload_training_ds()
                    except Exception as ex:
                        print(f"Excepción no controlada")
                # Calcular pérdida y precisión en el conjunto de validación
                while True:
                    try:
                        loss, accuracy = self.red_n.classifier.evaluate(self.red_n.training_dataset, steps=self.steps/2)
                        print(f"Época {epoch + 1} completada - Pérdida: {loss:.4f} - Precisión: {accuracy:.4f}")
                        break
                    except Exception as ex:
                        print(f"Imagen dañada en conjunto de validación: se ha omitido")
                        image_path = self.red_n.training_dataset.filenames[step]
                        # Construir la ruta completa a la imagen dañada
                        image_path = os.path.join(self.images_folder, image_path)
                        # Obtener la ruta absoluta de la imagen dañada
                        absolute_image_path = os.path.abspath(image_path)
                        # Eliminar la imagen dañada
                        if os.path.exists(absolute_image_path):
                            os.remove(absolute_image_path)
                            print(f"Imagen dañada eliminada: {absolute_image_path}")
                            self.red_n.reload_training_ds()
                        continue
            # Guardar modelo
            self.red_n.classifier.save(self.modelo_path)
        else:
            # Calcular pérdida y precisión en el conjunto de validación
            while True:
                loaded_model = load_model(filepath=self.modelo_file_path)
                for epoch in range(self.epochs):
                    for step in range(self.steps):
                        try:
                            batch_x, batch_y = next(self.red_n.training_dataset)
                            loaded_model.train_on_batch(batch_x, batch_y)
                            if step % 25 == 0 or step == 0 or step == self.steps - 1:
                                print(f"Epoch: {epoch + 1} - Step: {step if step < self.steps - 1 else self.steps}.")
                        except Exception as ex:
                            print(f"Error en la época {epoch + 1}, step {step}: {ex}")
                            continue
                    try:
                        # Calcular métricas (accuracy y loss) en el conjunto de validación
                        val_loss, val_accuracy = loaded_model.evaluate(self.red_n.training_dataset, steps=self.steps/2)
                        # Mostrar métricas
                        print(f"Época {epoch + 1} completada - Pérdida: {val_loss:.4f} - Precisión: {val_accuracy:.4f}")
                        continue
                    except Exception as ex:
                        print(f"Imagen dañada en conjunto de validación: se ha omitido.")
                        image_path = self.red_n.training_dataset.filenames[step]
                        # Construir la ruta completa a la imagen dañada
                        image_path = os.path.join(self.images_folder, image_path)
                        # Obtener la ruta absoluta de la imagen dañada
                        absolute_image_path = os.path.abspath(image_path)
                        # Eliminar la imagen dañada
                        if os.path.exists(absolute_image_path):
                            os.remove(absolute_image_path)
                            print(f"Imagen dañada eliminada: {absolute_image_path}")
                            self.red_n.reload_training_ds()
                        continue
                    
                print("Entrenamiento completado sin errores en todas las épocas.")
                break
            # Guardar el modelo entrenado
            loaded_model.save(self.modelo_path)

    def predict(self, image_path):
        # Cargar el modelo desde el archivo H5
        loaded_model = load_model(filepath=self.modelo_file_path)
        # Cargar la imagen que deseas predecir
        image = functions.load_and_preprocess_image(image_path)  # Implementa esta función según tus necesidades
        # Realizar la predicción en la imagen
        prediction = loaded_model.predict(image)
        
        # Definir etiquetas descriptivas
        labels = ["Gato", "Perro"]  # Agrega etiquetas correspondientes a las clases

        # Determinar la clase predicha
        predicted_class = labels[int(round(prediction[0][0]))]

        # Imprimir la predicción
        print(f"La imagen es un: {predicted_class}")



if __name__ == "__main__":
    oTrain = Train()
    os.system('cls')
    while True:
        print("\n------  ¿Qué desea hacer?  ------")
        print("1) Entrenar (actualizar)")
        print("2) Entrenar (nuevo)")
        print("3) Predecir")
        # print("4) Chequear modelo h5")
        print("5) Salir")
        print("---------------------------------")
        opcion = int(input("Opción:"))
        if opcion == 1:
            oTrain.train()
        elif opcion == 2:
            oTrain.train(force=True, disabled_force=True)
        elif opcion == 3:
            image_path = os.path.join("predict_images", "dog6.jpg")
            absolute_image_path = os.path.abspath(image_path)
            oTrain.predict(image_path=absolute_image_path)
        # elif opcion == 4:
        #     functions.check_if_H5("model/")
        elif opcion == 5:
            break
        else:
            print("Opción inválida, intente denuevo")
            continue
    